"""
Script to train a pointwise reward model.

Scipt adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
https://huggingface.co/docs/transformers/en/tasks/sequence_classification
"""

import warnings
import logging
import time
import numpy as np
from dataclasses import dataclass

import torch
from datasets import load_dataset, DatasetDict, Value
from tqdm import tqdm

import evaluate
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    DataCollatorWithPadding,
)
from trl import (
    ModelConfig,
    RewardConfig,
    setup_chat_format,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from unpaired_rlhf.utils.runtime import set_seed
from unpaired_rlhf.trainer.utils import wrap_peft


tqdm.pandas()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the pairwise reward training script.
    """

    dataset_name: str = "sahandrez/ultrafeedback_binarized_unpaired"
    train_split: str = "train"
    test_split: str = "test"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, reward_config, model_config = parser.parse_args_into_dataclasses()
    reward_config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    reward_config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    reward_config.output_dir = reward_config.output_dir.replace("_", "-")
    reward_config.run_name = reward_config.output_dir

    # Set seed everywhere
    set_seed(reward_config.seed)

    # Define label2id and id2label
    id2label = {0: "BAD", 1: "GOOD"}
    label2id = {v: k for k, v in id2label.items()}

    ################
    # Model & Tokenizer
    ################
    logger.info("Loading the pretrained model...")
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    # Get the PEFT model
    if model_config.use_peft:
        model = wrap_peft(model, reward_config, get_peft_config(model_config))

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    raw_datasets = load_dataset(script_args.dataset_name)

    raw_datasets = DatasetDict(
        {
            "train": raw_datasets[script_args.train_split],
            "test": raw_datasets[script_args.test_split],
        }
    )

    # Apply chat template if the dataset requires it
    if isinstance(raw_datasets["train"].features["completion"], list):
        logger.info("Applying chat template to the dataset...")

        def concat_prompt_completion(example):
            return {"completion": example["prompt"] + example["completion"]}

        def format_dataset(example):
            example["completion"] = tokenizer.apply_chat_template(
                example["completion"], tokenize=False
            )
            if "prompt" in example:
                example["prompt"] = tokenizer.apply_chat_template(
                    example["prompt"], tokenize=False
                )
            return example

        # Concat the prompt and completion if the dataset has a prompt column
        if "prompt" in raw_datasets["train"].features:
            raw_datasets = raw_datasets.map(
                concat_prompt_completion, remove_columns=["prompt"], batched=False
            )

        raw_datasets = raw_datasets.map(format_dataset)

    # Convert bool labels to int
    if raw_datasets["train"].features["label"].dtype == "bool":
        raw_datasets = raw_datasets.cast_column("label", Value("int64"))

        def convert_bool_to_int(example):
            example["label"] = int(example["label"])
            return example

        raw_datasets = raw_datasets.map(convert_bool_to_int)

    # Tokenize completions inputs
    def preprocess_function(examples):
        return tokenizer(examples["completion"], padding=False, truncation=True)

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=["completion"],
    )

    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids"]) <= reward_config.max_length
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    # Define the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ################
    # Training
    ################
    # Evaluation metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(reward_config.output_dir)
    if reward_config.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(metrics)
    logger.info("Evaluation complete.")
