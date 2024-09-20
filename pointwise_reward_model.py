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

import torch
from tqdm import tqdm

import evaluate
from datasets import load_dataset, DatasetDict
from accelerate import PartialState
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
from trl.commands.cli_utils import RewardScriptArguments
from trl.extras.dataset_formatting import conversations_formatting_function

from unpaired_rlhf.utils.runtime import set_seed
from unpaired_rlhf.trainer.utils import wrap_peft


tqdm.pandas()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = HfArgumentParser((RewardScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    config.output_dir = config.output_dir.replace("_", "-")
    config.run_name = config.output_dir

    # Set seed everywhere
    set_seed(config.seed)

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
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=torch_dtype,
        attn_implementation=model_config.attn_implementation,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_config.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    # Get the PEFT model
    if model_config.use_peft:
        model = wrap_peft(model, config, get_peft_config(model_config))

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    dataset = load_dataset(args.dataset_name)
    dataset = DatasetDict(
        {
            args.dataset_train_split: dataset[args.dataset_train_split],
            args.dataset_test_split: dataset[args.dataset_test_split],
        }
    )

    def preprocess_function(examples):
        return tokenizer(examples["completion"], padding=False)

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the completion columns are in the OpenAI messages format.
        completion_fn = conversations_formatting_function(tokenizer, "completion")
        dataset = dataset.map(
            lambda x: {"completion": completion_fn(x)},
            num_proc=config.dataset_num_proc,
        )
        # Tokenize inputs
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
            remove_columns=["prompt", "completion"],
        )
        # Filter out examples that are too long
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ################
    # Training
    ################
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(metrics)
    logger.info("Evaluation complete.")
