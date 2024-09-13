"""
Script to train a pairwise reward model.

Scipt adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
"""

import warnings
import logging
import time
from dataclasses import dataclass

import torch
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    setup_chat_format,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)

from unpaired_rlhf.utils.runtime import set_seed


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

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    test_split: str = "test_prefs"


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
        model_config.model_name_or_path, num_labels=1, **model_kwargs
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
    if isinstance(raw_datasets["train"].features["chosen"], list):
        logger.info("Applying chat template to the dataset...")

        def format_dataset(example):
            example["chosen"] = tokenizer.apply_chat_template(
                example["chosen"], tokenize=False
            )
            example["rejected"] = tokenizer.apply_chat_template(
                example["rejected"], tokenize=False
            )
            return example

        raw_datasets = raw_datasets.map(format_dataset)

    # Tokenize chosen/rejected pairs of inputs
    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
            tokenized_chosen = tokenizer(chosen)
            tokenized_rejected = tokenizer(rejected)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(
                tokenized_chosen["attention_mask"]
            )
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(
                tokenized_rejected["attention_mask"]
            )

        return new_examples

    # Preprocess the dataset and filter out examples that are longer than args.max_length
    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    raw_datasets = raw_datasets.filter(
        lambda x: len(x["input_ids_chosen"]) <= reward_config.max_length
        and len(x["input_ids_rejected"]) <= reward_config.max_length
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=get_peft_config(model_config),
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
