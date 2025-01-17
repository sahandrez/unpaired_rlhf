"""
Script to train a pairwise reward model.

Scipt adapted from the TRL library:
https://github.com/huggingface/trl/blob/v0.13.0/examples/scripts/reward_modeling.py
"""

import warnings
import logging
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)

from unpaired_rlhf.utils.runtime import set_seed


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    training_args.output_dir += f"-{model_args.model_name_or_path.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    training_args.output_dir = training_args.output_dir.replace("_", "-")
    training_args.run_name = training_args.output_dir

    # Set seed everywhere
    set_seed(training_args.seed)

    ################
    # Model & Tokenizer
    ################
    logger.info("Loading the pretrained model...")
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    dataset = load_dataset(script_args.dataset_name)

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        peft_config=get_peft_config(model_args),
    )
    logger.info("Starting training...")
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    logger.info("Evaluation complete.")
