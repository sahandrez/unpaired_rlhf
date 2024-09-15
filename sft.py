"""
Script for SFT training

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import logging
import time

from trl.commands.cli_utils import SFTScriptArguments, TrlParser
from datasets import load_dataset, DatasetDict

from tqdm.rich import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    setup_chat_format,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from unpaired_rlhf.utils.runtime import set_seed


tqdm.pandas()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    # Add dataset name and a timestamp to the output directory
    training_args.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    training_args.output_dir = training_args.output_dir.replace("_", "-")
    training_args.run_name = training_args.output_dir

    # Set seed everywhere
    set_seed(training_args.seed)

    ################
    # Model init kwargs & Tokenizer
    ################
    logger.info("Loading the pretrained model...")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
        model, tokenizer = setup_chat_format(model, tokenizer)

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    raw_datasets = load_dataset(args.dataset_name)
    raw_datasets = DatasetDict(
        {
            "train": raw_datasets[args.dataset_train_split],
            "test": raw_datasets[args.dataset_test_split],
        }
    )

    # Apply chat template if the dataset requires it
    if isinstance(
        raw_datasets["train"].features[training_args.dataset_text_field],
        list,
    ):
        logger.info("Applying chat template to the dataset...")

        def format_dataset(example):
            example[training_args.dataset_text_field] = tokenizer.apply_chat_template(
                example[training_args.dataset_text_field], tokenize=False
            )
            return example

        raw_datasets = raw_datasets.map(format_dataset)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    ################
    # Evaluation
    ################
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
