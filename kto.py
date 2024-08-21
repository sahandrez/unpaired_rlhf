"""
Script to finetune an LLM with KTO.

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py
"""

import time
import logging
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format

from unpaired_rlhf.utils.runtime import set_seed


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    This script also works with the "trl-lib/kto-mix-14k" dataset.
    """

    dataset_name: str = "sahandrez/ultrafeedback_binarized_kto"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Add dataset name and a timestamp to the output directory
    kto_args.output_dir += (
        f"_{script_args.dataset_name.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    kto_args.run_name = kto_args.output_dir

    # Set seed everywhere
    set_seed(kto_args.seed)

    # Load a pretrained model
    logger.info("Loading the pretrained model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    ).to("cuda")
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    logger.info("Loading the dataset...")
    dataset = load_dataset(script_args.dataset_name)

    # Apply chat template
    def format_dataset(example):
        example["prompt"] = tokenizer.apply_chat_template(
            example["prompt"], tokenize=False
        )
        example["completion"] = tokenizer.apply_chat_template(
            example["completion"], tokenize=False
        )
        return example

    formatted_dataset = dataset.map(format_dataset)

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        ref_model,
        args=kto_args,
        train_dataset=formatted_dataset["train"],
        eval_dataset=formatted_dataset["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    kto_trainer.train()
    kto_trainer.save_model(kto_args.output_dir)
    if kto_args.push_to_hub:
        kto_trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = kto_trainer.evaluate()
    metrics["eval_samples"] = len(formatted_dataset["test"])
    kto_trainer.log_metrics("eval", metrics)
    kto_trainer.save_metrics("eval", metrics)
    logger.info("Evaluation complete.")
