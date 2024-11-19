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
from accelerate import PartialState
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

# from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.extras.dataset_formatting import conversations_formatting_function

from unpaired_rlhf.utils.runtime import set_seed


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    The arguments for the Reward Model training script.
    """

    dataset_name: str = "trl-internal-testing/tldr-preference-trl-style"
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
    unpaired: bool = False


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    config.output_dir = config.output_dir.replace("_", "-").replace("--", "-")
    config.run_name = config.output_dir

    # Set seed everywhere
    set_seed(config.seed)

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_config.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
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

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the chosen/rejected columns are in the OpenAI messages format.
        chosen_fn = conversations_formatting_function(tokenizer, "chosen")
        rejected_fn = conversations_formatting_function(tokenizer, "rejected")
        dataset = dataset.map(
            lambda x: {"chosen": chosen_fn(x), "rejected": rejected_fn(x)},
            num_proc=config.dataset_num_proc,
        )
        # Tokenize inputs
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=config.dataset_num_proc,
        )
        # Filter out examples that are too long
        dataset = dataset.filter(
            lambda x: len(x["input_ids_chosen"]) <= config.max_length
            and len(x["input_ids_rejected"]) <= config.max_length,
            num_proc=config.dataset_num_proc,
        )

    ################
    # Training
    ################
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=config,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        peft_config=get_peft_config(model_config),
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
