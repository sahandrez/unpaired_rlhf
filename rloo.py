"""
Script to finetune an LLM with RLOO.

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/v0.13.0/examples/scripts/rloo/rloo.py
"""

import logging
import time
import wandb
from dataclasses import dataclass

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from trl import (
    ModelConfig,
    RLOOConfig,
    RLOOTrainer,
    ScriptArguments,
    get_peft_config,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from unpaired_rlhf.trainer.unpaired_rloo_trainer import UnpairedRLOOTrainer
from unpaired_rlhf.trainer.utils import wrap_peft
from unpaired_rlhf.utils.runtime import set_seed


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AdditionalArguments:
    """Additional arguments for the RLOO script."""

    unpaired: bool = False


if __name__ == "__main__":
    parser = HfArgumentParser(
        (AdditionalArguments, ScriptArguments, RLOOConfig, ModelConfig)
    )
    addtional_args, script_args, training_args, model_args = (
        parser.parse_args_into_dataclasses()
    )

    # Add dataset name and a timestamp to the output directory
    training_args.output_dir += f"-{model_args.model_name_or_path.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    training_args.output_dir = training_args.output_dir.replace("_", "-")
    training_args.run_name = training_args.output_dir

    # RLOO Trainer does not suppor run_name
    if "wandb" in training_args.report_to:
        wandb.init(name=training_args.run_name)

    # Set seed everywhere
    set_seed(training_args.seed)

    # Unpaired or paired feedback setup
    if addtional_args.unpaired:
        trainer_cls = UnpairedRLOOTrainer
        num_labels = 2
    else:
        trainer_cls = RLOOTrainer
        num_labels = 1

    ################
    # Model & Tokenizer
    ################
    logger.info("Loading the pretrained models...")
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
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        training_args.reward_model_path, num_labels=num_labels, **model_kwargs
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, **model_kwargs
    )
    policy = AutoModelForCausalLM.from_pretrained(
        training_args.sft_model_path, **model_kwargs
    )

    # Get the PEFT models
    if model_args.use_peft:
        policy = wrap_peft(policy, training_args, get_peft_config(model_args))

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    eval_samples = 100
    dataset = load_dataset(script_args.dataset_name)
    train_dataset = dataset[script_args.dataset_train_split]
    eval_dataset = dataset[script_args.dataset_test_split]
    eval_dataset = eval_dataset.select(range(eval_samples))
    dataset_text_field = "prompt"
    assert (
        dataset_text_field in train_dataset.column_names
    ), f"Dataset does not have the {dataset_text_field} text field."

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=training_args.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    logger.info("Starting training...")
    trainer = trainer_cls(
        config=training_args,
        processing_class=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
