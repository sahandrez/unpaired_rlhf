"""
Script to finetune an LLM with RLOO.

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/rloo/rloo.py
"""

import logging
import time
from dataclasses import dataclass
import wandb

from accelerate import PartialState
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, get_peft_config, setup_chat_format
from trl.extras.dataset_formatting import conversations_formatting_function
from unpaired_rlhf.trainer.rloo_trainer import RLOOTrainer
from unpaired_rlhf.trainer.rloo_config import RLOOConfig

# from unpaired_rlhf.trainer.unpaired_rloo_trainer import UnpairedRLOOTrainer
from unpaired_rlhf.trainer.utils import wrap_peft
from unpaired_rlhf.utils.runtime import set_seed


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    The arguments for the RLOO training script.
    """

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    dataset_train_split: str = "train_prefs"
    dataset_test_split: str = "test_prefs"
    dataset_text_field: str = "prompt"
    unpaired: bool = False


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RLOOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()

    # Add dataset name and a timestamp to the output directory
    config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    config.output_dir = config.output_dir.replace("_", "-")
    config.run_name = config.output_dir

    # RLOO Trainer does not suppor run_name
    if "wandb" in config.report_to:
        wandb.init(name=config.run_name)

    # Set seed everywhere
    set_seed(config.seed)

    # Unpaired or paired feedback setup
    if args.unpaired:
        raise NotImplementedError("Unpaired feedback is not yet supported.")
    else:
        trainer_cls = RLOOTrainer
        num_labels = 1

    ################
    # Model & Tokenizer
    ################
    logger.info("Loading the pretrained models...")
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        use_cache=False,
        torch_dtype=model_config.torch_dtype,
        attn_implementation=model_config.attn_implementation,
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, num_labels=num_labels, **model_kwargs
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, **model_kwargs
    )
    policy = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)

    # Get the PEFT models
    if model_config.use_peft:
        policy = wrap_peft(policy, config, get_peft_config(model_config))

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )

    # Align padding tokens between tokenizer and model
    reward_model.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        reward_model, tokenizer = setup_chat_format(reward_model, tokenizer)
        ref_policy, _ = setup_chat_format(ref_policy, tokenizer)
        policy, _ = setup_chat_format(policy, tokenizer)

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    eval_samples = 20
    dataset = load_dataset(args.dataset_name)
    train_dataset = dataset[args.dataset_train_split]
    eval_dataset = dataset[args.dataset_test_split]
    eval_dataset = eval_dataset.select(range(eval_samples))

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[args.dataset_text_field],
                padding=False,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=config.dataset_num_proc,
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
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
