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
#from trl.trainer.rloo_trainer import RLOOConfig, RLOOTrainer
from unpaired_rlhf.trainer.rloo_trainer import RLOOConfig, RLOOTrainer

# from unpaired_rlhf.trainer.unpaired_rloo_trainer import UnpairedRLOOTrainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

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

    dataset_name: str = "trl-internal-testing/tldr-preference-sft-trl-style"
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    ################
    # Dataset
    ################
    raw_datasets = load_dataset(args.dataset_name)
    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1000))
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer.apply_chat_template(
                element["messages"][:1],
                padding=False,
                add_generation_prompt=True,
            )
            return {"input_ids": input_ids, "lengths": len(input_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            load_from_cache_file=not config.sanity_check,
            num_proc=config.dataset_num_proc,
        )

    # Compute that only on the main process for faster data processing.
    # see: https://github.com/huggingface/trl/pull/1255
    with PartialState().local_main_process_first():
        train_dataset = prepare_dataset(train_dataset, tokenizer)
        eval_dataset = prepare_dataset(eval_dataset, tokenizer)
        # filtering
        train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=config.dataset_num_proc)
        eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512, num_proc=config.dataset_num_proc)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"
    ################
    # Training
    ################
    trainer = RLOOTrainer(
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