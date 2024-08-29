import shutil
from dataclasses import dataclass
import logging

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)

from trl import ModelConfig, get_peft_config
from trl.trainer.ppov2_trainer import PPOv2Config, PPOv2Trainer
from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE

from unpaired_rlhf.utils.runtime import set_seed, log_memory_usage
from unpaired_rlhf.trainer.utils import wrap_peft


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    The arguments for the PPOv2 training script.
    """

    dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"
    train_split: str = "train_prefs"
    test_split: str = "test_prefs"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, PPOv2Config, ModelConfig))
    script_args, config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    # Set seed everywhere
    set_seed(config.seed)

    ################
    # Model & Tokenizer
    ################
    logger.info("Loading the pretrained models...")

    torch_dtype = model_config.torch_dtype
    value_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, num_labels=1, torch_dtype=torch_dtype
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path, num_labels=1, torch_dtype=torch_dtype
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, torch_dtype=torch_dtype
    )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, torch_dtype=torch_dtype
    )
    log_memory_usage(logger)

    # Get the PEFT models
    if model_config.use_peft:
        value_model = wrap_peft(value_model, config, get_peft_config(model_config))
        policy = wrap_peft(policy, config, get_peft_config(model_config))

    # Get the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    if policy.config.pad_token_id is None:
        value_model.config.pad_token_id = value_model.config.eos_token_id
        reward_model.config.pad_token_id = reward_model.config.eos_token_id
        policy.config.pad_token_id = policy.config.eos_token_id
        ref_policy.config.pad_token_id = ref_policy.config.eos_token_id

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    raw_datasets = load_dataset(script_args.dataset_name)
    train_dataset = raw_datasets[script_args.train_split]
    eval_dataset = raw_datasets[script_args.test_split]
    dataset_text_field = "prompt"

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            outputs = tokenizer(
                element[dataset_text_field],
                padding=False,
                truncation=True,
            )
            return {"input_ids": outputs["input_ids"]}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            batched=True,
            num_proc=4,  # multiprocessing.cpu_count(),
            load_from_cache_file=False,
        )

    ################
    # Training
    ################
    log_memory_usage(logger)
    logger.info("Starting training...")
    trainer = PPOv2Trainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=prepare_dataset(train_dataset, tokenizer),
        eval_dataset=prepare_dataset(eval_dataset, tokenizer),
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
