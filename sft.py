"""
Script for SFT training

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/sft.py
"""

import logging
import time
from tqdm.rich import tqdm

from accelerate import PartialState
from datasets import load_dataset, DatasetDict
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
from trl.extras.dataset_formatting import conversations_formatting_function
from trl.commands.cli_utils import SFTScriptArguments, TrlParser

from unpaired_rlhf.utils.runtime import set_seed


tqdm.pandas()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, config, model_config = parser.parse_args_and_config()
    config.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    config.output_dir = config.output_dir.replace("_", "-")
    config.run_name = config.output_dir

    # Set seed everywhere
    set_seed(config.seed)

    ################
    # Model init kwargs & Tokenizer
    ################
    logger.info("Loading the pretrained model...")
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False,
        torch_dtype=model_config.torch_dtype,
        attn_implementation=model_config.attn_implementation,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

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

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the columns are in the OpenAI messages format.
        completion_fn = conversations_formatting_function(
            tokenizer, config.dataset_text_field
        )
        dataset = dataset.map(
            lambda x: {config.dataset_text_field: completion_fn(x)},
            num_proc=config.dataset_num_proc,
        )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    ################
    # Evaluation
    ################
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    metrics["eval_samples"] = len(dataset[args.dataset_test_split])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
