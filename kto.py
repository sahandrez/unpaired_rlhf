"""
Script to finetune an LLM with KTO.

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py
"""

import time
import logging
from dataclasses import dataclass

from datasets import load_dataset
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    get_peft_config,
    setup_chat_format,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)
from trl.extras.dataset_formatting import conversations_formatting_function

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
    dataset_train_split: str = "train"
    dataset_test_split: str = "test"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    args, config, model_config = parser.parse_args_into_dataclasses()

    # Add dataset name and a timestamp to the output directory
    config.output_dir += f"-{model_config.model_name_or_path.split('/')[-1]}-{args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    config.output_dir = config.output_dir.replace("_", "-")
    config.run_name = config.output_dir

    # Set seed everywhere
    set_seed(config.seed)

    # Load a pretrained model
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
    # model.enable_input_require_grads()
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        use_fast=True,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    logger.info("Loading the dataset...")
    dataset = load_dataset(args.dataset_name)

    # Apply chat template
    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the prompt/completion columns are in the OpenAI messages format.
        prompt_fn = conversations_formatting_function(tokenizer, "prompt")
        completion_fn = conversations_formatting_function(tokenizer, "completion")
        dataset = dataset.map(
            lambda x: {"prompt": prompt_fn(x), "completion": completion_fn(x)},
            num_proc=config.dataset_num_proc,
        )

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        ref_model,
        args=config,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=dataset[args.dataset_test_split],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    kto_trainer.train()
    kto_trainer.save_model(config.output_dir)
    if config.push_to_hub:
        kto_trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = kto_trainer.evaluate()
    metrics["eval_samples"] = len(dataset[args.dataset_test_split])
    kto_trainer.log_metrics("eval", metrics)
    kto_trainer.save_metrics("eval", metrics)
    logger.info("Evaluation complete.")
