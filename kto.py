"""
Script to finetune an LLM with KTO.

Script adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/kto.py
"""

import time
import logging

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from trl import (
    KTOConfig,
    KTOTrainer,
    ModelConfig,
    ScriptArguments,
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


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()

    # Add dataset name and a timestamp to the output directory
    training_args.output_dir += f"-{model_args.model_name_or_path.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    training_args.output_dir = training_args.output_dir.replace("_", "-")
    training_args.run_name = training_args.output_dir

    # Set seed everywhere
    set_seed(training_args.seed)

    # Load a pretrained model
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
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Align padding tokens between tokenizer and model
    # model.config.pad_token_id = tokenizer.pad_token_id

    # If we are aligning a base model, we use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    # Load the dataset
    logger.info("Loading the dataset...")
    dataset = load_dataset(script_args.dataset_name)

    # # Apply chat template
    # with PartialState().local_main_process_first():
    #     # Wrap inputs with chat template.
    #     # This assumes the prompt/completion columns are in the OpenAI messages format.
    #     prompt_fn = conversations_formatting_function(tokenizer, "prompt")
    #     completion_fn = conversations_formatting_function(tokenizer, "completion")
    #     dataset = dataset.map(
    #         lambda x: {"prompt": prompt_fn(x), "completion": completion_fn(x)},
    #         num_proc=training_args.dataset_num_proc,
    #     )

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    kto_trainer.train()
    kto_trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        kto_trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = kto_trainer.evaluate()
    metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
    kto_trainer.log_metrics("eval", metrics)
    kto_trainer.save_metrics("eval", metrics)
    logger.info("Evaluation complete.")
