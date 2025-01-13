"""
Script to train a pointwise reward model.

Scipt adapted from the TRL library:
https://github.com/huggingface/trl/blob/main/examples/scripts/reward_modeling.py
https://huggingface.co/docs/transformers/en/tasks/sequence_classification
"""

import warnings
import logging
import time
import numpy as np

import evaluate
from datasets import load_dataset, DatasetDict
from accelerate import PartialState
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    DataCollatorWithPadding,
)
from trl import (
    ModelConfig,
    RewardConfig,
    ScriptArguments,
    setup_chat_format,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.extras.dataset_formatting import conversations_formatting_function

from unpaired_rlhf.utils.runtime import set_seed
from unpaired_rlhf.trainer.utils import wrap_peft


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    # Add dataset name and a timestamp to the output directory
    training_args.output_dir += f"-{model_args.model_name_or_path.split('/')[-1]}-{script_args.dataset_name.split('/')[-1]}-{time.strftime('%Y%m%d_%H%M%S')}"
    training_args.output_dir = training_args.output_dir.replace("_", "-")
    training_args.run_name = training_args.output_dir

    # Set seed everywhere
    set_seed(training_args.seed)

    # Define label2id and id2label
    id2label = {0: "BAD", 1: "GOOD"}
    label2id = {v: k for k, v in id2label.items()}

    ################
    # Model & Tokenizer
    ################
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script."
        )

    # Get the PEFT model
    if model_args.use_peft:
        model = wrap_peft(model, training_args, get_peft_config(model_args))

    ################
    # Dataset
    ################
    logger.info("Loading the dataset...")
    dataset = load_dataset(script_args.dataset_name)
    dataset = DatasetDict(
        {
            script_args.dataset_train_split: dataset[script_args.dataset_train_split],
            script_args.dataset_test_split: dataset[script_args.dataset_test_split],
        }
    )

    def preprocess_function(examples):
        return tokenizer(examples["completion"], padding=False)

    with PartialState().local_main_process_first():
        # Wrap inputs with chat template.
        # This assumes the completion columns are in the OpenAI messages format.
        completion_fn = conversations_formatting_function(tokenizer, "completion")
        dataset = dataset.map(
            lambda x: {"completion": completion_fn(x)},
            num_proc=training_args.dataset_num_proc,
        )
        # Tokenize inputs
        dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=training_args.dataset_num_proc,
            remove_columns=["prompt", "completion"],
        )
        # Filter out examples that are too long
        dataset = dataset.filter(
            lambda x: len(x["input_ids"]) <= training_args.max_length,
            num_proc=training_args.dataset_num_proc,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    ################
    # Training
    ################
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train and push the model to the Hub
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()
    logger.info("Training complete.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(metrics)
    logger.info("Evaluation complete.")
