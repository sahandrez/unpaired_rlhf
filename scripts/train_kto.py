# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
In general, the optimal configuration for KTO will be similar to that of DPO.
"""

import time
import logging
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format
from utils import log_memory_usage


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the KTO training script.
    """

    dataset_name: str = "trl-lib/kto-mix-14k"


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, KTOConfig, ModelConfig))
    script_args, kto_args, model_args = parser.parse_args_into_dataclasses()

    # Add dataset name and a timestamp to the output directory
    kto_args.output_dir += (
        f"_{script_args.dataset_name.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    kto_args.run_name = kto_args.output_dir

    # Load a pretrained model
    logger.info("Loading the pretrained model...")
    log_memory_usage(logger=logger)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    ).to("cuda")
    log_memory_usage(logger=logger)
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    ).to("cuda")
    log_memory_usage(logger=logger)

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

    # Split the dataset if necessary
    if "test" not in dataset.keys():
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

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
    log_memory_usage(logger=logger)

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
    # kto_trainer.push_to_hub()
