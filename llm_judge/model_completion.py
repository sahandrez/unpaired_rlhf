import numpy as np
import concurrent.futures
import os
from huggingface_hub import login
token = os.getenv('HF_TOKEN')
# Authenticate
login(token)
from dataclasses import dataclass, field
from typing import Optional
from datasets import load_dataset
from transformers import HfArgumentParser
from vllm import LLM, SamplingParams
from trl import HfPairwiseJudge, OpenAIPairwiseJudge
import torch
import re
from trl import BasePairwiseJudge
from vllm import LLM, SamplingParams  # Ensure these imports match your actual library usage
import argparse
import pandas as pd


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    num_examples: Optional[int] = field(default=None, metadata={"help": "The number of examples to evaluate."})


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# Load the dataset
raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
if args.num_examples is not None:
    raw_dataset = raw_dataset.select(range(args.num_examples))
# Extract the prompts and reference completions
prompts = raw_dataset["prompt"]
reference_completions = raw_dataset["chosen"]
reference_completions = [completion[1]['content'].strip() for completion in reference_completions]
assert type(reference_completions[0])==str

# ----- tldr dataset -----
# raw_dataset = load_dataset("trl-lib/tldr", split="validation")
# if args.num_examples is not None:
#     raw_dataset = raw_dataset.select(range(args.num_examples))
# # Extract the prompts and reference completions
# prompts = raw_dataset["prompt"]
# reference_completions = raw_dataset["completion"]


# Generate the model completions
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=200)  # very generous max token length
llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
model_completions = [output.outputs[0].text.strip() for output in outputs]

df = pd.DataFrame(model_completions, columns=["Text"])
os.makedirs('model_completion', exist_ok=True)
file_name = f"model_completion/{'_'.join(args.model_name_or_path.split('/'))}.csv"
df.to_csv(file_name, index=False)

# read data
# read_data = pd.read_csv(file_name)["Text"].tolist()
