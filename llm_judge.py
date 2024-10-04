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


DEFAULT_PAIRWISE_SYSTEM_PROMPT = '''I require a leaderboard for various large language models. I'll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.

## Instruction

{{
    "instruction": """{prompt}""",
}}

## Model Outputs

Here are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.

{{
    {{
        "model_identifier": "0",
        "output": """{response0}"""
    }},
    {{
        "model_identifier": "1",
        "output": """{response1}"""
    }}
}}

## Task

Evaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).
'''


"""
Examples:

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --num_examples 1000
Model win rate: 31.40%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --judge_model gpt-3.5-turbo-0125 --num_examples 1000
Model win rate: 51.60%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/rloo_tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 51.20%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --num_examples 1000
Model win rate: 46.30%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --judge_model gpt-3.5-turbo-0125 --num_examples 1000
Model win rate: 52.50%

python examples/scripts/evals/judge_tldr.py --model_name_or_path vwxyzjn/ppo_tldr --judge_model gpt-4o-mini --num_examples 1000
Model win rate: 63.00%
"""



import torch
import re
from trl import BasePairwiseJudge
from vllm import LLM, SamplingParams  # Ensure these imports match your actual library usage
import argparse


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    dataset_name: str = field(default="trl-lib/tldr")
    num_examples: Optional[int] = field(default=None, metadata={"help": "The number of examples to evaluate."})


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]


# load datasets
if args.dataset_name=="HuggingFaceH4/ultrafeedback_binarized":
    raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    if args.num_examples is not None:
        raw_dataset = raw_dataset.select(range(args.num_examples))
    # Extract the prompts and reference completions
    prompts = raw_dataset["prompt"]
    reference_completions = raw_dataset["chosen"]
    reference_completions = [completion[1]['content'].strip() for completion in reference_completions]
    assert type(reference_completions[0])==str


elif args.dataset_name=="trl-lib/tldr":
    raw_dataset = load_dataset("trl-lib/tldr", split="validation")
    if args.num_examples is not None:
        raw_dataset = raw_dataset.select(range(args.num_examples))
    # Extract the prompts and reference completions
    prompts = raw_dataset["prompt"]
    reference_completions = raw_dataset["completion"]


# load llm model
llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1)
# Generate the model completions
sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=200)  # very generous max token length
outputs = llm.generate(prompts, sampling_params)
model_completions = [output.outputs[0].text.strip() for output in outputs]


# # Judge the outputs
if "gpt" in args.judge_model:
    judge = OpenAIPairwiseJudge(args.judge_model)
else:
    judge = HfPairwiseJudge(args.judge_model)

completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]


report_list = []
batch = 10
import time
for i in range(0, len(completions), batch):
    best_idxs = judge.judge(prompts[i:i+batch], completions[i:i+batch])
    model_win_rate = best_idxs.count(1) / len(best_idxs)
    print(f"Model win rate: {model_win_rate:.2f}%")
    report_list.append(model_win_rate)
    #time.sleep(2)
print(f"Model win rate: {np.mean(report_list)*100:.2f}%")