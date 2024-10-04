import numpy as np
import pandas as pd
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
class LlamaJudge(BasePairwiseJudge):
    def __init__(self, model_name, model_revision="main", dtype="float32", temperature=0.7, 
                 max_tokens=200, top_p=0.9, tensor_parallel_size=1, sampling_params=None):
        super().__init__()
        self.llm = LLM(
            model=model_name,
            revision=model_revision,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,   # This changes the GPU support to 2
            trust_remote_code=True,
            #max_model_len=512, # uses the model default if not defined
        )
        self.tokenizer = self.llm.get_tokenizer()
        if sampling_params is None:
            self.sampling_params = SamplingParams(temperature=0.0,
                max_tokens=1024,
                top_p=0.95,
                n=1,
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                stop=[ '[[A]]', '[[B]]', '[[C]]' ], include_stop_str_in_output=True
            )
        else:
            self.sampling_params = sampling_params

        

    def get_rank(self, prompt, candidates):
        id_map = {'A':"0", 'B':"1"}
        content = self.make_prompt(prompt, candidates[0], candidates[1])
        self.sampling_params = SamplingParams(temperature=0.0,
                max_tokens=1024,
                top_p=0.95,
                n=1,
                stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                stop=[ '[[A]]', '[[B]]', '[[C]]' ], include_stop_str_in_output=True
            )
        outputs = self.llm.generate(content, self.sampling_params)
        try:
            response = outputs[0].outputs[0].text
            response = self.extract_answer(response)
            response = id_map.get(response, "error")

            if response in ["0", "1"]:
                return int(response)
            return "error"
        except:
            return "error" 


    def get_rank_batch(self, prompts, candidates):
        id_map = {'A':0, 'B':1}
        contents = []
        for prompt, candidate in zip(prompts, candidates):
            contents.append(self.make_prompt(prompt, candidate[0], candidate[1]))


        outputs = self.llm.generate(contents, self.sampling_params)
        batch_response = []
        for output in outputs:
            response = output.outputs[0].text
            response = self.extract_answer(response)
            response = id_map.get(response, "error")
            if response!="error":
                batch_response.append(response)
            else:
                print("error")

        return batch_response

    def make_prompt(self, article, answer_a, answer_b):
        JUDGE_PROMPT = ("Please act as an impartial judge and evaluate the quality of the responses provided "
                        "by two AI assistants to the user question displayed below. You should choose the "
                        "assistant that follows the user's instructions and answers the user's question better. "
                        "Your evaluation should consider factors such as the helpfulness, relevance, accuracy, "
                        "depth, creativity, and level of detail of their responses. Begin your evaluation by "
                        "comparing the two responses and provide a short explanation. Avoid any position biases "
                        "and ensure that the order in which the responses were presented does not influence your "
                        "decision. Do not allow the length of the responses to influence your evaluation. Do not "
                        "favor certain names of the assistants. Be as objective as possible. After providing your "
                        "explanation, output your final verdict by strictly following this format: \"The answer is "
                        "[[A]]\" if assistant A is better, \"The answer is [[B]]\" if assistant B is better, and "
                        "\"The answer is [[C]]\" for a tie.")
        return f"{JUDGE_PROMPT}\n[User Question]\n{article}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"

    def extract_answer(self, output, prefix = "The answer is"):
        #The answer is [[B]] or something 
        match = re.match(r'\[\[(A|B|C)\]\]', output.split(prefix)[-1].strip())
        return match.group(1) if match else "error"

    def judge(self, prompts, completions, shuffle_order=False):
        batch_response = self.get_rank_batch(prompts, completions)
        return batch_response

        # NOT USING Shuffle-order for now, requires testing
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

            ranks = []
            for p,c in zip(prompts, completions):
                r = self.get_rank(p, c)
                ranks.append(r)
            filtered_result = []
            if shuffle_order:
                for i, flip in enumerate(flip_mask):
                    if ranks[i]!="error":
                        filtered_result.append(ranks[i] if not flip else 1 - ranks[i])

            return filtered_result
        
        else:
            ranks = []
            for p,c in zip(prompts, completions):
                r = self.get_rank(p, c)
                if r!="error":
                    ranks.append(r)
            return ranks

        


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model name or path to the model to evaluate."})
    judge_model: str = field(
        default="meta-llama/Meta-Llama-3-70B-Instruct",
        metadata={
            "help": "The model name or path to the model to use as a judge. E.g., 'gpt-3.5-turbo-0125', 'meta-llama/Meta-Llama-3-70B-Instruct'."
        },
    )
    num_examples: Optional[int] = field(default=None, metadata={"help": "The number of examples to evaluate."}),
    reference_completions: str = field(default='chosen_response_ultrafeedback',metadata={"help": "TBD."})


# Parse the arguments
parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

# Load the dataset
if args.reference_completions == "chosen_response_ultrafeedback":
    raw_dataset = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="test_prefs")
    if args.num_examples is not None:
        raw_dataset = raw_dataset.select(range(args.num_examples))
    # Extract the prompts and reference completions
    prompts = raw_dataset["prompt"]
    reference_completions = raw_dataset["chosen"]
    reference_completions = [completion[1]['content'].strip() for completion in reference_completions]
    assert type(reference_completions[0])==str

elif args.reference_completions == "chosen_response_trl_lib":
    raw_dataset = load_dataset("trl-lib/tldr", split="validation")
    if args.num_examples is not None:
        raw_dataset = raw_dataset.select(range(args.num_examples))
    # Extract the prompts and reference completions
    prompts = raw_dataset["prompt"]
    reference_completions = raw_dataset["completion"]


# load pre-computed model completion 
file_name = f"model_completion/{'_'.join(args.model_name_or_path.split('/'))}.csv"
model_completions = pd.read_csv(file_name)["Text"].tolist()
completions = [[c0, c1] for c0, c1 in zip(reference_completions, model_completions)]

# initialize llm judge
judge = LlamaJudge(
    model_name=args.judge_model,
    model_revision="main",
    dtype="float16",
    tensor_parallel_size=1,
)
best_idxs = judge.judge(prompts, completions) 
model_win_rate = best_idxs.count(1) / len(best_idxs)
print(f"Model win rate: {model_win_rate:.2f}%")

# save result
df = pd.DataFrame([model_win_rate*100], columns=["Text"])
os.makedirs('model_win_rate', exist_ok=True)
file_name = f"model_win_rate/{'_'.join(args.model_name_or_path.split('/'))}.csv"
df.to_csv(file_name, index=False)