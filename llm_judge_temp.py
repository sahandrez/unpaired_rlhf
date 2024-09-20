import re
from trl.trl import BasePairwiseJudge
from vllm import LLM, SamplingParams  # Ensure these imports match your actual library usage
import argparse

class LlamaJudge(BasePairwiseJudge):
    def __init__(self, model_name, model_revision, dtype, temperature, max_tokens, top_p, tensor_parallel_size=1):
        super().__init__()
        self.llm = LLM(
            model=model_name,
            revision=model_revision,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,   # This changes the GPU support to 2
            trust_remote_code=True,
            max_model_len=1024,
        )
        self.tokenizer = self.llm.get_tokenizer()
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            n=1,
            stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            #stop_token_ids=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("")]
            # stop_token_ids=[self.tokenizer.eos_token_id]  # Only use EOS token ID, 
            # By including the appropriate stop tokens, it ensures that the generated responses are of the desired length and quality.
        )

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

        # JUDGE_PROMPT = "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"The answer is [[A]]\" if assistant A is better, \"The answer is [[B]]\" if assistant B is better, and \"The answer is [[C]]\" for a tie."
        # preamble = JUDGE_PROMPT
        # prompt = f"{preamble}\n[User Question]\n{article}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"
        # return prompt



    def extract_answer(self, output, prefix = "The answer is"):
        #The answer is [[B]] or something 
        match = re.match(r'\[\[(A|B|C)\]\]', output.split(prefix)[-1].strip())
        return match.group(1) if match else ""

    def judge(self, prompts, completions, shuffle_order=False):
        # Generate the prompt
        prompt = self.make_prompt(prompts['question'], completions['a'], completions['b'])
        
        # Generate the evaluation using VLLM
        # sample_prompt = 'I require a leaderboard for various large language models. I\'ll provide you with prompts given to these models and their corresponding outputs. Your task is to assess these responses, and select the model that produces the best output from a human perspective.\n\n## Instruction\n\n{\n    "instruction": """What is the capital of France?""",\n}\n\n## Model Outputs\n\nHere are the unordered outputs from the models. Each output is associated with a specific model, identified by a unique model identifier.\n\n{\n    {\n        "model_identifier": "0",\n        "output": """Paris"""\n    },\n    {\n        "model_identifier": "1",\n        "output": """Lyon"""\n    }\n}\n\n## Task\n\nEvaluate the models on the basis of the quality and relevance of their results, and select the model that generated the best result. Reply with the identifier of the best model. Our evaluation will only take into account the first character of your answer, so make sure it contains only one of the identifiers and nothing else (no quotation marks, no spaces, no new lines, ...).\n'
        # sample_outputs = self.llm.generate(sample_prompt)
        
        outputs = self.llm.generate(prompt, self.sampling_params)
        # Print the outputs.
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            #print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        # Extract the answer from the result
        return self.extract_answer(generated_text)

    


def main():
    parser = argparse.ArgumentParser(description="Judge AI responses using LLaMA model.")
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8b-Instruct', help="Name of the LLaMA model.")
    parser.add_argument('--model_revision', type=str, default='main', help="Revision of the LLaMA model.")
    parser.add_argument('--model_dtype', type=str, default='float32', help="Data type for the model.")
    parser.add_argument('--model_temperature', type=float, default=0.7, help="Temperature parameter for sampling.")
    parser.add_argument('--max_new_tokens', type=int, default=50, help="Maximum number of new tokens to generate.")
    parser.add_argument('--top_p', type=float, default=0.9, help="Top-p parameter for sampling.")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="number of GPUs to utilize")



    args = parser.parse_args()

    # Create an instance of LlamaJudge with these arguments
    judge = LlamaJudge(
        model_name=args.model_name,
        model_revision=args.model_revision,
        dtype=args.model_dtype,
        temperature=args.model_temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Example input for judging
    prompts = {
        'question': 'What is the capital of France?'
    }
    completions = {
        'a': 'Paris is the capital of France.',
        'b': 'The capital city of France is Paris.'
    }

    # Get the evaluation result
    result = judge.judge(prompts, completions)
    print(f"The better answer is: {result}")


if __name__ == "__main__":
    main()
