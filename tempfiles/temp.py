from vllm import LLM, SamplingParams


def check_simple_vllm():


    prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    ]
    # prompts = ["Act as an expert in Reinforcement learning and give a brief idea what is reinforcement learning and what are important topics one should leard on them"
    # ]

    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    llm = LLM(model="meta-llama/Meta-Llama-3.1-8b-Instruct")

    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print('complete!')


def check_trl():
    from trl.trl import HfPairwiseJudge

    judge = HfPairwiseJudge()
    outputs = judge.judge(
        prompts=["What is the capital of France?", "What is the biggest planet in the solar system?"],
        completions=[["Paris", "Lyon"], ["Saturn", "Jupiter"]],
    ) 


    print('complete!')



if __name__=='__main__':
    check_simple_vllm()
    #check_trl()