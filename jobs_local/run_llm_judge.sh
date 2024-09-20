

# Llama3 70B model
#python llm_judge_temp.py --model_name='meta-llama/Meta-Llama-3.1-70B-Instruct' --model_dtype = "bfloat16"

# Llama3 8B model
#python llm_judge_temp.py --model_name='meta-llama/Meta-Llama-3.1-8b-Instruct'

# for trl-rloo
python llm_judge.py --dataset_name trl-lib/tldr \
                    --model_name_or_path=saminyeasar/rloo \
                    --judge_model=meta-llama/Meta-Llama-3.1-70b-Instruct \
                    --num_examples=1000



# for ultrachat-gemma
# python llm_judge.py --dataset_name HuggingFaceH4/ultrafeedback_binarized \
#                     --model_name_or_path=sahandrez/sft-gemma-2-2b-ultrafeedback \
#                     --judge_model=meta-llama/Meta-Llama-3.1-70b-Instruct \
#                     --num_examples=1000