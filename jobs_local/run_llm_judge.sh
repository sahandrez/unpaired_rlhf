#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export WANDB_ENTITY="unpaired_rlhf"
export WANDB_PROJECT="samin"
# export WANDB_API_KEY=""
# export HF_TOKEN=""

# -----------------------------------
# compare model with chosen response
# -----------------------------------
# model_list=("sahandrez/kto-gemma-2-2b-ultrafeedback" "sahandrez/sft-gemma-2-2b-ultrafeedback" "google/gemma-2-2b")
# for model in "${model_list[@]}"; do
#     python ./llm_judge/judge_rloo.py --model_name_or_path="${model}" --num_examples=1000 --judge_model=meta-llama/Meta-Llama-3.1-8b-Instruct
# done

# -----------------------------
# compute 1v1 response
# -----------------------------
# python ./llm_judge/judge_1v1.py --model_1=sahandrez/kto-gemma-2-2b-ultrafeedback \
#                     --model_0=sahandrez/sft-gemma-2-2b-ultrafeedback \
#                     --num_examples=1000 \
#                     --judge_model=meta-llama/Meta-Llama-3.1-8b-Instruct

# python ./llm_judge/judge_1v1.py --model_1=google/gemma-2-2b \
#                     --model_0=sahandrez/sft-gemma-2-2b-ultrafeedback \
#                     --num_examples=1000 \
#                     --judge_model=meta-llama/Meta-Llama-3.1-8b-Instruct

# python ./llm_judge/judge_1v1.py --model_1=sahandrez/kto-gemma-2-2b-ultrafeedback \
#                     --model_0=google/gemma-2-2b \
#                     --num_examples=1000 \
#                     --judge_model=meta-llama/Meta-Llama-3.1-8b-Instruct

python ./llm_judge/judge_1v1.py --model_0=sahandrez/rloo-paired-gemma-2-2b-ultrafeedback-binarized-20241010-141032 \
                    --model_1=sahandrez/sft-gemma-2-2b-ultrafeedback \
                    --num_examples=1000 \
                    --judge_model=meta-llama/Meta-Llama-3.1-8b-Instruct