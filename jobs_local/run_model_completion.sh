#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export WANDB_ENTITY="unpaired_rlhf"
export WANDB_PROJECT="samin"
# export WANDB_API_KEY=""
export HF_TOKEN="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"

# model_list=("sahandrez/kto-gemma-2-2b-ultrafeedback" "sahandrez/sft-gemma-2-2b-ultrafeedback" "google/gemma-2-2b")
model_list=("sahandrez/rloo-paired-gemma-2-2b-ultrafeedback-binarized-20241010-141032")
for model in "${model_list[@]}"; do
    python ./llm_judge/model_completion.py --model_name_or_path="${model}" --num_examples=1000
done