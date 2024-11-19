# SFT full training onn RLHF dataset
# Tested with google/gemma-2-2b-it on a single A100 GPU

python reward_model_tldr.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --dataset_name trl-internal-testing/tldr-preference-trl-style \
    --dataset_train_split train \
    --dataset_test_split validation \
    --output_dir logs/pairwise-reward \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --remove_unused_columns False \
    --optim adamw_torch \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 250 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --logging_first_step \
    --bf16 \
    --use_peft \
    --load_in_4bit \
    --lora_task_type SEQ_CLS \
    --lora_r 128 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj