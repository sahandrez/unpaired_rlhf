# Trains a pairwise reward model

python pairwise_reward_model.py \
    --model_name_or_path=alignment-handbook/zephyr-7b-sft-qlora \
    --dataset_name=Anthropic/hh-rlhf \
    --output_dir=logs/reward-modeling-zephyr-7b-sft-qlora \
    --torch_dtype=bfloat16 \
    --attn_implementation=flash_attention_2 \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --num_train_epochs=1 \
    --learning_rate=1.5e-5 \
    --max_length=512 \
    --remove_unused_columns=False \
    --optim=adamw_torch \
    --gradient_checkpointing \
    --logging_steps=10 \
    --eval_strategy=steps \
    --eval_steps=500 \
    --report_to=wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_task_type SEQ_CLS \
    --lora_r=128 \
    --lora_alpha=128 \
    --lora_dropout=0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
