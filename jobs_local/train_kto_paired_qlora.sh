# KTO with paired feedback with QLoRA
# Tested with alignment-handbook/zephyr-7b-sft-qlora on a single A100 GPU

python kto.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --dataset_name sahandrez/ultrafeedback_binarized_unpaired \
    --train_split "train" \
    --test_split "test" \
    --output_dir logs/kto-zephyr-7b-sft-qlora \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --learning_rate 5.0e-6 \
    --lr_scheduler_type cosine \
    --max_length 1024 \
    --max_prompt_length 512 \
    --optim paged_adamw_32bit \
    --gradient_checkpointing \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --push_to_hub \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_r 128 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj