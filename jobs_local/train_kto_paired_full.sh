# KTO with paired feedback (full training)
# Tested with 

python kto.py \
    --model_name_or_path sahandrez/sft-gemma-2-2b-ultrafeedback \
    --dataset_name sahandrez/ultrafeedback_kto \
    --dataset_train_split "train" \
    --dataset_test_split "test" \
    --output_dir logs/kto \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --learning_rate 5.0e-6 \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --max_length 2048 \
    --max_prompt_length 512 \
    --optim adamw_torch \
    --gradient_checkpointing \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
