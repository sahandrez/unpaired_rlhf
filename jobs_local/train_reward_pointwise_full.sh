# Trains a pointwise reward model (full training)
# Dataset options: 
#   * sahandrez/ultrafeedback_unpaired (train, test)
# Model options:
#   * Qwen/Qwen2.5-1.5B
#   * sahandrez/Qwen2.5-1.5B-sft-uf

python pointwise_reward_model.py \
    --model_name_or_path sahandrez/Qwen2.5-1.5B-sft-uf \
    --dataset_name sahandrez/ultrafeedback_unpaired \
    --dataset_train_split train \
    --dataset_test_split test \
    --output_dir logs/pointwise-reward \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --max_length 2048 \
    --remove_unused_columns False \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 250 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
