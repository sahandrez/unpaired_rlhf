# Trains a pairwise reward model (full training)
# Dataset options: 
#   * trl-lib/ultrafeedback_binarized (train, test)
# Model options:
#   * Qwen/Qwen2.5-1.5B
#   * sahandrez/Qwen2.5-1.5B-sft-uf

python pairwise_reward_model.py \
    --model_name_or_path sahandrez/Qwen2.5-1.5B-sft-uf \
    --dataset_name trl-lib/ultrafeedback_binarized  \
    --dataset_train_split train \
    --dataset_test_split test \
    --output_dir logs/pairwise-reward \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --max_length 2048 \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
