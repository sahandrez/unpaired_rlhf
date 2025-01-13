# RLOO full training with paired feedback 
# Dataset options: 
#   * HuggingFaceH4/ultrafeedback_binarized (train_prefs, test_prefs)
# Model options:
#   * Qwen/Qwen2.5-1.5B

python rloo.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --sft_model_path sahandrez/Qwen2.5-1.5B-sft-uf \
    --reward_model_path sahandrez/pairwise-reward-Qwen2.5-1.5B-sft-uf \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split "train_prefs" \
    --dataset_test_split "test_prefs" \
    --output_dir logs/rloo-paired \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --num_ppo_epochs 1 \
    --rloo_k 2 \
    --num_mini_batches 1 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --local_rollout_forward_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --total_episodes 50000 \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
