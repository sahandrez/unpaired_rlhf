# PPO with paired feedback (full training)
# Tested with EleutherAI/pythia-1b-deduped on a single A100 GPU

python ppo.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --train_split "train_prefs" \
    --test_split "test_prefs" \
    --output_dir logs/ppo-pythia-1b-deduped \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --total_episodes 1000000 \
    --local_rollout_forward_batch_size 4 \
    --non_eos_penalty \
    --report_to wandb \
    --push_to_hub \
    --bf16 \
    --logging_first_step \