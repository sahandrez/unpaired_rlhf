# RLOO with paired feedback with QLoRA
# Tested with google/gemma-2-2b-it on a single A100 GPU

python rloo.py \
    --model_name_or_path google/gemma-2b-it \
    --sft_model_path google/gemma-2b-it \
    --reward_model_path sahandrez/pairwise-reward-gemma-2b-it-ultrafeedback \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --unpaired False \
    --train_split "train_prefs" \
    --test_split "test_prefs" \
    --output_dir logs/rloo-paired \
    --num_ppo_epochs 1 \
    --rloo_k 2 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --local_rollout_forward_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --total_episodes 100000 \
    --non_eos_penalty \
    --report_to wandb \
    --push_to_hub \
    --bf16 \
    --logging_first_step \
