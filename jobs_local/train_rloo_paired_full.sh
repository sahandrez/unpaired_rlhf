# RLOO with paired feedback with QLoRA
# Tested with google/gemma-2-2b-it on a single A100 GPU

python rloo.py \
    --model_name_or_path google/gemma-2-2b \
    --sft_model_path sahandrez/sft-gemma-2-2b-ultrafeedback \
    --reward_model_path sahandrez/pairwise-reward-gemma-2-2b-it-ultrafeedback \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --unpaired False \
    --dataset_train_split "train_prefs" \
    --dataset_test_split "test_prefs" \
    --dataset_text_field "prompt" \
    --output_dir logs/rloo-paired \
    --num_ppo_epochs 1 \
    --rloo_k 2 \
    --num_mini_batches 1 \
    --learning_rate 5e-6 \
    --attn_implementation eager \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --local_rollout_forward_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --total_episodes 50000 \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step