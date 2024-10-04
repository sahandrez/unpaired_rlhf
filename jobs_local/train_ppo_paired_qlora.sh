# PPO with paired feedback with QLoRA
# Tested with alignment-handbook/zephyr-7b-sft-qlora on a single A100 GPU

python ppo.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --sft_model_path alignment-handbook/zephyr-7b-sft-qlora \
    --reward_model_path sahandrez/pairwise-reward-zephyr-7b-sft-qlora-ultrafeedback \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --train_split "train_prefs" \
    --test_split "test_prefs" \
    --output_dir logs/ppo \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --local_rollout_forward_batch_size 1 \
    --gradient_checkpointing \
    --total_episodes 10000 \
    --non_eos_penalty \
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