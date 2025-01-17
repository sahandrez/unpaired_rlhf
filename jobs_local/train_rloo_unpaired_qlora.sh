# RLOO with paired feedback with QLoRA
# Dataset options: 
#   * HuggingFaceH4/ultrafeedback_binarized (train_prefs, test_prefs)
# Model options:
#   * alignment-handbook/zephyr-7b-sft-qlora

python rloo.py \
    --unpaired True \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --sft_model_path alignment-handbook/zephyr-7b-sft-qlora \
    --reward_model_path sahandrez/pointwise-reward-zephyr-7b-sft-qlora-uf \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --unpaired False \
    --dataset_train_split "train_prefs" \
    --dataset_test_split "test_prefs" \
    --dataset_text_field "prompt" \
    --output_dir logs/rloo-unpaired \
    --num_ppo_epochs 1 \
    --rloo_k 2 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --local_rollout_forward_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing \
    --total_episodes 50000 \
    --missing_eos_penalty 1.0 \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_r 128 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj