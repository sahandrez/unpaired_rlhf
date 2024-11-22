# SFT full training onn RLHF dataset
# Dataset options: 
#   * trl-lib/ultrafeedback_binarized (train, test)
# Model options:
#   * alignment-handbook/zephyr-7b-sft-qlora

python sft.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split train_prefs \
    --dataset_test_split test_prefs \
    --dataset_text_field messages \
    --output_dir logs/sft \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size=16 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --num_train_epochs=1 \
    --learning_rate 5e-6 \
    --max_seq_length 2048 \
    --max_steps=-1 \
    --logging_steps=10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj