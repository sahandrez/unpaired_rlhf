# SFT full training
# Dataset options: 
#   * HuggingFaceH4/ultrafeedback_binarized (train_prefs, test_prefs)
# Model options:
#   * Qwen/Qwen2.5-1.5B

python sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split train_prefs \
    --dataset_test_split test_prefs \
    --dataset_text_field messages \
    --output_dir logs/sft \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --gradient_checkpointing \
    --num_train_epochs=1 \
    --learning_rate 5e-6 \
    --max_seq_length 2048 \
    --packing \
    --max_steps=-1 \
    --logging_steps=10 \
    --eval_strategy steps \
    --eval_steps 50 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \