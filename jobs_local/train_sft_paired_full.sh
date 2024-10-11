# SFT full training onn RLHF dataset
# Tested with google/gemma-2-2b-it on a single A100 GPU

python sft.py \
    --model_name_or_path google/gemma-2-2b \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split train_sft \
    --dataset_test_split test_sft \
    --dataset_text_field chosen \
    --output_dir logs/sft \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=16 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --logging_steps=1 \
    --num_train_epochs=1 \
    --max_steps=-1 \
    --eval_strategy steps \
    --eval_steps 20 \
    --save_steps 20 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \