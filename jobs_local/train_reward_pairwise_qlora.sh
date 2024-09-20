# Trains a pairwise reward model with QLoRA
# Tested with alignment-handbook/zephyr-7b-sft-qlora on a single A100 GPU

python pairwise_reward_model.py \
    --model_name_or_path alignment-handbook/zephyr-7b-sft-qlora \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --dataset_train_split "train_prefs" \
    --dataset_test_split "test_prefs" \
    --output_dir logs/pairwise-reward \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1.5e-5 \
    --max_length 2048 \
    --remove_unused_columns False \
    --optim adamw_torch \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 200 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_task_type SEQ_CLS \
    --lora_r 128 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
