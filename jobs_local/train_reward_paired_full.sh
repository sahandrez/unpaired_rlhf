# Trains a pairwise reward model (full training)
# Tested with EleutherAI/pythia-1.4b-deduped on a single A100 GPU

python pairwise_reward_model.py \
    --model_name_or_path EleutherAI/pythia-1.4b-deduped \
    --dataset_name HuggingFaceH4/ultrafeedback_binarized \
    --train_split "train_prefs" \
    --test_split "test_prefs" \
    --output_dir logs/pairwise-reward-pythia-1.4b-deduped \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --max_length 512 \
    --remove_unused_columns False \
    --optim adamw_torch \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub \
    --bf16 \
    --logging_first_step \