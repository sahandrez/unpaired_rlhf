# Trains a pointwise reward model (full training)
# Tested with google/gemma-2-2b-it on a single A100 GPU

python pointwise_reward_model.py \
    --model_name_or_path google/gemma-2-2b-it \
    --dataset_name sahandrez/ultrafeedback_unpaired \
    --dataset_train_split "train" \
    --dataset_test_split "test" \
    --output_dir logs/pointwise-reward \
    --torch_dtype bfloat16 \
    --attn_implementation eager \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --max_length 2048 \
    --remove_unused_columns False \
    --optim adamw_torch \
    --gradient_checkpointing \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --load_best_model_at_end \
    --report_to wandb \
    --push_to_hub True \
    --bf16 \
    --logging_first_step \
