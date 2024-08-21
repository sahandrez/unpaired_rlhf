# Trains a pairwise reward model

python pairwise_reward_model.py \
    --model_name_or_path=facebook/opt-350m \
    --dataset_name=Anthropic/hh-rlhf \
    --output_dir=logs/reward-modeling-opt-350m \
    --per_device_train_batch_size=16 \
    --num_train_epochs=1 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=True \
    --learning_rate=1.41e-5 \
    --report_to=wandb \
    --remove_unused_columns=False \
    --optim=adamw_torch \
    --logging_steps=10 \
    --eval_strategy=steps \
    --eval_steps=500 \
    --max_length=512 
