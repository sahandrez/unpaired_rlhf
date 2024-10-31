# SFT full training onn RLHF dataset
# Tested with google/gemma-2-2b-it on a single A100 GPU

python sft.py \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --dataset_name trl-internal-testing/tldr-preference-sft-trl-style \
    --dataset_train_split train \
    --dataset_test_split validation \
    --dataset_text_field chosen \
    --output_dir logs/sft \
    --torch_dtype bfloat16 \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=8 \
    --gradient_accumulation_steps=16 \
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
    --logging_first_step \
    --attn_implementation eager \
    --bf16 \