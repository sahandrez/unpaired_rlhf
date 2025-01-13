# Unpaired RLHF
This codebase explores Reinforcement Learning with Human Feedback (RLHF) using unpaired preferences. 
Unlike the standard approach, where preferences are determined by comparing two prompt completions, 
we fine-tune the model based on preferences expressed as thumbs-up or thumbs-down ratings.

## Usages
* Install the requirements from `requirements.txt`. 
* All scripts can run on a single A100-80GB GPU and have two variants, using full training or QLoRA. 

### RLHF with Paired Preferences: 
* Supervised fine-tuning (SFT):
```bash
# Full SFT training of Qwen2.5-1.5B
bash jobs_local/train_sft_full.sh

# QLoRA SFT training of Zephyr-7b
bash jobs_local/train_sft_qlora.sh
```

* **Pairwise** reward model (RM) training: 
```bash
# Full pairwise RM training of Qwen2.5-1.5B
bash jobs_local/train_reward_pairwise_full.sh

# QLoRA pairwise RM training of Zephyr-7b
bash jobs_local/train_reward_pairwise_qlora.sh
```

* RLOO training with **paired** preferences: 
```bash
# Full training of standard RLOO of Qwen2.5-1.5B
bash jobs_local/train_rloo_paired_full.sh

# QLoRA training of standard RLOO of Zephyr-7b
bash jobs_local/train_rloo_paired_qlora.sh
```

### RLHF with Unpaired Preferences:
* Use the same SFT scripts as above.

* **Pointwise** reward model (RM) training: 
```bash
# Full training of pointwise RM of Qwen2.5-1.5B
bash jobs_local/train_reward_pointwise_full.sh

# QLoRA training of pointwise RM of Zephyr-7b
bash jobs_local/train_reward_pointwise_qlora.sh
```

* RLOO training with **unpaired** preferences: 
```bash
# Full training of unpaired RLOO of Qwen2.5-1.5B
bash jobs_local/train_rloo_unpaired_full.sh

# QLoRA training of poinunpairedtwise RLOO of Zephyr-7b
bash jobs_local/train_rloo_unpaired_qlora.sh
```

* KTO:
```bash
# Full training of KTO of Qwen2.5-1.5B
bash jobs_local/train_kto_full.sh

# QLoRA training of KTO of 
bash jobs_local/train_kto_full.sh
```
