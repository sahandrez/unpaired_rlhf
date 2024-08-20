# Unpaired RLHF

## Usages
* Train Zephyr-7B with KTO on a single A100 GPU with QLoRA: 
```bash
bash jobs_local/train_kto_qlora.sh
```
* Preprocess the pairwise UltraFeedback dataset to an unpaired dataset: 
```bash
python preprocess_data.py
```
