# Unpaired RLHF

## Usages
* All scripts can be run on a single A100 GPU with QLoRA: 
* Train Zephyr-7B with KTO:
```bash
bash jobs_local/train_kto_qlora.sh
```
* Train Zephyr-7B as a pairwise reward model:
```bash
bash jobs_local/train_pairwise_reward.sh
```
* Train Zephyr-7B as a pointwise reward model:
```bash
bash jobs_local/train_pointwise_reward.sh
```

---

* Collect model completion for llm judge
```
./jobs_local/run_model_completion.sh
```

* Use LLM Judge
```
./jobs_local/run_llm_judge.sh
```
