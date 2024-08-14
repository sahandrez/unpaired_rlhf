"""
Script to preprocess the any preference dataset for the KTO training script.
"""

from datasets import load_dataset, DatasetDict


def convert_to_kto_format(batch: dict) -> dict:
    return {
        "prompt": [prompt for prompt in batch["prompt"]]
        + [prompt for prompt in batch["prompt"]],
        "completion": [chosen for chosen in batch["chosen"]]
        + [rejected for rejected in batch["rejected"]],
        "label": [True for _ in batch["chosen"]] + [False for _ in batch["rejected"]],
    }


if __name__ == "__main__":
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"

    train_dataset = load_dataset(dataset_name, split="train_prefs")
    test_dataset = load_dataset(dataset_name, split="test_prefs")

    # Remove unused columns
    train_dataset = train_dataset.remove_columns(
        ["prompt_id", "messages", "score_chosen", "score_rejected"]
    )
    test_dataset = test_dataset.remove_columns(
        ["prompt_id", "messages", "score_chosen", "score_rejected"]
    )

    # Convert to KTO format
    kto_train_dataset = train_dataset.map(
        convert_to_kto_format,
        batched=True,
        batch_size=1,
        num_proc=4,
        remove_columns=train_dataset.column_names,
    )
    kto_test_dataset = test_dataset.map(
        convert_to_kto_format,
        batched=True,
        batch_size=1,
        num_proc=4,
        remove_columns=test_dataset.column_names,
    )

    # Save and push to the hub
    kto_dataset = DatasetDict({"train": kto_train_dataset, "test": kto_test_dataset})
    kto_dataset.save_to_disk("ultrafeedback_binarized_kto")
    kto_dataset.push_to_hub("ultrafeedback_binarized_kto")
