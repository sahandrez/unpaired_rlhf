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


def convert_prompt_to_chat_format(row: dict) -> dict:
    row["prompt"] = [{"role": "user", "content": row["prompt"]}]
    return row


def remove_user_messages_in_completions(row: dict) -> dict:
    row["completion"] = [
        message for message in row["completion"] if message["role"] == "assistant"
    ]
    return row


if __name__ == "__main__":
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"

    train_dataset = load_dataset(dataset_name, split="train_prefs")
    test_dataset = load_dataset(dataset_name, split="test_prefs")

    kto_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Remove unused columns
    kto_dataset = kto_dataset.remove_columns(
        ["prompt_id", "messages", "score_chosen", "score_rejected"]
    )

    # Convert to chat format
    kto_dataset = kto_dataset.map(
        convert_prompt_to_chat_format,
        batched=False,
        num_proc=4,
    )

    # Convert to KTO format
    kto_dataset = kto_dataset.map(
        convert_to_kto_format,
        batched=True,
        batch_size=1,
        num_proc=4,
        remove_columns=kto_dataset["train"].column_names,
    )

    # Remove user messages from completions
    kto_dataset = kto_dataset.map(
        remove_user_messages_in_completions,
        batched=False,
        num_proc=4,
    )

    # Save and push to the hub
    kto_dataset.save_to_disk("ultrafeedback_binarized_kto")
    kto_dataset.push_to_hub("ultrafeedback_binarized_kto")
