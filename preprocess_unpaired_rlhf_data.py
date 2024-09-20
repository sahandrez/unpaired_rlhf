"""
Utilities to preprocess preference datasets to an unpaired format.
"""

from datasets import load_dataset, DatasetDict


def convert_to_kto_format(batch: dict) -> dict:
    return {
        "prompt": [prompt for prompt in batch["prompt"]]
        + [prompt for prompt in batch["prompt"]],
        "completion": [chosen for chosen in batch["chosen"]]
        + [rejected for rejected in batch["rejected"]],
        "label": [1 for _ in batch["chosen"]] + [0 for _ in batch["rejected"]],
    }


def convert_to_unpaired_reward_format(batch: dict) -> dict:
    return {
        "completion": [chosen for chosen in batch["chosen"]]
        + [rejected for rejected in batch["rejected"]],
        "label": [1 for _ in batch["chosen"]] + [0 for _ in batch["rejected"]],
    }


def convert_prompt_to_chat_format(row: dict) -> dict:
    row["prompt"] = [{"role": "user", "content": row["prompt"]}]
    return row


def remove_user_messages_in_completions(row: dict) -> dict:
    row["completion"] = [
        message for message in row["completion"] if message["role"] == "assistant"
    ]
    return row


def create_unpaired_rlhf_dataset(
    dataset_name: str,
    new_dataset_name: str,
    train_split: str,
    test_split: str,
    remove_user_messages: bool = False,
    convert_prompt_to_chat: bool = False,
):
    """
    Converts a preference dataset to the KTO format and saves it to the hub.
    The KTO format should have the following columns:
        - prompt: list of prompts
        - completion: list of completions
        - label: list of labels

    More details: https://huggingface.co/docs/trl/kto_trainer

    Note: Only tested on the `HuggingFaceH4/ultrafeedback_binarized` dataset.
    """
    train_dataset = load_dataset(dataset_name, split=train_split)
    test_dataset = load_dataset(dataset_name, split=test_split)

    unpaired_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Remove unused columns
    unpaired_dataset = unpaired_dataset.remove_columns(
        ["prompt_id", "messages", "score_chosen", "score_rejected"]
    )

    # Convert to chat format
    if convert_prompt_to_chat:
        unpaired_dataset = unpaired_dataset.map(
            convert_prompt_to_chat_format,
            batched=False,
        )

    # Convert to KTO format
    unpaired_dataset = unpaired_dataset.map(
        convert_to_kto_format,
        batched=True,
        batch_size=1,
        remove_columns=unpaired_dataset["train"].column_names,
    )

    # Remove user messages from completions
    if remove_user_messages:
        unpaired_dataset = unpaired_dataset.map(
            remove_user_messages_in_completions,
            batched=False,
        )

    # Save and push to the hub
    print(f"Pushing the dataset to the hub: {new_dataset_name}")
    unpaired_dataset.push_to_hub(new_dataset_name)


if __name__ == "__main__":
    # Convert "HuggingFaceH4/ultrafeedback_binarized" to an unpaired dataset for the KTO model
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    new_dataset_name = "ultrafeedback_kto"
    train_split = "train_prefs"
    test_split = "test_prefs"
    create_unpaired_rlhf_dataset(
        dataset_name=dataset_name,
        new_dataset_name=new_dataset_name,
        train_split=train_split,
        test_split=test_split,
        remove_user_messages=True,
        convert_prompt_to_chat=True,
    )

    # Convert "HuggingFaceH4/ultrafeedback_binarized" to a general purpose unpaired dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    new_dataset_name = "ultrafeedback_unpaired"
    train_split = "train_prefs"
    test_split = "test_prefs"
    create_unpaired_rlhf_dataset(
        dataset_name=dataset_name,
        new_dataset_name=new_dataset_name,
        train_split=train_split,
        test_split=test_split,
        remove_user_messages=False,
        convert_prompt_to_chat=True,
    )
