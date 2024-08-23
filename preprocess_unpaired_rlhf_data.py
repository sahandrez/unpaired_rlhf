"""
Utilities to preprocess preference datasets to an unpaired format.
"""

from datasets import load_dataset, DatasetDict

from unpaired_rlhf.utils.data import (
    convert_prompt_to_chat_format,
    convert_to_kto_format,
    convert_to_unpaired_reward_format,
    remove_user_messages_in_completions,
)


def create_unpaired_rlhf_dataset(dataset_name: str, train_split: str, test_split: str):
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
    unpaired_dataset = unpaired_dataset.map(
        convert_prompt_to_chat_format,
        batched=False,
        num_proc=4,
    )

    # Convert to KTO format
    unpaired_dataset = unpaired_dataset.map(
        convert_to_kto_format,
        batched=True,
        batch_size=1,
        num_proc=4,
        remove_columns=unpaired_dataset["train"].column_names,
    )

    # Remove user messages from completions
    unpaired_dataset = unpaired_dataset.map(
        remove_user_messages_in_completions,
        batched=False,
        num_proc=4,
    )

    # Save and push to the hub
    unpaired_dataset_name = dataset_name.split("/")[-1] + "_unpaired"
    print(f"Pushing the dataset to the hub: {unpaired_dataset_name}")
    unpaired_dataset.push_to_hub(unpaired_dataset_name)


def create_unpaired_reward_dataset(
    dataset_name: str, train_split: str, test_split: str
):
    """
    Converts a preference dataset to the unpaired format and saves it to the hub.
    The unpaired format should have the following columns:
        - prompt and completion: list of prompts and completions concatenated
        - label: list of labels

    Note: Only tested on the `Anthropic/hh-rlhf` dataset.
    """
    train_dataset = load_dataset(dataset_name, split=train_split)
    test_dataset = load_dataset(dataset_name, split=test_split)

    unpaired_dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Convert to unpaired format
    unpaired_dataset = unpaired_dataset.map(
        convert_to_unpaired_reward_format,
        batched=True,
        batch_size=1,
        num_proc=4,
        remove_columns=unpaired_dataset["train"].column_names,
    )

    # Save and push to the hub
    unpaired_dataset_name = dataset_name.split("/")[-1] + "_unpaired"
    print(f"Pushing the dataset to the hub: {unpaired_dataset_name}")
    unpaired_dataset.push_to_hub(unpaired_dataset_name)


if __name__ == "__main__":
    # Convert "HuggingFaceH4/ultrafeedback_binarized" to an unpaired dataset
    dataset_name = "HuggingFaceH4/ultrafeedback_binarized"
    train_split = "train_prefs"
    test_split = "test_prefs"
    create_unpaired_rlhf_dataset(
        dataset_name=dataset_name, train_split=train_split, test_split=test_split
    )

    # Convert "Anthropic/hh-rlhf" to an unpaired dataset for the reward model
    dataset_name = "Anthropic/hh-rlhf"
    train_split = "train"
    test_split = "test"
    create_unpaired_reward_dataset(
        dataset_name=dataset_name, train_split=train_split, test_split=test_split
    )
