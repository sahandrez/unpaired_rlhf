"""
Utilities for data processing.
"""


def convert_to_kto_format(batch: dict) -> dict:
    return {
        "prompt": [prompt for prompt in batch["prompt"]]
        + [prompt for prompt in batch["prompt"]],
        "completion": [chosen for chosen in batch["chosen"]]
        + [rejected for rejected in batch["rejected"]],
        "label": [True for _ in batch["chosen"]] + [False for _ in batch["rejected"]],
    }


def convert_to_unpaired_reward_format(batch: dict) -> dict:
    return {
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
