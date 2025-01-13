import warnings
import inspect
from typing import Union

import torch.nn as nn
from transformers import (
    PreTrainedModel,
)
from transformers.utils import is_peft_available
from trl.trainer.reward_config import RewardConfig
from peft import (
    PeftModel,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def wrap_peft(
    model: Union[PreTrainedModel, nn.Module],
    args: RewardConfig,
    peft_config: PeftConfig,
):
    """
    Wraps the model in a PeftModel if it is not already a PeftModel.
    Code is adapted from the SFTTrainer:
    https://github.com/huggingface/trl/blob/v0.13.0/trl/trainer/reward_trainer.py
    """

    if not is_peft_available() and peft_config is not None:
        raise ValueError(
            "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
        )

    if is_peft_available() and peft_config is not None:
        if not isinstance(model, PeftModel):
            if getattr(model, "is_loaded_in_8bit", False) or getattr(
                model, "is_quantized", False
            ):
                _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {
                    "use_gradient_checkpointing": args.gradient_checkpointing
                }

                if (
                    not _supports_gc_kwargs
                    and args.gradient_checkpointing_kwargs is not None
                ):
                    warnings.warn(
                        "You passed `gradient_checkpointing_kwargs` in the trainer's kwargs, but your peft version does not support it. "
                        "please update to the latest version of peft to use `gradient_checkpointing_kwargs`.",
                        UserWarning,
                    )
                elif (
                    _supports_gc_kwargs
                    and args.gradient_checkpointing_kwargs is not None
                ):
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                        args.gradient_checkpointing_kwargs
                    )

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

            model = get_peft_model(model, peft_config)
    return model
