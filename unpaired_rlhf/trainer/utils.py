import inspect
import warnings
from typing import Dict, Union

import torch.nn as nn
from transformers import (
    PreTrainedModel,
)
from trl.trainer.reward_config import RewardConfig
from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training


def wrap_peft(
    model: Union[PreTrainedModel, nn.Module], args: RewardConfig, peft_config: Dict
):
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
                    "please update to the latest version of peft to use `gradient_checkpointing_kwargs`."
                )
            elif _supports_gc_kwargs and args.gradient_checkpointing_kwargs is not None:
                prepare_model_kwargs["gradient_checkpointing_kwargs"] = (
                    args.gradient_checkpointing_kwargs
                )

            model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)

        model = get_peft_model(model, peft_config)
    return model
