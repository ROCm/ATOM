# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Architecture lookup for existing native input builders and model hooks.

Implementations are imported only when their architecture is selected.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from atom.config import Config

_MULTIMODAL_ARCH_TO_MODEL: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
    "Qwen4ExpForConditionalGeneration": (
        "atom.models.qwen4_exp.Qwen4ExpForConditionalGeneration"
    ),
}

_MULTIMODAL_ARCH_TO_INPUT_BUILDER: dict[str, str] = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.image_processing.build_inputs"
    ),
    "KimiK3ForConditionalGeneration": "atom.models.kimi_k3_vl.build_kimi_k3_inputs",
}

# Architectures whose checkpoint ships its own processor rather than one
# `AutoProcessor` can build. Selected before that fallback, so a model listed
# here never reaches it.
_MULTIMODAL_ARCH_TO_PROCESSOR: dict[str, str] = {
    "DeepseekV41ForCausalLM": (
        "atom.models.deepseek_v41.image_processing.DeepseekV41ImageProcessor"
    ),
}


def _resolve(qualname: str) -> Any:
    module, _, name = qualname.rpartition(".")
    return getattr(import_module(module), name)


def get_native_multimodal_processor(atom_config, tokenizer, encoder):
    """The checkpoint's own processor, or None to fall back to `AutoProcessor`."""
    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    factory = (
        _MULTIMODAL_ARCH_TO_PROCESSOR.get(architectures[0]) if architectures else None
    )
    return (
        None if factory is None else _resolve(factory)(atom_config, tokenizer, encoder)
    )


def get_multimodal_input_builder(
    atom_config: Config, *, is_text_prompt: bool = False
) -> Callable:
    """Select an existing builder, retaining the Qwen convention as fallback.

    The registered special builders currently require a chat conversation.
    Validate this before loading media.
    """
    hf_config = getattr(atom_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    builder = (
        _MULTIMODAL_ARCH_TO_INPUT_BUILDER.get(architectures[0])
        if architectures
        else None
    )
    if builder is not None and is_text_prompt:
        raise ValueError(
            "This model's multimodal processor requires a chat conversation"
        )
    return _resolve(builder or "atom.models.qwen3_5_vl.build_qwen_vl_inputs")


def get_mrope_input_positions(
    atom_config: Config,
    input_tokens: list[int],
    multimodal_data: dict,
) -> tuple[np.ndarray | None, int]:
    """Return request-level MRoPE positions via the model's MRoPE interface."""

    architectures = getattr(atom_config.hf_config, "architectures", None) or []
    if not architectures:
        return None, 0

    model_qualname = _MULTIMODAL_ARCH_TO_MODEL.get(architectures[0])
    if model_qualname is None:
        return None, 0

    model_cls = _resolve(model_qualname)
    mrope_getter = getattr(model_cls, "get_mrope_input_positions", None)
    if mrope_getter is None:
        return None, 0

    return mrope_getter(atom_config, input_tokens, multimodal_data)
