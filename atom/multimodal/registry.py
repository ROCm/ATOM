# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal helpers shared by the OpenAI server and the offline examples.

Two model-specific hooks live here:

* :func:`get_mrope_input_positions` — request-level MRoPE positions, for models
  whose language side consumes 3D positions (Qwen3.5).
* :func:`build_multimodal_inputs` — turning chat messages + images into
  ``(input_ids, multimodal_data)``. Most Hugging Face processors follow the
  Qwen convention (``processor(text=..., images=...)`` returning already
  expanded image placeholders), which the callers implement inline; models that
  deviate register a builder below.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from atom.config import Config
from atom.utils import resolve_obj_by_qualname

_MULTIMODAL_ARCH_TO_MODEL: dict[str, str] = {
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
}

_MULTIMODAL_ARCH_TO_INPUT_BUILDER: dict[str, str] = {
    "KimiK3ForConditionalGeneration": ("atom.models.kimi_k3_vl.build_kimi_k3_inputs"),
    # DeepSeek-V4 is one architecture with an optional vision tower, so this
    # builder returns None when the config has none and the caller falls back
    # to its text path.
    "DeepseekV4ForCausalLM": ("atom.models.deepseek_v4_vl.build_deepseek_v4_inputs"),
}


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

    model_cls = resolve_obj_by_qualname(model_qualname)
    mrope_getter = getattr(model_cls, "get_mrope_input_positions", None)
    if mrope_getter is None:
        return None, 0

    return mrope_getter(atom_config, input_tokens, multimodal_data)


def build_multimodal_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict] | None:
    """Tokenize a chat + its images with the architecture's own processor API.

    Returns ``(input_ids, multimodal_data)``, or ``None`` when the architecture
    has no registered builder and the caller should fall back to the default
    Qwen-style ``processor(text=..., images=...)`` path.
    """
    hf_config = getattr(atom_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        return None

    builder_qualname = _MULTIMODAL_ARCH_TO_INPUT_BUILDER.get(architectures[0])
    if builder_qualname is None:
        return None

    builder: Callable = resolve_obj_by_qualname(builder_qualname)
    return builder(
        atom_config,
        processor,
        messages,
        images,
        chat_template_kwargs,
        tools=tools,
    )


def expand_media_placeholders(
    input_ids: Sequence[int],
    tokens_per_media: Sequence[int],
    placeholder_token_id: int,
) -> list[int]:
    """Repeat each single placeholder token into its media item's token run.

    Processors that leave the expansion to the model (Kimi-K3) emit exactly one
    placeholder per image, but ATOM needs one token per image embedding: the
    scheduler allocates KV blocks and positions from the token count, and the
    prefill scatter matches embeddings against placeholder positions.
    """
    num_placeholders = sum(1 for token in input_ids if token == placeholder_token_id)
    if num_placeholders != len(tokens_per_media):
        raise ValueError(
            f"prompt has {num_placeholders} media placeholder tokens but "
            f"{len(tokens_per_media)} media items were preprocessed"
        )

    expanded: list[int] = []
    media_index = 0
    for token in input_ids:
        if token == placeholder_token_id:
            expanded.extend([token] * tokens_per_media[media_index])
            media_index += 1
        else:
            expanded.append(token)
    return expanded
