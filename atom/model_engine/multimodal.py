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
    "Glm5NextForConditionalGeneration": (
        "atom.models.glm5_next.Glm5NextMultimodalModel"
    ),
    "Qwen3_5ForConditionalGeneration": "atom.models.qwen3_5.Qwen3_5MultimodalModel",
    "Qwen3_5MoeForConditionalGeneration": (
        "atom.models.qwen3_5.Qwen3_5MoeMultimodalModel"
    ),
}

_MULTIMODAL_ARCH_TO_INPUT_BUILDER: dict[str, str] = {
    "KimiK3ForConditionalGeneration": (
        "atom.model_engine.multimodal.build_kimi_k3_inputs"
    ),
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


def _as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return (int(value[0]), int(value[1]))


def kimi_k3_tokens_per_image(grid_thws, merge_kernel_size) -> list[int]:
    """Image-token count per grid after the ``sd2_tpool`` merge.

    The merge pools the temporal axis away and downsamples each spatial axis by
    the merge kernel, so a ``(t, h, w)`` patch grid yields ``(h // kh) * (w //
    kw)`` tokens regardless of ``t``.
    """
    kernel_h, kernel_w = _as_pair(merge_kernel_size)
    grids = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    return [(int(h) // kernel_h) * (int(w) // kernel_w) for _, h, w in grids]


def build_kimi_k3_inputs(
    atom_config: Config,
    processor: Any,
    messages: list[dict],
    images: list,
    chat_template_kwargs: dict,
    tools: Any = None,
) -> tuple[list[int], dict]:
    """Build Kimi-K3 inputs via ``KimiK3Processor``.

    The K3 processor takes messages plus a separate ``medias`` list (its chat
    encoder is Python, not Jinja), returns ``grid_thws`` rather than
    ``image_grid_thw``, and emits a single ``<|media_pad|>`` per image that the
    reference model expands while merging embeddings. Normalize all three so the
    engine sees the same contract as every other multimodal model.
    """
    multimodal_config = getattr(atom_config, "multimodal_config", None)
    if multimodal_config is None:
        raise ValueError(
            "Kimi-K3 image requests need the full HF config; start the server "
            "with --trust-remote-code."
        )

    template_kwargs = dict(chat_template_kwargs)
    template_kwargs.pop("tokenize", None)
    if tools:
        template_kwargs["tools"] = tools

    medias = [{"type": "image", "image": image} for image in images]
    inputs = processor(
        messages=messages,
        medias=medias,
        return_tensors="pt",
        **template_kwargs,
    )

    grid_thws = inputs["grid_thws"]
    input_ids = inputs["input_ids"][0].tolist()
    placeholder_token_id = int(
        getattr(multimodal_config, "media_placeholder_token_id", 163605)
    )
    tokens_per_image = kimi_k3_tokens_per_image(
        grid_thws, multimodal_config.vision_config.merge_kernel_size
    )
    input_ids = expand_media_placeholders(
        input_ids, tokens_per_image, placeholder_token_id
    )

    multimodal_data = {
        "pixel_values": inputs["pixel_values"],
        "image_grid_thw": grid_thws,
    }
    return input_ids, multimodal_data
