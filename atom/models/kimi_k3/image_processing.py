# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 image request inputs: the prompt with its image tokens expanded and
the processor outputs the vision tower takes. Host-side only, so building a
request does not import the vision model or AITER."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.multimodal.processing import expand_media_placeholders

if TYPE_CHECKING:
    from atom.config import Config


def as_pair(value) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, Sequence) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError(f"Expected an int or a length-2 sequence, got {value!r}")


def kimi_k3_tokens_per_image(grid_thws, merge_kernel_size) -> list[int]:
    """Image-token count per grid after the ``sd2_tpool`` merge.

    The merge pools the temporal axis away and downsamples each spatial axis by
    the merge kernel, so a ``(t, h, w)`` patch grid yields ``(h // kh) * (w //
    kw)`` tokens regardless of ``t``.
    """
    kernel_h, kernel_w = as_pair(merge_kernel_size)
    grids = grid_thws.tolist() if hasattr(grid_thws, "tolist") else grid_thws
    return [(int(h) // kernel_h) * (int(w) // kernel_w) for _, h, w in grids]


def build_kimi_k3_inputs(
    atom_config: "Config",
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
