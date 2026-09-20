# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal input preparation and shared media placeholder operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.multimodal import MediaLoader, MultiModalDataDict, normalize_multimodal_data
from atom.multimodal.media import load_multimodal_data
from atom.multimodal.registry import get_multimodal_input_builder

if TYPE_CHECKING:
    from atom.config import Config


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


def _bind_images_to_messages(
    conversation: list[dict[str, Any]], images: list
) -> list[dict[str, Any]]:
    """Bind the nth image marker to the nth image across all chat turns."""
    messages = []
    image_index = 0
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if part["type"] == "image":
                    if image_index >= len(images):
                        raise ValueError(
                            "Conversation has more image markers than images"
                        )
                    part = {**part, "image": images[image_index]}
                    image_index += 1
                elif part["type"] != "text":
                    raise ValueError(
                        "Expected a normalized conversation with image/text parts"
                    )
                parts.append(dict(part))
            content = parts
        # Retain the native processors' handling of assistant/tool messages.
        messages.append({**message, "content": "" if content is None else content})
    if image_index != len(images):
        raise ValueError(
            f"Conversation has {image_index} image markers but {len(images)} images"
        )
    return messages


def prepare_multimodal_inputs(
    atom_config: Config,
    processor: Any,
    prompt: str | list[dict[str, Any]],
    multi_modal_data: MultiModalDataDict,
    chat_template_kwargs: dict[str, Any] | None = None,
    tools: Any = None,
    *,
    media_loader: MediaLoader | None = None,
) -> tuple[list[int], dict[str, Any]]:
    """Prepare a prompt and separate media using the existing native builders.

    Chat callers supply a conversation with image markers, in the same order as
    multi_modal_data['image']. Qwen-style processors also accept a rendered text
    prompt, with one image placeholder per item and no chat template applied.
    Video/audio are rejected before any media is loaded.
    """
    media_items = normalize_multimodal_data(multi_modal_data)
    if not media_items:
        raise ValueError("Multimodal request did not contain any media")
    unsupported = media_items.keys() - {"image"}
    if unsupported:
        raise ValueError(
            f"Native multimodal inference does not support {', '.join(sorted(unsupported))} "
            "inputs yet; supported modalities: image"
        )
    builder = get_multimodal_input_builder(
        atom_config, is_text_prompt=isinstance(prompt, str)
    )
    media = load_multimodal_data(media_items, media_loader or MediaLoader())
    images = media["image"]
    if not isinstance(prompt, str):
        prompt = _bind_images_to_messages(prompt, images)
    return builder(
        atom_config,
        processor,
        prompt,
        images,
        dict(chat_template_kwargs or {}),
        tools=tools,
    )
