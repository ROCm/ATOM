# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Multimodal input preparation and shared media placeholder operations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from atom.multimodal import MediaLoader, MultiModalDataDict, normalize_multimodal_data
from atom.multimodal.media import load_multimodal_data
from atom.multimodal.registry import (
    get_multimodal_input_builder,
    get_multimodal_modalities,
)

if TYPE_CHECKING:
    from atom.config import Config


def expand_media_placeholders(
    input_ids: Sequence[int],
    tokens_per_media: Sequence[int],
    placeholder_token_id: int,
) -> tuple[list[int], tuple[tuple[int, int], ...]]:
    """Repeat each single placeholder token into its media item's token run.

    Processors that leave the expansion to the model (Kimi-K3) emit exactly one
    placeholder per image, but ATOM needs one token per image embedding: the
    scheduler allocates KV blocks and positions from the token count, and the
    prefill scatter matches embeddings against placeholder positions.

    Also returns each run as ``(start, count)`` in the expanded ids, in media
    order: the ``embedding_spans`` that let an image prefill be chunked.
    """
    num_placeholders = sum(1 for token in input_ids if token == placeholder_token_id)
    if num_placeholders != len(tokens_per_media):
        raise ValueError(
            f"prompt has {num_placeholders} media placeholder tokens but "
            f"{len(tokens_per_media)} media items were preprocessed"
        )

    expanded: list[int] = []
    spans: list[tuple[int, int]] = []
    media_index = 0
    for token in input_ids:
        if token == placeholder_token_id:
            count = int(tokens_per_media[media_index])
            spans.append((len(expanded), count))
            expanded.extend([token] * count)
            media_index += 1
        else:
            expanded.append(token)
    return expanded, tuple(spans)


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
    Video is accepted for architectures the registry lists with it (chat
    conversations only); audio is rejected before any media is loaded.
    """
    media_items = normalize_multimodal_data(multi_modal_data)
    if not media_items:
        raise ValueError("Multimodal request did not contain any media")
    builder = get_multimodal_input_builder(
        atom_config, is_text_prompt=isinstance(prompt, str)
    )
    supported = get_multimodal_modalities(atom_config)
    unsupported = media_items.keys() - supported
    if unsupported:
        raise ValueError(
            f"Native multimodal inference does not support {', '.join(sorted(unsupported))} "
            f"inputs yet; supported modalities: {', '.join(sorted(supported))}"
        )
    media = load_multimodal_data(media_items, media_loader or MediaLoader())
    images = media.get("image", [])
    videos = media.get("video", [])
    if not isinstance(prompt, str):
        # Videos reach the builder bound to their conversation parts.
        prompt = _bind_media_to_messages(prompt, images, videos)
    return builder(
        atom_config,
        processor,
        prompt,
        images,
        dict(chat_template_kwargs or {}),
        tools=tools,
    )


def _bind_media_to_messages(
    conversation: list[dict[str, Any]], images: list, videos: list = ()
) -> list[dict[str, Any]]:
    """Bind the nth image (and video) marker to the nth item, across all turns."""
    items = {"image": list(images), "video": list(videos)}
    used = {"image": 0, "video": 0}
    allowed = "image/video/text" if videos else "image/text"
    messages = []
    for message in conversation:
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                kind = part["type"]
                if kind == "image" or (kind == "video" and videos):
                    if used[kind] >= len(items[kind]):
                        raise ValueError(
                            f"Conversation has more {kind} markers than {kind}s"
                        )
                    part = {**part, kind: items[kind][used[kind]]}
                    used[kind] += 1
                elif kind != "text":
                    raise ValueError(
                        f"Expected a normalized conversation with {allowed} parts"
                    )
                parts.append(dict(part))
            content = parts
        # Retain the native processors' handling of assistant/tool messages.
        messages.append({**message, "content": "" if content is None else content})
    for kind, values in items.items():
        if used[kind] != len(values):
            raise ValueError(
                f"Conversation has {used[kind]} {kind} markers but {len(values)} {kind}s"
            )
    return messages
