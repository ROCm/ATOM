# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Adapt chat content parts to a conversation and separate media data."""

import base64
import binascii
from collections.abc import Mapping, Sequence
from typing import Any

from atom.multimodal import EncodedMedia, Modality, MultiModalDataItems

_PART_MODALITIES: dict[str, Modality] = {
    "image": "image",
    "image_url": "image",
    "video": "video",
    "video_url": "video",
    "audio": "audio",
    "audio_url": "audio",
    "input_audio": "audio",
}


def has_multimodal_content(messages: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        isinstance(part, Mapping)
        and isinstance(part.get("type"), str)
        and part["type"] in _PART_MODALITIES
        for message in messages
        if isinstance(message.get("content"), list)
        for part in message["content"]
    )


def _parse_media_part(part: Mapping[str, Any]) -> tuple[Modality, Any, dict[str, Any]]:
    part_type = part["type"]
    modality = _PART_MODALITIES[part_type]
    data = part.get(part_type)
    options = {k: v for k, v in part.items() if k not in ("type", part_type)}
    if part_type.endswith("_url"):
        if not isinstance(data, Mapping) or not isinstance(data.get("url"), str):
            raise ValueError(f"{part_type} must include {part_type}.url")
        options.update({k: v for k, v in data.items() if k != "url"})
        data = data["url"]
    elif part_type == "input_audio":
        if (
            not isinstance(data, Mapping)
            or not isinstance(data.get("data"), str)
            or not isinstance(data.get("format"), str)
            or not data["format"]
        ):
            raise ValueError("input_audio must include base64 data and format")
        try:
            data = EncodedMedia(
                base64.b64decode(data["data"], validate=True), data["format"]
            )
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Invalid base64 data for input_audio") from exc
    if data is None:
        raise ValueError(f"{part_type} must include media data")
    return modality, data, options


def parse_chat_messages(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], MultiModalDataItems]:
    """Extract media in encounter order, leaving typed markers in the chat.

    The nth marker of a modality refers to the nth item of that modality across
    the whole conversation. Metadata stays on messages/content parts. This
    adapter performs no network/file I/O or media decoding.
    """
    conversation = []
    media: MultiModalDataItems = {}
    for message in messages:
        if not isinstance(message, Mapping):
            raise ValueError("Messages must be mappings")  # noqa: TRY004
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("Each message must have a non-empty role")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, Mapping):
                    raise ValueError(  # noqa: TRY004
                        "Message content parts must be mappings"
                    )
                part_type = part.get("type")
                if part_type == "text":
                    if not isinstance(part.get("text"), str):
                        raise ValueError("text content part must include text")
                    parts.append(dict(part))
                elif isinstance(part_type, str) and part_type in _PART_MODALITIES:
                    modality, data, options = _parse_media_part(part)
                    media.setdefault(modality, []).append(data)
                    parts.append({**options, "type": modality})
                else:
                    raise ValueError(f"Unsupported content part type: {part_type!r}")
            content = parts
        elif content is not None and not isinstance(content, str):
            raise ValueError("Message content must be text, a list of parts or None")
        conversation.append({**message, "content": content})
    return conversation, media
