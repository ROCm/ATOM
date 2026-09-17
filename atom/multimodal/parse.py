# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Normalize media grouped by modality, independently of chat protocols."""

from collections.abc import Mapping
from os import PathLike, fspath
from typing import cast

from .inputs import (
    AudioData,
    EncodedMedia,
    MediaData,
    Modality,
    MultiModalDataDict,
    MultiModalDataItems,
    VideoData,
)


def _normalize_item(modality: Modality, data: MediaData) -> MediaData:
    if isinstance(data, PathLike):
        data = fspath(data)
    if isinstance(data, bytes):
        data = EncodedMedia(data)
    if modality == "audio" and isinstance(data, tuple) and len(data) == 2:
        data = AudioData(*data)

    if isinstance(data, str):
        if not data.strip():
            raise ValueError(f"{modality} URL/path must not be empty")
    elif not isinstance(data, EncodedMedia):
        if modality == "image":
            from PIL import Image

            valid = isinstance(data, Image.Image)
        else:
            valid = isinstance(data, AudioData if modality == "audio" else VideoData)
        if not valid:
            raise ValueError(
                f"Invalid {modality} input: expected a URL/path, encoded bytes or decoded {modality} data"
            )
    return data


def normalize_multimodal_data(data: MultiModalDataDict) -> MultiModalDataItems:
    """Normalize single items and lists without loading media or copying arrays.

    Empty modalities are omitted. Audio tuples are single items; lists contain
    separate items. Sampling rates and video timing must be supplied explicitly.
    """
    if not isinstance(data, Mapping):
        raise ValueError(  # noqa: TRY004
            "Multimodal data must be a mapping keyed by modality"
        )
    normalized: MultiModalDataItems = {}
    for name, value in data.items():
        if name not in ("image", "audio", "video"):
            raise ValueError(f"Unsupported modality: {name!r}")
        modality = cast(Modality, name)
        if value is None:
            continue
        items = value if isinstance(value, list) else [value]
        if items:
            normalized[modality] = [_normalize_item(modality, item) for item in items]
    return normalized
