# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from ..inputs import EncodedMedia, ImageItem
from .connector import MediaConnector

if TYPE_CHECKING:
    from PIL import Image


def load_image(data: ImageItem, connector: MediaConnector) -> Image.Image:
    from PIL import Image, UnidentifiedImageError

    if isinstance(data, Image.Image):
        return data.convert("RGB")
    if isinstance(data, EncodedMedia):
        image_bytes = data.data
    elif isinstance(data, bytes):
        image_bytes = data
    else:
        image_bytes = connector.read(data)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data") from exc
