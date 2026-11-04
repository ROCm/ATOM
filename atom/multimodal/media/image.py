# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import io
from typing import TYPE_CHECKING

from ..inputs import EncodedMedia, ImageItem
from .connector import MediaConnector

if TYPE_CHECKING:
    from PIL import Image


def load_image(
    data: ImageItem, connector: MediaConnector, *, keep_alpha: bool = False
) -> Image.Image:
    """Decode an image to RGB, or to RGBA when ``keep_alpha`` and it is transparent.

    Converting a transparent image straight to RGB composites it on black, so
    dark content on a transparent background disappears. Processors that apply
    their model's own background fill (Kimi-K3) need the alpha channel; callers
    opt in per model through the input builder's ``keeps_alpha``.
    """
    from PIL import Image, UnidentifiedImageError

    if isinstance(data, Image.Image):
        return _decoded(data, keep_alpha)
    if isinstance(data, EncodedMedia):
        image_bytes = data.data
    elif isinstance(data, bytes):
        image_bytes = data
    else:
        image_bytes = connector.read(data)
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return _decoded(image, keep_alpha)
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image data") from exc


def _decoded(image: Image.Image, keep_alpha: bool) -> Image.Image:
    if keep_alpha and ("A" in image.getbands() or "transparency" in image.info):
        return image.convert("RGBA")
    return image.convert("RGB")
