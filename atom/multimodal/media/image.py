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
    """Decode an image and return it in its decoded mode.

    Color conversion is left to the model's processor: converting here would
    drop the alpha channel and palette transparency before a processor that
    handles them sees the image.
    """
    from PIL import Image

    if isinstance(data, Image.Image):
        return data
    if isinstance(data, EncodedMedia):
        image_bytes = data.data
    elif isinstance(data, bytes):
        image_bytes = data
    else:
        image_bytes = connector.read(data)
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Decode now, so invalid data fails here rather than in the processor.
        image.load()
    except OSError as exc:  # UnidentifiedImageError included
        raise ValueError("Invalid image data") from exc
    return image
