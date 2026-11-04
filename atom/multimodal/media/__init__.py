# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Media loading, independent of model processors and serving protocols."""

from __future__ import annotations

from ..inputs import (
    AudioData,
    AudioItem,
    ImageItem,
    MediaData,
    Modality,
    MultiModalDataItems,
    VideoData,
    VideoItem,
)
from .connector import MediaConnector
from .image import load_image


class MediaLoader:
    """Decode images and accept decoded audio/video with timing metadata.

    Encoded audio/video require a codec implementation of load_audio/load_video.
    The native engine rejects unsupported modalities before loading media.
    """

    def __init__(self, timeout: float = 30):
        self.connector = MediaConnector(timeout)

    def load_image(self, data: ImageItem, *, keep_alpha: bool = False):
        return load_image(data, self.connector, keep_alpha=keep_alpha)

    def load_audio(self, data: AudioItem) -> AudioData:
        if isinstance(data, tuple):
            data = AudioData(*data)
        if isinstance(data, AudioData):
            return data
        raise ValueError(
            "Audio decoding requires a media loader; pass decoded AudioData "
            "or implement MediaLoader.load_audio"
        )

    def load_video(self, data: VideoItem) -> VideoData:
        if isinstance(data, VideoData):
            return data
        raise ValueError(
            "Video decoding requires a media loader; pass decoded VideoData "
            "or implement MediaLoader.load_video"
        )

    def load(self, modality: Modality, data: MediaData, *, keep_alpha: bool = False):
        if modality == "image":
            return self.load_image(data, keep_alpha=keep_alpha)
        loaders = {"audio": self.load_audio, "video": self.load_video}
        return loaders[modality](data)


def load_multimodal_data(
    data: MultiModalDataItems, loader: MediaLoader, *, keep_alpha: bool = False
) -> MultiModalDataItems:
    """Decode normalized media, retaining the order within each modality.

    ``keep_alpha`` keeps transparent images as RGBA (see ``load_image``).
    """
    # Pass keep_alpha only when it applies, so loaders that override ``load``
    # without it keep working for every other request.
    extra = {"keep_alpha": True} if keep_alpha else {}
    return {
        modality: [
            loader.load(modality, item, **(extra if modality == "image" else {}))
            for item in items
        ]
        for modality, items in data.items()
    }
