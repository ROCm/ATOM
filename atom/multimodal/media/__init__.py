# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Media loading, independent of model processors and serving protocols."""

from __future__ import annotations

from ..inputs import (
    AudioData,
    AudioItem,
    EncodedMedia,
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
    """Decode images, read encoded video, accept decoded audio/video.

    Encoded video is read but not decoded: sampling and frame size are the
    model's, so its input builder decodes it (see ``media.video``). Encoded
    audio requires a codec implementation of load_audio. The native engine
    rejects unsupported modalities before loading media.
    """

    def __init__(self, timeout: float = 30):
        self.connector = MediaConnector(timeout)

    def load_image(self, data: ImageItem):
        return load_image(data, self.connector)

    def load_audio(self, data: AudioItem) -> AudioData:
        if isinstance(data, tuple):
            data = AudioData(*data)
        if isinstance(data, AudioData):
            return data
        raise ValueError(
            "Audio decoding requires a media loader; pass decoded AudioData "
            "or implement MediaLoader.load_audio"
        )

    def load_video(self, data: VideoItem) -> VideoData | EncodedMedia:
        """Decoded ``VideoData`` as given, or the encoded video's bytes."""
        if isinstance(data, (VideoData, EncodedMedia)):
            return data
        return EncodedMedia(self.connector.read(data))

    def load(self, modality: Modality, data: MediaData):
        loaders = {
            "image": self.load_image,
            "audio": self.load_audio,
            "video": self.load_video,
        }
        return loaders[modality](data)


def load_multimodal_data(
    data: MultiModalDataItems, loader: MediaLoader
) -> MultiModalDataItems:
    """Decode normalized media, retaining the order within each modality."""
    return {
        modality: [loader.load(modality, item) for item in items]
        for modality, items in data.items()
    }
