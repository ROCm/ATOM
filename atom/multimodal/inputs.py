# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Model-independent media inputs, before HF processing or token expansion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from typing import TYPE_CHECKING, Literal, TypeAlias, Union

import numpy as np

if TYPE_CHECKING:
    from PIL import Image

Modality = Literal["image", "video", "audio"]


@dataclass(frozen=True)
class EncodedMedia:
    data: bytes
    format: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("Encoded media must contain non-empty bytes")


@dataclass(frozen=True)
class AudioData:
    """Decoded mono waveform and its sampling rate; no implicit resampling."""

    waveform: np.ndarray
    sampling_rate: int

    def __post_init__(self):
        if (
            not isinstance(self.waveform, np.ndarray)
            or self.waveform.ndim != 1
            or not self.waveform.size
            or not np.issubdtype(self.waveform.dtype, np.number)
            or np.iscomplexobj(self.waveform)
        ):
            raise ValueError(
                "AudioData.waveform must be a non-empty 1D real numeric array"
            )
        if (
            isinstance(self.sampling_rate, bool)
            or not isinstance(self.sampling_rate, int)
            or self.sampling_rate <= 0
        ):
            raise ValueError("AudioData.sampling_rate must be a positive integer")


@dataclass(frozen=True)
class VideoData:
    """Decoded RGB frames in THWC order, with fps or per-frame timestamps."""

    frames: np.ndarray
    fps: float | None = None
    timestamps: tuple[float, ...] | None = None

    def __post_init__(self):
        if (
            not isinstance(self.frames, np.ndarray)
            or self.frames.ndim != 4
            or not self.frames.size
            or self.frames.shape[-1] != 3
            or self.frames.dtype != np.uint8
        ):
            raise ValueError(
                "VideoData.frames must be a non-empty uint8 RGB THWC array"
            )
        if self.fps is None and self.timestamps is None:
            raise ValueError("VideoData requires fps or timestamps")
        if self.fps is not None and (
            isinstance(self.fps, bool) or not math.isfinite(self.fps) or self.fps <= 0
        ):
            raise ValueError("VideoData.fps must be finite and positive")
        if self.timestamps is not None:
            if len(self.timestamps) != len(self.frames) or any(
                not math.isfinite(t) or t < 0 for t in self.timestamps
            ):
                raise ValueError(
                    "VideoData requires one finite non-negative timestamp per frame"
                )
            if any(a > b for a, b in zip(self.timestamps, self.timestamps[1:])):
                raise ValueError("VideoData.timestamps must be non-decreasing")


MediaSource: TypeAlias = str | PathLike[str] | bytes | EncodedMedia
ImageItem: TypeAlias = Union[MediaSource, "Image.Image"]
AudioItem: TypeAlias = MediaSource | AudioData | tuple[np.ndarray, int]
VideoItem: TypeAlias = MediaSource | VideoData
MediaData: TypeAlias = ImageItem | AudioItem | VideoItem

# A modality may contain one item or an ordered list of items. Conversation
# structure and protocol-specific content parts belong to the entrypoint.
MultiModalDataDict: TypeAlias = Mapping[str, MediaData | list[MediaData] | None]
MultiModalDataItems: TypeAlias = dict[Modality, list[MediaData]]
