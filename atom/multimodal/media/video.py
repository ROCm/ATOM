# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Encoded video: probe it, then decode only the sampled frames (PyAV).

The model's input builder chooses the sampling and the frame size; this module
only reads the container. Probing demuxes packets without decoding them, so a
request can be refused before any frame is decoded.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# A request may carry an arbitrary container; refuse absurd frame counts while
# probing, before anything is decoded.
MAX_SOURCE_FRAMES = 100_000


@dataclass(frozen=True)
class VideoInfo:
    """What probing reports: frame count, average frame rate and frame size."""

    num_frames: int
    fps: float
    width: int
    height: int


def sample_frame_indices(
    total_frames: int, avg_fps: float, sample_fps: float, max_frames: int | None
) -> list[int]:
    """Uniform indices at most ``sample_fps`` per second of source video.

    ``round(total * fps / avg_fps)`` frames (at least one, at most
    ``max_frames`` and ``total``), spread over the whole clip with ``linspace``
    and rounded to the nearest frame.
    """
    if total_frames <= 0:
        raise ValueError("Video input must contain at least one frame")
    if not avg_fps > 0:
        raise ValueError("Video input must report a positive frame rate")
    fps = min(float(sample_fps), avg_fps)
    count = max(round(total_frames * fps / avg_fps), 1)
    if max_frames is not None:
        count = min(count, max(1, int(max_frames)))
    count = min(count, total_frames)
    return np.linspace(0, total_frames - 1, count).round().astype(int).tolist()


def _import_av():
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ValueError(
            "Video inputs need PyAV: `pip install av`, or install ATOM with "
            "its vision extra (`pip install -e '.[vision]'`)"
        ) from exc
    return av


def _open(data: bytes):
    av = _import_av()
    try:
        container = av.open(io.BytesIO(data), mode="r")
    except (av.error.FFmpegError, ValueError) as exc:  # type: ignore[attr-defined]
        raise ValueError("Invalid video data") from exc
    if not container.streams.video:
        container.close()
        raise ValueError("Video data has no video stream")
    return container


def probe_video(data: bytes) -> VideoInfo:
    """Count the video stream's frames by demuxing, without decoding."""
    av = _import_av()
    container = _open(data)
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        width, height = stream.codec_context.width, stream.codec_context.height
        count = 0
        try:
            for packet in container.demux(stream):
                # The demuxer ends with an empty flush packet.
                if packet.size:
                    count += 1
                    if count > MAX_SOURCE_FRAMES:
                        raise ValueError(
                            f"Video has more than {MAX_SOURCE_FRAMES} frames"
                        )
        except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
            raise ValueError("Invalid video data") from exc
    finally:
        container.close()
    if count == 0:
        raise ValueError("Video input must contain at least one frame")
    if not width or not height:
        raise ValueError("Video stream reports no frame size")
    return VideoInfo(count, float(rate) if rate else 0.0, width, height)


def decode_video_frames(
    data: bytes,
    indices: Sequence[int],
    transform: Callable[[int, Any], np.ndarray],
) -> tuple[list[np.ndarray], int]:
    """Decode once, keeping ``transform(position, image)`` of each wanted frame.

    ``indices`` are ascending source frame indices; ``position`` is the
    frame's place in them and ``image`` is the decoded RGB ``PIL.Image``.
    Only the transformed frames are kept, so host memory follows the size the
    transform produces, not the source resolution. Returns the kept frames and
    the number of frames decoded, which can fall short of the probed count
    when trailing packets carry no frame.
    """
    av = _import_av()
    wanted = {index: position for position, index in enumerate(indices)}
    last = indices[-1]
    frames: list[np.ndarray] = []
    decoded = 0
    container = _open(data)
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        try:
            for index, frame in enumerate(container.decode(stream)):
                decoded = index + 1
                position = wanted.get(index)
                if position is not None:
                    frames.append(transform(position, frame.to_image()))
                if index >= last:
                    break
        except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
            raise ValueError("Invalid video data") from exc
    finally:
        container.close()
    return frames, decoded
