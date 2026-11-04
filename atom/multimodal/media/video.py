# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode an encoded video into uniformly sampled RGB frames (PyAV)."""

from __future__ import annotations

import io

import numpy as np

from ..inputs import VideoData

# A request may carry an arbitrary container; refuse absurd frame counts
# before any frame is converted to RGB.
MAX_SOURCE_FRAMES = 100_000


def sample_frame_indices(
    total_frames: int, avg_fps: float, sample_fps: float, max_frames: int | None
) -> list[int]:
    """Uniform indices at most ``sample_fps`` per second of source video.

    Matches the Kimi-K3 reference sampler: ``round(total * fps / avg_fps)``
    frames (at least one, at most ``max_frames`` and ``total``), spread over
    the whole clip with ``linspace`` and rounded to the nearest frame.
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


def decode_video(
    data: bytes, *, sample_fps: float, max_frames: int | None = None
) -> VideoData:
    """Sampled frames (THWC uint8 RGB) and each frame's source timestamp.

    Two passes: the first counts decodable frames without converting them,
    the second keeps only the sampled indices, so memory scales with the
    sampled frames rather than the clip length.
    """
    av = _import_av()

    container = _open(data)
    try:
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        avg_fps = float(rate) if rate else 0.0
        total = 0
        try:
            for _ in container.decode(stream):
                total += 1
                if total > MAX_SOURCE_FRAMES:
                    raise ValueError(f"Video has more than {MAX_SOURCE_FRAMES} frames")
        except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
            raise ValueError("Invalid video data") from exc
    finally:
        container.close()

    indices = sample_frame_indices(total, avg_fps, sample_fps, max_frames)
    wanted = set(indices)
    last = indices[-1]
    frames = []
    container = _open(data)
    try:
        stream = container.streams.video[0]
        try:
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if index >= last:
                    break
        except av.error.FFmpegError as exc:  # type: ignore[attr-defined]
            raise ValueError("Invalid video data") from exc
    finally:
        container.close()
    if len(frames) != len(indices):
        raise ValueError("Video frames changed between decoding passes")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Video frames must share one size")
    return VideoData(
        frames=np.stack(frames).astype(np.uint8, copy=False),
        timestamps=tuple(index / avg_fps for index in indices),
    )
