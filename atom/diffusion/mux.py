# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Mux decoded frames and audio into an MP4.

MiniMax-H3's output contract, confirmed by ffprobe on the reference server's
own MP4s: **H.264 video at 24 fps plus one AAC stereo stream at 32 kHz**, in a
single file. A video-only file is not a valid H3 result -- the audio track is
half the model.
"""

import logging
from fractions import Fraction

import numpy as np
import torch

logger = logging.getLogger(__name__)

VIDEO_FPS = 24
AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2


def frames_to_uint8(frames: torch.Tensor) -> np.ndarray:
    """``[T, H, W, 3]`` uint8 from a float tensor in **[0, 1]**.

    [0, 1], not [-1, 1]: the VAE's ``transform_rev`` already returns display
    range. Assuming [-1, 1] here halves contrast and lifts black, which is
    structurally invisible and cost ~22 dB before it was found.
    """
    if frames.ndim == 5:
        if int(frames.shape[0]) != 1:
            raise ValueError(f"expected batch 1, got {int(frames.shape[0])}")
        frames = frames[0]
    if frames.ndim != 4:
        raise ValueError(f"expected [C, T, H, W], got {list(frames.shape)}")
    channels = int(frames.shape[0])
    if channels != 3:
        raise ValueError(f"expected 3 colour channels, got {channels}")

    video = frames.detach().float().permute(1, 2, 3, 0)  # T,H,W,C
    video = (video.clamp(0.0, 1.0) * 255.0).round().clamp(0, 255)
    return video.to(torch.uint8).cpu().numpy()


def _audio_to_float32(waveform: torch.Tensor) -> np.ndarray:
    """Normalise a decoded waveform to ``[channels, samples]`` float32."""
    w = waveform.detach().float().cpu()
    while w.ndim > 2:
        if int(w.shape[0]) == 1:
            w = w[0]
        else:
            w = w.reshape(w.shape[0], -1)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    if int(w.shape[0]) > int(w.shape[1]):
        # Samples-major input; make it channel-major.
        w = w.transpose(0, 1)
    return w.clamp(-1.0, 1.0).numpy().astype(np.float32)


def write_video_with_audio(
    path: str,
    frames: torch.Tensor,
    audio: torch.Tensor | None = None,
    *,
    fps: int = VIDEO_FPS,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> str:
    """Write an H.264 + AAC MP4 and return the path.

    Requires PyAV. ffmpeg/ffprobe must also be importable by whatever validates
    the artefact afterwards -- the reference server rejects its own output if
    ffprobe is absent.
    """
    import av

    video = frames_to_uint8(frames)
    num_frames, height, width, _ = video.shape

    container = av.open(path, mode="w")
    try:
        vstream = container.add_stream("libx264", rate=fps)
        vstream.width = width
        vstream.height = height
        vstream.pix_fmt = "yuv420p"
        vstream.time_base = Fraction(1, fps)

        astream = None
        if audio is not None:
            samples = _audio_to_float32(audio)
            channels = int(samples.shape[0])
            layout = "stereo" if channels == 2 else "mono"
            astream = container.add_stream("aac", rate=sample_rate)
            astream.layout = layout

        for index in range(num_frames):
            frame = av.VideoFrame.from_ndarray(video[index], format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            container.mux(vstream.encode(frame))
        container.mux(vstream.encode(None))

        if astream is not None:
            aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
            aframe.sample_rate = sample_rate
            aframe.time_base = Fraction(1, sample_rate)
            aframe.pts = 0
            for packet in astream.encode(aframe):
                container.mux(packet)
            for packet in astream.encode(None):
                container.mux(packet)
    finally:
        container.close()

    logger.info(
        "wrote %s (%d frames %dx%d @%dfps, audio=%s)",
        path,
        num_frames,
        width,
        height,
        fps,
        "yes" if audio is not None else "no",
    )
    return path
