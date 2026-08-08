# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Encode recipes follow the reference in sgl-project/sglang
# (.../minimax_h3/reference_encoding.py, Apache-2.0).

"""MiniMax-H3 ref2va reference material -> packed conditioning rows.

Unlike an fl2va keyframe, a reference never binds the target canvas.

* ``image`` -- keep the ratio, short edge to 2048 (upscaling if needed), round
  both dims to the nearest 32, then the fl2va keyframe encode.
* ``audio`` -- stereo float PCM at the source rate, one resample to 32 kHz, then
  the audio VAE posterior **mean** (no sampling), normalised, channel-major.
* ``video``/``video_audio`` -- one ffmpeg pass feeds *both* Qwen and the visual
  VAE, so the two cannot disagree. Soundtrack extracted separately at 44.1 kHz.

Two asymmetries that look like oversights but are not: the video encode
*samples* the posterior (seed 42) while the audio encode takes the mean, and the
audio encode needs :class:`audio_vae_determinism` or the same waveform yields
different latents run to run.
"""

import functools
import math
import subprocess
from typing import Any, Self

import numpy as np
import torch

from atom.diffusion.stages.minimax_h3.keyframe import scoped_encode_rng
from atom.diffusion.stages.minimax_h3.packed_tokens import patchify_video_latent

REFERENCE_IMAGE_SHORT_EDGE = 2048
REFERENCE_IMAGE_MULTIPLE = 32
REFERENCE_IMAGE_MAX_RATIO = 4.0

AUDIO_SAMPLE_RATE = 32000
AUDIO_CHANNELS = 2
VIDEO_SOURCE_AUDIO_RATE = 44100

SUPPORTED_FPS = 24
REFERENCE_VIDEO_ENCODE_SEED = 42
REFERENCE_VIDEO_PATCH_SIZE = (1, 2, 2)
LATENT_CHANNELS = 24

# Qwen sees a 2 FPS strided view of the same frames the VAE encodes.
QWEN_VIDEO_SAMPLE_FPS = 2.0
QWEN_TEMPORAL_PATCH = 2

AUDIO_MATERIAL_CHAIN = "audio"
VIDEO_MATERIAL_CHAINS = ("video.reference_preserve", "video_audio.reference_preserve")


class audio_vae_determinism:
    """Scoped determinism for the audio encode.

    Disables TF32 and cuDNN and pins SDP to math, then restores. Without it the
    same waveform yields different latents run to run. Re-entrant, so a caller
    encoding several references can wrap the whole loop.
    """

    _depth = 0
    _saved: tuple | None = None

    def __enter__(self) -> Self:
        if audio_vae_determinism._depth == 0:
            b = torch.backends
            audio_vae_determinism._saved = (
                b.cuda.matmul.allow_tf32,
                b.cudnn.allow_tf32,
                b.cudnn.benchmark,
                b.cudnn.deterministic,
                b.cudnn.enabled,
                b.cuda.flash_sdp_enabled(),
                b.cuda.mem_efficient_sdp_enabled(),
                b.cuda.math_sdp_enabled(),
            )
            b.cuda.matmul.allow_tf32 = False
            b.cudnn.allow_tf32 = False
            b.cudnn.benchmark = False
            b.cudnn.deterministic = True
            b.cudnn.enabled = False
            b.cuda.enable_flash_sdp(False)
            b.cuda.enable_mem_efficient_sdp(False)
            b.cuda.enable_math_sdp(True)
        audio_vae_determinism._depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        audio_vae_determinism._depth -= 1
        if audio_vae_determinism._depth or audio_vae_determinism._saved is None:
            return
        b = torch.backends
        (
            b.cuda.matmul.allow_tf32,
            b.cudnn.allow_tf32,
            b.cudnn.benchmark,
            b.cudnn.deterministic,
            b.cudnn.enabled,
            flash,
            mem_efficient,
            math_sdp,
        ) = audio_vae_determinism._saved
        b.cuda.enable_flash_sdp(flash)
        b.cuda.enable_mem_efficient_sdp(mem_efficient)
        b.cuda.enable_math_sdp(math_sdp)
        audio_vae_determinism._saved = None


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, round(float(value) / multiple) * multiple)


def resolve_reference_image_shape(*, width: float, height: float) -> dict[str, Any]:
    """Reference-image geometry, resolved independently of the target canvas.

    Note there is no area cap here, unlike the target/video shape policy: a
    reference always lands on a 2048px short edge.
    """
    try:
        source_width, source_height = float(width), float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        ) from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError(
            "reference image width and height must be positive finite numbers"
        )
    if (
        source_width > REFERENCE_IMAGE_MAX_RATIO * source_height
        or source_height > REFERENCE_IMAGE_MAX_RATIO * source_width
    ):
        raise ValueError(
            "reference image ratio must be within 1:4 to 4:1, got "
            f"{source_width:g}x{source_height:g}"
        )

    scale = REFERENCE_IMAGE_SHORT_EDGE / min(source_width, source_height)
    target_width = _nearest_multiple(source_width * scale, REFERENCE_IMAGE_MULTIPLE)
    target_height = _nearest_multiple(source_height * scale, REFERENCE_IMAGE_MULTIPLE)
    return {
        "width": target_width,
        "height": target_height,
        "short_edge": min(target_width, target_height),
        "multiple": REFERENCE_IMAGE_MULTIPLE,
    }


def resize_reference_image(image: Any, *, target_width: int, target_height: int) -> Any:
    """Resize to the resolved reference shape (LANCZOS, no crop)."""
    from PIL import Image

    if target_width <= 0 or target_height <= 0:
        raise ValueError("reference image target dimensions must be positive")
    if (
        target_width % REFERENCE_IMAGE_MULTIPLE
        or target_height % REFERENCE_IMAGE_MULTIPLE
    ):
        raise ValueError(
            "reference image target dimensions must be aligned to "
            f"{REFERENCE_IMAGE_MULTIPLE}"
        )
    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------


def load_reference_waveform(
    path: str,
    *,
    material_chain: str = AUDIO_MATERIAL_CHAIN,
    max_duration_seconds: float | None = None,
    start_time_seconds: float = 0.0,
    source_sample_rate: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Decode to stereo float PCM ``[2, N]`` and report the source rate.

    Pure-audio references keep their source rate (the single 32 kHz resample
    happens at the VAE boundary); video-bearing references are pulled at
    44.1 kHz. ffmpeg writes interleaved float PCM to stdout, so there is no
    temporary lossless file and no second decode.
    """
    if max_duration_seconds is not None:
        max_duration_seconds = float(max_duration_seconds)
        if not math.isfinite(max_duration_seconds) or max_duration_seconds <= 0:
            raise ValueError("reference audio duration bound must be positive")
    start_time_seconds = float(start_time_seconds)
    if not math.isfinite(start_time_seconds) or start_time_seconds < 0:
        raise ValueError("reference audio start time must be non-negative")

    if material_chain == AUDIO_MATERIAL_CHAIN:
        if source_sample_rate is None or int(source_sample_rate) <= 0:
            raise ValueError("reference audio sample rate must be positive")
        source_rate = int(source_sample_rate)
    elif material_chain in VIDEO_MATERIAL_CHAINS:
        source_rate = VIDEO_SOURCE_AUDIO_RATE
    else:
        raise ValueError(f"unsupported audio material chain {material_chain!r}")

    command = ["ffmpeg", "-v", "error"]
    if start_time_seconds > 0:
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += ["-i", str(path), "-map", "0:a:0", "-vn", "-ac", str(AUDIO_CHANNELS)]
    if material_chain != AUDIO_MATERIAL_CHAIN:
        command += ["-ar", str(source_rate)]
    if max_duration_seconds is not None:
        command += ["-t", f"{max_duration_seconds:.9g}"]
    command += ["-f", "f32le", "pipe:1"]

    payload = subprocess.run(command, check=True, capture_output=True).stdout
    frame_bytes = AUDIO_CHANNELS * torch.float32.itemsize
    if len(payload) % frame_bytes:
        raise ValueError(
            f"ffmpeg returned a partial audio sample frame: {len(payload)} bytes"
        )
    waveform = torch.from_numpy(
        np.frombuffer(payload, dtype=np.float32).reshape(-1, AUDIO_CHANNELS).T.copy()
    )
    return waveform, source_rate


@functools.lru_cache(maxsize=8)
def _resampler(source_rate: int):
    import torchaudio

    return torchaudio.transforms.Resample(source_rate, AUDIO_SAMPLE_RATE)


@torch.inference_mode()
def encode_reference_audio_rows(
    audio_vae: Any,
    audio_path: str,
    *,
    latents_mean: list[float],
    latents_std: list[float],
    material_chain: str = AUDIO_MATERIAL_CHAIN,
    max_duration_seconds: float | None = None,
    start_time_seconds: float = 0.0,
    source_sample_rate: int | None = None,
) -> dict[str, Any]:
    """Audio file -> ``{"rows": [2*T, 32] fp32 cpu, "ref_audio_t", "duration"}``.

    Uses the posterior **mean**, not a sample: unlike the video VAE there is no
    seeded draw here, and calling ``encode`` (which samples) instead would give
    plausible-but-different conditioning.
    """
    device = next(audio_vae.parameters()).device
    waveform, source_rate = load_reference_waveform(
        audio_path,
        material_chain=material_chain,
        max_duration_seconds=max_duration_seconds,
        start_time_seconds=start_time_seconds,
        source_sample_rate=source_sample_rate,
    )
    if waveform.numel() == 0:
        raise ValueError(f"reference audio is empty: {audio_path}")
    if int(source_rate) != AUDIO_SAMPLE_RATE:
        waveform = _resampler(int(source_rate))(waveform)
    waveform = waveform.to(device)

    with audio_vae_determinism():
        audio_data = audio_vae.preprocess(waveform.unsqueeze(1), AUDIO_SAMPLE_RATE)
        z = audio_vae.encoder(audio_data)
        if bool(getattr(audio_vae, "attn_proj", False)):
            z = audio_vae.pre_block(z.transpose(1, 2)).transpose(1, 2)
        if not hasattr(audio_vae, "mean_proj"):
            raise AttributeError(
                "audio VAE must expose mean_proj for a deterministic mean encode"
            )
        latent = audio_vae.mean_proj(z).float()

    if latent.ndim != 3:
        raise ValueError(f"expected a 3-D audio latent, got {list(latent.shape)}")
    channels = len(latents_mean)
    if int(latent.shape[-1]) != channels:
        if int(latent.shape[1]) != channels:
            raise ValueError(f"cannot canonicalise audio latent {list(latent.shape)}")
        latent = latent.transpose(1, 2).contiguous()  # -> [2, T, C]
    latent = latent.cpu()

    mean = torch.tensor(latents_mean, dtype=torch.float32).view(1, 1, channels)
    std = torch.tensor(latents_std, dtype=torch.float32).view(1, 1, channels)
    latent = (latent - mean) / std
    return {
        "rows": latent.reshape(-1, channels).to(torch.float32).contiguous(),
        "ref_audio_t": int(latent.shape[1]),
        "duration_seconds": float(waveform.shape[-1]) / float(AUDIO_SAMPLE_RATE),
    }


# ---------------------------------------------------------------------------
# video
# ---------------------------------------------------------------------------


def decode_reference_video_frames(
    video_path: str,
    *,
    target_width: int,
    target_height: int,
    target_frame_count: int,
    fps: float = SUPPORTED_FPS,
    start_time_seconds: float = 0.0,
) -> np.ndarray:
    """Decode, rotate, CFR-sample, scale and truncate in a single ffmpeg pass.

    The returned ``[T, H, W, 3]`` uint8 array is shared by Qwen and the visual
    VAE; re-decoding for the second consumer would let the two disagree.
    """
    if target_frame_count <= 0:
        raise ValueError("target_frame_count must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("reference video dimensions must be positive")
    if not math.isfinite(float(fps)) or float(fps) <= 0:
        raise ValueError("reference video fps must be positive")
    start_time_seconds = float(start_time_seconds)
    if not math.isfinite(start_time_seconds) or start_time_seconds < 0:
        raise ValueError("reference video start time must be non-negative")

    filters = (
        f"fps={float(fps):g},"
        f"scale={target_width}:{target_height}:flags=lanczos,"
        "setsar=1"
    )
    command = ["ffmpeg", "-v", "error"]
    if start_time_seconds > 0:
        # Input seeking stays accurate while transcoding and avoids decoding
        # the unused prefix of a long reference into RGB frames.
        command += ["-ss", f"{start_time_seconds:.9g}"]
    command += [
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filters,
        "-frames:v",
        str(target_frame_count),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    payload = subprocess.run(command, check=True, capture_output=True).stdout
    frame_bytes = target_width * target_height * 3
    if len(payload) % frame_bytes:
        raise ValueError(
            f"ffmpeg returned a partial video frame: {len(payload)} bytes for "
            f"{target_width}x{target_height} RGB24"
        )
    frame_count = len(payload) // frame_bytes
    if frame_count <= 0:
        raise ValueError(f"reference video has no frames: {video_path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        frame_count, target_height, target_width, 3
    )


@torch.inference_mode()
def encode_reference_video_rows(
    video_vae: Any,
    frames: np.ndarray,
    *,
    latents_mean: list[float],
    latents_std: list[float],
    seed: int = REFERENCE_VIDEO_ENCODE_SEED,
) -> tuple[torch.Tensor, int, int, int]:
    """Frames -> ``(rows [n, 96] fp32 cpu, latent_t, latent_h, latent_w)``.

    Same recipe as the fl2va keyframe sink -- fp32 weights, seeded RNG because
    the encode *samples* the posterior, fp16 latent -- then normalise and
    patchify. The VAE's ``clip_length=17`` / ``token_drop=3`` produce the
    17-frames-per-5-latents grouping.
    """
    frames = np.asarray(frames)
    if (
        frames.ndim != 4
        or int(frames.shape[-1]) != 3
        or frames.dtype != np.uint8
        or int(frames.shape[0]) <= 0
    ):
        raise ValueError(
            "reference video frames must be non-empty [T, H, W, 3] uint8, got "
            f"shape={list(frames.shape)} dtype={frames.dtype}"
        )

    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with scoped_encode_rng(seed, parameter.device):
            z = video_vae.encode_videos(frames, use_fp16_latent=True)[0]
    finally:
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)

    z = z.cpu().float()
    if z.dim() == 4:
        z = z[None]
    if z.dim() != 5 or int(z.shape[1]) != LATENT_CHANNELS:
        raise ValueError(f"unexpected reference video latent shape {list(z.shape)}")
    latent_t, latent_h, latent_w = (int(z.shape[i]) for i in (2, 3, 4))

    view = (1, LATENT_CHANNELS, 1, 1, 1)
    mean = torch.tensor(latents_mean, dtype=torch.float32).view(view)
    std = torch.tensor(latents_std, dtype=torch.float32).view(view)
    z = (z - mean) / std

    rows = patchify_video_latent(z, patch_size=REFERENCE_VIDEO_PATCH_SIZE)
    return rows.to(torch.float32), latent_t, latent_h, latent_w


def sample_reference_video_frames(frames: np.ndarray) -> dict[str, Any]:
    """Strided 2 FPS view of the shared frames, plus Qwen block timestamps.

    Qwen pairs frames along its temporal patch, so the index list is padded to
    a multiple of the patch with the *last* frame and each block's timestamp is
    the mean of its pair. Those become ``"<{ts:.1f} seconds>"`` labels in the
    presentation.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or int(frames.shape[0]) <= 0:
        raise ValueError("Qwen video sampling requires non-empty [T, H, W, C] frames")

    stride = int(SUPPORTED_FPS / QWEN_VIDEO_SAMPLE_FPS)
    sampled = frames[::stride]
    stamps = [i / QWEN_VIDEO_SAMPLE_FPS for i in range(int(sampled.shape[0]))]
    stamps += [stamps[-1]] * ((-len(stamps)) % QWEN_TEMPORAL_PATCH)
    block_timestamps = [
        (stamps[i] + stamps[i + QWEN_TEMPORAL_PATCH - 1]) / 2
        for i in range(0, len(stamps), QWEN_TEMPORAL_PATCH)
    ]
    return {"frames": sampled, "block_timestamps": block_timestamps}
