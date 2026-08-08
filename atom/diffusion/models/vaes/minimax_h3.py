# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 VAE decode.

The H3 checkpoint **ships its own VAE implementation** -- ``video_vae/*.py``
and ``audio_vae/*.py``, each with an ``auto_map`` entry and a `from_pretrained`
that wires up config, tiling options and weights. We load that rather than
re-implementing ~4,200 LOC: it is the reference numerics by construction, and
it tracks the checkpoint if MiniMax revises it.

``AutoModel.from_pretrained`` does **not** work on these directories -- their
config.json has an ``auto_map`` but no ``model_type``, so the auto-class
registry cannot dispatch. Load the class through
``get_class_from_dynamic_module`` and call the wrapper's own
``from_pretrained``.

Decode contract:

1. rows -> latent tensor (unpatchify for video, channel-major unpack for audio)
2. de-normalize per channel: ``z = z * std + mean``
3. ``vae.decode(z)``
4. crop video back to the target canvas -- the VAE pads the latent grid up to
   its tile multiples, so a non-tile-aligned request decodes *larger* than
   asked for, with the padding at bottom/right.
"""

import json
import logging
import os
from typing import Any

import torch

from atom.diffusion.stages.minimax_h3.packed_tokens import (
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

logger = logging.getLogger(__name__)


def load_checkpoint_vae(path: str, *, device: torch.device | str = "cpu") -> Any:
    """Instantiate a VAE from the code bundled in the checkpoint directory."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_path = os.path.join(path, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    auto_map = config.get("auto_map", {})
    ref = auto_map.get("AutoModel")
    if not ref:
        raise ValueError(
            f"{config_path} has no auto_map.AutoModel entry; cannot locate the "
            f"bundled VAE class"
        )

    cls = get_class_from_dynamic_module(ref, path)
    model = cls.from_pretrained(path)
    model = model.to(device).eval()
    logger.info("loaded %s from %s on %s", type(model).__name__, path, device)
    return model


def latent_stats(path: str) -> tuple[list[float], list[float]] | None:
    """Read ``latents_mean``/``latents_std`` from a VAE config.

    Raises if the config is missing -- that is a wrong checkpoint path and
    should say so. Returns ``None`` only when the config exists but declares no
    stats, which is the audio VAE's case on some partitions.
    """
    with open(os.path.join(path, "config.json")) as f:
        config = json.load(f)
    mean = config.get("latents_mean")
    std = config.get("latents_std")
    if mean is None or std is None:
        return None
    return list(mean), list(std)


def denormalize_latents(
    latents: torch.Tensor,
    *,
    mean: list[float] | torch.Tensor,
    std: list[float] | torch.Tensor,
    name: str = "vae",
) -> torch.Tensor:
    """``z * std + mean`` over the channel axis (dim 1). Returns a new tensor."""
    mean_t = torch.as_tensor(mean, device=latents.device, dtype=latents.dtype)
    std_t = torch.as_tensor(std, device=latents.device, dtype=latents.dtype)
    if mean_t.ndim != 1 or std_t.ndim != 1:
        raise ValueError(f"{name} latents_mean/std must be 1-D")
    if mean_t.shape != std_t.shape:
        raise ValueError(
            f"{name} mean/std shape mismatch: {tuple(mean_t.shape)} vs "
            f"{tuple(std_t.shape)}"
        )
    if latents.ndim < 2:
        raise ValueError(f"{name} latents need a channel dimension")
    if int(latents.shape[1]) != int(mean_t.shape[0]):
        raise ValueError(
            f"{name} channel mismatch: latents.shape[1]="
            f"{int(latents.shape[1])} vs {int(mean_t.shape[0])} stats"
        )
    view = [1] * latents.ndim
    view[1] = int(mean_t.shape[0])
    return latents * std_t.view(*view) + mean_t.view(*view)


def crop_to_canvas(frames: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """Crop ``[B, C, T, H, W]`` down to the requested canvas.

    The VAE pads the latent grid up to tile multiples and the padding lands at
    the bottom/right, so cropping from the origin is correct.
    """
    if frames.ndim != 5:
        raise ValueError(f"frames must be rank 5, got {list(frames.shape)}")
    h, w = int(frames.shape[-2]), int(frames.shape[-1])
    if h < height or w < width:
        raise ValueError(
            f"decoded frames {h}x{w} are smaller than the target canvas "
            f"{height}x{width}"
        )
    if h == height and w == width:
        return frames
    return frames[..., :height, :width].contiguous()


def denormalize_pixels(frames: torch.Tensor, vae: Any) -> torch.Tensor:
    """ImageNet-normalized decoder output -> pixels in [0, 1].

    The VAE decoder does **not** emit displayable pixels. It emits values in
    the model's normalized pixel space (``pixel_norm_type="imagenet"``), and
    both the reference and the checkpoint's own processor finish with
    ``transform_rev(x).clamp(0, 1)``. Skipping this and treating the output as
    [-1, 1] still produces a plausible-looking video -- the error is
    per-channel, so it survives every structural check and only shows up in a
    pixel comparison.

    ``transform_rev`` is a 4-D (N, C, H, W) transform, so the temporal axis is
    folded into the batch first, mirroring the reference processor.
    """
    transform_rev = getattr(vae, "transform_rev", None)
    if transform_rev is None:
        raise AttributeError(
            "video VAE has no transform_rev; cannot map decoder output out of "
            "normalized pixel space"
        )
    if frames.ndim != 5:
        raise ValueError(f"frames must be rank 5, got {list(frames.shape)}")

    b, c, t, h, w = frames.shape
    flat = frames.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    flat = transform_rev(flat).clamp_(0.0, 1.0)
    return flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


@torch.no_grad()
def decode_video_rows(
    vae: Any,
    rows: torch.Tensor,
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    height: int,
    width: int,
    mean: list[float],
    std: list[float],
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> torch.Tensor:
    """DiT video rows -> decoded frames ``[B, C, T, H, W]``."""
    pt, ph, pw = patch_size
    latent = unpatchify_video_tokens(
        rows.float(),
        latent_shape=(latent_t // pt, latent_h // ph, latent_w // pw, 24),
        patch_size=patch_size,
    )
    latent = denormalize_latents(latent, mean=mean, std=std, name="video_vae")
    z = latent.to(next(vae.parameters()).dtype)

    # Use the clip-aware temporal path, not the base `decode`. The base path
    # upsamples uniformly by vae_ratio_t (37 latents -> 148 frames), whereas H3
    # frames live on the 17n+5 lattice (37 -> 124). `decode_temporal` honours
    # clip_length=17 / token_drop=3 and produces the right count; measured
    # side by side on this checkpoint, decode()->148 and
    # decode_temporal()->124. Nothing downstream catches the difference -- the
    # file is still a valid MP4, just with the wrong frames.
    decode_fn = getattr(vae, "decode_temporal", None) or vae.decode
    frames = decode_fn(z)
    frames = getattr(frames, "sample", frames)
    frames = denormalize_pixels(frames.float(), vae)
    return crop_to_canvas(frames, height=height, width=width)


@torch.no_grad()
def decode_audio_rows(
    vae: Any,
    rows: torch.Tensor,
    *,
    audio_channel: int = 2,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> torch.Tensor:
    """DiT audio rows -> decoded waveform."""
    latent = unpack_audio_tokens(
        rows.float(), audio_t=int(rows.shape[0]), audio_channel=audio_channel
    )
    if mean is not None and std is not None:
        latent = denormalize_latents(latent, mean=mean, std=std, name="audio_vae")
    waveform = vae.decode(latent.to(next(vae.parameters()).dtype))
    return getattr(waveform, "sample", waveform)
