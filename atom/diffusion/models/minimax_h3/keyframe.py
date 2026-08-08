# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Canvas and encode recipes follow the reference in sgl-project/sglang
# (.../minimax_h3/{canvas,keyframe_encoding}.py, Apache-2.0).

"""MiniMax-H3 fl2va keyframe conditioning.

canvas (LANCZOS cover crop) -> ``encode_images(use_fp16_latent=True)`` on fp32
weights under a forked RNG seeded to **42** -> ``(z - mean) / std`` -> patchify.

The seed is contract, not convenience: the posterior is *sampled*, so another
seed gives different-but-plausible rows and a silently different video. The
normalise step is the exact inverse of decode's; reversing it yields
conditioning of roughly the right magnitude and is correspondingly hard to spot.
"""

import contextlib
from typing import Any

import torch

from atom.diffusion.models.minimax_h3.packed_tokens import patchify_video_latent

KEYFRAME_ENCODE_SEED = 42
KEYFRAME_PATCH_SIZE = (1, 2, 2)
LATENT_CHANNELS = 24


def cover_crop_plan(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
    allow_upscale: bool = False,
) -> dict[str, Any]:
    """Deterministic aspect-preserving cover-crop transform."""
    if source_width <= 0 or source_height <= 0:
        raise ValueError("cover crop requires positive source dimensions")
    scale = max(
        target_width / float(source_width), target_height / float(source_height)
    )
    if scale > 1.0 and not allow_upscale:
        raise ValueError(
            f"cover crop would upscale {source_width}x{source_height} to "
            f"{target_width}x{target_height}; pass allow_upscale=True"
        )
    resized_w = max(target_width, round(source_width * scale))
    resized_h = max(target_height, round(source_height * scale))
    left = max(0, (resized_w - target_width) // 2)
    top = max(0, (resized_h - target_height) // 2)
    return {
        "scale": scale,
        "resized_size": (resized_w, resized_h),
        "crop_box": (left, top, left + target_width, top + target_height),
    }


def prepare_keyframe_canvas(
    image: Any,
    *,
    target_width: int,
    target_height: int,
    allow_upscale: bool = False,
) -> Any:
    """Cover-crop a PIL image onto the target canvas."""
    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    plan = cover_crop_plan(
        source_width=image.size[0],
        source_height=image.size[1],
        target_width=target_width,
        target_height=target_height,
        allow_upscale=allow_upscale,
    )
    return image.resize(plan["resized_size"], Image.Resampling.LANCZOS).crop(
        plan["crop_box"]
    )


def stretch_keyframe_canvas(
    image: Any, *, target_width: int, target_height: int
) -> Any:
    """Stretch (do not crop) an image onto the target canvas."""
    from PIL import Image

    image = image.convert("RGB")
    if image.size == (target_width, target_height):
        return image
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


@contextlib.contextmanager
def scoped_encode_rng(seed: int, device: torch.device | None = None):
    """Seed the default generators for one sampled encode, then restore them."""
    devices: list[torch.device] = []
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        devices = [device]
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(int(seed))
        for dev in devices:
            with torch.cuda.device(dev):
                torch.cuda.manual_seed(int(seed))
        yield


@torch.inference_mode()
def encode_keyframe_cond_rows(
    video_vae: Any,
    image: Any,
    *,
    latents_mean: list[float],
    latents_std: list[float],
    seed: int = KEYFRAME_ENCODE_SEED,
) -> torch.Tensor:
    """Canvas-sized PIL image -> packed cond rows ``[n_rows, 96]`` fp32 (CPU)."""
    parameter = next(video_vae.parameters())
    prev_dtype = parameter.dtype
    if prev_dtype != torch.float32:
        video_vae.to(torch.float32)
    try:
        with scoped_encode_rng(seed, parameter.device):
            z = video_vae.encode_images(image, use_fp16_latent=True)[0]
    finally:
        if prev_dtype != torch.float32:
            video_vae.to(prev_dtype)

    z = z.cpu().float()
    if z.dim() == 4:
        z = z[None]
    if z.dim() != 5 or int(z.shape[1]) != LATENT_CHANNELS:
        raise ValueError(f"unexpected keyframe latent shape {list(z.shape)}")

    view = (1, LATENT_CHANNELS, 1, 1, 1)
    mean = torch.tensor(latents_mean).view(view)
    std = torch.tensor(latents_std).view(view)
    z = (z - mean) / std

    return patchify_video_latent(z, patch_size=KEYFRAME_PATCH_SIZE).to(torch.float32)
