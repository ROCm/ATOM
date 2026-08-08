# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Noise semantics follow the reference in sgl-project/sglang
# (.../minimax_h3/stages/latent_preparation.py, Apache-2.0).

"""Initial noise for a MiniMax-H3 t2va request.

The RNG contract has to be reproduced exactly or the output diverges from the
reference for the same seed, and there is nothing downstream to catch it:

* **CPU** generators, fp32, always -- not the device RNG;
* video noise is drawn on the **raw latent tensor** ``[1, 24, T, H, W]`` and
  *then* patchified into row order. Drawing directly in row shape gives a
  different sample for the same seed;
* audio uses a **second, independently constructed** generator re-seeded with
  the *same* seed -- not a continuation of the video generator;
* default seed is 42 when the request omits one.
"""

import torch

from atom.diffusion.stages.minimax_h3.packed_tokens import patchify_video_latent

DEFAULT_SEED = 42
VIDEO_LATENT_CHANNELS = 24
AUDIO_LATENT_CHANNELS = 32
AUDIO_CHANNELS = 2


def build_initial_latents(
    *,
    latent_t: int,
    latent_h: int,
    latent_w: int,
    audio_t: int,
    seed: int | None = None,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(video_rows [Nv, 96], audio_rows [Na, 32])`` fp32 on CPU."""
    if seed is None:
        seed = DEFAULT_SEED
    seed = int(seed)

    pt, ph, pw = patch_size
    if latent_t % pt or latent_h % ph or latent_w % pw:
        raise ValueError(
            f"latent grid {latent_t}x{latent_h}x{latent_w} not divisible by "
            f"patch {patch_size}"
        )

    gen_v = torch.Generator().manual_seed(seed)
    video_tensor = torch.randn(
        1,
        VIDEO_LATENT_CHANNELS,
        latent_t,
        latent_h,
        latent_w,
        generator=gen_v,
        dtype=torch.float32,
    )
    video_rows = patchify_video_latent(video_tensor, patch_size=patch_size).to(
        torch.float32
    )

    # Independent generator, same seed -- each modality re-seeds its own.
    gen_a = torch.Generator().manual_seed(seed)
    audio_rows = torch.randn(
        audio_t * AUDIO_CHANNELS,
        AUDIO_LATENT_CHANNELS,
        generator=gen_a,
        dtype=torch.float32,
    )

    expected_video = (
        (latent_t // pt) * (latent_h // ph) * (latent_w // pw),
        VIDEO_LATENT_CHANNELS * pt * ph * pw,
    )
    if tuple(video_rows.shape) != expected_video:
        raise ValueError(
            f"video noise shape {tuple(video_rows.shape)} != {expected_video}"
        )
    return video_rows, audio_rows


def scatter_rows_into_packed(
    *,
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-form latents -> the full-length ``x`` / ``audio_x`` the DiT takes.

    The DiT reads ``[1, S, 96]`` and ``[1, S, 32]`` buffers indexed by global
    row id, even though only the media rows carry data; padding and text rows
    stay zero.
    """
    device = video_rows.device
    x = torch.zeros(
        1, seq_len, video_rows.shape[-1], dtype=video_rows.dtype, device=device
    )
    audio_x = torch.zeros(
        1, seq_len, audio_rows.shape[-1], dtype=audio_rows.dtype, device=device
    )
    x[0].index_copy_(0, img_pos.to(device), video_rows)
    audio_x[0].index_copy_(0, audio_pos.to(device), audio_rows)
    return x, audio_x
