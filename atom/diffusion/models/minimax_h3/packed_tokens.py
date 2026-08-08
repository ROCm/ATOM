# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Packing order follows the reference in sgl-project/sglang
# (.../minimax_h3/packed_tokens.py, Apache-2.0).

"""Convert between VAE latent tensors and MiniMax-H3 DiT token rows.

Video: ``[B, C, T, H, W]`` <-> ``[B*t*h*w, C*pt*ph*pw]`` with patch (1, 2, 2),
so 24 channels become the 96-wide rows the DiT consumes.

The permutation is the part worth being careful about: channel varies *fastest*
within a row and the patch offsets trail it (``nctrhpwq -> nthwcrpq``). Getting
the axis order wrong preserves both the shape and the value histogram, so it
survives every check except a pixel comparison.
"""

from collections.abc import Sequence

import torch


def _int_tuple(value: Sequence[int], name: str, length: int) -> tuple[int, ...]:
    if len(value) != length:
        raise ValueError(f"{name} must have length {length}, got {list(value)!r}")
    out = tuple(int(v) for v in value)
    if any(v <= 0 for v in out):
        raise ValueError(f"{name} values must be positive, got {list(value)!r}")
    return out


def patchify_video_latent(
    latent: torch.Tensor, *, patch_size: Sequence[int] = (1, 2, 2)
) -> torch.Tensor:
    """``[B, C, T, H, W]`` -> ``[B*t*h*w, C*pt*ph*pw]``."""
    if latent.ndim != 5:
        raise ValueError(f"video latent must be rank 5, got shape {list(latent.shape)}")
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    b, c, full_t, full_h, full_w = (int(d) for d in latent.shape)
    if full_t % pt or full_h % ph or full_w % pw:
        raise ValueError(
            f"latent dims {list(latent.shape)} not divisible by patch "
            f"{[pt, ph, pw]}"
        )
    t, h, w = full_t // pt, full_h // ph, full_w // pw
    packed = latent.reshape(b, c, t, pt, h, ph, w, pw)
    packed = torch.einsum("nctrhpwq->nthwcrpq", packed)
    return packed.reshape(b * t * h * w, c * pt * ph * pw).contiguous()


def unpatchify_video_tokens(
    rows: torch.Tensor,
    *,
    latent_shape: Sequence[int],
    patch_size: Sequence[int] = (1, 2, 2),
) -> torch.Tensor:
    """``[N, C*pt*ph*pw]`` -> ``[B, C, T, H, W]``. ``latent_shape`` is (t,h,w,C)."""
    if rows.ndim != 2:
        raise ValueError(f"token rows must be rank 2, got {list(rows.shape)}")
    t, h, w, channel = _int_tuple(latent_shape, "latent_shape", 4)
    pt, ph, pw = _int_tuple(patch_size, "patch_size", 3)
    expected = pt * ph * pw * channel
    if int(rows.shape[-1]) != expected:
        raise ValueError(
            f"token width {int(rows.shape[-1])} != patch volume x channel {expected}"
        )
    per_sample = t * h * w
    if int(rows.shape[0]) % per_sample:
        raise ValueError(
            f"row count {int(rows.shape[0])} not divisible by t*h*w {per_sample}"
        )
    packed = rows.reshape(-1, t, h, w, channel, pt, ph, pw)
    latent = torch.einsum("nthwcrpq->nctrhpwq", packed)
    return latent.reshape(-1, channel, t * pt, h * ph, w * pw).contiguous()


def unpack_audio_tokens(
    rows: torch.Tensor, *, audio_t: int, audio_channel: int = 2
) -> torch.Tensor:
    """``[audio_t, D]`` -> ``[C, D, audio_t // C]`` for the audio VAE.

    ``audio_t`` here is the *row* count (latent steps x channels); rows are
    channel-major, matching how the packed sequence lays them out.
    """
    if rows.ndim != 2:
        raise ValueError(f"audio token rows must be rank 2, got {list(rows.shape)}")
    audio_t = int(audio_t)
    audio_channel = int(audio_channel)
    if audio_t <= 0 or audio_channel <= 0:
        raise ValueError(
            f"audio_t and audio_channel must be positive, got {audio_t}, "
            f"{audio_channel}"
        )
    if int(rows.shape[0]) != audio_t:
        raise ValueError(f"audio rows {int(rows.shape[0])} != audio_t {audio_t}")
    if audio_t % audio_channel:
        raise ValueError(
            f"audio_t {audio_t} not divisible by audio_channel {audio_channel}"
        )
    native = rows.reshape(audio_channel, audio_t // audio_channel, int(rows.shape[-1]))
    return native.permute(0, 2, 1).contiguous()
