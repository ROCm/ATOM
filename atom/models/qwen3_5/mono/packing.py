# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The layouts the kernel reads: aiter's ``shuffle_weight`` (16, 16) FP8 and MXFP4 banks and their
E8M0 scales, as ATOM holds them (``tests/test_qwen3_5_mono_layout.py`` checks them against aiter),
and the BF16 MFMA tiles of the small matrices the runner packs (router, GDN b | a)."""

import torch


def shuffle_fp8(q: torch.Tensor) -> torch.Tensor:
    """FP8 ``[..., N, K]`` in aiter's ``shuffle_weight(layout=(16, 16))`` order, the layout ATOM holds its
    FP8 GEMM / MoE weights in: 16-row, 32-K tiles of two 16-K halves, each 16 rows x 16 contiguous K.
    """

    *_lead, rows, k = q.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"FP8 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w8 = q.view(torch.uint8).reshape(-1, rows // 16, 16, k // 32, 2, 16)
    return w8.permute(0, 1, 3, 4, 2, 5).contiguous().view(-1)


def pack_bf16(w: torch.Tensor) -> torch.Tensor:
    """Pack BF16 ``[N, K]`` for the kernel's 16-row, 64-K MFMA tiles."""

    if w.ndim != 2:
        raise ValueError(f"BF16 packing expects a matrix, got shape {tuple(w.shape)}")
    rows, k = w.shape
    if rows % 16 or k % 64:
        raise ValueError(
            f"BF16 matrix dimensions must be divisible by (16, 64), got {(rows, k)}"
        )
    w16 = w.view(torch.int16).reshape(rows // 16, 16, k // 64, 2, 4, 8)
    return w16.permute(0, 2, 3, 4, 1, 5).contiguous().view(-1)


def pack_mxfp4(q: torch.Tensor) -> torch.Tensor:
    """An MXFP4 bank [E, N, K / 2] in ATOM's gfx950 layout for gate / up NOT interleaved (aiter
    ``moe_shuffle_weight(is_guinterleave=False)``: a [gate | up] bank shuffles as plain rows).
    """
    q = q.view(torch.uint8)
    e, n, kp = q.shape
    if n % 16 or kp % 64:
        raise ValueError(
            f"MXFP4 bank dimensions must be divisible by (16, 128), got {(n, kp * 2)}"
        )
    return (
        q.view(e, n // 16, 16, kp // 64, 4, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(e, n, kp)
    )


def pack_mxfp4_scales(s: torch.Tensor) -> torch.Tensor:
    """E8M0 scales [E, N, K / 32] in ATOM's gfx950 layout (aiter ``moe_shuffle_scale``,
    ``is_guinterleave=False``); K / 32 padded to a multiple of 8 with 1.0 (0x7F)."""
    s = s.view(torch.uint8)
    e, n, kb = s.shape
    kb8 = -(-kb // 8) * 8
    if kb8 != kb:
        s = torch.cat(
            [s, torch.full((e, n, kb8 - kb), 0x7F, dtype=torch.uint8, device=s.device)],
            dim=-1,
        )
    return (
        s.view(e, n // 32, 2, 16, kb8 // 8, 2, 4)
        .permute(0, 1, 4, 6, 3, 5, 2)
        .contiguous()
        .view(e, n, kb8)
    )
