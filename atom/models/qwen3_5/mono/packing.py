# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The BF16 MFMA tiles of the small matrices the runner packs (router, GDN b | a). The FP8 and MXFP4
weights the kernel reads in place, in ATOM's own layout (``tests/test_qwen3_5_mono_layout.py`` checks
that layout against aiter's shuffles)."""

import torch


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
