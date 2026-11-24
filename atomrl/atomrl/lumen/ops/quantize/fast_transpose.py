# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fast FP8 matrix transpose dispatch.

Dispatches to AITER's Triton tiled transpose kernel which replaces
``tensor.t().contiguous()`` (a full ``aten::copy_`` kernel) with a single
fused kernel launch.
"""

from __future__ import annotations

import torch


def fast_transpose_fp8(x: torch.Tensor) -> torch.Tensor:
    """Transpose a 2D FP8 tensor via AITER Triton kernel.

    Returns a contiguous (N, M) tensor from a (M, N) input.
    Falls back to ``x.t().contiguous()`` if AITER is unavailable.
    """
    from aiter.ops.triton.quant.fast_transpose import fast_transpose_2d

    return fast_transpose_2d(x)
