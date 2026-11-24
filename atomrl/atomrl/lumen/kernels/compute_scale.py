# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Single-element Triton kernel to compute FP8 quantisation scale.

Replaces the 4-op PyTorch sequence ``div + gt + ones_like + where`` inside
``ScalingManager._compute_scale`` with a single GPU launch.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _compute_scale_kernel(amax_ptr, out_ptr, effective_max_recip: tl.constexpr):
    """Compute ``scale = amax / effective_max`` with fallback to 1.0 when the
    result is not a safe normal float. A zero or subnormal scale overflows
    ``1/scale`` and turns an underflowed activation into NaN."""
    amax = tl.load(amax_ptr).to(tl.float32)
    scale = amax * effective_max_recip
    # Subnormal scales overflow ``1/scale`` to inf, and ``0 * inf`` is NaN.
    scale = tl.where(scale > 1.0e-30, scale, 1.0)
    tl.store(out_ptr, scale)


def compute_scale(amax: torch.Tensor, fp8_max: float, margin: int = 0) -> torch.Tensor:
    """Fused scale computation — one kernel launch instead of four.

    Args:
        amax: Scalar or 1-element tensor (GPU, any float dtype).
        fp8_max: Maximum representable FP8 value.
        margin: Power-of-two head-room exponent.

    Returns:
        ``(1,)`` float32 tensor on the same device as *amax*.  Returning
        a 1-D tensor avoids the downstream ``float().reshape(1).contiguous()``
        copies in cast_transpose and GEMM scale preparation.
    """
    effective_max = fp8_max / (2**margin)
    effective_max_recip = 1.0 / effective_max

    amax_f32 = amax.float().reshape(1).contiguous()
    out = torch.empty(1, dtype=torch.float32, device=amax.device)
    _compute_scale_kernel[(1,)](amax_f32, out, effective_max_recip=effective_max_recip)
    return out
