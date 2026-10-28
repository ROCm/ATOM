# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gemma RMSNorm and per-token FP8 using the AITER fused kernel."""

import torch
from aiter.utility import dtypes

try:
    import aiter.ops.fused_qk_rmsnorm_group_quant as _fused_norm_ops
except ModuleNotFoundError as exc:
    if exc.name != "aiter.ops.fused_qk_rmsnorm_group_quant":
        raise
    _fused_norm_ops = None

_fused_per_token_quant = getattr(
    _fused_norm_ops, "fused_qk_rmsnorm_per_token_quant", None
)

# The fused AITER kernel keeps one row in registers.
MAX_HIDDEN = 8192


def gemma_rms_norm_fp8(x, weight, epsilon, residual=None):
    assert _fused_per_token_quant is not None
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] <= MAX_HIDDEN and x.stride(1) == 1
    assert dtypes.fp8 == torch.float8_e4m3fn
    out = torch.empty(x.shape, dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else None
    # AITER quantizes the FP32 norm result directly, without rounding to BF16.
    if x.shape[0] > 0:
        _fused_per_token_quant(
            out,
            scale,
            x,
            weight,
            epsilon,
            q_res_out=res_out,
            q_residual=residual,
            gemma_norm=True,
        )
    # First decoder layer keeps the original residual stream by reference.
    return out, scale, x if residual is None else res_out


def gemma_rms_norm_fp8_supported(hidden_width: int) -> bool:
    return _fused_per_token_quant is not None and hidden_width <= MAX_HIDDEN
