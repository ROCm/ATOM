# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Fused SwiGLU + FP8 quantization bridge.

When enabled via ``LUMEN_FUSED_SWIGLU_QUANT=1``, the SwiGLU activation
stores an FP8-quantized MLP hidden state. The next
:func:`~atomrl.lumen.modules.parallel_linear._do_gemm` that quantizes
activations for FP8 picks it up via :func:`pop_swiglu_fp8_cache` and skips
a separate SiLU + mul + quant pass.

Uses AITER ``dynamic_per_tensor_quant_fp8_i8`` on the BF16 SwiGLU output,
fusing amax computation and quantization in a single pass.
"""

from __future__ import annotations

import threading

import torch
from aiter.ops.enum import QuantType
from aiter.ops.quant import get_hip_quant
from aiter.ops.triton.quant import dynamic_per_tensor_quant_fp8_i8

_cache = threading.local()

# Activation scale granularity the downstream FP8 GEMM expects. Set once by
# enable_fp8_for_parallel_linear so the cached SwiGLU scale layout matches the
# consuming fc2 GEMM (per-tensor → 1D scalar scale; blockwise → 2D 1×block scale).
_fused_scaling = {"type": "dynamic", "block_size": 128}


def set_fused_swiglu_scaling(scaling_type, block_size: int = 128) -> None:
    """Record the global FP8 activation scale granularity for the fusion bridge.

    ``scaling_type`` may be a ``ScalingType`` enum or its string value. Called
    from :func:`atomrl.lumen.models.megatron.enable_fp8_for_parallel_linear` when modules
    are switched to FP8, so :func:`try_fused_swiglu_fp8` produces a scale whose
    layout matches what the fc2 GEMM consumes (avoids feeding a per-tensor 1D
    scale into a blockwise GEMM, which expects a 2D ``(M, K/block)`` scale).
    """
    _fused_scaling["type"] = getattr(scaling_type, "value", scaling_type)
    _fused_scaling["block_size"] = block_size or 128


def try_fused_swiglu_fp8(swiglu_input: torch.Tensor, bf16_output: torch.Tensor) -> None:
    """Run fused SiLU*mul + FP8 quant and cache ``(fp8, scale, amax)`` for the next FP8 GEMM.

    Uses AITER dynamic per-tensor quantization with amax output on the
    already-computed bf16_output, fusing amax computation and quantization
    in one pass.  The amax is passed through to the scaling manager so it
    can skip its own ``fused_amax_abs`` call.
    """
    if not swiglu_input.is_cuda:
        return

    from atomrl.lumen.quantize.config import _get_float8_e4m3

    fp8_dtype = _get_float8_e4m3()

    out_2d = bf16_output.detach().reshape(-1, bf16_output.shape[-1])
    if not out_2d.is_contiguous():
        out_2d = out_2d.contiguous()

    scaling_type = _fused_scaling["type"]

    if scaling_type in ("mxfp4", "mxfp8"):
        # This bridge only emits FP8 with a float scale; an MX GEMM needs E8M0
        # block scales and would have to discard it. Skip the quant pass.
        return

    if scaling_type in ("blockwise", "blockwise2d"):
        # blockwise activation quant: 1×block scale, matching
        # _quant_blockwise2d_activation so the consuming fc2 blockscale GEMM
        # (incl. bpreshuffle, which does scale_a.transpose(0,1)) receives a 2D
        # (M, K/block) scale. A per-tensor 1D scale here would be misread by the
        # blockscale kernel and crash. Only per_1x128 (block_size=128) is
        # supported by the HIP kernel; for any other block size, skip fusion and
        # let fc2 re-quantize via its normal blockwise2d path.
        block_size = _fused_scaling["block_size"]
        if block_size != 128 or out_2d.shape[-1] % block_size != 0:
            return
        try:
            quant_fn = get_hip_quant(QuantType.per_1x128)
            qx, scale_out = quant_fn(out_2d, quant_dtype=fp8_dtype)
        except RuntimeError:
            return

        _cache.fp8_data = qx
        _cache.scale = scale_out
        # blockwise is dynamic per-block; there is no delayed amax history to feed.
        _cache.amax = None
        _cache.valid = True
        return

    # per-tensor (dynamic/delayed): single scalar scale + fused amax in one pass.
    try:
        qx = torch.empty_like(out_2d, dtype=fp8_dtype)
        scale_out = torch.empty(1, dtype=torch.float32, device=out_2d.device)
        dynamic_per_tensor_quant_fp8_i8(qx, out_2d, scale_out)
        amax_out = scale_out * torch.finfo(fp8_dtype).max

        _cache.fp8_data = qx
        _cache.scale = scale_out
        _cache.amax = amax_out
        _cache.valid = True
    except RuntimeError:
        return


def discard_swiglu_fp8_cache() -> None:
    """Drop any pending fused SwiGLU FP8 cache without consuming it."""
    _cache.valid = False
    _cache.fp8_data = None
    _cache.scale = None
    _cache.amax = None


def pop_swiglu_fp8_cache():
    """Pop cached ``(fp8_data, scale)`` if set; otherwise return ``None``.

    Also returns ``amax`` as the third element when available.
    """
    if getattr(_cache, "valid", False):
        result = (_cache.fp8_data, _cache.scale, getattr(_cache, "amax", None))
        _cache.valid = False
        _cache.fp8_data = None
        _cache.scale = None
        _cache.amax = None
        return result
    return None
