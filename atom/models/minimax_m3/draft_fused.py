# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 activation producer for the MiniMax-M3.1 DSpark draft's linears.

Every draft linear is native MXFP8 (E4M3 values, an E8M0 scale per 32
columns), and each one quantized its bf16 input in a pass of its own
(`quantize_fp8`) right after the Gemma RMSNorm in front of qkv_proj and
gate_up_proj. This kernel writes the quantized pair directly. The value
quantized is the bf16 the unfused norm would have written, and the groups go
through the same `fp8_groups`, so the pair matches norm-then-quantize up to the
norm's own fp32 reduction order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE, fp8_groups

_E8M0 = torch.float8_e8m0fnu


@triton.jit
def _gemma_rmsnorm_mxfp8_kernel(
    x_ptr,
    res_ptr,  # residual in, or x_ptr when HAS_RES is False
    w_ptr,
    q_ptr,  # [M, K] fp8 out
    s_ptr,  # [M, K / 32] uint8 E8M0 codes out
    res_out_ptr,  # [M, K] bf16 out: x + residual (HAS_RES)
    stride_x,
    stride_r,
    eps,
    K: tl.constexpr,
    G: tl.constexpr,  # K / 32
    GP: tl.constexpr,  # G rounded up to a power of two
    HAS_RES: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    groups = tl.arange(0, GP)
    offs = groups[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = offs < K
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        # the norm reads the fp32 sum; the residual carried on is its bf16
        # (aiter's fused_qk_rmsnorm_group_quant, which GemmaRMSNorm runs)
        r = tl.load(res_ptr + row * stride_r + offs, mask=mask, other=0.0)
        x = x + r.to(tl.float32)
        tl.store(res_out_ptr + row * K + offs, x.to(tl.bfloat16), mask=mask)
    var = tl.sum(tl.sum(x * x, 1), 0) / K
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * tl.rsqrt(var + eps) * (1.0 + w)).to(tl.bfloat16).to(tl.float32)
    q, code, _ = fp8_groups(y)
    tl.store(q_ptr + row * K + offs, q, mask=mask)
    tl.store(s_ptr + row * G + groups, code.to(tl.uint8), mask=groups < G)


def gemma_rmsnorm_mxfp8(
    x: torch.Tensor,  # [M, K] bf16
    weight: torch.Tensor,  # [K]
    eps: float,
    residual: torch.Tensor | None = None,
):
    """``GemmaRMSNorm(x [+ residual])`` quantized to MXFP8: ``(values [M, K]
    E4M3, scales [M, K / 32] E8M0, residual out or None)``."""
    M, K = x.shape
    assert x.dtype == torch.bfloat16 and K % 32 == 0 and x.stride(1) == 1
    assert residual is None or (
        residual.shape == x.shape
        and residual.dtype == x.dtype
        and residual.stride(1) == 1
    )
    q = torch.empty(M, K, dtype=FP8_DTYPE, device=x.device)
    s = torch.empty(M, K // 32, dtype=torch.uint8, device=x.device)
    res_out = torch.empty_like(x) if residual is not None else None
    if M:
        _gemma_rmsnorm_mxfp8_kernel[(M,)](
            x, x if residual is None else residual, weight, q, s,
            x if res_out is None else res_out,
            x.stride(0), 0 if residual is None else residual.stride(0), float(eps),
            K=K, G=K // 32, GP=triton.next_power_of_2(K // 32),
            HAS_RES=residual is not None, num_warps=8,
        )  # fmt: skip
    return q, s.view(_E8M0), res_out
