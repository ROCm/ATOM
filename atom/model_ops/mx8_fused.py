# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Producer-fused MXFP8 activation quantization for MiniMax-M3.1 (ATOM).

Every function returns exactly the bytes the unfused ATOM path produces:
  swiglu_oai_quant_fp8(gate_up)  == quantize_fp8(swiglu_oai_split(gate_up))
The bf16 rounding of the intermediate the unfused path stores is reproduced before
quantizing, and the group math is ATOM's own ``fp8_groups``.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE, fp8_groups


@triton.jit
def _swiglu_oai_quant_kernel(
    GU, Y, S,
    total_groups, n_inter, stride_gm,
    alpha, beta, limit,
    HAS_LIMIT: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    # flat group index over [rows, n_inter / 32]; Y and S are contiguous
    gidx = tl.program_id(0).to(tl.int64) * BLOCK_G + tl.arange(0, BLOCK_G)
    gmask = gidx < total_groups
    gpr = n_inter // 32
    row = gidx // gpr
    gcol = gidx - row * gpr
    src = row[:, None] * stride_gm + gcol[:, None] * 32 + tl.arange(0, 32)[None, :]
    mask = gmask[:, None]
    gate = tl.load(GU + src, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GU + src + n_inter, mask=mask, other=0.0).to(tl.float32)
    if HAS_LIMIT:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    out = out.to(tl.bfloat16).to(tl.float32)  # the unfused path stores bf16
    q, code, _ = fp8_groups(out)
    tl.store(Y + gidx[:, None] * 32 + tl.arange(0, 32)[None, :], q, mask=mask)
    tl.store(S + gidx, code.to(tl.uint8), mask=gmask)


def swiglu_oai_quant_fp8(gate_up: torch.Tensor, alpha: float, beta: float, limit: float | None):
    """SwiGLU-OAI of a split ``[..., 2I]`` bf16 tensor, quantized to E4M3 with E8M0 group-32
    scales: (``[..., I]`` fp8, ``[..., I/32]`` e8m0)."""
    two_i = gate_up.shape[-1]
    n_inter = two_i // 2
    assert n_inter % 32 == 0
    x2 = gate_up.reshape(-1, two_i)
    assert x2.stride(1) == 1
    m = x2.shape[0]
    y = torch.empty((m, n_inter), dtype=FP8_DTYPE, device=gate_up.device)
    s = torch.empty((m, n_inter // 32), dtype=torch.uint8, device=gate_up.device)
    total = m * (n_inter // 32)
    block_g = 8 if total <= 4096 else 32
    if m:
        _swiglu_oai_quant_kernel[(triton.cdiv(total, block_g),)](
            x2, y, s,
            total, n_inter, x2.stride(0),
            float(alpha), float(beta), 0.0 if limit is None else float(limit),
            HAS_LIMIT=limit is not None,
            BLOCK_G=block_g,
            num_warps=4,
        )
    shape = gate_up.shape[:-1]
    return y.view(*shape, n_inter), s.view(torch.float8_e8m0fnu).view(*shape, n_inter // 32)
