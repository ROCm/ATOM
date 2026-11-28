# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""SwiGLU-OAI fused with the MXFP8 quant of its output (MiniMax-M3.1 TP1).

``_swiglu_oai_kernel``'s expressions, rounded to bf16 exactly as it stores them,
then ``fp8_groups`` (``quantize_fp8``'s group math) on those bf16 values in
registers: the e4m3 bytes and e8m0 scales equal ``quantize_fp8(swiglu_oai_split(
gate_up))`` byte for byte, without the bf16 round trip.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.blockscale_kernels.quantization import fp8_groups

_FP8 = torch.float8_e4m3fn


@triton.jit
def _swiglu_oai_mxfp8_kernel(
    gate_up_ptr,
    y_ptr,
    s_ptr,
    n_inter: tl.constexpr,
    stride_gm: tl.constexpr,
    stride_gn: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
    block_i: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * block_i + tl.arange(0, block_i)
    mask = cols < n_inter
    gate = tl.load(
        gate_up_ptr + row * stride_gm + cols * stride_gn, mask=mask, other=0.0
    ).to(tl.float32)
    up = tl.load(
        gate_up_ptr + row * stride_gm + (n_inter + cols) * stride_gn,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    ob = out.to(tl.bfloat16).to(tl.float32)
    G: tl.constexpr = block_i // 32
    q, code, _scale = fp8_groups(tl.reshape(ob, (G, 32)))
    qcols = tile * block_i + tl.arange(0, G)[:, None] * 32 + tl.arange(0, 32)[None, :]
    tl.store(y_ptr + row.to(tl.int64) * n_inter + qcols, q, qcols < n_inter)
    g = tile * G + tl.arange(0, G)
    tl.store(
        s_ptr + row.to(tl.int64) * (n_inter // 32) + g,
        code.to(tl.uint8),
        g < n_inter // 32,
    )


def swiglu_oai_mxfp8(gate_up: torch.Tensor, alpha, beta, limit):
    """[m, 2I] split-layout gate_up -> (e4m3 [m, I], e8m0 bytes [m, I/32])."""
    x2 = gate_up.reshape(-1, gate_up.shape[-1])
    m = x2.shape[0]
    n_inter = x2.shape[1] // 2
    if n_inter % 512:
        raise ValueError(f"SwiGLU+MXFP8 needs I % 512 == 0, got {n_inter}")
    y = torch.empty((m, n_inter), dtype=_FP8, device=x2.device)
    s = torch.empty((m, n_inter // 32), dtype=torch.uint8, device=x2.device)
    if m:
        block_i = 512 if n_inter >= 2048 else 256
        _swiglu_oai_mxfp8_kernel[(m, triton.cdiv(n_inter, block_i))](
            x2,
            y,
            s,
            n_inter,
            x2.stride(0),
            x2.stride(1),
            float(alpha),
            float(beta),
            0.0 if limit is None else float(limit),
            has_limit=limit is not None,
            block_i=block_i,
            num_warps=4,
        )
    return y, s
