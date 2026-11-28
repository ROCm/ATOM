# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""SwiGLU-OAI fused with the MXFP8 quant of its output (MiniMax-M3.1 TP1).

``_swiglu_oai_kernel``'s expressions, rounded to bf16 exactly as it stores them,
then ``fp8_groups`` (``quantize_fp8``'s group math) on those bf16 values in
registers: the e4m3 bytes and e8m0 scales equal ``quantize_fp8(swiglu_oai_split(
gate_up))`` byte for byte, without the bf16 round trip. The w4 variant writes the
scales in the shuffle_scale_w4 layout the gfx950 8-wave MXFP8 GEMM reads (32-row x
256-column tiles: a tile's 256 scale bytes are one block of that layout).
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


@triton.jit
def _swiglu_oai_mxfp8_w4_kernel(
    gate_up_ptr,
    y_ptr,
    s_ptr,
    M,
    n_inter: tl.constexpr,
    stride_gm: tl.constexpr,
    alpha,
    beta,
    limit,
    has_limit: tl.constexpr,
):
    rb = tl.program_id(0)
    kb = tl.program_id(1)
    rows = rb * 32 + tl.arange(0, 32)
    cols = kb * 256 + tl.arange(0, 256)
    rmask = (rows < M)[:, None]
    base = rows.to(tl.int64)[:, None] * stride_gm
    gate = tl.load(gate_up_ptr + base + cols[None, :], rmask, other=0.0).to(tl.float32)
    up = tl.load(gate_up_ptr + base + (n_inter + cols)[None, :], rmask, other=0.0).to(
        tl.float32
    )
    if has_limit:
        gate = tl.minimum(gate, limit)
        up = tl.minimum(tl.maximum(up, -limit), limit)
    out = gate * tl.sigmoid(alpha * gate) * (up + beta)
    ob = out.to(tl.bfloat16).to(tl.float32)
    q, code, _scale = fp8_groups(tl.reshape(ob, (256, 32)))
    tl.store(
        y_ptr + rows.to(tl.int64)[:, None] * n_inter + cols[None, :],
        tl.reshape(q, (32, 256)),
        rmask,
    )
    gi = tl.arange(0, 256)
    r = gi // 8
    g = gi % 8
    # word (g % 4) * 16 + r % 16 of the block, byte (g // 4) * 2 + r // 16
    dst = (
        (rb * (n_inter // 256) + kb) * 256
        + ((g % 4) * 16 + r % 16) * 4
        + (g // 4) * 2
        + r // 16
    )
    tl.store(s_ptr + dst, code.to(tl.uint8))


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


def swiglu_oai_mxfp8_w4(gate_up: torch.Tensor, alpha, beta, limit):
    """As ``swiglu_oai_mxfp8``, the scales in the shuffle_scale_w4 layout over m
    padded to 256 (rows past m unset; the GEMM drops them)."""
    x2 = gate_up.reshape(-1, gate_up.shape[-1])
    m = x2.shape[0]
    n_inter = x2.shape[1] // 2
    if n_inter % 256 or x2.stride(1) != 1:
        raise ValueError("SwiGLU+MXFP8 w4 needs I % 256 == 0 and unit column stride")
    y = torch.empty((m, n_inter), dtype=_FP8, device=x2.device)
    s = torch.empty(
        ((m + 255) // 256 * 256, n_inter // 32), dtype=torch.uint8, device=x2.device
    )
    if m:
        _swiglu_oai_mxfp8_w4_kernel[(triton.cdiv(m, 32), n_inter // 256)](
            x2,
            y,
            s,
            m,
            n_inter,
            x2.stride(0),
            float(alpha),
            float(beta),
            0.0 if limit is None else float(limit),
            has_limit=limit is not None,
            num_warps=4,
        )
    return y, s
