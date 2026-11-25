# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_EMBED_NORM0_MX32=1 (default off): a decode forward's embedding lookup, layer 0's input
Gemma RMSNorm and the 1x32 MXFP8 quant of layer 0's first projection input in ONE launch.

Served (flag off) layer 0 runs three kernels: F.embedding (torch index_select, ~6-8 us at c64
row counts: a slow small-index gather), aiter's Gluon `_gluon_fused_rms_kernel` (HAS_RES 0,
GEMMA 1) and the standalone quant of the MXFP8 in_proj / qkv_proj (`quant_standalone`, ~2 us).
Here a program a row loads the row's token id, gathers its embedding row, stores it as the
residual (the bf16 rows F.embedding would have returned; layer 0's residual IS the embedding),
normalizes it with `_gluon_fused_rms_kernel`'s code line for line in the same register layout
(the same [1, 8] x 32 lanes x WARPS_N waves BlockedLayout, so gl.sum reduces in the same
order), rounds to bf16 as that kernel stores, and quantizes the bf16 row to fp8 e4m3 + one
e8m0 byte per 32 columns with mxfp8_1x32.mx32_quant_groups' steps (the Gluon form used by
moe_gather_tail_gluon._mx32_store_row), which the module docstring of mxfp8_1x32 states is the
standalone quant's bytes. Result: the same residual, x_fp8 and x_scale bytes.

Scope: rows M < 32 (row-major x_scale, no n32k4 pad rows), local (TP 1) bf16 table, and the
Gluon norm config of (M, N) a non-TDM one with BLOCK_M 1; anything else keeps the served path.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import Tensor
from triton.experimental import gluon
from triton.experimental.gluon import language as gl


def enabled() -> bool:
    # read at call (graph capture) time, so an in-process A/B can toggle it
    return os.environ.get("ATOM_EMBED_NORM0_MX32", "0") == "1"


class EmbedNormMx32(NamedTuple):
    """Layer 0's input as `Qwen3NextDecoderLayer._input_norm_mx32` returns it."""

    x_fp8: Tensor  # [M, N] fp8 e4m3
    x_scale: Tensor  # flat uint8, mxfp8_1x32.alloc layout (row-major first M * N / 32 bytes)
    residual: Tensor  # [M, N] bf16 = embedding rows


@gluon.jit
def _embed_rms_mx32_kernel(
    ids_ptr,
    table_ptr,
    w_ptr,
    res_ptr,
    q_ptr,
    s_ptr,
    M,
    stride_t,
    stride_res,
    eps,
    N: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WARPS_N: gl.constexpr,
):
    # _gluon_fused_rms_kernel's layout with BLOCK_M = 1: a lane holds 8 contiguous elements.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, WARPS_N], [1, 0])
    pid_m = gl.program_id(0)
    offs_m = pid_m + gl.arange(0, 1, layout=gl.SliceLayout(1, layout))
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    rows = gl.minimum(offs_m, M - 1)
    mask_n = offs_n < N
    mask = mask_n[None, :]

    tok = gl.load(ids_ptr + rows).to(gl.int64)
    xb = gl.load(
        table_ptr + tok[:, None] * stride_t + offs_n[None, :], mask=mask, other=0.0
    )
    gl.store(res_ptr + rows[:, None] * stride_res + offs_n[None, :], xb, mask=mask)
    x = xb.to(gl.float32)
    w = gl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(gl.float32)

    # _gluon_fused_rms_kernel (HAS_RES 0, GEMMA 1), line for line
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / N + eps)
    w = w + 1.0
    out = x * rstd[:, None] * w[None, :]
    normed = out.to(res_ptr.dtype.element_ty)

    # mxfp8_1x32.mx32_quant_groups on the bf16 row (moe_gather_tail_gluon._mx32_store_row,
    # with the row-major x_scale of M < 32)
    KG: gl.constexpr = N // 32
    GRP_L: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [WARPS_N, 1], [1, 0])
    KG_L: gl.constexpr = gl.SliceLayout(1, GRP_L)
    y = gl.convert_layout(gl.reshape(normed.to(gl.float32), [KG, 32]), GRP_L)
    amax = gl.max(gl.abs(y), axis=1)
    amax = gl.minimum(gl.maximum(amax, 1e-10), 448.0 * 6.646139978924579e35)  # 2^119
    u = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
    e = ((u >> 23) & 0xFF).to(gl.int32)
    e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).to(gl.int32)
    inv = ((254 - gl.minimum(e, 253)).to(gl.uint32) << 23).to(gl.float32, bitcast=True)
    v = gl.minimum(gl.maximum(y * gl.expand_dims(inv, 1), -448.0), 448.0)
    gq = gl.arange(0, KG, layout=KG_L)
    cq = gl.arange(0, 32, layout=gl.SliceLayout(0, GRP_L))
    row = gl.minimum(pid_m, M - 1).to(gl.int64)
    gl.store(q_ptr + row * N + gq[:, None] * 32 + cq[None, :], v.to(gl.float8e4nv))
    gl.store(s_ptr + row * KG + gq, e.to(gl.uint8))


def _norm_config(m: int, n: int) -> dict | None:
    try:
        from aiter.ops.triton.fusions.fused_rmsnorm_add import _get_config

        return _get_config(m, n)
    except Exception:  # noqa: BLE001
        return None


def embed_norm_mx32(
    input_ids: Tensor, table: Tensor, norm_weight: Tensor, eps: float
) -> EmbedNormMx32 | None:
    """(x_fp8, x_scale, residual) of layer 0, or None where the fused launch does not apply
    (the caller then runs the served embedding + norm + Linear quant)."""
    from atom.model_ops import mxfp8_1x32

    if input_ids.dim() != 1 or input_ids.dtype not in (torch.int32, torch.int64):
        return None
    m = input_ids.shape[0]
    n = table.shape[1]
    if not (0 < m < 32) or n % 128 or n & (n - 1) or n > 8192:
        return None
    if (
        table.dtype != torch.bfloat16
        or norm_weight.dtype != table.dtype
        or table.stride(1) != 1
        or not norm_weight.is_contiguous()
        or not input_ids.is_contiguous()
    ):
        return None
    cfg = _norm_config(m, n)
    if cfg is None or cfg.get("USE_TDM") or cfg.get("BLOCK_M") != 1:
        return None
    q, s = mxfp8_1x32.alloc(m, n, table.device)
    res = torch.empty((m, n), dtype=table.dtype, device=table.device)
    _embed_rms_mx32_kernel[(m,)](
        input_ids,
        table,
        norm_weight,
        res,
        q,
        s,
        m,
        table.stride(0),
        res.stride(0),
        eps,
        N=n,
        BLOCK_N=n,
        WARPS_N=cfg["WARPS_N"],
        num_warps=cfg["WARPS_N"],
    )
    return EmbedNormMx32(q, s, res)
