# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_TAIL_LMHEAD_QUANT: the MTP draft LM head's activation quant in the
final norm's epilogue.

With ATOM_MTP_DRAFT_LM_HEAD_FP8 a draft pass runs the MoE gather + tail
RMSNorm (`_moe_gather_tail_rmsnorm_kernel`, one program a row), then aiter's
`dynamic_per_group_scaled_quant` (1 x 128 groups, fp8 e4m3, e8m0 scales,
transposed) on the normed rows, then the FP8 head GEMM. Each tail program
already holds its whole normed row, so it also writes the row's fp8 codes and
its 16 scale bytes (at `g * rows + row`, the transposed layout): one launch
less, the same bytes (`mx32_quant_groups_bits`' arithmetic, which is aiter's
e8m0 group quant per element, over 128-wide groups). The bf16 normed rows are
still stored (the next pass's hidden input reads them).

The pair is handed to `DraftLMHeadFP8.mm` keyed by the normed tensor (as the
folded router hands its routing to the experts), so the head skips its quant
when it is given exactly those rows and quantizes as before otherwise."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from atom.model_ops.moe_tail_norm import _gather_tail_row, can_fuse_gather_tail
from atom.model_ops.mxfp8_1x32 import mx32_e4m3_bits, mx32_e8m0

ENABLED = os.getenv("ATOM_MTP_TAIL_LMHEAD_QUANT", "0") == "1"
GROUP = 128
# Decode sizes only (one program a row); prefill keeps its kernels.
MAX_ROWS = 1024

# (normed, q, s): the quant of `normed` for the next head GEMM on it.
_PREPARED: list = []


def take_prepared(x: torch.Tensor):
    """(q, s) prepared for exactly `x`, else None (consumed either way)."""
    if not _PREPARED:
        return None
    p = _PREPARED.pop()
    n = p[0]
    if (
        n.data_ptr() != x.data_ptr()
        or n.shape != x.shape
        or n.stride() != x.stride()
        or n.dtype != x.dtype
    ):
        return None
    return p[1], p[2]


@triton.jit
def _moe_gather_tail_rmsnorm_fp8g128_kernel(
    grouped_ptr,
    route_rows_ptr,
    route_w_ptr,
    nvt_ptr,
    shared_ptr,
    gate_ptr,
    residual_ptr,
    weight_ptr,
    rows_ptr,
    out_ptr,
    q_ptr,
    s_ptr,
    stride_grouped,
    stride_shared,
    stride_gate,
    stride_residual,
    stride_out,
    n_out,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_GATE: tl.constexpr,
    GATHER: tl.constexpr,
    HAS_NVT: tl.constexpr,
):
    """`_moe_gather_tail_rmsnorm_kernel` (no residual store) that also writes
    the normed row's 1 x 128 e8m0 fp8 quant: codes at q[row], scale byte of
    group g at s[g * n_out + row]."""
    row = tl.program_id(0).to(tl.int64)
    if GATHER:
        src = tl.load(rows_ptr + row).to(tl.int64)
    else:
        src = row
    normed = _gather_tail_row(
        row,
        src,
        grouped_ptr,
        route_rows_ptr,
        route_w_ptr,
        nvt_ptr,
        shared_ptr,
        gate_ptr,
        residual_ptr,
        weight_ptr,
        out_ptr,
        stride_grouped,
        stride_shared,
        stride_gate,
        stride_residual,
        0,
        eps,
        N,
        BLOCK,
        TOPK,
        HAS_GATE,
        False,
        HAS_NVT,
    )
    cols = tl.arange(0, BLOCK)
    tl.store(out_ptr + row * stride_out + cols, normed)
    KG: tl.constexpr = N // 128
    y = tl.reshape(normed.to(tl.float32), (KG, 128))
    amax = tl.max(tl.abs(y), axis=1)
    amax = tl.minimum(tl.maximum(amax, 1e-10), 448.0 * 6.646139978924579e35)  # 2^119
    e, inv = mx32_e8m0(amax)
    v = tl.minimum(tl.maximum(y * inv[:, None], -448.0), 448.0)
    q = mx32_e4m3_bits(v)
    g = tl.arange(0, KG)
    qcols = tl.max_contiguous(g[:, None] * 128 + tl.arange(0, 128)[None, :], [1, 8])
    tl.store(q_ptr + row * N + qcols, q)
    tl.store(s_ptr + g * n_out + row, e.to(tl.uint8))


def applies(moe, residual, weight, rows, write_residual) -> bool:
    if not ENABLED or write_residual:
        return False
    if not can_fuse_gather_tail(moe, residual, weight, rows):
        return False
    hidden = moe.shared.shape[1]
    return hidden == triton.next_power_of_2(hidden) and hidden % GROUP == 0


_COMPILED: dict = {}


def moe_gather_tail_rmsnorm_fp8g128(moe, residual, weight, eps, rows=None):
    """`moe_gather_tail_rmsnorm(..., rows, write_residual=False)` (bitwise the
    same rows), plus their `per_1x128_e8m0_quant(transpose_scale=True)` handed
    to the next `take_prepared` on the returned tensor."""
    from aiter import dtypes

    g, shared, gate = moe
    tokens, hidden = shared.shape
    n_out = tokens if rows is None else rows.shape[0]
    dev = shared.device
    out = torch.empty((n_out, hidden), dtype=shared.dtype, device=dev)
    q = torch.empty((n_out, hidden), dtype=dtypes.fp8, device=dev)
    s = torch.empty((n_out, hidden // GROUP), dtype=dtypes.fp8_e8m0, device=dev)
    if n_out == 0:
        return out
    nvt = g.num_valid_tokens
    args = (
        g.grouped,
        g.route_rows,
        g.weights,
        nvt if nvt is not None else g.route_rows,
        shared,
        gate if gate is not None else shared,
        residual,
        weight,
        rows if rows is not None else g.route_rows,
        out,
        q.view(torch.uint8),
        s.view(torch.uint8),
        g.grouped.stride(0),
        shared.stride(0),
        gate.stride(0) if gate is not None else 0,
        residual.stride(0),
        out.stride(0),
        n_out,
        eps,
    )
    _moe_gather_tail_rmsnorm_fp8g128_kernel[(n_out,)](
        *args,
        N=hidden,
        BLOCK=hidden,
        TOPK=g.route_rows.shape[1],
        HAS_GATE=gate is not None,
        GATHER=rows is not None,
        HAS_NVT=nvt is not None,
        num_warps=4 if hidden <= 2048 else 8,
        enable_fp_fusion=False,
    )
    _PREPARED.clear()
    _PREPARED.append((out, q, s))
    return out
