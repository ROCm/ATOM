# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_MTP_ROWS_IN_POST_ATTN_NORM: the MTP drafter's step-0 row selection in the post-attention norm.

Step 0 of a draft keeps one row a sequence after the attention (`select_rows`, ATOM_MTP_SELECT_ROWS_BEFORE_MOE): the
decoder layer runs the post-attention residual add + Gemma RMSNorm over every row (aiter's `_gluon_fused_rms_kernel`,
HAS_RES), then two `index_select`s (normed rows, residual rows). Here the norm kernel itself reads only the selected
rows and writes them compacted: one launch instead of three. Row-wise math in the kernel's own layout (BLOCK_M 1,
WARPS_N 4, 8 contiguous elements a lane), so each kept row is bitwise the unfused one."""

import os

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

ENABLED = os.environ.get("ATOM_MTP_ROWS_IN_POST_ATTN_NORM", "0") == "1"


@gluon.jit
def _gluon_fused_rms_rows_kernel(
    x_ptr, res_ptr, w_ptr, out_ptr, out_res_ptr, rows_ptr,
    stride_xm, stride_resm, stride_outm, stride_out_resm, eps,
    N: gl.constexpr, BLOCK_N: gl.constexpr, WARPS_N: gl.constexpr, GEMMA: gl.constexpr,
):
    # aiter _gluon_fused_rms_kernel (BLOCK_M 1, HAS_RES) reading source row rows[i], writing row i.
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [1, WARPS_N], [1, 0])
    pid = gl.program_id(0)
    offs_m = pid + gl.arange(0, 1, layout=gl.SliceLayout(1, layout))
    src = gl.load(rows_ptr + offs_m).to(gl.int64)
    offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, layout))
    mask_n = offs_n < N
    mask = mask_n[None, :]
    x = gl.load(x_ptr + src[:, None] * stride_xm + offs_n[None, :], mask=mask, other=0.0).to(gl.float32)
    res = gl.load(res_ptr + src[:, None] * stride_resm + offs_n[None, :], mask=mask, other=0.0)
    w = gl.load(w_ptr + offs_n, mask=mask_n, other=0.0).to(gl.float32)
    x = x + res.to(gl.float32)
    dst = offs_m.to(gl.int64)
    gl.store(out_res_ptr + dst[:, None] * stride_out_resm + offs_n[None, :], x.to(out_res_ptr.dtype.element_ty),
             mask=mask)
    rstd = gl.rsqrt(gl.sum(x * x, axis=1) / N + eps)
    if GEMMA:
        w = w + 1.0
    out = x * rstd[:, None] * w[None, :]
    gl.store(out_ptr + dst[:, None] * stride_outm + offs_n[None, :], out.to(out_ptr.dtype.element_ty), mask=mask)


def rms_add_rows(x, res, weight, eps, rows, gemma=True):
    """(norm(x + res)[rows], (x + res)[rows]) in one launch (aiter fused_rmsnorm_add's gfx1250 gluon kernel math)."""
    n = x.shape[1]
    m = rows.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    out_res = torch.empty((m, n), dtype=res.dtype, device=x.device)
    if m == 0:
        return out, out_res
    _gluon_fused_rms_rows_kernel[(m,)](
        x, res, weight, out, out_res, rows,
        x.stride(0), res.stride(0), out.stride(0), out_res.stride(0), eps,
        N=n, BLOCK_N=n, WARPS_N=4, GEMMA=gemma, num_warps=4,
    )
    return out, out_res


def try_post_attn_norm_rows(norm, hidden_states, residual, rows):
    """The decoder layer's `norm(hidden, residual)` + `index_select(.., rows)` of both, or None where the norm is not
    the plain gfx1250 gluon Gemma kernel this reproduces (2048-wide bf16 rows, TP-local, no quant / all-reduce)."""
    if not ENABLED or torch.compiler.is_compiling():
        return None
    from atom.model_ops.layernorm import GemmaRMSNorm, gluon_gemma_rmsnorm

    if (
        not isinstance(norm, GemmaRMSNorm)
        or gluon_gemma_rmsnorm() is None
        or norm.weight.dtype != hidden_states.dtype
        or getattr(norm, "use_fused_quant", False)
        or getattr(norm, "fused_allreduce", False)
        or residual is None
        or hidden_states.dim() != 2
        or hidden_states.shape != residual.shape
        or hidden_states.shape[1] != 2048
        or hidden_states.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or hidden_states.stride(1) != 1
        or residual.stride(1) != 1
        or rows.dim() != 1
        or rows.dtype not in (torch.int32, torch.int64)
    ):
        return None
    return rms_add_rows(hidden_states, residual, norm.weight, norm.variance_epsilon, rows)
