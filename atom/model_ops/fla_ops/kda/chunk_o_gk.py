# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/gla/chunk.py). The original source code was licensed
# under the MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted for ATOM:
#   - Only chunk_gla_fwd_kernel_o and its wrapper are copied; the kernel itself
#     is unmodified.
#   - The wrapper accepts a caller-provided `o`, replacing
#     `o = torch.zeros_like(v)` and the caller's subsequent `out.copy_`.
#
# CALLER CONTRACT: when `o` is passed, a varlen padding tail is NOT written.
# In the varlen case the grid covers only NT = len(chunk_indices) chunks, which
# tile [cu_seqlens[0], cu_seqlens[-1]); rows at t >= cu_seqlens[-1] have NO
# program launched for them at all -- the store mask is irrelevant, the store
# never executes. Upstream's zeros_like hid this. Verified: T_buf=192 with
# cu_seqlens[-1]=130 leaves rows 130..191 untouched. The caller MUST zero (or
# otherwise own) any rows at t >= cu_seqlens[-1] before the call.
# The NON-varlen case has no such tail: the grid is NT = cdiv(T, BT) and tiles
# all of T, so every row of `o` is covered.
#
# WARNING: base-2 (`exp2`). See this package's __init__ docstring.

import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import autotune_cache_kwargs


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@fla_cache_autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["BT", "HV", "STATE_V_FIRST"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(
            chunk_indices + i_t * 2 + 1
        ).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    m_s = tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]

    q += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    v += (bos * HV + i_hv) * V
    o += (bos * HV + i_hv) * V
    h += (i_tg * HV + i_hv).to(tl.int64) * K * V
    A += (bos * HV + i_hv) * BT

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    o_t = i_t * BT + tl.arange(0, BT)
    o_v = i_v * BV + tl.arange(0, BV)
    o_i = tl.arange(0, BT)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_A = m_t[:, None] & (o_i[None, :] < BT)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qk = m_t[:, None] & m_k[None, :]
        p_q = q + o_t[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_t[:, None] * (HV * K) + o_k[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            m_h = m_v[:, None] & m_k[None, :]
        else:
            p_h = h + o_k[:, None] * V + o_v[None, :]
            m_h = m_k[:, None] & m_v[None, :]

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qk, other=0.0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=m_qk, other=0.0).to(tl.float32)
        # [BT, BK]
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        if i_k >= 0:
            if STATE_V_FIRST:
                b_o += tl.dot(b_qg, tl.trans(b_h).to(b_qg.dtype))
            else:
                b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
    p_o = o + o_t[:, None] * (HV * V) + o_v[None, :]
    p_A = A + o_t[:, None] * (HV * BT) + o_i[None, :]
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=m_A, other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    o: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns the output tensor; `o` itself when provided."""
    B, T, H, K, HV, V = *q.shape, v.shape[2], v.shape[-1]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if o is None:
        # Upstream zero-fills here ("Please ensure zeros, since vllm will use
        # padding v"). The reason is the GRID, not the store mask: in the varlen
        # case no program is launched for rows at t >= cu_seqlens[-1], so those
        # rows are never stored to under any mask and must already be zero.
        o = torch.zeros_like(v)
    else:
        # Same discipline as fla_ops/chunk_o.py:151,176-186: the kernel assumes
        # stride (HV*V, 1) on the (T, V) plane, so a non-contiguous or
        # wrong-dtype buffer would silently corrupt rather than fail.
        assert o.shape == v.shape, (
            f"chunk_gla_fwd_o_gk: caller-provided o.shape {tuple(o.shape)} != "
            f"v.shape {tuple(v.shape)}"
        )
        assert o.dtype == v.dtype, (
            f"chunk_gla_fwd_o_gk: caller-provided o.dtype {o.dtype} != v.dtype "
            f"{v.dtype}"
        )
        assert (
            o.is_contiguous()
        ), "chunk_gla_fwd_o_gk: caller-provided o must be contiguous"

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * HV)

    chunk_gla_fwd_kernel_o[grid](
        q=q,
        v=v,
        g=g,
        h=h,
        o=o,
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return o
