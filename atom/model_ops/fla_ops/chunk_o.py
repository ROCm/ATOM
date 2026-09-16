# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501


import torch

import triton
import triton.language as tl

from .index import prepare_chunk_indices
from .op import exp
from .utils import FLA_GDN_FIX_BT, check_shared_mem, is_nvidia_hopper

BKV_LIST = [64, 128] if check_shared_mem() else [32, 64]
NUM_WARPS = [2, 4] if is_nvidia_hopper else [2, 4, 8]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in BKV_LIST
        for BV in BKV_LIST
        for num_warps in NUM_WARPS
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    q += (bos * Hg + i_h // (H // Hg)) * K
    k += (bos * Hg + i_h // (H // Hg)) * K
    v += (bos * H + i_h) * V
    o += (bos * H + i_h) * V
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        # [BT, BK]
        _p_q_0 = (i_t * BT) + tl.arange(0, BT)
        _p_q_1 = (i_k * BK) + tl.arange(0, BK)
        b_q = tl.load(
            q + _p_q_0[:, None] * (Hg * K) + _p_q_1[None, :] * (1),
            mask=(_p_q_0[:, None] < (T)) & (_p_q_1[None, :] < (K)),
            other=0.0,
        )
        # [BK, BT]
        _p_k_0 = (i_k * BK) + tl.arange(0, BK)
        _p_k_1 = (i_t * BT) + tl.arange(0, BT)
        b_k = tl.load(
            k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (Hg * K),
            mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
            other=0.0,
        )
        # [BK, BV]
        _p_h_0 = (i_k * BK) + tl.arange(0, BK)
        _p_h_1 = (i_v * BV) + tl.arange(0, BV)
        b_h = tl.load(
            h + _p_h_0[:, None] * (V) + _p_h_1[None, :] * (1),
            mask=(_p_h_0[:, None] < (K)) & (_p_h_1[None, :] < (V)),
            other=0.0,
        )

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        _p_g_0 = (i_t * BT) + tl.arange(0, BT)
        b_g = tl.load(g + _p_g_0 * (H), mask=(_p_g_0 < (T)), other=0.0)
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    _p_v_0 = (i_t * BT) + tl.arange(0, BT)
    _p_v_1 = (i_v * BV) + tl.arange(0, BV)
    b_v = tl.load(
        v + _p_v_0[:, None] * (H * V) + _p_v_1[None, :] * (1),
        mask=(_p_v_0[:, None] < (T)) & (_p_v_1[None, :] < (V)),
        other=0.0,
    )

    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    _p_o_0 = (i_t * BT) + tl.arange(0, BT)
    _p_o_1 = (i_v * BV) + tl.arange(0, BV)
    tl.store(
        o + _p_o_0[:, None] * (H * V) + _p_o_1[None, :] * (1),
        b_o.to(o.dtype.element_ty),
        mask=(_p_o_0[:, None] < (T)) & (_p_o_1[None, :] < (V)),
    )


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,  # cumsum of log decay
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    o: torch.Tensor | None = None,
) -> torch.Tensor:
    """Returns the attention output tensor.

    If ``o`` is provided, the kernel writes into it inplace and ``o`` is
    returned. The caller's buffer MUST match ``v``'s shape and dtype and
    be contiguous — the Triton kernel assumes stride ``(H * V, 1)`` along
    ``(T, V)`` for a ``[B, T, H, V]`` layout. The public chunk_gated_delta_rule
    entry point asserts these contracts before .apply() (so input_guard's
    silent .contiguous() clone can't defeat them); we re-assert here as a
    defense-in-depth backstop for any caller that bypasses the public API.
    """
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = 64 if FLA_GDN_FIX_BT else min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    if o is None:
        o = torch.empty_like(v)
    else:
        assert o.shape == v.shape, (
            f"chunk_fwd_o: caller-provided o.shape {tuple(o.shape)} != "
            f"v.shape {tuple(v.shape)}"
        )
        assert o.dtype == v.dtype, (
            f"chunk_fwd_o: caller-provided o.dtype {o.dtype} != v.dtype " f"{v.dtype}"
        )
        assert o.is_contiguous(), (
            "chunk_fwd_o: caller-provided o must be contiguous (kernel "
            "assumes stride (H*V, 1) on the (T, V) plane)"
        )

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), NT, B * H)

    chunk_fwd_kernel_o[grid](
        q,
        k,
        v,
        h,
        g,
        o,
        cu_seqlens,
        chunk_indices,
        scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return o
