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


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T
    _p_beta_0 = (i_t * BT) + tl.arange(0, BT)
    b_beta = tl.load(
        beta + bos * H + i_h + _p_beta_0 * (H), mask=(_p_beta_0 < (T)), other=0.0
    )
    _p_A_0 = (i_t * BT) + tl.arange(0, BT)
    _p_A_1 = (0) + tl.arange(0, BT)
    b_A = tl.load(
        A + (bos * H + i_h) * BT + _p_A_0[:, None] * (H * BT) + _p_A_1[None, :] * (1),
        mask=(_p_A_0[:, None] < (T)) & (_p_A_1[None, :] < (BT)),
        other=0.0,
    )
    _p_g_0 = (i_t * BT) + tl.arange(0, BT)
    b_g = tl.exp(
        tl.load(g + (bos * H + i_h) + _p_g_0 * (H), mask=(_p_g_0 < (T)), other=0.0)
    )

    for i_v in range(tl.cdiv(V, BV)):
        _p_v_0 = (i_t * BT) + tl.arange(0, BT)
        _p_v_1 = (i_v * BV) + tl.arange(0, BV)
        b_v = tl.load(
            v + (bos * H + i_h) * V + _p_v_0[:, None] * (H * V) + _p_v_1[None, :] * (1),
            mask=(_p_v_0[:, None] < (T)) & (_p_v_1[None, :] < (V)),
            other=0.0,
        )
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_u = tl.dot(b_A, b_vb, allow_tf32=False)
        _p_u_0 = (i_t * BT) + tl.arange(0, BT)
        _p_u_1 = (i_v * BV) + tl.arange(0, BV)
        tl.store(
            u + (bos * H + i_h) * V + _p_u_0[:, None] * (H * V) + _p_u_1[None, :] * (1),
            b_u.to(u.dtype.element_ty),
            mask=(_p_u_0[:, None] < (T)) & (_p_u_1[None, :] < (V)),
        )

    for i_k in range(tl.cdiv(K, BK)):
        _p_k_0 = (i_t * BT) + tl.arange(0, BT)
        _p_k_1 = (i_k * BK) + tl.arange(0, BK)
        b_k = tl.load(
            k
            + (bos * Hg + i_h // (H // Hg)) * K
            + _p_k_0[:, None] * (Hg * K)
            + _p_k_1[None, :] * (1),
            mask=(_p_k_0[:, None] < (T)) & (_p_k_1[None, :] < (K)),
            other=0.0,
        )
        b_kb = (b_k * b_beta[:, None] * b_g[:, None]).to(b_k.dtype)
        b_w = tl.dot(b_A, b_kb)
        _p_w_0 = (i_t * BT) + tl.arange(0, BT)
        _p_w_1 = (i_k * BK) + tl.arange(0, BK)
        tl.store(
            w + (bos * H + i_h) * K + _p_w_0[:, None] * (H * K) + _p_w_1[None, :] * (1),
            b_w.to(w.dtype.element_ty),
            mask=(_p_w_0[:, None] < (T)) & (_p_w_1[None, :] < (K)),
        )


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    BK = 64
    BV = 64
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    recompute_w_u_fwd_kernel[(NT, B * H)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
    )
    return w, u
