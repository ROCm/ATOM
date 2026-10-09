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

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp
from .utils import use_cuda_graph

NUM_WARPS = [2, 4, 8, 16]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT"],
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BK, BV]
    b_h1 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([64, BV], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += ((boh * H + i_h) * K * V).to(tl.int64)
    v += ((bos * H + i_h) * V).to(tl.int64)
    k += ((bos * Hg + i_h // (H // Hg)) * K).to(tl.int64)
    w += ((bos * H + i_h) * K).to(tl.int64)
    if SAVE_NEW_VALUE:
        v_new += ((bos * H + i_h) * V).to(tl.int64)
    stride_v = H * V
    stride_h = H * K * V
    stride_k = Hg * K
    stride_w = H * K
    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        _p_h0_1_0 = (0) + tl.arange(0, 64)
        _p_h0_1_1 = (i_v * BV) + tl.arange(0, BV)
        b_h1 += tl.load(
            h0 + _p_h0_1_0[:, None] * (V) + _p_h0_1_1[None, :] * (1),
            mask=(_p_h0_1_0[:, None] < (K)) & (_p_h0_1_1[None, :] < (V)),
            other=0.0,
        ).to(tl.float32)
        if K > 64:
            _p_h0_2_0 = (64) + tl.arange(0, 64)
            _p_h0_2_1 = (i_v * BV) + tl.arange(0, BV)
            b_h2 += tl.load(
                h0 + _p_h0_2_0[:, None] * (V) + _p_h0_2_1[None, :] * (1),
                mask=(_p_h0_2_0[:, None] < (K)) & (_p_h0_2_1[None, :] < (V)),
                other=0.0,
            ).to(tl.float32)
        if K > 128:
            _p_h0_3_0 = (128) + tl.arange(0, 64)
            _p_h0_3_1 = (i_v * BV) + tl.arange(0, BV)
            b_h3 += tl.load(
                h0 + _p_h0_3_0[:, None] * (V) + _p_h0_3_1[None, :] * (1),
                mask=(_p_h0_3_0[:, None] < (K)) & (_p_h0_3_1[None, :] < (V)),
                other=0.0,
            ).to(tl.float32)
        if K > 192:
            _p_h0_4_0 = (192) + tl.arange(0, 64)
            _p_h0_4_1 = (i_v * BV) + tl.arange(0, BV)
            b_h4 += tl.load(
                h0 + _p_h0_4_0[:, None] * (V) + _p_h0_4_1[None, :] * (1),
                mask=(_p_h0_4_0[:, None] < (K)) & (_p_h0_4_1[None, :] < (V)),
                other=0.0,
            ).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        _p_h1_0 = (0) + tl.arange(0, 64)
        _p_h1_1 = (i_v * BV) + tl.arange(0, BV)
        tl.store(
            h + i_t * stride_h + _p_h1_0[:, None] * (V) + _p_h1_1[None, :] * (1),
            b_h1.to(h.dtype.element_ty),
            mask=(_p_h1_0[:, None] < (K)) & (_p_h1_1[None, :] < (V)),
        )
        if K > 64:
            _p_h2_0 = (64) + tl.arange(0, 64)
            _p_h2_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                h + i_t * stride_h + _p_h2_0[:, None] * (V) + _p_h2_1[None, :] * (1),
                b_h2.to(h.dtype.element_ty),
                mask=(_p_h2_0[:, None] < (K)) & (_p_h2_1[None, :] < (V)),
            )
        if K > 128:
            _p_h3_0 = (128) + tl.arange(0, 64)
            _p_h3_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                h + i_t * stride_h + _p_h3_0[:, None] * (V) + _p_h3_1[None, :] * (1),
                b_h3.to(h.dtype.element_ty),
                mask=(_p_h3_0[:, None] < (K)) & (_p_h3_1[None, :] < (V)),
            )
        if K > 192:
            _p_h4_0 = (192) + tl.arange(0, 64)
            _p_h4_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                h + i_t * stride_h + _p_h4_0[:, None] * (V) + _p_h4_1[None, :] * (1),
                b_h4.to(h.dtype.element_ty),
                mask=(_p_h4_0[:, None] < (K)) & (_p_h4_1[None, :] < (V)),
            )

        _p_w_0 = (i_t * BT) + tl.arange(0, BT)
        _p_w_1 = (0) + tl.arange(0, 64)
        b_w = tl.load(
            w + _p_w_0[:, None] * (stride_w) + _p_w_1[None, :] * (1),
            mask=(_p_w_0[:, None] < (T)) & (_p_w_1[None, :] < (K)),
            other=0.0,
        )
        b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            _p_w_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w_1 = (64) + tl.arange(0, 64)
            b_w = tl.load(
                w + _p_w_0[:, None] * (stride_w) + _p_w_1[None, :] * (1),
                mask=(_p_w_0[:, None] < (T)) & (_p_w_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            _p_w_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w_1 = (128) + tl.arange(0, 64)
            b_w = tl.load(
                w + _p_w_0[:, None] * (stride_w) + _p_w_1[None, :] * (1),
                mask=(_p_w_0[:, None] < (T)) & (_p_w_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            _p_w_0 = (i_t * BT) + tl.arange(0, BT)
            _p_w_1 = (192) + tl.arange(0, 64)
            b_w = tl.load(
                w + _p_w_0[:, None] * (stride_w) + _p_w_1[None, :] * (1),
                mask=(_p_w_0[:, None] < (T)) & (_p_w_1[None, :] < (K)),
                other=0.0,
            )
            b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        _p_v_0 = (i_t * BT) + tl.arange(0, BT)
        _p_v_1 = (i_v * BV) + tl.arange(0, BV)
        b_v = (
            tl.load(
                v + _p_v_0[:, None] * (stride_v) + _p_v_1[None, :] * (1),
                mask=(_p_v_0[:, None] < (T)) & (_p_v_1[None, :] < (V)),
                other=0.0,
            )
            - b_v
        )

        if SAVE_NEW_VALUE:
            _p_v_0 = (i_t * BT) + tl.arange(0, BT)
            _p_v_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                v_new + _p_v_0[:, None] * (stride_v) + _p_v_1[None, :] * (1),
                b_v.to(v_new.dtype.element_ty),
                mask=(_p_v_0[:, None] < (T)) & (_p_v_1[None, :] < (V)),
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            _p_g_0 = (i_t * BT) + tl.arange(0, BT)
            b_g = tl.load(
                g + bos * H + i_h + _p_g_0 * (H), mask=(_p_g_0 < (T)), other=0.0
            )
            b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
            b_g_last = exp(b_g_last)
            b_h1 *= b_g_last
            if K > 64:
                b_h2 *= b_g_last
            if K > 128:
                b_h3 *= b_g_last
            if K > 192:
                b_h4 *= b_g_last

        if USE_GK:
            o_k1 = tl.arange(0, 64)
            b_gk_last1 = tl.load(
                gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            )
            b_h1 *= exp(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                )
                b_h2 *= exp(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                )
                b_h3 *= exp(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                )
                b_h4 *= exp(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        _p_k_0 = (0) + tl.arange(0, 64)
        _p_k_1 = (i_t * BT) + tl.arange(0, BT)
        b_k = tl.load(
            k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (stride_k),
            mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
            other=0.0,
        )
        b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            _p_k_0 = (64) + tl.arange(0, 64)
            _p_k_1 = (i_t * BT) + tl.arange(0, BT)
            b_k = tl.load(
                k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (stride_k),
                mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
                other=0.0,
            )
            b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            _p_k_0 = (128) + tl.arange(0, 64)
            _p_k_1 = (i_t * BT) + tl.arange(0, BT)
            b_k = tl.load(
                k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (stride_k),
                mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
                other=0.0,
            )
            b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            _p_k_0 = (192) + tl.arange(0, 64)
            _p_k_1 = (i_t * BT) + tl.arange(0, BT)
            b_k = tl.load(
                k + _p_k_0[:, None] * (1) + _p_k_1[None, :] * (stride_k),
                mask=(_p_k_0[:, None] < (K)) & (_p_k_1[None, :] < (T)),
                other=0.0,
            )
            b_h4 += tl.dot(b_k, b_v)
    # epilogue
    if STORE_FINAL_STATE:
        _p_ht_0 = (0) + tl.arange(0, 64)
        _p_ht_1 = (i_v * BV) + tl.arange(0, BV)
        tl.store(
            ht + _p_ht_0[:, None] * (V) + _p_ht_1[None, :] * (1),
            b_h1.to(ht.dtype.element_ty),
            mask=(_p_ht_0[:, None] < (K)) & (_p_ht_1[None, :] < (V)),
        )
        if K > 64:
            _p_ht_0 = (64) + tl.arange(0, 64)
            _p_ht_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                ht + _p_ht_0[:, None] * (V) + _p_ht_1[None, :] * (1),
                b_h2.to(ht.dtype.element_ty),
                mask=(_p_ht_0[:, None] < (K)) & (_p_ht_1[None, :] < (V)),
            )
        if K > 128:
            _p_ht_0 = (128) + tl.arange(0, 64)
            _p_ht_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                ht + _p_ht_0[:, None] * (V) + _p_ht_1[None, :] * (1),
                b_h3.to(ht.dtype.element_ty),
                mask=(_p_ht_0[:, None] < (K)) & (_p_ht_1[None, :] < (V)),
            )
        if K > 192:
            _p_ht_0 = (192) + tl.arange(0, 64)
            _p_ht_1 = (i_v * BV) + tl.arange(0, BV)
            tl.store(
                ht + _p_ht_0[:, None] * (V) + _p_ht_1[None, :] * (1),
                b_h4.to(ht.dtype.element_ty),
                mask=(_p_ht_0[:, None] < (K)) & (_p_ht_1[None, :] < (V)),
            )


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,  # SY: remove this argument and force chunk size 64?
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, chunk_size)
        if cu_seqlens is not None
        else None
    )
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = (
        k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None
    )

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
    )
    return h, v_new, final_state
