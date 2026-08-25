# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from .index import prepare_chunk_indices, prepare_chunk_offsets
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
        "STORE_CKPT": lambda args: args["ckpt"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in [2, 3, 4]
        for BV in [32, 64]
    ],
    key=["H", "HV", "K", "V", "BT", "STATE_V_FIRST"],
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
    ckpt,
    ckpt_slots,
    ckpt_base,
    cu_seqlens,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_CKPT: tl.constexpr,
    CKPT_EVERY: tl.constexpr,
    MAX_CKPT: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // HV, i_nh % HV
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(
            cu_seqlens + i_n + 1
        ).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    if STATE_V_FIRST:
        b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([BV, 64], dtype=tl.float32)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 64:
            b_h2 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 128:
            b_h3 = tl.zeros([64, BV], dtype=tl.float32)
        if K > 192:
            b_h4 = tl.zeros([64, BV], dtype=tl.float32)

    # calculate offset
    h += (boh * HV + i_h).to(tl.int64) * K * V
    v += (bos * HV + i_h).to(tl.int64) * V
    k += (bos * H + i_h // (HV // H)).to(tl.int64) * K
    w += (bos * HV + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * HV + i_h).to(tl.int64) * V

    if USE_INITIAL_STATE:
        h0 = h0 + i_nh * K * V
    if STORE_FINAL_STATE:
        ht = ht + i_nh * K * V

    # load initial state
    if USE_INITIAL_STATE:
        if STATE_V_FIRST:
            p_h0_1 = tl.make_block_ptr(
                h0, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_h0_1 = tl.make_block_ptr(
                h0, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        b_h1 += tl.load(p_h0_1, boundary_check=(0, 1)).to(tl.float32)
        if K > 64:
            if STATE_V_FIRST:
                p_h0_2 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_h0_2 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            b_h2 += tl.load(p_h0_2, boundary_check=(0, 1)).to(tl.float32)
        if K > 128:
            if STATE_V_FIRST:
                p_h0_3 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_h0_3 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            b_h3 += tl.load(p_h0_3, boundary_check=(0, 1)).to(tl.float32)
        if K > 192:
            if STATE_V_FIRST:
                p_h0_4 = tl.make_block_ptr(
                    h0, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_h0_4 = tl.make_block_ptr(
                    h0, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            b_h4 += tl.load(p_h0_4, boundary_check=(0, 1)).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        if STATE_V_FIRST:
            p_h1 = tl.make_block_ptr(
                h + i_t_int64 * HV * K * V,
                (V, K),
                (K, 1),
                (i_v * BV, 0),
                (BV, 64),
                (1, 0),
            )
        else:
            p_h1 = tl.make_block_ptr(
                h + i_t_int64 * HV * K * V,
                (K, V),
                (V, 1),
                (0, i_v * BV),
                (64, BV),
                (1, 0),
            )
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 64),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h2 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (64, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 128),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h3 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (128, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (V, K),
                    (K, 1),
                    (i_v * BV, 192),
                    (BV, 64),
                    (1, 0),
                )
            else:
                p_h4 = tl.make_block_ptr(
                    h + i_t_int64 * HV * K * V,
                    (K, V),
                    (V, 1),
                    (192, i_v * BV),
                    (64, BV),
                    (1, 0),
                )
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), boundary_check=(0, 1))

        # Scatter the running fp32 state at every absolute M-token boundary.
        # Slot -1 means the scheduler could not reserve this checkpoint; fold
        # that skip into the store mask because a nested runtime conditional in
        # this loop crashes the Triton AMD backend (llvm iota_range assertion).
        if STORE_CKPT:
            i_abs = i_t + tl.load(ckpt_base + i_n).to(tl.int32)
            if i_abs > 0 and i_abs % CKPT_EVERY == 0:
                i_ck = i_abs // CKPT_EVERY - 1
                i_slot = tl.load(ckpt_slots + i_n * MAX_CKPT + i_ck).to(tl.int64)
                m_ck = i_slot >= 0
                p_base = ckpt + tl.maximum(i_slot, 0) * HV * K * V + i_h * K * V
                o_v = i_v * BV + tl.arange(0, BV)
                o_k = tl.arange(0, 64)
                if STATE_V_FIRST:
                    m_v = m_ck & (o_v[:, None] < V)
                    tl.store(
                        p_base + o_v[:, None] * K + o_k[None, :],
                        b_h1.to(ckpt.dtype.element_ty),
                        mask=m_v & (o_k[None, :] < K),
                    )
                    if K > 64:
                        tl.store(
                            p_base + o_v[:, None] * K + (o_k[None, :] + 64),
                            b_h2.to(ckpt.dtype.element_ty),
                            mask=m_v & ((o_k[None, :] + 64) < K),
                        )
                    if K > 128:
                        tl.store(
                            p_base + o_v[:, None] * K + (o_k[None, :] + 128),
                            b_h3.to(ckpt.dtype.element_ty),
                            mask=m_v & ((o_k[None, :] + 128) < K),
                        )
                    if K > 192:
                        tl.store(
                            p_base + o_v[:, None] * K + (o_k[None, :] + 192),
                            b_h4.to(ckpt.dtype.element_ty),
                            mask=m_v & ((o_k[None, :] + 192) < K),
                        )
                else:
                    m_v = m_ck & (o_v < V)
                    tl.store(
                        p_base + o_k[:, None] * V + o_v[None, :],
                        b_h1.to(ckpt.dtype.element_ty),
                        mask=m_v[None, :] & (o_k[:, None] < K),
                    )
                    if K > 64:
                        tl.store(
                            p_base + (o_k[:, None] + 64) * V + o_v[None, :],
                            b_h2.to(ckpt.dtype.element_ty),
                            mask=m_v[None, :] & ((o_k[:, None] + 64) < K),
                        )
                    if K > 128:
                        tl.store(
                            p_base + (o_k[:, None] + 128) * V + o_v[None, :],
                            b_h3.to(ckpt.dtype.element_ty),
                            mask=m_v[None, :] & ((o_k[:, None] + 128) < K),
                        )
                    if K > 192:
                        tl.store(
                            p_base + (o_k[:, None] + 192) * V + o_v[None, :],
                            b_h4.to(ckpt.dtype.element_ty),
                            mask=m_v[None, :] & ((o_k[:, None] + 192) < K),
                        )

        p_w = tl.make_block_ptr(w, (T, K), (HV * K, 1), (i_t * BT, 0), (BT, 64), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 64), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 128), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = tl.make_block_ptr(
                w, (T, K), (HV * K, 1), (i_t * BT, 192), (BT, 64), (1, 0)
            )
            b_w = tl.load(p_w, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = tl.make_block_ptr(
            v, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
        )
        b_v = tl.load(p_v, boundary_check=(0, 1)) - b_v

        if SAVE_NEW_VALUE:
            p_v = tl.make_block_ptr(
                v_new, (T, V), (HV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)
            )
            tl.store(p_v, b_v.to(p_v.dtype.element_ty), boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            m_t = (i_t * BT + tl.arange(0, BT)) < T
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = tl.make_block_ptr(
                g + (bos * HV + i_h).to(tl.int64), (T,), (HV,), (i_t * BT,), (BT,), (0,)
            )
            b_g = tl.load(p_g, boundary_check=(0,)).to(tl.float32)
            b_v = b_v * tl.where(m_t, tl.exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = tl.exp2(b_g_last)
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
                gk + (bos + last_idx) * HV * K + i_h * K + o_k1,
                mask=(o_k1 < K),
                other=0.0,
            ).to(tl.float32)
            if STATE_V_FIRST:
                b_h1 *= tl.exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= tl.exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= tl.exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= tl.exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= tl.exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= tl.exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= tl.exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= tl.exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_t * BT), (64, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (64, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (128, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = tl.make_block_ptr(
                k, (K, T), (1, H * K), (192, i_t * BT), (64, BT), (0, 1)
            )
            b_k = tl.load(p_k, boundary_check=(0, 1))
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if STATE_V_FIRST:
            p_ht = tl.make_block_ptr(
                ht, (V, K), (K, 1), (i_v * BV, 0), (BV, 64), (1, 0)
            )
        else:
            p_ht = tl.make_block_ptr(
                ht, (K, V), (V, 1), (0, i_v * BV), (64, BV), (1, 0)
            )
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 64:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 64), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (64, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 128:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 128), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (128, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), boundary_check=(0, 1))
        if K > 192:
            if STATE_V_FIRST:
                p_ht = tl.make_block_ptr(
                    ht, (V, K), (K, 1), (i_v * BV, 192), (BV, 64), (1, 0)
                )
            else:
                p_ht = tl.make_block_ptr(
                    ht, (K, V), (V, 1), (192, i_v * BV), (64, BV), (1, 0)
                )
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    ckpt: torch.Tensor | None = None,
    ckpt_slots: torch.LongTensor | None = None,
    ckpt_base: torch.LongTensor | None = None,
    ckpt_every: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
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

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = (
            k.new_zeros(N, HV, V, K, dtype=torch.float32)
            if output_final_state
            else None
        )
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = (
            k.new_zeros(N, HV, K, V, dtype=torch.float32)
            if output_final_state
            else None
        )

    v_new = torch.empty_like(u) if save_new_value else None

    if ckpt is not None:
        assert ckpt_slots is not None and ckpt_base is not None
        assert ckpt_every > 0 and ckpt_every % BT == 0, (
            f"checkpoint interval {ckpt_every} must be a positive multiple of "
            f"the scan chunk size {BT}"
        )
        ckpt_every_chunks = ckpt_every // BT
        max_ckpt = ckpt_slots.shape[-1]
    else:
        ckpt_every_chunks, max_ckpt = 1, 1

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

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
        ckpt=ckpt,
        ckpt_slots=ckpt_slots,
        ckpt_base=ckpt_base,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        CKPT_EVERY=ckpt_every_chunks,
        MAX_CKPT=max_ckpt,
    )
    return h, v_new, final_state
