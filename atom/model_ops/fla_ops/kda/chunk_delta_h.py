# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project
# (version 0.5.2, fla/ops/common/chunk_delta_h.py). The original source code
# was licensed under the MIT license and included the following copyright
# notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Adapted for ATOM:
#   - Forward path only; the backward kernel and its wrapper are not copied.
#   - `h0` / `ht` are addressed through `h0_indices` (a per-sequence cache-slot
#     index) rather than the dense `i_nh`, so the caller no longer gathers the
#     initial state or scatters the final state. Follows the same shape as
#     `fla_ops/fused_sigmoid_gating.py:108-126`.
#   - `has_initial_state` skips the h0 load for fresh sequences, absorbing what
#     `model_ops/kimi_k3/kda_state.py` used to do in a separate pass.
#   - `inplace_final_state` lets `ht` alias `initial_state`.
#   - `restore_value=["h0"]` added to the autotune decorator. NOT optional: with
#     `inplace_final_state=True` ht aliases h0, so without it every autotune
#     trial re-reads what the previous trial wrote and the real launch consumes
#     f(f(...f(h0))). Cold-cache-only, silent, and numerically wrong. See the
#     inline note at the decorator.
#   - Both public names carry a `_log2` suffix
#     (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64_log2`,
#     `chunk_gated_delta_rule_fwd_h_log2`) so they cannot be confused with the
#     identically-named base-e functions in `fla_ops/chunk_delta_h.py`.
#   - `@dispatch('common')` dropped from the wrapper: no backend is installed on
#     ROCm, and the indirection would make the parity test ambiguous.
#
# WARNING: base-2 (`exp2`). See this package's __init__ docstring before
# swapping anything here with `fla_ops/chunk_delta_h.py` (same filename, same
# original function names) or `fla_ops/chunk_delta_h_vk.py`, which are base-e.

import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_BLACKWELL, autotune_cache_kwargs, check_shared_mem

# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2] if IS_NVIDIA_BLACKWELL else [2, 4]


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_H0_INDICES": lambda args: args["h0_indices"] is not None,
        "USE_HAS_INITIAL_STATE": lambda args: args["has_initial_state"] is not None,
    }
)
@fla_cache_autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in GATED_DELTA_RULE_FWD_H_NUM_WARPS
        for num_stages in ([2, 3, 4] if check_shared_mem("ampere") else [2, 1])
        for BV in ([32, 64] if check_shared_mem("ada") else [32])
    ],
    key=[
        "H",
        "HV",
        "K",
        "V",
        "BT",
        "STATE_V_FIRST",
        "USE_H0_INDICES",
        "INPLACE_FINAL_STATE",
    ],
    # When inplace_final_state=True, ht aliases h0 (same tensor, same memory).
    # Without restore_value, each autotuner benchmark trial re-reads the state
    # that the previous trial just wrote, producing f(f(...f(h0))) after all
    # configs are benchmarked. The real launch then reads this corrupted state
    # and produces wrong output. restore_value="h0" clones h0 before each
    # trial and restores it after, so every trial starts from clean state.
    # ht is not separately listed because it IS h0 (same data_ptr) when
    # inplace_final_state=True; restoring h0 restores ht implicitly.
    # When inplace_final_state=False, h0 and ht are independent and no writes
    # to h0 occur during the kernel, so restore_value is a no-op in that case.
    restore_value=["h0"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64_log2(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    h0_indices,
    has_initial_state,
    cu_seqlens,
    chunk_offsets,
    T,
    stride_state_slot,
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
    USE_H0_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    # Unused in the body. Kept solely as an autotune-key discriminator (see the
    # `key=` list above) so the ht-aliases-h0 case and the disjoint case get
    # separate cache entries. Triton resolves key names against the bound args
    # and *silently skips* names it cannot find (autotuner.py:223,
    # `if key in _args`), so deleting this parameter would not raise -- it would
    # quietly collapse the two cases onto one tuning result. Keep it.
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
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

    # Resolve the state slot. Upstream indexes densely by i_nh, which requires
    # the caller to gather ssm_state[state_indices] into a packed buffer first.
    # With h0_indices we read/write the cache slot directly.
    #
    # When USE_H0_INDICES=True, valid_h0 / valid_ht are runtime booleans that
    # track whether the load / store should proceed. They are folded into the
    # mask= operand of every tl.load / tl.store in the h0 and ht sections below
    # so that no Python `if` branches on them (which would not compile in Triton
    # — only tl.constexpr conditions can be used as Python-level if guards).
    # See fla_ops/fused_sigmoid_gating.py:108-126 for the in-repo precedent for
    # the slot-index pattern; this kernel folds mask rather than early-return
    # because the ht scatter at the end of the kernel also needs gating.
    if USE_H0_INDICES:
        i_slot = tl.load(h0_indices + i_n).to(tl.int64)
        # PAD_SLOT_ID (-1) marks an idle slot: no state to read, none to write.
        valid_slot = i_slot >= 0
        valid_h0 = valid_slot
        valid_ht = valid_slot
        if USE_HAS_INITIAL_STATE:
            # A fresh sequence starts from zeros; skip the load.
            has_h0 = tl.load(has_initial_state + i_n) != 0
            valid_h0 = valid_h0 & has_h0
        i_state_base = i_slot * stride_state_slot + (i_h * K * V).to(tl.int64)
        if USE_INITIAL_STATE:
            h0 = h0 + i_state_base
        if STORE_FINAL_STATE:
            ht = ht + i_state_base
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + i_nh * K * V
        if STORE_FINAL_STATE:
            ht = ht + i_nh * K * V

    # load initial state
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    if USE_INITIAL_STATE:
        if USE_H0_INDICES:
            if STATE_V_FIRST:
                p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
                m_h0_1 = m_v[:, None] & m_k1[None, :] & valid_h0
            else:
                p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
                m_h0_1 = m_k1[:, None] & m_v[None, :] & valid_h0
            b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
            if K > 64:
                if STATE_V_FIRST:
                    p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                    m_h0_2 = m_v[:, None] & m_k2[None, :] & valid_h0
                else:
                    p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                    m_h0_2 = m_k2[:, None] & m_v[None, :] & valid_h0
                b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
            if K > 128:
                if STATE_V_FIRST:
                    p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                    m_h0_3 = m_v[:, None] & m_k3[None, :] & valid_h0
                else:
                    p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                    m_h0_3 = m_k3[:, None] & m_v[None, :] & valid_h0
                b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
            if K > 192:
                if STATE_V_FIRST:
                    p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                    m_h0_4 = m_v[:, None] & m_k4[None, :] & valid_h0
                else:
                    p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                    m_h0_4 = m_k4[:, None] & m_v[None, :] & valid_h0
                b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)
        else:
            if STATE_V_FIRST:
                p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
                m_h0_1 = m_v[:, None] & m_k1[None, :]
            else:
                p_h0_1 = h0 + o_k1[:, None] * V + o_v[None, :]
                m_h0_1 = m_k1[:, None] & m_v[None, :]
            b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
            if K > 64:
                if STATE_V_FIRST:
                    p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
                    m_h0_2 = m_v[:, None] & m_k2[None, :]
                else:
                    p_h0_2 = h0 + o_k2[:, None] * V + o_v[None, :]
                    m_h0_2 = m_k2[:, None] & m_v[None, :]
                b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
            if K > 128:
                if STATE_V_FIRST:
                    p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
                    m_h0_3 = m_v[:, None] & m_k3[None, :]
                else:
                    p_h0_3 = h0 + o_k3[:, None] * V + o_v[None, :]
                    m_h0_3 = m_k3[:, None] & m_v[None, :]
                b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
            if K > 192:
                if STATE_V_FIRST:
                    p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
                    m_h0_4 = m_v[:, None] & m_k4[None, :]
                else:
                    p_h0_4 = h0 + o_k4[:, None] * V + o_v[None, :]
                    m_h0_4 = m_k4[:, None] & m_v[None, :]
                b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence
    for i_t in range(NT):
        i_t_int64 = i_t.to(tl.int64)
        o_t = i_t * BT + tl.arange(0, BT)
        m_t = o_t < T
        if STATE_V_FIRST:
            p_h1 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
        else:
            p_h1 = h + i_t_int64 * HV * K * V + o_k1[:, None] * V + o_v[None, :]
            m_h1 = m_k1[:, None] & m_v[None, :]
        tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
        if K > 64:
            if STATE_V_FIRST:
                p_h2 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
            else:
                p_h2 = h + i_t_int64 * HV * K * V + o_k2[:, None] * V + o_v[None, :]
                m_h2 = m_k2[:, None] & m_v[None, :]
            tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
        if K > 128:
            if STATE_V_FIRST:
                p_h3 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
            else:
                p_h3 = h + i_t_int64 * HV * K * V + o_k3[:, None] * V + o_v[None, :]
                m_h3 = m_k3[:, None] & m_v[None, :]
            tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
        if K > 192:
            if STATE_V_FIRST:
                p_h4 = h + i_t_int64 * HV * K * V + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
            else:
                p_h4 = h + i_t_int64 * HV * K * V + o_k4[:, None] * V + o_v[None, :]
                m_h4 = m_k4[:, None] & m_v[None, :]
            tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)

        p_w = w + o_t[:, None] * (HV * K) + o_k1[None, :]
        b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
        if STATE_V_FIRST:
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
        else:
            b_v = tl.dot(b_w, b_h1.to(b_w.dtype))
        if K > 64:
            p_w = w + o_t[:, None] * (HV * K) + o_k2[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h2.to(b_w.dtype))
        if K > 128:
            p_w = w + o_t[:, None] * (HV * K) + o_k3[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h3.to(b_w.dtype))
        if K > 192:
            p_w = w + o_t[:, None] * (HV * K) + o_k4[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
            if STATE_V_FIRST:
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            else:
                b_v += tl.dot(b_w, b_h4.to(b_w.dtype))
        p_v = v + o_t[:, None] * (HV * V) + o_v[None, :]
        b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

        if SAVE_NEW_VALUE:
            p_v = v_new + o_t[:, None] * (HV * V) + o_v[None, :]
            tl.store(
                p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :]
            )

        last_idx = min((i_t + 1) * BT, T) - 1
        if USE_G:
            b_g_last = tl.load(g + (bos * HV + last_idx * HV + i_h).to(tl.int64)).to(
                tl.float32
            )
            p_g = g + (bos * HV + i_h).to(tl.int64) + o_t * HV
            b_g = tl.load(p_g, mask=m_t, other=0.0).to(tl.float32)
            b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
            b_g_last = exp2(b_g_last)
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
                b_h1 *= exp2(b_gk_last1)[None, :]
            else:
                b_h1 *= exp2(b_gk_last1)[:, None]
            if K > 64:
                o_k2 = 64 + o_k1
                b_gk_last2 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k2,
                    mask=(o_k2 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h2 *= exp2(b_gk_last2)[None, :]
                else:
                    b_h2 *= exp2(b_gk_last2)[:, None]
            if K > 128:
                o_k3 = 128 + o_k1
                b_gk_last3 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k3,
                    mask=(o_k3 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h3 *= exp2(b_gk_last3)[None, :]
                else:
                    b_h3 *= exp2(b_gk_last3)[:, None]
            if K > 192:
                o_k4 = 192 + o_k1
                b_gk_last4 = tl.load(
                    gk + (bos + last_idx) * HV * K + i_h * K + o_k4,
                    mask=(o_k4 < K),
                    other=0.0,
                ).to(tl.float32)
                if STATE_V_FIRST:
                    b_h4 *= exp2(b_gk_last4)[None, :]
                else:
                    b_h4 *= exp2(b_gk_last4)[:, None]
        b_v = b_v.to(k.dtype.element_ty)

        p_k = k + o_k1[:, None] + o_t[None, :] * (H * K)
        b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
        if STATE_V_FIRST:
            b_h1 += tl.trans(tl.dot(b_k, b_v))
        else:
            b_h1 += tl.dot(b_k, b_v)
        if K > 64:
            p_k = k + o_k2[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h2 += tl.dot(b_k, b_v)
        if K > 128:
            p_k = k + o_k3[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h3 += tl.dot(b_k, b_v)
        if K > 192:
            p_k = k + o_k4[:, None] + o_t[None, :] * (H * K)
            b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
            if STATE_V_FIRST:
                b_h4 += tl.trans(tl.dot(b_k, b_v))
            else:
                b_h4 += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        if USE_H0_INDICES:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k1[None, :]
                m_ht = m_v[:, None] & m_k1[None, :] & valid_ht
            else:
                p_ht = ht + o_k1[:, None] * V + o_v[None, :]
                m_ht = m_k1[:, None] & m_v[None, :] & valid_ht
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 64:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                    m_ht = m_v[:, None] & m_k2[None, :] & valid_ht
                else:
                    p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                    m_ht = m_k2[:, None] & m_v[None, :] & valid_ht
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 128:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                    m_ht = m_v[:, None] & m_k3[None, :] & valid_ht
                else:
                    p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                    m_ht = m_k3[:, None] & m_v[None, :] & valid_ht
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 192:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                    m_ht = m_v[:, None] & m_k4[None, :] & valid_ht
                else:
                    p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                    m_ht = m_k4[:, None] & m_v[None, :] & valid_ht
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)
        else:
            if STATE_V_FIRST:
                p_ht = ht + o_v[:, None] * K + o_k1[None, :]
                m_ht = m_v[:, None] & m_k1[None, :]
            else:
                p_ht = ht + o_k1[:, None] * V + o_v[None, :]
                m_ht = m_k1[:, None] & m_v[None, :]
            tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 64:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k2[None, :]
                    m_ht = m_v[:, None] & m_k2[None, :]
                else:
                    p_ht = ht + o_k2[:, None] * V + o_v[None, :]
                    m_ht = m_k2[:, None] & m_v[None, :]
                tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 128:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k3[None, :]
                    m_ht = m_v[:, None] & m_k3[None, :]
                else:
                    p_ht = ht + o_k3[:, None] * V + o_v[None, :]
                    m_ht = m_k3[:, None] & m_v[None, :]
                tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
            if K > 192:
                if STATE_V_FIRST:
                    p_ht = ht + o_v[:, None] * K + o_k4[None, :]
                    m_ht = m_v[:, None] & m_k4[None, :]
                else:
                    p_ht = ht + o_k4[:, None] * V + o_v[None, :]
                    m_ht = m_k4[:, None] & m_v[None, :]
                tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def chunk_gated_delta_rule_fwd_h_log2(
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
    h0_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Returns (h, v_new, final_state)."""
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

    if inplace_final_state:
        if h0_indices is None:
            raise ValueError("inplace_final_state requires h0_indices.")
        if initial_state is None:
            raise ValueError("inplace_final_state requires initial_state.")
        if not output_final_state:
            raise ValueError("inplace_final_state requires output_final_state.")
    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
    else:
        h = k.new_empty(B, NT, HV, K, V)
    if not output_final_state:
        final_state = None
    elif inplace_final_state:
        # ht aliases the cache; the kernel writes the indexed slots in place.
        final_state = initial_state
    elif state_v_first:
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32)
    else:
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32)

    v_new = torch.empty_like(u) if save_new_value else None

    if h0_indices is not None:
        if h0_indices.ndim != 1:
            raise ValueError(
                f"h0_indices must be 1D (one cache slot per sequence), got shape "
                f"{tuple(h0_indices.shape)}. 2D spec-decode indices are not "
                f"supported on the chunked prefill path."
            )
        if h0_indices.shape[0] != N:
            raise ValueError(
                f"h0_indices has {h0_indices.shape[0]} entries but there are {N} "
                f"sequences."
            )
    if has_initial_state is not None:
        if h0_indices is None:
            raise ValueError("has_initial_state requires h0_indices.")
        # The kernel does an unguarded tl.load(has_initial_state + i_n) for
        # i_n in [0, N); a short tensor reads out of bounds silently.
        if has_initial_state.shape[0] != N:
            raise ValueError(
                f"has_initial_state has {has_initial_state.shape[0]} entries but "
                f"there are {N} sequences."
            )
    stride_state_slot = initial_state.stride(0) if initial_state is not None else 0

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * HV)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64_log2[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        h0_indices=h0_indices,
        has_initial_state=has_initial_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
        stride_state_slot=stride_state_slot,
        INPLACE_FINAL_STATE=inplace_final_state,
    )
    return h, v_new, final_state
