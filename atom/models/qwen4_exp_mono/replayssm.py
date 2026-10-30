# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen Mono split-V ReplaySSM with race-free flush publication.

The serial kernel is specialized from the shared ReplaySSM implementation:
new k/g records stay in the reserved tail until every V CTA finishes replay.
The same-stream publication (or the fused GDN finish) then copies them to the
front. Checkpoint/u remain V-owned, and the shared commit owns the cursor.
Keep this variant local until the shared implementation adopts the fix; do
not replace the process-wide ReplaySSM functions. Arithmetic helpers, the UT
route, state layout, and cursor commit are reused from the shared module.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.model_ops.fla_ops.op import exp
from atom.model_ops.fla_ops.replayssm import (
    UT_MIN_QUERY_LEN,
    _replay_dot,
    _replay_dot_mode,
    _replay_tiles_kmajor,
    _replayssm_ut_fwd_kernel,
    _serial_launch_config,
    flush_threshold_ok,
)


@triton.jit(do_not_specialize=["T_MAX", "CAP"])
def _publish_flush_records_kernel(
    buf_k,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    T_MAX,
    CAP,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_n, i_hv = tl.program_id(0), tl.program_id(1)
    T = tl.load(cu_seqlens + i_n + 1) - tl.load(cu_seqlens + i_n)
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if T == 0 or slot < 0:
        return
    h = tl.maximum(tl.load(write_pos + slot), 0).to(tl.int64)
    if h + 2 * T_MAX <= CAP:
        return

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    mask = (o_t[:, None] < T) & (o_k[None, :] < K)
    p_k = (
        buf_k
        + slot * stride_bufk_slot
        + i_hv * stride_bufk_hv
        + o_t[:, None] * stride_bufk_pos
        + o_k[None, :]
    )
    b_k = tl.load(p_k + h * stride_bufk_pos, mask=mask, other=0)
    p_g = buf_g + slot * stride_bufg_slot + i_hv * stride_bufg_hv
    if IS_KDA:
        p_g = p_g + o_t[:, None] * stride_bufg_pos + o_k[None, :]
        b_g = tl.load(p_g + h * stride_bufg_pos, mask=mask, other=0)
    else:
        p_g = p_g + o_t * stride_bufg_pos
        b_g = tl.load(p_g + h * stride_bufg_pos, mask=o_t < T, other=0)
    # At the minimum capacity h may be smaller than T, so source and
    # destination overlap. All waves must finish both loads before any store.
    tl.debug_barrier()
    tl.store(p_k, b_k, mask=mask)
    if IS_KDA:
        tl.store(p_g, b_g, mask=mask)
    else:
        tl.store(p_g, b_g, mask=o_t < T)


def _publish_flush_records(
    buf_k, buf_g, write_pos, slot_idx, cu_seqlens, max_query_len
):
    """Publish staged k/g only after every V tile has finished replaying.

    The existing capacity invariant reserves [h, h + T) for this step.
    Split-V kernels use that free tail while old [0, h) records are being
    read, then this kernel moves the new records to [0, T) on a flush.
    Checkpoint/u are V-owned; neither needs staging. The cursor is unchanged.
    """
    bk = triton.next_power_of_2(buf_k.shape[-1])
    bt = triton.next_power_of_2(max_query_len)
    _publish_flush_records_kernel[(cu_seqlens.numel() - 1, buf_k.shape[1])](
        buf_k,
        buf_g,
        write_pos,
        slot_idx,
        cu_seqlens,
        max_query_len,
        buf_k.shape[2],
        buf_k.shape[-1],
        bk,
        bt,
        buf_k.stride(0),
        buf_k.stride(2),
        buf_k.stride(1),
        buf_g.stride(0),
        buf_g.stride(2),
        buf_g.stride(1),
        IS_KDA=buf_g.ndim == buf_k.ndim,
        num_warps=4 if bk * bt > 512 else 1,
    )


@triton.jit(do_not_specialize=["N", "T_TOT", "T_MAX", "CAP"])
def _replayssm_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    N,
    T_TOT,
    T_MAX,
    CAP,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    stride_ckpt_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    stride_beta_tok: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_KDA: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    DOT_MODE: tl.constexpr,
    T1_FAST: tl.constexpr,
    T1_TILED: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return

    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:  # padded / idle batch entry
        return

    h = tl.load(write_pos + slot).to(tl.int32)
    # Clamp the prefill sentinel.  In the normal order the commit kernel has
    # already turned -1 into 0, but graph capture wires the buffers up and
    # replays dummy batches without committing; folding nothing and writing
    # from 0 is both the right answer and what keeps `base` off -1, which would
    # index a record row outside this slot.
    h = tl.maximum(h, 0)
    do_flush = h + 2 * T_MAX > CAP
    base = tl.where(do_flush, 0, h).to(tl.int64)
    # Shared k/g cannot overwrite history until all V tiles finish reading.
    kg_base = h.to(tl.int64) if V > BV else base

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    if T1_TILED:
        # Scalar-gate T == 1.  Same algebra as the T1_FAST block below, but
        # specialised: the gate is per record, so it stays a [BH] vector and
        # the checkpoint's decay stays a scalar, rather than going through
        # `_replay_tiles_kmajor`'s [BK]-shaped form.  Worth 111.5 -> 104 us.
        #
        # This was written to be dstate-tiled -- walking the checkpoint in
        # [BKT, BV] slices -- on the theory that holding the 8 KiB tile whole
        # (128 VGPRs a lane at BK=128/BV=64) was capping occupancy, which is
        # also the fix upstream names for exactly this regime: their fp32-state
        # results do not transfer to a 16-bit state ("FP16/BF16 state is
        # currently ~parity with baseline latency (high register pressure ->
        # low bandwidth utilization); the planned fix is dstate-tiling"), and
        # ATOM's GDN state is bf16.  Measured here it does not hold: BKT of
        # 16/32/64/128 gave 116.6/107.2/136.2/104.0 us, so the untiled form
        # wins and the loop was collapsed.  Re-tuning BV and num_warps for the
        # new shape likewise landed back on the existing BV=64/num_warps=1
        # (against 110.2 at BV=128, 111.8 at BV=32, 116.2 at num_warps=2).
        o_h = tl.arange(0, BH)
        m_h = o_h < h

        # Replay weights: per record, so a [BH] vector, computed once.
        b_g1 = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h * stride_bufg_pos
            + i_hv * stride_bufg_hv,
            mask=m_h,
            other=0.0,
        ).to(tl.float32)
        b_ctot1 = tl.sum(b_g1, axis=0)
        b_w = exp(b_ctot1 - tl.cumsum(b_g1, axis=0))
        b_decay = exp(b_ctot1)

        b_ru = tl.load(
            buf_u
            + slot * stride_bufu_slot
            + o_h[:, None] * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v[None, :],
            mask=m_h[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)

        b_gt = tl.load(g + bos * HV + i_hv).to(tl.float32)
        b_eg = exp(b_gt)

        # The l2 norms need the whole vector, so take them before tiling; the
        # per-tile reloads below are [BKT] slices of the same two vectors.
        if USE_QK_L2NORM_IN_KERNEL:
            b_qf = tl.load(q + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
                tl.float32
            )
            b_kf = tl.load(k + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
                tl.float32
            )
            q_rn = 1.0 / tl.sqrt(tl.sum(b_qf * b_qf) + 1e-6)
            k_rn = 1.0 / tl.sqrt(tl.sum(b_kf * b_kf) + 1e-6)
        else:
            q_rn = 1.0
            k_rn = 1.0

        b_sk = tl.zeros([BV], dtype=tl.float32)
        b_sq = tl.zeros([BV], dtype=tl.float32)
        b_ck = tl.zeros([BH], dtype=tl.float32)
        b_cq = tl.zeros([BH], dtype=tl.float32)
        b_kq = 0.0
        p_ckpt = ckpt + slot * stride_ckpt_slot + i_hv * K * V
        b_qt = (
            tl.load(q + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
                tl.float32
            )
            * q_rn
            * scale
        )
        b_kt = (
            tl.load(k + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
                tl.float32
            )
            * k_rn
        )
        b_kq += tl.sum(b_kt * b_qt)
        b_xk = b_eg * b_kt
        b_xq = b_eg * b_qt

        p_s = p_ckpt + o_k[:, None] * V + o_v[None, :]
        b_s = tl.load(p_s, mask=mask_k[:, None] & mask_v[None, :], other=0.0).to(
            tl.float32
        )
        b_sk += tl.sum(b_s * (b_decay * b_xk)[:, None], 0)
        b_sq += tl.sum(b_s * (b_decay * b_xq)[:, None], 0)

        b_kwt = (
            tl.load(
                buf_k
                + slot * stride_bufk_slot
                + o_h[None, :] * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k[:, None],
                mask=m_h[None, :] & mask_k[:, None],
                other=0.0,
            ).to(tl.float32)
            * b_w[None, :]
        )
        b_ck += tl.sum(b_kwt * b_xk[:, None], 0)
        b_cq += tl.sum(b_kwt * b_xq[:, None], 0)

        if do_flush:
            tl.store(
                p_s,
                (b_s * b_decay + _replay_dot(b_kwt, b_ru, True, DOT_MODE)).to(
                    p_s.dtype.element_ty
                ),
                mask=mask_k[:, None] & mask_v[None, :],
            )
        if i_v == 0:
            p_bkt = (
                buf_k
                + slot * stride_bufk_slot
                + kg_base * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bkt, b_kt.to(p_bkt.dtype.element_ty), mask=mask_k)

        # record half, once the per-record weights are complete
        b_sk += tl.sum(b_ck[:, None] * b_ru, 0)
        b_sq += tl.sum(b_cq[:, None] * b_ru, 0)

        b_v = tl.load(v + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0).to(
            tl.float32
        )
        if IS_BETA_HEADWISE:
            b_beta = tl.load(
                beta + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0
            ).to(tl.float32)
        else:
            b_beta = tl.load(beta + bos * HV + i_hv).to(tl.float32)
        b_u = b_beta * (b_v - b_sk)

        p_o = o + (bos * HV + i_hv) * V + o_v
        tl.store(p_o, (b_sq + b_u * b_kq).to(p_o.dtype.element_ty), mask=mask_v)

        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + base * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        tl.store(p_bu, b_u.to(p_bu.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            p_bg2 = buf_g + slot * stride_bufg_slot + kg_base * stride_bufg_pos
            tl.store(p_bg2 + i_hv * stride_bufg_hv, b_gt.to(p_bg2.dtype.element_ty))
        return

    # ---- 1. the checkpoint and the committed records -----------------------
    p_ckpt = ckpt + slot * stride_ckpt_slot + i_hv * K * V
    p_ckpt_hv = p_ckpt + o_k[:, None] * V + o_v[None, :]
    b_s0 = tl.load(p_ckpt_hv, mask=mask_h, other=0.0).to(tl.float32)

    b_kw, b_ru, b_decay = _replay_tiles_kmajor(
        buf_k,
        buf_u,
        buf_g,
        slot,
        h,
        stride_bufk_slot,
        stride_bufk_pos,
        stride_bufk_hv,
        stride_bufu_slot,
        stride_bufu_pos,
        stride_bufu_hv,
        stride_bufg_slot,
        stride_bufg_pos,
        stride_bufg_hv,
        i_hv,
        o_k,
        o_v,
        mask_k,
        mask_v,
        K,
        V,
        BK,
        BH,
        IS_KDA,
    )
    if T1_FAST:
        # T == 1: nothing chains from token to token, so S_h is never needed as
        # a value -- only its two contractions, S_h^T k and S_h^T q, and both
        # distribute over the record sum because (k_j (x) u_j)^T x = (k_j.x)u_j:
        #
        #   S_h^T x = e^C (S_0^T x) + sum_j (k~_j . x) u_j
        #
        # so two [BH]-long matvecs replace the [BK,BH]x[BH,BV] rebuild.  That
        # GEMM measured 66-92 us of a 173 us kernel and would not come down:
        # casting the operands to bf16 (an eighth the MFMA cost) only moved it
        # 92 -> 63 us, and loading the k tile k-major to drop the `tl.trans`
        # bought ~0, because Triton stages the layout conversion through LDS
        # either way -- the record axis is the strided one in the buffer, whichever
        # orientation it is read in.  Not doing the GEMM is the way past it.
        # A flush is the one step that still needs S_h as a value, and it is
        # one step in `cap - 1`.
        p_o = o + (bos * HV + i_hv) * V + o_v
        b_q = tl.load(q + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_k = tl.load(k + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(
            tl.float32
        )
        b_v = tl.load(v + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0).to(
            tl.float32
        )
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        if IS_KDA:
            b_g = tl.load(g + (bos * HV + i_hv) * K + o_k, mask=mask_k, other=0.0).to(
                tl.float32
            )
        else:
            b_g = tl.load(g + bos * HV + i_hv).to(tl.float32)
        # (diag(e^g) S_h)^T x == S_h^T (e^g . x), so this step's gate rides on
        # the contraction vector instead of scaling a whole [BK, BV] tile.
        b_eg = exp(b_g)
        b_xk = b_eg * b_k
        b_xq = b_eg * b_q

        # checkpoint half; e^C likewise rides on the vector, not the tile
        b_sk = tl.sum(b_s0 * (b_decay * b_xk)[:, None], 0)
        b_sq = tl.sum(b_s0 * (b_decay * b_xq)[:, None], 0)
        # record half: one weight per record, then scattered onto u
        b_sk += tl.sum(tl.sum(b_kw * b_xk[:, None], 0)[:, None] * b_ru, 0)
        b_sq += tl.sum(tl.sum(b_kw * b_xq[:, None], 0)[:, None] * b_ru, 0)

        if IS_BETA_HEADWISE:
            b_beta = tl.load(
                beta + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0
            ).to(tl.float32)
        else:
            b_beta = tl.load(beta + bos * HV + i_hv).to(tl.float32)
        b_u = b_beta * (b_v - b_sk)
        # o = (diag(e^g) S_h + k (x) u)^T q
        b_o = b_sq + b_u * tl.sum(b_k * b_q)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + base * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        tl.store(p_bu, b_u.to(p_bu.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + kg_base * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg2 = buf_g + slot * stride_bufg_slot + kg_base * stride_bufg_pos
            if IS_KDA:
                tl.store(
                    p_bg2 + i_hv * stride_bufg_hv + o_k,
                    b_g.to(p_bg2.dtype.element_ty),
                    mask=mask_k,
                )
            else:
                tl.store(p_bg2 + i_hv * stride_bufg_hv, b_g.to(p_bg2.dtype.element_ty))

        if do_flush:
            b_h = b_s0 * b_decay[:, None] + _replay_dot(b_kw, b_ru, True, DOT_MODE)
            tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)
        return

    # State is [BK, BV] here and the k tile arrives k-major, so the contraction
    # is a bare k~ @ u with no transpose to stage through LDS.
    b_h = b_s0 * b_decay[:, None]
    b_h += _replay_dot(b_kw, b_ru, True, DOT_MODE)

    # ---- 2. flush: the checkpoint absorbs the committed prefix --------------
    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)

    # ---- 3. this step's tokens ---------------------------------------------
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if IS_KDA:
        p_g = g + (bos * HV + i_hv) * K + o_k
    else:
        p_g = g + bos * HV + i_hv

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        if IS_KDA:
            b_g = tl.load(p_g, mask=mask_k, other=0.0).to(tl.float32)
            b_h *= exp(b_g)[:, None]
        else:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0.0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

        tl.store(
            p_o, tl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty), mask=mask_v
        )

        # ---- append the record ---------------------------------------------
        pos = base + i_t
        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + pos * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        tl.store(p_bu, b_v.to(p_bu.dtype.element_ty), mask=mask_v)
        # k and g do not depend on the V split; let one program own the store
        # instead of having every V-block redundantly write the same bytes.
        if i_v == 0:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + (kg_base + i_t) * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg2 = buf_g + slot * stride_bufg_slot + (kg_base + i_t) * stride_bufg_pos
            if IS_KDA:
                tl.store(
                    p_bg2 + i_hv * stride_bufg_hv + o_k,
                    b_g.to(p_bg2.dtype.element_ty),
                    mask=mask_k,
                )
            else:
                tl.store(p_bg2 + i_hv * stride_bufg_hv, b_g.to(p_bg2.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_beta += stride_beta_tok
        p_g += HV * K if IS_KDA else HV


def replayssm_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    route: str = "auto",
    o: torch.Tensor | None = None,
) -> torch.Tensor:
    """ReplaySSM forward for one linear-attention layer.

    Args:
        q, k: ``[1, T_tot, H, K]``
        v: ``[1, T_tot, HV, V]``
        g: ``[1, T_tot, HV]`` (GDN) or ``[1, T_tot, HV, K]`` (KDA)
        beta: ``[1, T_tot, HV]`` or ``[1, T_tot, HV, V]``
        ckpt: ``[num_slots, HV, K, V]`` checkpoint pool (mutated on flush)
        buf_k / buf_u / buf_g: record buffers, see ``replayssm_buffer_shapes``
        write_pos: ``[num_slots]`` int32 committed-record cursor; advance with
            :func:`replayssm_commit` once per forward, never here
        slot_idx: ``[N]`` int32, ``PAD_SLOT_ID`` for idle entries
        max_query_len: the ``T`` used by the flush predicate; must match the
            value passed to :func:`replayssm_commit` in the same step
        route: ``"serial"`` walks tokens one at a time; ``"ut"`` uses the
            chunked delta-rule UT transform (scalar gate, non-headwise beta,
            multi-token steps only); ``"auto"`` picks ``"ut"`` when eligible.
        o: optional contiguous ``[T_tot, HV, V]`` destination (e.g. a slice of
            the layer's output buffer); the kernel writes into it directly
            instead of a fresh tensor, saving the caller a device copy.

    Returns:
        ``o`` of shape ``[1, T_tot, HV, V]`` -- same leading-batch convention
        as :func:`fused_recurrent_gated_delta_rule`, so this is a drop-in
        replacement at the call sites in ``attention_gdn.py``.
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    if scale is None:
        scale = K**-0.5
    assert flush_threshold_ok(cap, max_query_len), (
        f"replayssm cache_len={cap} must be >= 2*max_query_len="
        f"{2 * max_query_len}; otherwise the record buffer overruns"
    )

    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"

    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if o is None:
        o = q.new_empty(1, T_tot, HV, V)
    else:
        assert o.shape == (T_tot, HV, V) and o.is_contiguous() and o.dtype == q.dtype
        o = o.unsqueeze(0)

    if route not in ("auto", "serial", "ut"):
        raise ValueError(f"unknown replayssm route {route!r}; expected auto|serial|ut")
    ut_eligible = (not is_kda) and beta.ndim != v.ndim and max_query_len > 1
    if route == "auto":
        # Measured crossover on gfx950 (MI355X), Qwen3.5-like shape
        # H=2 HV=16 K=V=128 bf16, bs=128, best config per route:
        #   T=  2   3   4   6   8  12  16
        #   ut/serial 1.70 1.63 1.40 1.45 1.08 0.98 0.78
        # The UT transform only pays once the verify window is long enough to
        # amortise the T x T solve and the GEMM setup; below that the serial
        # chain is just four dependent rank-1 updates and wins outright.
        # Practical MTP windows (mtp_k = 1..3, i.e. T = 2..4) sit firmly on
        # the serial side, so `auto` only reaches for UT at long windows.
        route = "ut" if ut_eligible and max_query_len >= UT_MIN_QUERY_LEN else "serial"
    if route == "ut":
        if not ut_eligible:
            raise ValueError(
                "replayssm route='ut' requires a scalar gate (not KDA), "
                "non-headwise beta, and max_query_len > 1"
            )
        # One program per (sequence, v-head): the [T, K] work (loads, l2norm,
        # the K.K^T and Q.K^T gemms) does not depend on the V split, so
        # splitting V would replicate it NV times.  Measured 2.9x worse at
        # BV=32/NV=4 than BV=V/NV=1.
        BV_UT = triton.next_power_of_2(V)
        BT = triton.next_power_of_2(max_query_len)
        BH_UT = max(16, triton.next_power_of_2(cap - max_query_len))
        _replayssm_ut_fwd_kernel[(1, N * HV)](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=o,
            ckpt=ckpt,
            buf_k=buf_k,
            buf_u=buf_u,
            buf_g=buf_g,
            write_pos=write_pos,
            slot_idx=slot_idx,
            cu_seqlens=cu_seqlens,
            scale=scale,
            N=N,
            T_TOT=T_tot,
            T_MAX=max_query_len,
            CAP=cap,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV_UT,
            BH=BH_UT,
            BT=BT,
            stride_ckpt_slot=ckpt.stride(0),
            stride_bufk_slot=buf_k.stride(0),
            stride_bufk_pos=buf_k.stride(2),
            stride_bufk_hv=buf_k.stride(1),
            stride_bufu_slot=buf_u.stride(0),
            stride_bufu_pos=buf_u.stride(2),
            stride_bufu_hv=buf_u.stride(1),
            stride_bufg_slot=buf_g.stride(0),
            stride_bufg_pos=buf_g.stride(2),
            stride_bufg_hv=buf_g.stride(1),
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            DOT_MODE=_replay_dot_mode(buf_u.dtype),
            num_warps=4 if BT >= 16 else 2,
            num_stages=2,
        )
        return o

    BV, num_warps, num_stages = _serial_launch_config(N, HV, V, max_query_len, is_kda)
    NV = triton.cdiv(V, BV)
    # Replay tile height: the cursor can reach cap - max_query_len before a
    # flush resets it, and `tl.dot` wants at least 16 rows on CDNA, so a
    # short buffer pads rather than shrinks.
    BH = max(16, triton.next_power_of_2(cap - max_query_len))
    _replayssm_fwd_kernel[(NV, N * HV)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        T_TOT=T_tot,
        T_MAX=max_query_len,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BH=BH,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(2),
        stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(2),
        stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(2),
        stride_bufg_hv=buf_g.stride(1),
        stride_beta_tok=beta.stride(1),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        T1_FAST=max_query_len == 1,
        T1_TILED=(max_query_len == 1) and not is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if NV > 1:
        _publish_flush_records(
            buf_k, buf_g, write_pos, slot_idx, cu_seqlens, max_query_len
        )
    return o
