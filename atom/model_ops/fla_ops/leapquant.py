# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""LeapQuant: an 8-bit recurrent state for gated-delta-rule decode.

Pan et al., "LeapQuant: Efficient Linear Attention with Accurate Recurrent
State Quantization", arXiv:2609.38166.

Built on ReplaySSM (`replayssm.py`), whose record buffer already is the
paper's per-window form: a boundary state plus the window's ``(k, u, g)``
records, with the boundary rewritten only when the buffer fills.  LeapQuant
changes what that boundary state is.  Instead of an fp32 matrix it is

    S_0 = diag(c) Z diag(b) + Kc^T Uc

* ``Kc [r, K]``, ``Uc [r, V]`` -- fp16 compensator tokens: the top-r rank-1
  part of the state, fitted by power iteration.  They take the outlier rows
  and columns that would otherwise set the quantisation range.
* ``c [K]`` -- fp32 smoothing, the square root of each key row's mean
  |residual|.
* ``Z [K, V]`` int8 and ``b [V]`` fp32 -- the smoothed residual, symmetric
  per value channel.

A slot then holds ~1.19 bytes per state element instead of 4, and the state
is rounded once per window instead of once per token, so the rounding error
does not compound from step to step.

Decode never dequantises the matrix.  For a contraction vector ``x``,

    S_0^T x = b * (Z^T (c * x)) + Uc^T (Kc x)

so the T == 1 path is ReplaySSM's with the checkpoint term replaced.  The
compensators decay with the checkpoint, so a window's state is

    S_h = e^{C_h} S_0 + sum_j e^{C_h - C_j} k_j u_j^T

exactly as in ReplaySSM.  A flush materialises ``S_h`` once in fp32, refits
the compensators (warm-started from the previous window's), recomputes the
smoothing and re-quantises.

A speculative verify window (``T > 1`` tokens a sequence) cannot stay
matrix-free: each token reads the state after the ones before it, so that
kernel materialises ``S_h`` once and walks the window, as ReplaySSM's serial
route does.  The record buffer is ``p + 2T - 1`` long, so a fold still lands
every ``p`` committed tokens.

A prefix-cache checkpoint only ever sits at a prompt position, where the
prefill has just quantised the whole state into the slot and left no
records: an image carries ``Z``, the scales and the compensators, and a fork
seeds all three.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .op import exp
from .replayssm import flush_threshold_ok

__all__ = [
    "COLD_ITERS",
    "WARM_ITERS",
    "leapquant_buffer_shapes",
    "leapquant_fold",
    "leapquant_gated_delta_rule",
    "leapquant_load_state",
    "leapquant_seed_forked_slots",
    "leapquant_store_state",
    "leapquant_write_checkpoints",
]

#: Power-iteration steps per compensator.  A prefill has nothing to start
#: from; a window boundary starts from the previous window's compensators,
#: which a window of tokens barely moves, and already takes one step chained
#: across windows (see `_lq_quantize`).
COLD_ITERS = 8
WARM_ITERS = 0
_COLD_ITERS = tl.constexpr(COLD_ITERS)

#: Lower bound on the smoothing factors, so an all-zero key row still divides.
SMOOTH_FLOOR = tl.constexpr(1e-6)
_FP16_MAX = tl.constexpr(65504.0)
#: Below this a compensator direction or a column scale counts as zero: its
#: reciprocal still fits fp32.
_TINY = tl.constexpr(1e-30)


def leapquant_buffer_shapes(
    num_v_heads: int, head_k_dim: int, head_v_dim: int, rank: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Per-slot shapes of the side buffers next to the int8 residual.

    ``scale`` packs ``b [V]`` then ``c [K]``; each compensator row packs its
    key ``[K]`` then its value ``[V]``.
    """
    return (
        (num_v_heads, head_v_dim + head_k_dim),
        (num_v_heads, rank, head_k_dim + head_v_dim),
    )


@triton.jit
def _lq_boundary(
    p_z,
    p_sc,
    p_comp,
    o_k,
    o_v,
    o_c,
    mask_k,
    mask_v,
    mask_c,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
):
    """The dequantised boundary state ``[BK, BV]`` and its compensator keys."""
    b_z = tl.load(
        p_z + o_k[:, None] * V + o_v[None, :],
        mask=mask_k[:, None] & mask_v[None, :],
        other=0,
    ).to(tl.float32)
    b_b = tl.load(p_sc + o_v, mask=mask_v, other=0.0)
    b_c = tl.load(p_sc + V + o_k, mask=mask_k, other=0.0)
    b_s = b_z * b_c[:, None] * b_b[None, :]
    p_row = p_comp + o_c[:, None] * (K + V)
    b_ck = tl.load(
        p_row + o_k[None, :], mask=mask_c[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    b_cu = tl.load(
        p_row + K + o_v[None, :], mask=mask_c[:, None] & mask_v[None, :], other=0.0
    ).to(tl.float32)
    for r in tl.static_range(R):
        b_kr = tl.sum(tl.where(o_c[:, None] == r, b_ck, 0.0), 0)
        b_ur = tl.sum(tl.where(o_c[:, None] == r, b_cu, 0.0), 0)
        b_s += b_kr[:, None] * b_ur[None, :]
    return b_s, b_ck


@triton.jit
def _lq_rebuild(
    buf_k,
    buf_u,
    buf_g,
    slot,
    h,
    p_z,
    p_sc,
    p_comp,
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
    K: tl.constexpr,
    V: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
):
    """The window's state ``S_h`` in fp32.

    Every term is added as a rank-1 update with its vectors loaded straight
    from memory, so the tile stays in the layout the fit reduces over.
    """
    o_h = tl.arange(0, BH)
    p_g = buf_g + slot * stride_bufg_slot + i_hv * stride_bufg_hv
    b_ctot = tl.sum(
        tl.load(p_g + o_h * stride_bufg_pos, mask=o_h < h, other=0.0).to(tl.float32),
        axis=0,
    )
    b_decay = exp(b_ctot)

    b_z = tl.load(
        p_z + o_k[:, None] * V + o_v[None, :],
        mask=mask_k[:, None] & mask_v[None, :],
        other=0,
    ).to(tl.float32)
    b_c = tl.load(p_sc + V + o_k, mask=mask_k, other=0.0) * b_decay
    b_b = tl.load(p_sc + o_v, mask=mask_v, other=0.0)
    b_s = b_z * b_c[:, None] * b_b[None, :]
    for r in tl.static_range(R):
        b_kr = tl.load(p_comp + r * (K + V) + o_k, mask=mask_k, other=0.0)
        b_ur = tl.load(p_comp + r * (K + V) + K + o_v, mask=mask_v, other=0.0)
        b_s += (b_decay * b_kr.to(tl.float32))[:, None] * b_ur.to(tl.float32)[None, :]

    p_k = buf_k + slot * stride_bufk_slot + i_hv * stride_bufk_hv + o_k
    p_u = buf_u + slot * stride_bufu_slot + i_hv * stride_bufu_hv + o_v
    b_cj = b_ctot * 0.0
    # One record of prefetch: the next record's loads are in flight while
    # this one is accumulated.
    b_gn = tl.load(p_g, mask=h > 0, other=0.0).to(tl.float32)
    b_kn = tl.load(p_k, mask=mask_k & (h > 0), other=0.0)
    b_un = tl.load(p_u, mask=mask_v & (h > 0), other=0.0)
    for j in range(h):
        b_gj, b_kj, b_uj = b_gn, b_kn, b_un
        m_n = j + 1 < h
        b_gn = tl.load(p_g + (j + 1) * stride_bufg_pos, mask=m_n, other=0.0).to(
            tl.float32
        )
        b_kn = tl.load(p_k + (j + 1) * stride_bufk_pos, mask=mask_k & m_n, other=0.0)
        b_un = tl.load(p_u + (j + 1) * stride_bufu_pos, mask=mask_v & m_n, other=0.0)
        b_cj += b_gj
        b_s += (exp(b_ctot - b_cj) * b_kj.to(tl.float32))[:, None] * b_uj.to(
            tl.float32
        )[None, :]
    return b_s


@triton.jit
def _lq_unit(x):
    """``x / |x|``, or zeros when ``x`` is negligible.

    Scaling by the max first keeps ``sum(x * x)`` in ``[1, len(x)]``: once the
    compensators have drained a (near) low-rank state, the power iterate on
    the residual is ~``sigma^2`` and its squares flush to zero in fp32, which
    a plain ``x / max(|x|, eps)`` turns into an overflowing fp16 key.
    """
    m = tl.max(tl.abs(x), 0)
    x = x * tl.where(m > _TINY, 1.0 / m, 0.0)
    return x / tl.maximum(tl.sqrt(tl.sum(x * x)), 1.0)


@triton.jit
def _lq_quantize(
    b_s,
    p_z,
    p_sc,
    p_comp,
    o_k,
    o_v,
    mask_k,
    mask_v,
    K: tl.constexpr,
    V: tl.constexpr,
    R: tl.constexpr,
    ITERS: tl.constexpr,
    WARM: tl.constexpr,
):
    """Store the fp32 state ``b_s`` as compensators plus an int8 residual.

    ``WARM`` starts each compensator from the previous window's: ``k = S u_old``
    is one power-iteration step chained across windows (``u_old = S_old^T
    k_old``), so tracking the top subspace costs one reduction per window
    beyond the projection.  ``ITERS`` adds full steps on top.  The old row is
    read before this component overwrites it.  Where ``S u_old`` vanishes (its
    fp16 value underflowed, say) the component starts over as a cold one.
    """
    for r in tl.static_range(R):
        if WARM:
            b_uo = tl.load(p_comp + r * (K + V) + K + o_v, mask=mask_v, other=0.0)
            b_kv = tl.sum(b_s * b_uo.to(tl.float32)[None, :], 1)
            # The restart has to come out of the tile as `b_kv` does. Mixing
            # in a separately loaded [K] vector (the old key) made the result
            # vary from run to run with 2 warps on ROCm, barriers or not.
            if tl.max(tl.abs(b_kv), 0) <= _TINY:
                b_kv = _lq_unit(tl.sqrt(tl.sum(b_s * b_s, 1)))
                for _ in tl.static_range(_COLD_ITERS):
                    b_uv = tl.sum(b_s * b_kv[:, None], 0)
                    b_kv = _lq_unit(tl.sum(b_s * b_uv[None, :], 1))
        else:
            # The row norms point at the outlier rows the tokens are for.
            b_kv = tl.sqrt(tl.sum(b_s * b_s, 1))
        b_kv = _lq_unit(b_kv)
        for _ in tl.static_range(ITERS):
            b_uv = tl.sum(b_s * b_kv[:, None], 0)
            b_kv = _lq_unit(tl.sum(b_s * b_uv[None, :], 1))
        b_uv = tl.sum(b_s * b_kv[:, None], 0)
        b_kh = b_kv.to(tl.float16)
        b_uh = tl.minimum(tl.maximum(b_uv, -_FP16_MAX), _FP16_MAX).to(tl.float16)
        tl.store(p_comp + r * (K + V) + o_k, b_kh, mask=mask_k)
        tl.store(p_comp + r * (K + V) + K + o_v, b_uh, mask=mask_v)
        # Deflate by the rounded pair, so the residual is exactly what the
        # stored compensators leave out.
        b_s -= b_kh.to(tl.float32)[:, None] * b_uh.to(tl.float32)[None, :]

    b_c = tl.maximum(tl.sqrt(tl.sum(tl.abs(b_s), 1) / V), SMOOTH_FLOOR)
    b_s = b_s * (1.0 / b_c)[:, None]
    b_amax = tl.max(tl.abs(b_s), 0)
    b_b = tl.where(b_amax > _TINY, b_amax / 127.0, 1.0)
    b_q = libdevice.rint(b_s * (1.0 / b_b)[None, :])
    b_q = tl.minimum(tl.maximum(b_q, -127.0), 127.0)
    tl.store(
        p_z + o_k[:, None] * V + o_v[None, :],
        b_q.to(tl.int8),
        mask=mask_k[:, None] & mask_v[None, :],
    )
    tl.store(p_sc + o_v, b_b, mask=mask_v)
    tl.store(p_sc + V + o_k, b_c, mask=mask_k)


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _leapquant_fold_kernel(
    state,
    z,
    sc,
    comp,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    N,
    T_MAX,
    CAP,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
    stride_z_layer: tl.constexpr,
    stride_z_slot: tl.constexpr,
    stride_sc_layer: tl.constexpr,
    stride_sc_slot: tl.constexpr,
    stride_comp_layer: tl.constexpr,
    stride_comp_slot: tl.constexpr,
    stride_bufk_layer: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_layer: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_layer: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    FROM_STATE: tl.constexpr,
    ITERS: tl.constexpr,
    WARM: tl.constexpr,
):
    """One program per (sequence, v-head, layer), holding the whole ``[K, V]``
    tile.

    ``FROM_STATE`` quantises a given fp32 state (the end of a prefill).
    Otherwise this is the flush of a decode step: sequences whose record
    buffer is full fold it into the boundary, everyone else returns at once.
    The cursor is shared by every layer, so one launch can flush them all.
    """
    i_nh, i_l = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_n, i_hv = i_nh // HV, i_nh % HV
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    if not FROM_STATE:
        # Same predicate as the decode kernel and `replayssm_commit`.
        h = tl.maximum(tl.load(write_pos + slot), 0)
        if h + 2 * T_MAX <= CAP:
            return

    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    p_z = z + i_l * stride_z_layer + slot * stride_z_slot + i_hv * K * V
    p_sc = sc + i_l * stride_sc_layer + slot * stride_sc_slot + i_hv * (V + K)
    p_comp = (
        comp + i_l * stride_comp_layer + slot * stride_comp_slot + i_hv * R * (K + V)
    )

    if FROM_STATE:
        b_s = tl.load(
            state
            + (i_n * HV + i_hv).to(tl.int64) * K * V
            + o_k[:, None] * V
            + o_v[None, :],
            mask=mask_k[:, None] & mask_v[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        b_s = _lq_rebuild(
            buf_k + i_l * stride_bufk_layer,
            buf_u + i_l * stride_bufu_layer,
            buf_g + i_l * stride_bufg_layer,
            slot,
            h,
            p_z,
            p_sc,
            p_comp,
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
            BH,
            R,
        )

    _lq_quantize(b_s, p_z, p_sc, p_comp, o_k, o_v, mask_k, mask_v, K, V, R, ITERS, WARM)


@triton.jit
def _leapquant_load_kernel(
    out,
    z,
    sc,
    comp,
    slot_idx,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BR: tl.constexpr,
    R: tl.constexpr,
    stride_z_slot: tl.constexpr,
    stride_sc_slot: tl.constexpr,
    stride_comp_slot: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_k = tl.arange(0, BK)
    o_v = tl.arange(0, BV)
    o_c = tl.arange(0, BR)
    mask_k = o_k < K
    mask_v = o_v < V
    p_out = (
        out + (i_n * HV + i_hv).to(tl.int64) * K * V + o_k[:, None] * V + o_v[None, :]
    )
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        tl.store(
            p_out,
            tl.zeros([BK, BV], dtype=tl.float32),
            mask=mask_k[:, None] & mask_v[None, :],
        )
        return
    b_s, _ = _lq_boundary(
        z + slot * stride_z_slot + i_hv * K * V,
        sc + slot * stride_sc_slot + i_hv * (V + K),
        comp + slot * stride_comp_slot + i_hv * R * (K + V),
        o_k,
        o_v,
        o_c,
        mask_k,
        mask_v,
        o_c < R,
        K,
        V,
        R,
    )
    tl.store(p_out, b_s, mask=mask_k[:, None] & mask_v[None, :])


@triton.jit(do_not_specialize=["N", "CAP"])
def _leapquant_decode_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    z,
    sc,
    comp,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    N,
    CAP,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    R: tl.constexpr,
    stride_z_slot: tl.constexpr,
    stride_sc_slot: tl.constexpr,
    stride_comp_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    if eos == bos:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    # T == 1, so a flush is h + 2 > CAP, and on a flush step the fold kernel
    # has already absorbed every record into the boundary state.
    h = tl.maximum(tl.load(write_pos + slot), 0)
    h = tl.where(h + 2 > CAP, 0, h)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    o_h = tl.arange(0, BH)
    o_c = tl.arange(0, BR)
    mask_k = o_k < K
    mask_v = o_v < V
    m_h = o_h < h
    mask_c = o_c < R

    b_q = tl.load(q + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k + (bos * H + i_h) * K + o_k, mask=mask_k, other=0.0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale
    b_gt = tl.load(g + bos * HV + i_hv).to(tl.float32)
    # (e^g S_h)^T x == S_h^T (e^g x): this step's gate rides on the vectors.
    b_eg = exp(b_gt)
    b_xk = b_eg * b_k
    b_xq = b_eg * b_q

    # ---- the window's records ------------------------------------------------
    b_g1 = tl.load(
        buf_g + slot * stride_bufg_slot + o_h * stride_bufg_pos + i_hv * stride_bufg_hv,
        mask=m_h,
        other=0.0,
    ).to(tl.float32)
    b_ctot = tl.sum(b_g1, axis=0)
    b_w = exp(b_ctot - tl.cumsum(b_g1, axis=0))
    b_decay = exp(b_ctot)
    b_kr = tl.load(
        buf_k
        + slot * stride_bufk_slot
        + o_h[None, :] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_k[:, None],
        mask=m_h[None, :] & mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_ur = tl.load(
        buf_u
        + slot * stride_bufu_slot
        + o_h[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v[None, :],
        mask=m_h[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    b_rk = b_w * tl.sum(b_kr * b_xk[:, None], 0)
    b_rq = b_w * tl.sum(b_kr * b_xq[:, None], 0)
    b_sk = tl.sum(b_rk[:, None] * b_ur, 0)
    b_sq = tl.sum(b_rq[:, None] * b_ur, 0)

    # ---- the boundary state, decayed by e^C ----------------------------------
    p_z = z + slot * stride_z_slot + i_hv * K * V
    p_sc = sc + slot * stride_sc_slot + i_hv * (V + K)
    p_comp = comp + slot * stride_comp_slot + i_hv * R * (K + V)
    b_c = tl.load(p_sc + V + o_k, mask=mask_k, other=0.0) * b_decay
    b_b = tl.load(p_sc + o_v, mask=mask_v, other=0.0)
    b_z = tl.load(
        p_z + o_k[:, None] * V + o_v[None, :],
        mask=mask_k[:, None] & mask_v[None, :],
        other=0,
    ).to(tl.float32)
    b_sk += b_b * tl.sum(b_z * (b_c * b_xk)[:, None], 0)
    b_sq += b_b * tl.sum(b_z * (b_c * b_xq)[:, None], 0)
    p_row = p_comp + o_c[:, None] * (K + V)
    b_ck = tl.load(
        p_row + o_k[None, :], mask=mask_c[:, None] & mask_k[None, :], other=0.0
    ).to(tl.float32)
    b_cu = tl.load(
        p_row + K + o_v[None, :], mask=mask_c[:, None] & mask_v[None, :], other=0.0
    ).to(tl.float32)
    b_sk += tl.sum((b_decay * tl.sum(b_ck * b_xk[None, :], 1))[:, None] * b_cu, 0)
    b_sq += tl.sum((b_decay * tl.sum(b_ck * b_xq[None, :], 1))[:, None] * b_cu, 0)

    # ---- this token ------------------------------------------------------------
    b_v = tl.load(v + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0).to(
        tl.float32
    )
    if IS_BETA_HEADWISE:
        b_beta = tl.load(beta + (bos * HV + i_hv) * V + o_v, mask=mask_v, other=0.0).to(
            tl.float32
        )
    else:
        b_beta = tl.load(beta + bos * HV + i_hv).to(tl.float32)
    b_u = b_beta * (b_v - b_sk)
    p_o = o + (bos * HV + i_hv) * V + o_v
    tl.store(
        p_o, (b_sq + b_u * tl.sum(b_k * b_q)).to(p_o.dtype.element_ty), mask=mask_v
    )

    base = h.to(tl.int64)
    p_bu = (
        buf_u
        + slot * stride_bufu_slot
        + base * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v
    )
    b_ub = tl.minimum(tl.maximum(b_u, -_FP16_MAX), _FP16_MAX)
    tl.store(p_bu, b_ub.to(p_bu.dtype.element_ty), mask=mask_v)
    # k and g do not depend on the V split; one program owns their store.
    if i_v == 0:
        p_bk = (
            buf_k
            + slot * stride_bufk_slot
            + base * stride_bufk_pos
            + i_hv * stride_bufk_hv
            + o_k
        )
        tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
        p_bg = buf_g + slot * stride_bufg_slot + base * stride_bufg_pos
        tl.store(p_bg + i_hv * stride_bufg_hv, b_gt.to(p_bg.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _leapquant_verify_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    z,
    sc,
    comp,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    N,
    T_MAX,
    CAP,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BH: tl.constexpr,
    R: tl.constexpr,
    stride_z_slot: tl.constexpr,
    stride_sc_slot: tl.constexpr,
    stride_comp_slot: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufu_slot: tl.constexpr,
    stride_bufu_pos: tl.constexpr,
    stride_bufu_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
):
    """A window of up to ``T_MAX`` tokens per sequence: the speculative verify.

    Each token's state is the one after the tokens before it in the window,
    so ``S_h`` is materialised once in fp32 -- the boundary decayed by
    ``e^C`` plus the committed records -- and the window is walked token by
    token, as ReplaySSM's serial route does.  Every token appends its record;
    `replayssm_commit` then keeps only the accepted ones.
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    # Same predicate as the fold kernel and `replayssm_commit`: on a fold step
    # the boundary state already absorbed every committed record.
    h = tl.maximum(tl.load(write_pos + slot), 0)
    base = tl.where(h + 2 * T_MAX > CAP, 0, h)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    b_h = _lq_rebuild(
        buf_k,
        buf_u,
        buf_g,
        slot,
        base,
        z + slot * stride_z_slot + i_hv * K * V,
        sc + slot * stride_sc_slot + i_hv * (V + K),
        comp + slot * stride_comp_slot + i_hv * R * (K + V),
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
        BH,
        R,
    )

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_beta = beta + bos * HV + i_hv
    p_g = g + bos * HV + i_hv
    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_gt = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_gt)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= tl.load(p_beta).to(tl.float32)
        b_h += b_k[:, None] * b_v[None, :]
        tl.store(
            p_o, tl.sum(b_h * b_q[:, None], 0).to(p_o.dtype.element_ty), mask=mask_v
        )

        pos = (base + i_t).to(tl.int64)
        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + pos * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        b_ub = tl.minimum(tl.maximum(b_v, -_FP16_MAX), _FP16_MAX)
        tl.store(p_bu, b_ub.to(p_bu.dtype.element_ty), mask=mask_v)
        # k and g do not depend on the V split; one program owns their store.
        if i_v == 0:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + pos * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg = buf_g + slot * stride_bufg_slot + pos * stride_bufg_pos
            tl.store(p_bg + i_hv * stride_bufg_hv, b_gt.to(p_bg.dtype.element_ty))

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_beta += HV
        p_g += HV


_MAX_INT_32 = 2**31 - 1


class _Span:
    """A tensor view that Triton may address with 32-bit buffer offsets.

    Triton's AMD backend only enables buffer loads for tensors whose *storage*
    fits in 2 GiB, and one layer's view of an all-layer pool inherits the
    whole pool's storage, which costs the decode kernel ~1.6x.  The kernels
    never address outside the view, so the view's own extent is the bound.
    """

    __slots__ = ("_t", "dtype")

    def __init__(self, t: torch.Tensor):
        self._t, self.dtype = t, t.dtype

    def data_ptr(self) -> int:
        return self._t.data_ptr()

    def ptr_range(self) -> int:
        return _extent(self._t)


def _extent(t: torch.Tensor) -> int:
    """Bytes from ``t.data_ptr()`` to the end of the view."""
    last = sum((s - 1) * st for s, st in zip(t.shape, t.stride()) if s > 0)
    return (last + 1) * t.element_size()


def _span(t: torch.Tensor):
    if t.untyped_storage().size() <= _MAX_INT_32 or _extent(t) > _MAX_INT_32:
        return t
    return _Span(t)


def _side_strides(z: torch.Tensor, sc: torch.Tensor, comp: torch.Tensor) -> dict:
    return {
        "stride_z_slot": z.stride(0),
        "stride_sc_slot": sc.stride(0),
        "stride_comp_slot": comp.stride(0),
    }


def _record_strides(buf_k, buf_u, buf_g) -> dict:
    return {
        "stride_bufk_slot": buf_k.stride(0),
        "stride_bufk_pos": buf_k.stride(2),
        "stride_bufk_hv": buf_k.stride(1),
        "stride_bufu_slot": buf_u.stride(0),
        "stride_bufu_pos": buf_u.stride(2),
        "stride_bufu_hv": buf_u.stride(1),
        "stride_bufg_slot": buf_g.stride(0),
        "stride_bufg_pos": buf_g.stride(2),
        "stride_bufg_hv": buf_g.stride(1),
    }


def _layered(x: torch.Tensor, ndim: int) -> torch.Tensor:
    """``x`` with a leading layer axis; a single layer's tensor gets a unit one."""
    return x if x.ndim == ndim + 1 else x.unsqueeze(0)


def _fold_strides(z, sc, comp, buf_k, buf_u, buf_g) -> dict:
    """Strides of layered (``[L, slot, ...]``) tensors for the fold kernel."""
    out = {}
    for name, x in (("z", z), ("sc", sc), ("comp", comp)):
        out[f"stride_{name}_layer"] = x.stride(0)
        out[f"stride_{name}_slot"] = x.stride(1)
    for name, x in (("bufk", buf_k), ("bufu", buf_u), ("bufg", buf_g)):
        out[f"stride_{name}_layer"] = x.stride(0)
        out[f"stride_{name}_slot"] = x.stride(1)
        out[f"stride_{name}_hv"] = x.stride(2)
        out[f"stride_{name}_pos"] = x.stride(3)
    return out


def _check_side_layout(z, sc, comp) -> None:
    HV, K, V = z.shape[-3:]
    rank = comp.shape[-2]
    assert z.dtype == torch.int8, f"residual pool must be int8, got {z.dtype}"
    assert z.stride()[-3:] == (K * V, V, 1), "residual heads must be contiguous"
    assert sc.shape[-2:] == (HV, V + K) and sc.stride()[-2:] == (V + K, 1)
    assert comp.shape[-3:] == (HV, rank, K + V)
    assert comp.stride()[-3:] == (rank * (K + V), K + V, 1)


def _rank_block(rank: int) -> int:
    assert rank >= 1, "LeapQuant needs at least one compensator token"
    return max(2, triton.next_power_of_2(rank))


def _record_block(cap: int) -> int:
    return max(16, triton.next_power_of_2(cap - 1))


def _verify_launch_config(n: int, hv: int, v: int) -> tuple[int, int]:
    """``(BV, num_warps)`` for `_leapquant_verify_kernel`.

    One warp, so each token's contractions over K reduce inside it: four
    warps at BV = 32 measured 3.4x slower at 128 sequences (gfx950, the
    Qwen3.5-9B heads, T = 4).  Up to 1024 (sequence, head) pairs a 32-wide
    V slice keeps more programs in flight; past that 64 wins (83 against
    96 us a layer at 48 sequences, 142 against 165 at 128).
    """
    return min(triton.next_power_of_2(v), 32 if n * hv <= 1024 else 64), 1


def leapquant_fold(
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    max_query_len: int = 1,
    iters: int = WARM_ITERS,
    num_warps: int = 2,
    waves_per_eu: int = 2,
) -> None:
    """Fold full record buffers into the boundary states, before a decode step.

    Run after the step's `replayssm_commit` and before any layer.  Takes one
    layer's tensors or every layer's stacked on a leading axis
    (``z [L, slots, HV, K, V]`` and so on): the cursor is shared, so a
    sequence flushes in every layer at the same step and one launch covers
    them all.  A fold is a few dozen microseconds of dependent reductions, so
    batching the layers is what turns that latency into throughput.

    ``max_query_len`` is the verify window, as passed to the step's layers and
    to `replayssm_commit`: all three decide a flush by ``h + 2T > cap``.
    """
    tensors = (
        _layered(z, 4),
        _layered(sc, 3),
        _layered(comp, 4),
        _layered(buf_k, 4),
        _layered(buf_u, 4),
        _layered(buf_g, 3),
    )
    _check_side_layout(*tensors[:3])
    L, _, HV, K, V = tensors[0].shape
    N = slot_idx.numel()
    if N == 0:
        return
    cap = tensors[3].shape[3]
    assert flush_threshold_ok(cap, max_query_len), f"record buffer {cap} too short"
    # As many layers per launch as keep every view within 32-bit offsets.
    group = L
    for x in tensors:
        step = x.stride(0) * x.element_size()
        group = min(group, max(1, (_MAX_INT_32 - _extent(x[0])) // step + 1))
    for l0 in range(0, L, group):
        part = [x[l0 : l0 + group] for x in tensors]
        _leapquant_fold_kernel[(N * HV, part[0].shape[0])](
            part[0],
            *(_span(x) for x in part),
            write_pos=write_pos,
            slot_idx=slot_idx,
            N=N,
            T_MAX=max_query_len,
            CAP=cap,
            HV=HV,
            K=K,
            V=V,
            BK=triton.next_power_of_2(K),
            BV=triton.next_power_of_2(V),
            BH=_record_block(cap),
            R=part[2].shape[3],
            **_fold_strides(*part),
            FROM_STATE=False,
            ITERS=iters,
            WARM=True,
            num_warps=num_warps,
            # The fp32 tile takes ~296 VGPRs at num_warps=2, one wave per
            # SIMD, and the fold is latency-bound there. Capping at 256 gives
            # two waves for ~25 spilled registers: the same bytes, ~25% faster.
            waves_per_eu=waves_per_eu,
        )


def leapquant_store_state(
    state: torch.Tensor,
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    slot_idx: torch.Tensor,
    iters: int = COLD_ITERS,
) -> None:
    """Quantise fp32 states ``[N, HV, K, V]`` into one layer's slots.

    ``z [slots, HV, K, V]`` int8, ``sc`` and ``comp`` as
    :func:`leapquant_buffer_shapes`.  Entries of ``slot_idx`` below 0 are
    skipped.
    """
    N, HV, K, V = state.shape
    if N == 0:
        return
    _check_side_layout(z, sc, comp)
    state = state.to(torch.float32).contiguous()
    z, sc, comp = _layered(z, 4), _layered(sc, 3), _layered(comp, 4)
    _leapquant_fold_kernel[(N * HV, 1)](
        state=state,
        z=_span(z),
        sc=_span(sc),
        comp=_span(comp),
        buf_k=z,
        buf_u=z,
        buf_g=z,
        write_pos=slot_idx,
        slot_idx=slot_idx,
        N=N,
        T_MAX=1,
        CAP=0,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BH=16,
        R=comp.shape[3],
        **_fold_strides(z, sc, comp, z, z, z),
        FROM_STATE=True,
        ITERS=iters,
        WARM=False,
        num_warps=4,
    )


def leapquant_load_state(
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    slot_idx: torch.Tensor,
) -> torch.Tensor:
    """Dequantise the boundary states of ``slot_idx`` into fp32 ``[N, HV, K, V]``.

    Records are not replayed: callers use this where the cursor is parked at
    the prefill sentinel, i.e. the boundary is the whole state.  Entries of
    ``slot_idx`` below 0 come back as zeros.
    """
    _check_side_layout(z, sc, comp)
    _, HV, K, V = z.shape
    N = slot_idx.numel()
    out = torch.empty(N, HV, K, V, dtype=torch.float32, device=z.device)
    if N == 0:
        return out
    rank = comp.shape[2]
    _leapquant_load_kernel[(N * HV,)](
        out=out,
        z=_span(z),
        sc=_span(sc),
        comp=_span(comp),
        slot_idx=slot_idx,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=triton.next_power_of_2(V),
        BR=_rank_block(rank),
        R=rank,
        **_side_strides(z, sc, comp),
        num_warps=8,
    )
    return out


def leapquant_seed_forked_slots(
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    fork_slots: torch.Tensor | None,
) -> None:
    """Copy a state fork's source boundary state onto its write slot.

    `seed_forked_slots` for the int8 pool, which is no state without its
    scales and compensators.  Every layer at once: the pools are layer-major,
    ``[L, slots, ...]``.  ``fork_slots`` is the builder's int64 ``[2, F]``
    (write, source) pairs; the caller parks the write slots' cursors, since
    a source holds no records.
    """
    if fork_slots is None:
        return
    dst, src = fork_slots[0], fork_slots[1]
    for pool in (z, sc, comp):
        pool[:, dst] = pool[:, src]


def leapquant_write_checkpoints(
    h: torch.Tensor,
    final_state: torch.Tensor,
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    rows: torch.Tensor,
    slots: torch.Tensor,
    offs: torch.Tensor,
    is_end: torch.Tensor,
    chunk_offsets: torch.Tensor,
    chunk_size: int,
) -> None:
    """The SSM half of `write_state_checkpoints`, for the int8 pool.

    Same targets: an interior one is the chunk boundary
    ``h[chunk_offsets[row] + off // chunk_size]``, one at the end of its row's
    tokens is that row's ``final_state`` (``[N, HV, K, V]``), which ``h`` does
    not hold.  Each is quantised into its slot as a prefill's final state is;
    a checkpoint is stored from its slot and restored into a prefill, so no
    cursor is involved.
    """
    if rows.numel() == 0:
        return
    HV, K, V = z.shape[-3:]
    rows = rows.long()
    h = h.reshape(-1, HV, K, V)
    src = chunk_offsets.long()[rows] + offs.long() // chunk_size
    inner = h.index_select(0, src.clamp(max=h.shape[0] - 1)).float()
    state = torch.where(
        is_end.bool()[:, None, None, None],
        final_state.index_select(0, rows).float(),
        inner,
    )
    leapquant_store_state(state, z, sc, comp, slots)


def leapquant_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    sc: torch.Tensor,
    comp: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    fold: bool = True,
    iters: int = WARM_ITERS,
    o: torch.Tensor | None = None,
    block_v: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """LeapQuant decode for one GDN layer; same contract as
    :func:`replayssm_gated_delta_rule` with the checkpoint pool replaced by
    ``(z, sc, comp)``.  ``write_pos`` is advanced by `replayssm_commit`.

    ``max_query_len`` is the verify window the server runs with, whatever
    this step's rows carry: the flush predicate is ``h + 2T > cap`` for the
    configured ``T``, the same one `replayssm_commit` and the fold use.  With
    no speculation (``T == 1``) a token needs only two contractions of the
    state; a verify window walks its tokens over the materialised state.

    ``fold=False`` when the caller already ran :func:`leapquant_fold` for this
    step over every layer at once.  ``o`` is an optional contiguous
    ``[T_tot, HV, V]`` destination.
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _check_side_layout(z, sc, comp)
    _, T_tot, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    rank = comp.shape[2]
    if scale is None:
        scale = K**-0.5
    assert flush_threshold_ok(cap, max_query_len), f"record buffer {cap} too short"
    assert beta.ndim != v.ndim, "LeapQuant decode expects a per-head beta"

    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if o is None:
        o = q.new_empty(1, T_tot, HV, V)
    else:
        assert o.is_contiguous(), "o must be contiguous"
        o = o.view(1, T_tot, HV, V)
    if N == 0:
        return o
    if fold:
        leapquant_fold(
            z,
            sc,
            comp,
            buf_k,
            buf_u,
            buf_g,
            write_pos,
            slot_idx[:N],
            max_query_len=max_query_len,
            iters=iters,
        )
    if max_query_len > 1:
        BV, warps = _verify_launch_config(N, HV, V)
        BV = min(triton.next_power_of_2(V), block_v or BV)
        _leapquant_verify_kernel[(triton.cdiv(V, BV), N * HV)](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=o,
            z=_span(z),
            sc=_span(sc),
            comp=_span(comp),
            buf_k=_span(buf_k),
            buf_u=_span(buf_u),
            buf_g=_span(buf_g),
            write_pos=write_pos,
            slot_idx=slot_idx,
            cu_seqlens=cu_seqlens,
            scale=scale,
            N=N,
            T_MAX=max_query_len,
            CAP=cap,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=triton.next_power_of_2(K),
            BV=BV,
            BH=_record_block(cap),
            R=rank,
            **_side_strides(z, sc, comp),
            **_record_strides(buf_k, buf_u, buf_g),
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            num_warps=num_warps or warps,
            num_stages=1,
        )
        return o
    BV = min(triton.next_power_of_2(V), block_v or 128)
    _leapquant_decode_kernel[(triton.cdiv(V, BV), N * HV)](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        z=_span(z),
        sc=_span(sc),
        comp=_span(comp),
        buf_k=_span(buf_k),
        buf_u=_span(buf_u),
        buf_g=_span(buf_g),
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        BH=_record_block(cap),
        BR=_rank_block(rank),
        R=rank,
        **_side_strides(z, sc, comp),
        **_record_strides(buf_k, buf_u, buf_g),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_BETA_HEADWISE=False,
        num_warps=num_warps or 1,
        num_stages=1,
    )
    return o
