# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ReplaySSM for gated-delta-rule / KDA linear attention.

Instead of materialising one recurrent state per speculative token, cache the
*inputs* that produced them and rebuild the state on demand.  The recurrence

    S_t = diag(a_t) S_{t-1} + k_t u_t^T ,   u_t = beta_t (v_t - (diag(a_t)S_{t-1})^T k_t)

is linear in the state, so

    S_h = diag(e^{G_h}) S_0 + sum_j diag(e^{G_h - G_j}) k_j u_j^T

with G_j the running sum of log-decays.  Keeping a checkpoint ``S_0`` plus the
last few ``(k, u, g)`` records is therefore equivalent to keeping the state,
at ~1/40 of the bytes for K = V = 128.

Two consequences:

* **Speculative rollback becomes a cursor move.**  A rejected draft is undone
  by not advancing the cursor past its record; no per-draft state snapshot is
  needed, so the state pool stops scaling with ``num_speculative_tokens``.
* **The full-state write moves off the per-step path.**  The checkpoint is only
  rewritten when the record buffer fills (every ``cache_len`` committed
  tokens), so most steps read the state and never write it.

Divergence from upstream ReplaySSM: the checkpoint absorbs the committed
records at the *start* of a step rather than the end.  Upstream folds at the
end, which strands the current step's speculative records at a non-zero offset
and forces a true ring with modular indexing; folding at the start means the
buffer always refills from offset 0, so it is a plain linear buffer with reset.
Same state traffic, same flush cadence, no wrap-around arithmetic.

Reference: https://tridao.me/blog/2026/replayssm/
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from aiter.jit.utils.chip_info import get_gfx

from .op import exp

__all__ = [
    "PAD_SLOT_ID",
    "flush_threshold_ok",
    "replayssm_buffer_shapes",
    "replayssm_commit",
    "replayssm_gated_delta_rule",
    "replayssm_sigmoid_gating_delta_rule",
]

PAD_SLOT_ID = -1

#: `route="auto"` switches to the UT-transform kernel at or above this verify
#: window.  See the measurement table in `replayssm_gated_delta_rule`.
UT_MIN_QUERY_LEN = 12

#: Replay-GEMM arithmetic, keyed by the record buffer's dtype.  See
#: `_replay_dot` for what the modes do and why the split is only one-sided.
_DOT_MODE_BY_RECORD_DTYPE = {
    torch.bfloat16: 2,
    torch.float16: 3,
}


def _replay_dot_mode(record_dtype: torch.dtype) -> int:
    """How to contract the records against the checkpoint on a flush.

    A 16-bit record buffer replays on the bf16 matrix cores: fp32 MFMA runs at
    an eighth of the bf16 rate on CDNA, and upcasting a bf16 record to fp32
    cannot add information it never had.  An fp32 buffer is the other way
    round -- a bf16 hi/lo pair tops out near 16 mantissa bits and would lose
    real precision -- so those callers keep the fp32 contraction.
    """
    return _DOT_MODE_BY_RECORD_DTYPE.get(record_dtype, 0)


# --------------------------------------------------------------------------- #
# Flush policy                                                                 #
# --------------------------------------------------------------------------- #
#
# The predicate below is evaluated in two places -- the layer kernel (to decide
# whether *this* step folds) and the commit kernel (to re-derive what the
# *previous* step decided, since the forward does not touch the cursor).  They
# must agree exactly, so both call the same expression; do not inline a variant.
#
# Firing at `h + 2T > L` rather than `h + T > L` keeps at least one full window
# free: a step that lands at h = L-T followed by a full accept would otherwise
# leave a single free slot and truncate the next window to one draft.  It also
# guarantees h + T <= L, i.e. appends never run off the end of the buffer.


def flush_threshold_ok(cache_len: int, max_query_len: int) -> bool:
    """`cache_len` must hold two full windows or the invariant above breaks."""
    return cache_len >= 2 * max_query_len


def replayssm_buffer_shapes(
    cache_len: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    is_kda: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Per-slot record buffer shapes: (k, u, g)."""
    return (
        (num_v_heads, cache_len, head_k_dim),
        (num_v_heads, cache_len, head_v_dim),
        (num_v_heads, cache_len, head_k_dim) if is_kda else (num_v_heads, cache_len),
    )


# --------------------------------------------------------------------------- #
# Record replay                                                                #
# --------------------------------------------------------------------------- #
#
# Unrolling the recurrence over the h committed records,
#
#     S_{j+1} = diag(exp(g_j)) S_j + u_j (x) k_j
#
# leaves a form with no sequential dependency in it at all:
#
#     S_h = exp(C) * S_0 + sum_j exp(C - C_j) * u_j (x) k_j,   C_j = sum_{i<=j} g_i
#
# so the replay is ONE diagonal scale plus ONE GEMM rather than h dependent
# rank-1 updates.  What makes this legal is that the buffer stores `u` (already
# delta-corrected) and not raw `v`: with `v`, each `u_j = beta_j (v_j - S_{j-1}^T
# k_j)` would depend on the previous state and the chain could not be cut.
#
# This is not a micro-optimisation.  The serial form measured 4.95 us per record
# at K = V = 128 -- not bandwidth, but h dependent full-tile multiplies with no
# instruction-level parallelism to hide the latency.  Against a 23 us baseline
# step at T=1 (mean h ~7.5) that made plain decoding 8% SLOWER than the kernel
# ReplaySSM is supposed to beat, which is the opposite of the whole point:
# skipping the state write-back on 14 of every 15 steps is only a win if
# rebuilding is cheap.


@triton.jit
def _replay_tiles(
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
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """The h committed records as dense `[BH, *]` tiles, ready for one GEMM.

    Returns ``(b_kw, b_u, b_decay)`` with ``b_kw[j] = exp(C - C_j) * k_j``,
    ``b_u[j] = u_j`` and ``b_decay = exp(C)``, the factor the incoming
    checkpoint is scaled by.

    Rows ``j >= h`` are zeroed rather than skipped, so they contribute nothing
    to the GEMM and no caller has to branch on how many records were live.
    Their gate loads default to 0, which leaves both the running sum and the
    total untouched -- `b_ctot` is the sum over the whole tile precisely
    because the padding is additively neutral.
    """
    o_h = tl.arange(0, BH)
    m_h = o_h < h

    if IS_KDA:
        b_g = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h[:, None] * stride_bufg_pos
            + i_hv * stride_bufg_hv
            + o_k[None, :],
            mask=m_h[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
    else:
        # Scalar gate: broadcast the per-record value across K so the cumsum
        # and the weighting below stay one code path.  Only the fused-gating
        # KDA kernel reaches this helper, and it always has a per-channel gate,
        # so in practice this branch is not instantiated -- the scalar-gate
        # callers go through `_replay_tiles_kmajor`, which keeps the gate a
        # [BH] vector instead of paying the broadcast.
        b_g1 = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h * stride_bufg_pos
            + i_hv * stride_bufg_hv,
            mask=m_h,
            other=0.0,
        ).to(tl.float32)
        b_g = b_g1[:, None] + tl.zeros([BH, BK], dtype=tl.float32)

    # Gates are log-decays (<= 0), so C - C_j <= 0 and every weight is in (0, 1]
    # -- the exponentials cannot overflow however long the buffer gets.
    b_c = tl.cumsum(b_g, axis=0)
    b_ctot = tl.sum(b_g, axis=0)
    b_w = exp(b_ctot[None, :] - b_c)

    b_k = tl.load(
        buf_k
        + slot * stride_bufk_slot
        + o_h[:, None] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_k[None, :],
        mask=m_h[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)
    b_u = tl.load(
        buf_u
        + slot * stride_bufu_slot
        + o_h[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v[None, :],
        mask=m_h[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)
    return b_k * b_w, b_u, exp(b_ctot)


@triton.jit
def _replay_tiles_kmajor(
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
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BH: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    """`_replay_tiles` with the k side already transposed, for a [BK, BV] state.

    Returns ``(b_kw [BK, BH], b_u [BH, BV], b_decay [BK])`` so the caller can
    contract with a bare ``tl.dot(b_kw, b_u)``.

    k-major is here for the T == 1 path's reductions, not for the GEMM.  It
    was originally tried as a way to drop the ``tl.trans`` ahead of `tl.dot`,
    on the theory that the LDS staging Triton emits for it (`shared=8192`,
    ~40 `ds_write` on the fp32 tile) was what made the replay GEMM cost 92 us
    of a 173 us kernel.  That theory was wrong, or at least incomplete: with
    an fp32 dot, k-major measured 171 us against 173 us -- nothing.  The
    record axis is the strided one in the buffer whichever orientation the
    tile is read in, so the layout conversion happens either way.

    What it does buy is the orientation of `S_h^T x` on the T == 1 path, where
    the contraction is a reduction rather than a dot: reducing a [BK, BH] tile
    along BK measured 111.5 us against 118.1 us for the [BH, BK] tile reduced
    along its last axis, despite k-major's loads being the less coalesced of
    the two (consecutive lanes walk `stride_bufk_pos`, 4 KiB apart).

    The scalar-gate branch also keeps the gate as a [BH] vector rather than
    broadcasting it to [BH, BK] first: the weights are per-record, so the wide
    tile made the cumsum and BK-1 of every BK exponentials redundant.
    """
    o_h = tl.arange(0, BH)
    m_h = o_h < h

    b_kt = tl.load(
        buf_k
        + slot * stride_bufk_slot
        + o_h[None, :] * stride_bufk_pos
        + i_hv * stride_bufk_hv
        + o_k[:, None],
        mask=m_h[None, :] & mask_k[:, None],
        other=0.0,
    ).to(tl.float32)
    b_u = tl.load(
        buf_u
        + slot * stride_bufu_slot
        + o_h[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v[None, :],
        mask=m_h[:, None] & mask_v[None, :],
        other=0.0,
    ).to(tl.float32)

    # Gates are log-decays (<= 0), so C - C_j <= 0 and every weight is in (0, 1]
    # -- the exponentials cannot overflow however long the buffer gets.
    if IS_KDA:
        b_g = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h[None, :] * stride_bufg_pos
            + i_hv * stride_bufg_hv
            + o_k[:, None],
            mask=m_h[None, :] & mask_k[:, None],
            other=0.0,
        ).to(tl.float32)
        b_c = tl.cumsum(b_g, axis=1)
        b_ctot = tl.sum(b_g, axis=1)
        b_kw = b_kt * exp(b_ctot[:, None] - b_c)
        b_decay = exp(b_ctot)
    else:
        b_g1 = tl.load(
            buf_g
            + slot * stride_bufg_slot
            + o_h * stride_bufg_pos
            + i_hv * stride_bufg_hv,
            mask=m_h,
            other=0.0,
        ).to(tl.float32)
        b_c1 = tl.cumsum(b_g1, axis=0)
        b_ctot1 = tl.sum(b_g1, axis=0)
        b_kw = b_kt * exp(b_ctot1 - b_c1)[None, :]
        b_decay = exp(b_ctot1) + tl.zeros([BK], dtype=tl.float32)
    return b_kw, b_u, b_decay


@triton.jit
def _record_owner(rec_ctr, i_nh, i_v, NV: tl.constexpr, do_flush):
    """Whether this V slice stores the shared k/g records of its head this step.

    The k and g records do not depend on the V split, so one slice stores them.
    Off a fold step they land past every record being read, and slice 0 does it.
    On a fold step they land at 0.., over committed records that the head's other
    slices read to rebuild the state, possibly later than slice 0 runs. There the
    store goes to the last slice to finish reading: each slice counts itself in
    after its record loads, and the last one resets the counter for the next launch.
    """
    owner = i_v == 0
    if NV > 1:
        if do_flush:
            tl.debug_barrier()
            arrived = tl.atomic_add(rec_ctr + i_nh, 1, sem="acq_rel")
            owner = arrived == NV - 1
            if owner:
                tl.store(rec_ctr + i_nh, 0)
    return owner


_ARRIVAL_CTRS: dict[tuple[torch.device, str], torch.Tensor] = {}


def _arrival_counters(device: torch.device, name: str, entries: int) -> torch.Tensor:
    """Zeroed int32 arrival counters for one kind of cross-program handoff.

    Each launch leaves them zero (the last arrival resets its own), so one buffer
    serves every layer: they run one after another on one stream.
    """
    key = (device, name)
    buf = _ARRIVAL_CTRS.get(key)
    if buf is None or buf.numel() < entries:
        buf = torch.zeros(max(entries, 1 << 16), dtype=torch.int32, device=device)
        _ARRIVAL_CTRS[key] = buf
    return buf


@triton.jit
def _replay_dot(
    lhs,
    rhs,
    SPLIT_LHS: tl.constexpr,
    DOT_MODE: tl.constexpr,
):
    """``tl.dot(lhs, rhs)`` for the replay contraction.

    Operands arrive dot-ready: callers in the ``[BK, BV]`` state layout get a
    k-major tile out of `_replay_tiles_kmajor`, KDA's ``[BV, BK]`` layout
    transposes at the call site.

    Reached on a flush, or on a multi-token (speculative) step.  At T == 1 the
    caller takes the state-free path and never gets here, which is the point:
    this contraction is expensive on CDNA for reasons that survive every fix
    tried at it -- fp32 dot 92 us of a 173 us kernel, bf16 operands (an eighth
    the MFMA cost) still 63 us, k-major to skip the ``tl.trans`` ~0.  Only one
    step in ``cap - 1`` needs it now.

    ``SPLIT_LHS`` says which side is ``b_kw``, the only operand genuinely wider
    than the record buffer: ``b_u`` is a raw record load, so narrowing it back
    to the buffer's own dtype is exact, while ``b_kw = k * w`` is a product and
    is not.

    ``DOT_MODE`` picks the arithmetic:
      0 -- one fp32 dot.  What an fp32 record buffer needs: bf16 hi/lo tops out
           near 16 mantissa bits and cannot carry fp32 records faithfully.
      1 -- one bf16 dot.  ~8 mantissa bits; too lossy, attribution only.
      2 -- bf16 hi/lo on the ``b_kw`` side only, 2 dots.  For a bf16 record
           buffer the other side's lo term is identically zero, so this is the
           whole split; measured 3.6e-06 relative against an fp64 reference,
           553x tighter than mode 1, for 2.9 us over it.
      3 -- also splits the record side, 3 dots (lo*lo is negligible).  Needed
           when the buffer is 16-bit but not bf16, i.e. fp16.
    """
    if DOT_MODE == 0:
        acc = tl.dot(lhs, rhs)
    elif SPLIT_LHS:
        b_hi = lhs.to(tl.bfloat16)
        b_other = rhs.to(tl.bfloat16)
        acc = tl.dot(b_hi, b_other)
        if DOT_MODE >= 2:
            b_lo = (lhs - b_hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_lo, b_other)
        if DOT_MODE == 3:
            b_other_lo = (rhs - b_other.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_hi, b_other_lo)
    else:
        b_hi = rhs.to(tl.bfloat16)
        b_other = lhs.to(tl.bfloat16)
        acc = tl.dot(b_other, b_hi)
        if DOT_MODE >= 2:
            b_lo = (rhs - b_hi.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_other, b_lo)
        if DOT_MODE == 3:
            b_other_lo = (lhs - b_other.to(tl.float32)).to(tl.bfloat16)
            acc += tl.dot(b_other_lo, b_hi)
    return acc


# --------------------------------------------------------------------------- #
# Commit kernel -- runs once per forward, shared by every linear-attn layer    #
# --------------------------------------------------------------------------- #


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _replayssm_commit_kernel(
    write_pos,
    slot_idx,
    num_accepted,
    N,
    T_MAX,
    CAP,
):
    i_n = tl.program_id(0)
    if i_n >= N:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:
        return
    h = tl.load(write_pos + slot)
    # -1 is the prefill sentinel: that forward rebuilt the checkpoint wholesale
    # and wrote no records, so there is nothing to commit and the first decode
    # has to start from an empty prefix.  Advancing by the 1 that `num_bonus ==
    # 0` yields would fold record 0, which on a reused slot is still the
    # previous tenant's last record.
    if h < 0:
        tl.store(write_pos + slot, 0)
        return
    # Re-derive the previous forward's flush decision.  The forward never
    # mutates write_pos, so `h` here is exactly what that forward branched on.
    prev_flushed = h + 2 * T_MAX > CAP
    base = tl.where(prev_flushed, 0, h)
    tl.store(write_pos + slot, base + tl.load(num_accepted + i_n))


def replayssm_commit(
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    num_accepted: torch.Tensor,
    max_query_len: int,
    cache_len: int,
) -> None:
    """Advance each sequence's record cursor by the previous step's accepts.

    Call exactly once per forward, before any linear-attention layer runs.
    Device-side so the accepted counts never round-trip to the host.
    """
    n = slot_idx.numel()
    if n == 0:
        return
    _replayssm_commit_kernel[(n,)](
        write_pos,
        slot_idx,
        num_accepted,
        n,
        max_query_len,
        cache_len,
        num_warps=1,
    )


# --------------------------------------------------------------------------- #
# Fused rebuild + decode/verify kernel                                         #
# --------------------------------------------------------------------------- #


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
    rec_ctr,
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
        owns_rec = _record_owner(rec_ctr, i_nh, i_v, tl.cdiv(V, BV), do_flush)

        if do_flush:
            tl.store(
                p_s,
                (b_s * b_decay + _replay_dot(b_kwt, b_ru, True, DOT_MODE)).to(
                    p_s.dtype.element_ty
                ),
                mask=mask_k[:, None] & mask_v[None, :],
            )
        if owns_rec:
            p_bkt = (
                buf_k
                + slot * stride_bufk_slot
                + base * stride_bufk_pos
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
        if owns_rec:
            p_bg2 = buf_g + slot * stride_bufg_slot + base * stride_bufg_pos
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
    owns_rec = _record_owner(rec_ctr, i_nh, i_v, tl.cdiv(V, BV), do_flush)
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
        if owns_rec:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + base * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg2 = buf_g + slot * stride_bufg_slot + base * stride_bufg_pos
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
        if owns_rec:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + pos * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg2 = buf_g + slot * stride_bufg_slot + pos * stride_bufg_pos
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


# --------------------------------------------------------------------------- #
# UT-transform verify route (chunked delta rule)                               #
# --------------------------------------------------------------------------- #
#
# The serial kernel above walks the T verify tokens one at a time because
# u_s depends on the state after token s-1.  Expanding every token from the
# same rebuilt S_h instead decouples them:
#
#   R_s     = beta_s (v_s - S_h^T (e^{G_s} . k_s))            <- all s in parallel
#   A_{s,s'}= beta_s <e^{G_s-G_{s'}} . k_{s'}, k_s>,  s' < s  <- strictly lower
#   U       = (I + A)^{-1} R                                  <- T x T solve
#   o_s     = S_h^T (e^{G_s} . q_s) + sum_{s'<=s} <e^{G_s-G_{s'}} . k_{s'}, q_s> U_{s'}
#
# with G_s the running sum of log-decays.  The only serial part left is the
# T x T triangular solve, which is O(T^2 V) instead of O(T K V).
#
# Scope: scalar-gate GDN only.  With KDA's per-channel gate, e^{G_s - G_{s'}}
# sits inside the contraction over K, so the pairwise weight cannot be folded
# into a single GEMM without splitting it into e^{G_s} * e^{-G_{s'}} -- and
# e^{-G_{s'}} grows without bound over the window.  KDA therefore stays on the
# serial route, which costs nothing in memory (the whole point of ReplaySSM)
# and only forgoes this throughput optimisation.  Upstream vLLM/SGLang draw
# the same line.


@triton.jit(do_not_specialize=["N", "T_TOT", "T_MAX", "CAP"])
def _replayssm_ut_fwd_kernel(
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
    BT: tl.constexpr,
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
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    DOT_MODE: tl.constexpr,
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
    if slot < 0:
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

    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_k = o_k < K
    m_v = o_v < V
    m_tk = m_t[:, None] & m_k[None, :]
    m_tv = m_t[:, None] & m_v[None, :]

    # ---- token tiles --------------------------------------------------------
    p_qk = (bos + o_t)[:, None] * (H * K) + i_h * K + o_k[None, :]
    b_k = tl.load(k + p_qk, mask=m_tk, other=0.0).to(tl.float32)
    b_q = tl.load(q + p_qk, mask=m_tk, other=0.0).to(tl.float32)
    if USE_QK_L2NORM_IN_KERNEL:
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k, 1) + 1e-6)[:, None]
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q, 1) + 1e-6)[:, None]
    b_q = b_q * scale
    b_v = tl.load(
        v + (bos + o_t)[:, None] * (HV * V) + i_hv * V + o_v[None, :],
        mask=m_tv,
        other=0.0,
    ).to(tl.float32)
    b_gt = tl.load(g + (bos + o_t) * HV + i_hv, mask=m_t, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + (bos + o_t) * HV + i_hv, mask=m_t, other=0.0).to(tl.float32)

    # running log-decay; padded rows contribute 0 so they never disturb the sum
    b_G = tl.cumsum(b_gt, axis=0)

    # ---- rebuild S_h --------------------------------------------------------
    p_ckpt_hv = (
        ckpt + slot * stride_ckpt_slot + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    )
    m_h = m_k[:, None] & m_v[None, :]
    b_h = tl.load(p_ckpt_hv, mask=m_h, other=0.0).to(tl.float32)
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
        m_k,
        m_v,
        K,
        V,
        BK,
        BH,
        False,
    )
    b_h = b_h * b_decay[:, None]
    b_h += _replay_dot(b_kw, b_ru, True, DOT_MODE)

    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=m_h)

    # ---- project the rebuilt state onto every token at once -----------------
    b_eG = exp(b_G)  # <= 1, decays are non-positive
    b_hk = tl.dot((b_k * b_eG[:, None]).to(b_h.dtype), b_h)
    b_hq = tl.dot((b_q * b_eG[:, None]).to(b_h.dtype), b_h)
    b_R = b_beta[:, None] * (b_v - b_hk)

    # ---- pairwise decay weights --------------------------------------------
    # dG = G_s - G_s' is <= 0 for s > s' (log-decays are negative); clamping at
    # 0 is exact there and keeps the masked upper triangle from overflowing.
    b_dG = exp(tl.minimum(b_G[:, None] - b_G[None, :], 0.0))
    m_lo = (o_t[:, None] > o_t[None, :]) & m_t[:, None] & m_t[None, :]
    m_le = (o_t[:, None] >= o_t[None, :]) & m_t[:, None] & m_t[None, :]
    b_kk = tl.dot(b_k.to(b_h.dtype), tl.trans(b_k).to(b_h.dtype))
    b_qk = tl.dot(b_q.to(b_h.dtype), tl.trans(b_k).to(b_h.dtype))
    b_A = tl.where(m_lo, b_beta[:, None] * b_dG * b_kk, 0.0)

    # ---- (I + A)^-1, built row by row (same recurrence as solve_tril) -------
    b_Ai = -b_A
    for i in range(1, BT):
        b_row = -tl.sum(tl.where((o_t == i)[:, None], b_A, 0.0), 0)
        b_row = b_row + tl.sum(b_row[:, None] * b_Ai, 0)
        b_Ai = tl.where((o_t == i)[:, None], b_row, b_Ai)
    b_Ai += (o_t[:, None] == o_t[None, :]).to(tl.float32)

    b_U = tl.dot(b_Ai.to(b_h.dtype), b_R.to(b_h.dtype))
    b_o = b_hq + tl.dot(
        tl.where(m_le, b_dG * b_qk, 0.0).to(b_h.dtype), b_U.to(b_h.dtype)
    )

    # ---- write output and records ------------------------------------------
    p_o = o + (bos + o_t)[:, None] * (HV * V) + i_hv * V + o_v[None, :]
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_tv)

    p_bu = (
        buf_u
        + slot * stride_bufu_slot
        + (base + o_t)[:, None] * stride_bufu_pos
        + i_hv * stride_bufu_hv
        + o_v[None, :]
    )
    tl.store(p_bu, b_U.to(p_bu.dtype.element_ty), mask=m_tv)
    if i_v == 0:
        p_bk = (
            buf_k
            + slot * stride_bufk_slot
            + (base + o_t)[:, None] * stride_bufk_pos
            + i_hv * stride_bufk_hv
            + o_k[None, :]
        )
        tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=m_tk)
        p_bg = (
            buf_g
            + slot * stride_bufg_slot
            + (base + o_t) * stride_bufg_pos
            + i_hv * stride_bufg_hv
        )
        tl.store(p_bg, b_gt.to(p_bg.dtype.element_ty), mask=m_t)


def _serial_launch_config(
    N: int, HV: int, V: int, max_query_len: int, is_kda: bool
) -> tuple[int, int, int]:
    """(BV, num_warps, num_stages) for `_replayssm_fwd_kernel`.

    Large batches (swept on gfx950 over bs 32..256, T 2..16, both the Qwen3.5
    and Kimi-K3 head shapes): BV=64 with a single warp wins across the board.
    BV=32 doubles the number of programs and replicates the per-token
    l2norm/loads; BV=128 overflows the register budget.

    Small multi-token verify batches are latency bound instead: at BV=64 the
    grid is only 2 * N * HV single-warp programs. Splitting V into 16-wide
    slices (8x the programs, 1/4 the state tile each) wins up to N * HV = 768
    (tests/bench_replayssm_decode.py, gfx950, H=4 HV=16 K=V=128 T=4, cold
    cache, us at BV=64 -> 16): bs1 19.9->9.3, bs6 20.2->12.2, bs16 23.3->19.2,
    bs32 33.3->28.7, bs48 38.5->36.4, bs64 even. T=1 takes the T1_* paths,
    where BV=64 stays best; num_stages is flat either way.

    gfx1250 crosses over far earlier (HV=32 and HV=8, K=V=128, T=4, a graph
    of 30 layers' calls, us per call at BV=64/32/16, no-fold step / fold step):
    N*HV=32 8.6/6.8/5.8, 9.2/7.8/6.7; N*HV=64 8.7/6.9/6.5, 9.3/8.1/7.0;
    N*HV=128 9.1/7.3/8.0, 10.1/10.5/10.3; N*HV=256 10.3/10.5/10.8,
    13.3/14.6/15.7. Weighting one fold step in three (cache_len 16, ~3.4
    accepted per step): BV=16 up to N*HV = 64, BV=32 up to 128, BV=64 above.

    gfx942 keeps the large-batch config; it has not been measured.
    """
    if not is_kda and max_query_len > 1:
        gfx = get_gfx()
        if gfx == "gfx950" and N * HV <= 768:
            return min(triton.next_power_of_2(V), 16), 1, 3
        if gfx == "gfx1250" and N * HV <= 128:
            return min(triton.next_power_of_2(V), 16 if N * HV <= 64 else 32), 1, 3
    return min(triton.next_power_of_2(V), 64), 1, 3


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
        rec_ctr=_arrival_counters(q.device, "rec", N * HV),
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
    return o


# --------------------------------------------------------------------------- #
# KDA variant (Kimi-K3): fused gating + transposed state layout                #
# --------------------------------------------------------------------------- #
#
# Kimi's linear-attention layer calls `fused_sigmoid_gating_delta_rule_update`
# rather than the GDN entry, and that kernel differs in two ways that matter
# here, so it gets its own ReplaySSM kernel instead of more constexpr branches
# on the GDN one (which is already carrying production validation):
#
# 1. The state is laid out [slot, HV, V, K] -- the transpose of the GDN pool's
#    [slot, HV, K, V]. Same recurrence, written on S^T:
#        S^T <- S^T diag(a) + u k^T
#    so the ReplaySSM expansion carries over unchanged; only the reduction axis
#    and the pointer arithmetic move.
# 2. The gate is computed *inside* the kernel from (A_log, a, dt_bias) instead
#    of being passed in pre-computed, and Kimi uses a lower-bounded sigmoid
#    gate rather than GDN's -exp(A)*softplus. `a` and `dt_bias` are per-K-channel.
#
# The records are still (k, u, g); `g` is a [K] vector per token here, which
# `replayssm_buffer_shapes(is_kda=True)` already accounts for.


@triton.jit
def _kda_gate(A_log_ptr, a_ptr, dt_bias_ptr, mask_k, LOWER_BOUND, USE_LOWER_BOUND):
    """Per-K-channel log-decay, matching fused_sigmoid_gating's two branches."""
    x = tl.load(a_ptr, mask=mask_k, other=0.0).to(tl.float32) + tl.load(
        dt_bias_ptr, mask=mask_k, other=0.0
    ).to(tl.float32)
    b_A = tl.load(A_log_ptr).to(tl.float32)
    if USE_LOWER_BOUND:
        return LOWER_BOUND * tl.sigmoid(tl.exp(b_A) * x)
    softplus_x = tl.where(x <= 20.0, tl.log(1 + tl.exp(x)), x)
    return -tl.exp(b_A) * softplus_x


@triton.jit(do_not_specialize=["N", "T_TOT", "T_MAX", "CAP"])
def _replayssm_kda_fwd_kernel(
    q,
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    o,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    scale,
    LOWER_BOUND,
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
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    DOT_MODE: tl.constexpr,
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
    if slot < 0:
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

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]  # [BV, BK] -- transposed layout

    # ---- 1. rebuild S_h -----------------------------------------------------
    p_ckpt_hv = (
        ckpt + slot * stride_ckpt_slot + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    )
    b_h = tl.load(p_ckpt_hv, mask=mask_h, other=0.0).to(tl.float32)

    b_kw, b_ru, b_decay = _replay_tiles(
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
        True,
    )
    # KDA keeps the state transposed as [BV, BK], so the contraction is u^T @ k~.
    b_h = b_h * b_decay[None, :]
    b_h += _replay_dot(tl.trans(b_ru), b_kw, False, DOT_MODE)

    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)

    # ---- 2. this step's tokens ---------------------------------------------
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_o = o + (bos * HV + i_hv) * V + o_v
    p_a = a + (bos * HV + i_hv) * K + o_k
    p_b = b + bos * HV + i_hv
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv * K + o_k

    for i_t in range(T):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = _kda_gate(p_A_log, p_a, p_dt_bias, mask_k, LOWER_BOUND, USE_LOWER_BOUND)
        b_beta = tl.sigmoid(tl.load(p_b).to(tl.float32))

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_h *= exp(b_g)[None, :]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        tl.store(
            p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mask_v
        )

        pos = base + i_t
        p_bu = (
            buf_u
            + slot * stride_bufu_slot
            + pos * stride_bufu_pos
            + i_hv * stride_bufu_hv
            + o_v
        )
        tl.store(p_bu, b_v.to(p_bu.dtype.element_ty), mask=mask_v)
        if i_v == 0:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + pos * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg = (
                buf_g
                + slot * stride_bufg_slot
                + pos * stride_bufg_pos
                + i_hv * stride_bufg_hv
                + o_k
            )
            tl.store(p_bg, b_g.to(p_bg.dtype.element_ty), mask=mask_k)

        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_a += stride_a_token
        p_b += stride_b_token


def replayssm_sigmoid_gating_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    o: torch.Tensor | None = None,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """ReplaySSM for the KDA (Kimi-K3) linear-attention layer.

    Drop-in for :func:`fused_sigmoid_gating_delta_rule_update` on the decode /
    verify path: same fused gating, same ``[slot, HV, V, K]`` state layout, but
    the pool holds one checkpoint per request instead of one state per
    speculative token.  See the module docstring for the algebra.

    ``o`` may be passed to write the output in place (K3 does this).
    """
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    if scale is None:
        scale = K**-0.5
    assert flush_threshold_ok(cap, max_query_len), (
        f"replayssm cache_len={cap} must be >= 2*max_query_len=" f"{2 * max_query_len}"
    )

    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"
    # KDA tile width.  Measured on Kimi-K3 (tp=8, conc=64, per-launch from a
    # trace): BV=16 25.8 us, BV=32 23.3 us, BV=64 25.7 us against a 22.3 us
    # baseline, and BV=32 with num_warps=4 -- the width and warp count the
    # baseline `fused_sigmoid_gating` kernel itself uses -- is 36.2 us, so
    # matching the baseline's launch config is the worst of the options here.
    BV = min(triton.next_power_of_2(V), 32)
    NV = triton.cdiv(V, BV)
    # Replay tile height: the cursor can reach cap - max_query_len before a
    # flush resets it, and `tl.dot` wants at least 16 rows on CDNA, so a
    # short buffer pads rather than shrinks.
    BH = max(16, triton.next_power_of_2(cap - max_query_len))

    q, k, v, a, b = (x.contiguous() for x in (q, k, v, a, b))
    out = q.new_empty(1, T_tot, HV, V) if o is None else o.unsqueeze(0)

    _replayssm_kda_fwd_kernel[(NV, N * HV)](
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        LOWER_BOUND=0.0 if lower_bound is None else lower_bound,
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
        stride_a_token=a.stride(-3),
        stride_b_token=b.stride(-2),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        num_warps=1,
        num_stages=3,
    )
    return out


# --------------------------------------------------------------------------- #
# Fused GDN verify: conv1d + gating + ReplaySSM [+ gated RMSNorm]             #
# --------------------------------------------------------------------------- #
#
# The MTP verify step of a GDN layer otherwise runs causal_conv1d_update (spec
# mode), fused_gdn_gating, the serial ReplaySSM kernel above and a gated
# RMSNorm, with q/k/v, g and beta making a round trip through memory between
# them. This kernel takes the in_proj output rows directly:
#
#   * conv1d: the spec-mode sliding window of `causal_conv1d_update` -- the
#     window starts at `num_accepted - 1` in the slot's conv state, the T
#     verify tokens extend it, and the slot keeps the window shifted by one
#     (`[old[off+1], old[off+2], x_0 .. x_{T-1}]`). Same products, same bf16
#     rounding of the outputs, so q/k/v are the bytes the separate kernel wrote.
#   * gating: `fused_gdn_gating`'s g (fp32) and beta (rounded to bf16).
#   * the recurrence: the serial T > 1 path of `_replayssm_fwd_kernel`.
#   * FUSE_NORM (needs BV == V): norm(o) * silu(z) on the bf16-rounded output,
#     as `layer_norm_fwd(norm_before_gate=True, is_rms_norm=True)` computes it.
#
# The conv state is updated in place, and a key head's q/k channels are read by
# every program of its group (the HV / H value heads times the V slices). The
# window must not be overwritten until all of them have read it, so each
# program bumps a per-(sequence, key head) counter after its reads and the last
# to arrive writes the q/k window and resets the counter. The v channels of a
# slice belong to one program, which reads them before it writes them.


@triton.jit(do_not_specialize=["N", "T_MAX", "CAP"])
def _replayssm_conv_fwd_kernel(
    x,
    conv_w,
    conv_b,
    conv_state,
    conv_idx,
    num_accepted,
    a,
    b,
    A_log,
    dt_bias,
    o,
    z,
    norm_w,
    ckpt,
    buf_k,
    buf_u,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    conv_ctr,
    rec_ctr,
    scale,
    eps,
    N,
    T_MAX,
    CAP,
    stride_x_tok,
    stride_ab_tok,
    stride_z_tok,
    stride_z_hv,
    stride_conv_idx,
    stride_cs_slot: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_tok: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
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
    HAS_CONV_BIAS: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    DOT_MODE: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    NV: tl.constexpr = V // BV
    GROUP: tl.constexpr = (HV // H) * NV  # programs sharing this key head's q/k

    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if slot < 0:  # padded / idle batch entry
        return
    cslot = tl.load(conv_idx + i_n * stride_conv_idx).to(tl.int64)
    if cslot < 0:
        return

    # Same cursor handling as `_replayssm_fwd_kernel`.
    h = tl.load(write_pos + slot).to(tl.int32)
    h = tl.maximum(h, 0)
    do_flush = h + 2 * T_MAX > CAP
    base = tl.where(do_flush, 0, h).to(tl.int64)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # ---- conv window: the three history columns before this step's tokens ----
    ch_q = i_h * K + o_k
    ch_k = H * K + i_h * K + o_k
    ch_v = 2 * H * K + i_hv * V + o_v
    off = tl.load(num_accepted + i_n).to(tl.int64) - 1
    p_cs = conv_state + cslot * stride_cs_slot + off * stride_cs_tok
    p_csq = p_cs + ch_q * stride_cs_dim
    p_csk = p_cs + ch_k * stride_cs_dim
    p_csv = p_cs + ch_v * stride_cs_dim
    q0 = tl.load(p_csq, mask=mask_k, other=0.0)
    q1 = tl.load(p_csq + stride_cs_tok, mask=mask_k, other=0.0)
    q2 = tl.load(p_csq + 2 * stride_cs_tok, mask=mask_k, other=0.0)
    k0 = tl.load(p_csk, mask=mask_k, other=0.0)
    k1 = tl.load(p_csk + stride_cs_tok, mask=mask_k, other=0.0)
    k2 = tl.load(p_csk + 2 * stride_cs_tok, mask=mask_k, other=0.0)
    v0 = tl.load(p_csv, mask=mask_v, other=0.0)
    v1 = tl.load(p_csv + stride_cs_tok, mask=mask_v, other=0.0)
    v2 = tl.load(p_csv + 2 * stride_cs_tok, mask=mask_v, other=0.0)
    tl.debug_barrier()
    # Every lane has read the window; count this program in. The last of the
    # group owns the q/k window store below.
    arrived = tl.atomic_add(conv_ctr + i_n * H + i_h, 1, sem="acq_rel")
    owns_qk = arrived == GROUP - 1
    if owns_qk:
        tl.store(conv_ctr + i_n * H + i_h, 0)

    # The slot keeps the window shifted by one: rows 0, 1 are old[off+1],
    # old[off+2]; row 2 + t is token t (stored in the token loop).
    p_dst = conv_state + cslot * stride_cs_slot
    p_dstv = p_dst + ch_v * stride_cs_dim
    tl.store(p_dstv, v1, mask=mask_v)
    tl.store(p_dstv + stride_cs_tok, v2, mask=mask_v)
    p_dstq = p_dst + ch_q * stride_cs_dim
    p_dstk = p_dst + ch_k * stride_cs_dim
    if owns_qk:
        tl.store(p_dstq, q1, mask=mask_k)
        tl.store(p_dstq + stride_cs_tok, q2, mask=mask_k)
        tl.store(p_dstk, k1, mask=mask_k)
        tl.store(p_dstk + stride_cs_tok, k2, mask=mask_k)

    # conv weights (dim, 4) and bias, as the separate kernel loads them
    wq0 = tl.load(conv_w + ch_q * stride_w_dim, mask=mask_k, other=0.0)
    wq1 = tl.load(conv_w + ch_q * stride_w_dim + stride_w_width, mask=mask_k, other=0.0)
    wq2 = tl.load(conv_w + ch_q * stride_w_dim + 2 * stride_w_width, mask=mask_k, other=0.0)
    wq3 = tl.load(conv_w + ch_q * stride_w_dim + 3 * stride_w_width, mask=mask_k, other=0.0)
    wk0 = tl.load(conv_w + ch_k * stride_w_dim, mask=mask_k, other=0.0)
    wk1 = tl.load(conv_w + ch_k * stride_w_dim + stride_w_width, mask=mask_k, other=0.0)
    wk2 = tl.load(conv_w + ch_k * stride_w_dim + 2 * stride_w_width, mask=mask_k, other=0.0)
    wk3 = tl.load(conv_w + ch_k * stride_w_dim + 3 * stride_w_width, mask=mask_k, other=0.0)
    wv0 = tl.load(conv_w + ch_v * stride_w_dim, mask=mask_v, other=0.0)
    wv1 = tl.load(conv_w + ch_v * stride_w_dim + stride_w_width, mask=mask_v, other=0.0)
    wv2 = tl.load(conv_w + ch_v * stride_w_dim + 2 * stride_w_width, mask=mask_v, other=0.0)
    wv3 = tl.load(conv_w + ch_v * stride_w_dim + 3 * stride_w_width, mask=mask_v, other=0.0)
    if HAS_CONV_BIAS:
        bq = tl.load(conv_b + ch_q, mask=mask_k, other=0.0).to(tl.float32)
        bk = tl.load(conv_b + ch_k, mask=mask_k, other=0.0).to(tl.float32)
        bv = tl.load(conv_b + ch_v, mask=mask_v, other=0.0).to(tl.float32)
    else:
        bq = tl.zeros([BK], dtype=tl.float32)
        bk = tl.zeros([BK], dtype=tl.float32)
        bv = tl.zeros([BV], dtype=tl.float32)

    # per-head gating constants
    A_h = -tl.exp(tl.load(A_log + i_hv).to(tl.float32))
    dtb_h = tl.load(dt_bias + i_hv).to(tl.float32)

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
        False,
    )
    owns_rec = _record_owner(rec_ctr, i_nh, i_v, NV, do_flush)
    b_h = b_s0 * b_decay[:, None]
    b_h += _replay_dot(b_kw, b_ru, True, DOT_MODE)

    # ---- 2. flush: the checkpoint absorbs the committed prefix --------------
    if do_flush:
        tl.store(p_ckpt_hv, b_h.to(p_ckpt_hv.dtype.element_ty), mask=mask_h)

    if FUSE_NORM:
        b_nw = tl.load(norm_w + o_v, mask=mask_v, other=0.0).to(tl.float32)

    # ---- 3. this step's tokens ---------------------------------------------
    for i_t in range(T):
        tok = bos + i_t
        p_x = x + tok * stride_x_tok
        xq = tl.load(p_x + ch_q, mask=mask_k, other=0.0)
        xk = tl.load(p_x + ch_k, mask=mask_k, other=0.0)
        xv = tl.load(p_x + ch_v, mask=mask_v, other=0.0)

        # conv1d + SiLU, accumulated as causal_conv1d_update does
        acc = bq
        acc += q0 * wq0
        acc += q1 * wq1
        acc += q2 * wq2
        acc += xq * wq3
        b_q = (acc / (1 + tl.exp(-acc))).to(x.dtype.element_ty).to(tl.float32)
        acc = bk
        acc += k0 * wk0
        acc += k1 * wk1
        acc += k2 * wk2
        acc += xk * wk3
        b_k = (acc / (1 + tl.exp(-acc))).to(x.dtype.element_ty).to(tl.float32)
        acc = bv
        acc += v0 * wv0
        acc += v1 * wv1
        acc += v2 * wv2
        acc += xv * wv3
        b_v = (acc / (1 + tl.exp(-acc))).to(x.dtype.element_ty).to(tl.float32)
        q0, q1, q2 = q1, q2, xq
        k0, k1, k2 = k1, k2, xk
        v0, v1, v2 = v1, v2, xv
        tl.store(p_dstv + (2 + i_t) * stride_cs_tok, xv, mask=mask_v)
        if owns_qk:
            tl.store(p_dstq + (2 + i_t) * stride_cs_tok, xq, mask=mask_k)
            tl.store(p_dstk + (2 + i_t) * stride_cs_tok, xk, mask=mask_k)

        # gating, as fused_gdn_gating computes it (beta is stored bf16 there)
        a_t = tl.load(a + tok * stride_ab_tok + i_hv).to(tl.float32) + dtb_h
        sp = tl.where(a_t <= 20.0, tl.log(1 + tl.exp(a_t)), a_t)
        b_g = A_h * sp
        b_beta = (
            tl.sigmoid(tl.load(b + tok * stride_ab_tok + i_hv).to(tl.float32))
            .to(b.dtype.element_ty)
            .to(tl.float32)
        )

        # the serial ReplaySSM step
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale
        b_h *= exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)

        p_o = o + (tok * HV + i_hv) * V + o_v
        if FUSE_NORM:
            b_o = b_o.to(p_o.dtype.element_ty).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(b_o * b_o) / V + eps)
            b_z = tl.load(
                z + tok * stride_z_tok + i_hv * stride_z_hv + o_v, mask=mask_v, other=0.0
            ).to(tl.float32)
            b_o = b_o * rstd * b_nw * (b_z * tl.sigmoid(b_z))
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

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
        if owns_rec:
            p_bk = (
                buf_k
                + slot * stride_bufk_slot
                + pos * stride_bufk_pos
                + i_hv * stride_bufk_hv
                + o_k
            )
            tl.store(p_bk, b_k.to(p_bk.dtype.element_ty), mask=mask_k)
            p_bg2 = buf_g + slot * stride_bufg_slot + pos * stride_bufg_pos
            tl.store(p_bg2 + i_hv * stride_bufg_hv, b_g.to(p_bg2.dtype.element_ty))


@triton.jit
def _gated_rmsnorm_strided_kernel(
    o,
    z,
    w,
    eps,
    stride_z_tok,
    stride_z_hv,
    HV: tl.constexpr,
    V: tl.constexpr,
):
    """norm(o) * silu(z) in place on o [T, HV, V]; z read through its strides."""
    i_t, i_hv = tl.program_id(0), tl.program_id(1)
    o_v = tl.arange(0, V)
    p_o = o + (i_t * HV + i_hv) * V + o_v
    b_o = tl.load(p_o).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(b_o * b_o) / V + eps)
    b_z = tl.load(z + i_t * stride_z_tok + i_hv * stride_z_hv + o_v).to(tl.float32)
    b_w = tl.load(w + o_v).to(tl.float32)
    tl.store(p_o, (b_o * rstd * b_w * (b_z * tl.sigmoid(b_z))).to(p_o.dtype.element_ty))


def fused_verify_variant(N: int, HV: int) -> str | None:
    """Which fused verify kernel serves N sequences of HV value heads.

    gfx1250 sweep (bench_fused.py, K = V = 128, T = 4, us per layer against the
    separate kernels, fold steps weighted 1/3): A only wins while the grid is a
    few dozen programs (N*HV <= 32: 12.8 vs 16.2 at HV=32 N=1, 10.9 vs 15.1 at
    HV=8 N=1); past that its per-slice conv and gating rework loses to B, one
    program per head, which wins from N*HV = 64 on (14.4 vs 17.5 at HV=32 N=2,
    31.0 vs 45.1 at N=32, 399 vs 420 at N=512). Both beat the separate kernels
    at every size measured, so this never returns None on gfx1250.
    """
    if get_gfx() != "gfx1250":
        return None
    return "A" if N * HV <= 32 else "B"


def fused_verify_launch_config(
    N: int, HV: int, V: int, fuse_norm: bool
) -> tuple[int, int]:
    """(BV, num_warps) for `_replayssm_conv_fwd_kernel` (gfx1250 sweep,
    bench_fused.py): B takes 2 warps up to N*HV = 512 and 1 above (23.1 vs 25.5
    us at HV=32 N=16; 31.0 vs 42.9 at N=32); A takes BV=16 to N*HV = 16 and 32
    above (10.9 vs 11.9 us at HV=8 N=1; 12.8 vs 13.0 at HV=32 N=1)."""
    if fuse_norm:
        return V, (2 if N * HV <= 512 else 1)
    return (16 if N * HV <= 16 else 32), 1


def replayssm_conv_gated_delta_rule(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    conv_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    o: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    num_k_heads: int,
    head_k_dim: int,
    *,
    z: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fuse_norm: bool = False,
    block_v: int | None = None,
    num_warps: int | None = None,
) -> torch.Tensor:
    """The GDN verify step in one launch: spec-mode conv1d, gating, ReplaySSM
    and, with ``fuse_norm``, the gated RMSNorm.

    Args:
        mixed_qkv: ``[T_tot, 2*H*K + HV*V]`` in_proj rows (token-strided is fine)
        conv_weight: ``[conv_dim, 4]``; conv_state: ``[slot, conv_dim, state_len]``
            (any strides), updated in place as `causal_conv1d_update` does
        conv_state_indices, num_accepted_tokens: ``[N]`` per verify sequence
        a, b: ``[T_tot, HV]`` gating inputs (token-strided, same stride)
        o: ``[T_tot, HV, V]`` output; z: ``[T_tot, HV, V]`` gate (strided)
        the rest: as `replayssm_gated_delta_rule`.
    Returns ``o``: the core output, or ``norm(core) * silu(z)`` with ``fuse_norm``.
    """
    T_tot, conv_dim = mixed_qkv.shape
    HV, V = o.shape[1], o.shape[2]
    H, K = num_k_heads, head_k_dim
    assert conv_dim == 2 * H * K + HV * V and conv_weight.shape == (conv_dim, 4)
    assert mixed_qkv.stride(1) == 1 and a.stride(1) == 1 and b.stride(1) == 1
    assert a.stride(0) == b.stride(0)
    assert o.is_contiguous() and o.shape[0] >= T_tot
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    assert flush_threshold_ok(cap, max_query_len)
    BK = triton.next_power_of_2(K)
    assert BK == K
    if fuse_norm or z is not None:
        assert z is not None and norm_weight is not None
        assert z.stride(2) == 1
    if block_v is None or num_warps is None:
        cfg_bv, cfg_nw = fused_verify_launch_config(N, HV, V, fuse_norm)
        block_v = block_v or cfg_bv
        num_warps = num_warps or cfg_nw
    if fuse_norm:
        assert block_v == V, "the fused norm needs the whole V row in one program"
    BH = max(16, triton.next_power_of_2(cap - max_query_len))
    _replayssm_conv_fwd_kernel[(triton.cdiv(V, block_v), N * HV)](
        x=mixed_qkv,
        conv_w=conv_weight,
        conv_b=conv_bias if conv_bias is not None else conv_weight,
        conv_state=conv_state,
        conv_idx=conv_state_indices,
        num_accepted=num_accepted_tokens,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=o,
        z=z if z is not None else o,
        norm_w=norm_weight if norm_weight is not None else A_log,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        conv_ctr=_arrival_counters(o.device, "conv", N * H),
        rec_ctr=_arrival_counters(o.device, "rec", N * HV),
        scale=K**-0.5,
        eps=norm_eps,
        N=N,
        T_MAX=max_query_len,
        CAP=cap,
        stride_x_tok=mixed_qkv.stride(0),
        stride_ab_tok=a.stride(0),
        stride_z_tok=z.stride(0) if z is not None else 0,
        stride_z_hv=z.stride(1) if z is not None else 0,
        stride_conv_idx=conv_state_indices.stride(0),
        stride_cs_slot=conv_state.stride(0),
        stride_cs_dim=conv_state.stride(1),
        stride_cs_tok=conv_state.stride(2),
        stride_w_dim=conv_weight.stride(0),
        stride_w_width=conv_weight.stride(1),
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=block_v,
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
        HAS_CONV_BIAS=conv_bias is not None,
        FUSE_NORM=fuse_norm,
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        num_warps=num_warps,
        num_stages=3,
    )
    if z is not None and not fuse_norm:
        _gated_rmsnorm_strided_kernel[(T_tot, HV)](
            o, z, norm_weight, norm_eps, z.stride(0), z.stride(1), HV=HV, V=V,
            num_warps=1,
        )
    return o
