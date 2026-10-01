# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Distributed top-k threshold selection for the DCP sparse-prefill indexer.

Each DCP rank scores only its own round-robin shard, so no rank can rank the
global candidates. What crosses the wire is therefore not the candidates but
the global K-th score: a token in the global top-K is in its own rank's local
top-K (fewer tokens outrank it locally than globally), so every rank already
holds every global winner it owns and only needs to know where to cut.

Two collectives agree on that cut:

  * an all-gather of four per-row scalars, reduced to a bracket [lo, hi] that
    provably contains the global K-th score;
  * an all-reduce of a per-row linear histogram over that bracket.

The cut admits the whole threshold bin, so the emitted set is a SUPERSET of the
exact global top-K -- nothing the single-rank path would select is lost.

The decode twin is ``dcp_ops.dcp_decode_candidate_exchange_fused``, which
exchanges the candidates themselves. That does not transfer: decode has ~10^2
rows, prefill has up to ``max_num_batched_tokens``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

NEG_INF = float("-inf")
POS_INF = float("inf")

# Column layout of the exchanged per-row bracket statistics.
_MAX, _KTH, _MIN, _CNT = 0, 1, 2, 3


def row_bracket_stats(local_val: torch.Tensor) -> torch.Tensor:
    """Per-row bracket scalars for one rank's local top-k values.

    ``local_val`` is ``[rows, K]`` fp32 as ``top_k_per_row_prefill`` wrote it:
    the K winners ordered by COLUMN INDEX rather than by score, with a row
    holding fewer than K candidates tail-padded ``-inf``. "Valid" means finite.
    Every reduction below is order-independent, so nothing here depends on that
    ordering.

    Returns ``[rows, 4]`` fp32:
      0 ``local_max``        largest valid value, ``-inf`` if the row is empty
      1 ``local_kth``        the K-th largest IF the row returned K valid
                             entries, else ``-inf``
      2 ``local_min_valid``  smallest valid value, ``+inf`` if the row is empty
      3 ``local_valid_count`` number of valid entries (exact in fp32 below 2^24)

    ``local_kth`` falls out of ``local_min_valid``: when the kernel returns a
    full K entries those entries ARE the row's top-K, so their minimum is the
    K-th largest. A short row proves nothing about where the global K-th sits,
    which is what ``-inf`` encodes -- ``reduce_bracket``'s ``max`` then ignores
    that rank.
    """
    assert local_val.dtype == torch.float32, local_val.dtype
    assert local_val.dim() == 2, local_val.shape
    valid = torch.isfinite(local_val)
    rows, k = local_val.shape
    out = local_val.new_empty((rows, 4))
    neg_inf = local_val.new_full((), NEG_INF)
    out[:, _MAX] = torch.where(valid, local_val, neg_inf).amax(1)
    min_valid = torch.where(valid, local_val, local_val.new_full((), POS_INF)).amin(1)
    count = valid.sum(1, dtype=torch.float32)
    out[:, _MIN] = min_valid
    out[:, _CNT] = count
    out[:, _KTH] = torch.where(count == k, min_valid, neg_inf)
    return out


def reduce_bracket(gathered: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce the all-gathered ``[W, rows, 4]`` statistics to ``(lo, hi)``.

    ``hi = max_r local_max`` is the global largest candidate.

    ``lo = max( max_r local_kth , min_r local_min_valid )``. The first term is
    the tightest valid lower bound: the rank achieving it holds K candidates at
    or above that value, so the global K-th is at least that, and nothing below
    it can enter the global top-K. The second term only takes over when NO rank
    has K local candidates -- every query token in the first K positions of a
    prefill -- where the correct answer is "select everything" and widening the
    bracket to the global minimum delivers exactly that.

    Deliberately a reduce over a gathered tensor rather than a reduction
    collective: every rank must land on bit-identical ``lo``/``hi`` or the
    per-rank owned sets stop being a partition of one global selection, and a
    fixed-order torch reduce over the same bytes guarantees that.
    """
    assert gathered.dim() == 3 and gathered.shape[-1] == 4, gathered.shape
    hi = gathered[:, :, _MAX].amax(0)
    lo = torch.maximum(gathered[:, :, _KTH].amax(0), gathered[:, :, _MIN].amin(0))
    return lo, hi


def _bin_scale(lo: torch.Tensor, hi: torch.Tensor, nbins: int) -> torch.Tensor:
    """``nbins / (hi - lo)``, with the degenerate and non-finite cases pinned.

    ``hi == lo`` (every candidate scored the same) and ``hi - lo`` non-finite
    (an empty row, where ``lo`` is ``+inf``) would divide by zero or produce a
    NaN. A scale of 0 maps every candidate into bin 0 instead, which is the
    right answer for both: the whole row is admitted together, or there is
    nothing to admit.
    """
    span = hi - lo
    ok = torch.isfinite(span) & (span > 0)
    return torch.where(
        ok,
        nbins / torch.where(ok, span, torch.ones_like(span)),
        torch.zeros_like(span),
    )


def local_histogram(
    local_val: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, nbins: int
) -> torch.Tensor:
    """``[rows, nbins]`` int32 counts of this rank's candidates over ``[lo, hi]``.

    Only finite candidates at or above ``lo`` are counted: ``reduce_bracket``
    proved nothing below ``lo`` can enter the global top-k, and the padding
    ``top_k_per_row_prefill`` writes is ``-inf``. Counts are int32 so the
    all-reduce that follows is an integer sum and therefore order-independent --
    every rank must scan a bit-identical histogram.

    Out-of-range rows are routed to a scratch column ``nbins`` that is dropped
    on return, so no masked ``scatter_add_`` and no second kernel are needed.
    """
    rows, _ = local_val.shape
    scale = _bin_scale(lo, hi, nbins)[:, None]
    idx = ((local_val - lo[:, None]) * scale).floor()
    keep = torch.isfinite(local_val) & (local_val >= lo[:, None])
    idx = torch.where(keep, idx, torch.zeros_like(idx))
    idx = idx.clamp(0, nbins - 1).to(torch.int64)
    idx = torch.where(keep, idx, torch.full_like(idx, nbins))
    hist = torch.zeros((rows, nbins + 1), dtype=torch.int32, device=local_val.device)
    hist.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.int32))
    return hist[:, :nbins].contiguous()


def threshold_from_histogram(
    hist: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, topk: int
) -> torch.Tensor:
    """Per-row cut value: the low edge of the bin the global k-th falls in.

    Scans bins high to low and stops at the first bin whose running count
    reaches ``topk``; the whole of that bin is admitted, which is what makes the
    result a SUPERSET of the exact top-k rather than an approximation of it. A
    row whose candidates never reach ``topk`` falls to bin 0, i.e. ``lo``, so it
    selects everything -- the common case for query tokens in the first ``topk``
    positions of a prefill.
    """
    nbins = hist.shape[1]
    # Inclusive running count from the top bin down.
    from_top = torch.cumsum(hist.flip(1).to(torch.int64), dim=1).flip(1)
    reached = from_top >= topk
    bins = torch.arange(nbins, device=hist.device, dtype=torch.int32)
    # The highest bin index whose inclusive-from-top count already reaches topk.
    # `reached` is monotone non-increasing in the bin index, so its last True is
    # that bin; rows that never reach topk take bin 0 and therefore `lo`.
    b_star = torch.where(reached, bins, torch.zeros_like(bins)).amax(1)
    span = hi - lo
    span = torch.where(torch.isfinite(span) & (span > 0), span, torch.zeros_like(span))
    thr = lo + b_star.to(lo.dtype) * (span / nbins)
    # An empty row has lo = +inf (no valid candidate on any rank); hand the emit
    # kernel a number rather than an inf so its `>=` compare is well defined.
    # Nothing is admitted either way -- every candidate in such a row is -inf.
    return torch.where(torch.isfinite(thr), thr, torch.full_like(thr, NEG_INF))


@triton.jit
def _emit_owned_slots_kernel(
    keep_mask,  # int8 [rows, K] 1 where the candidate is admitted
    local_idx,  # int32 [rows, K] absolute column in the LOCAL flat plane
    local_ks,  # int32 [rows] this row's local-plane region start
    batch_id_per_q_token,  # int32 [rows]
    block_table,  # int32 [num_req, cols]
    out_kv_indptr,  # int32 [rows + 1] prefix offsets, already cumsummed
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    BLOCK_SIZE: tl.constexpr,  # runner (physical) block size
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    km_stride0: tl.int64,
    li_stride0: tl.int64,
    bt_stride0: tl.int64,
):
    """Pack this row's admitted slots to the front of its region.

    Mirrors ``dcp_ops._compact_filter_dcp_prefill_kernel``'s two invariants --
    no ``-1`` holes (they break aiter's lse path) and an order-preserving
    compaction so the fp accumulation order is deterministic -- but with the
    simple local slot formula (see ``emit_owned_slots``' docstring).

    ``keep_mask`` arrives precomputed rather than being re-derived here. The
    counts that built ``out_kv_indptr`` came from the same tensor, so the two
    passes cannot drift: a row whose count and whose writes disagree would
    overflow into the next row's region, which is silent corruption.
    """
    row = tl.program_id(0)
    req_id = tl.load(batch_id_per_q_token + row)
    req_id = tl.maximum(req_id, 0)  # pad rows are fully masked by keep_mask
    base = tl.load(local_ks + row)
    out_start = tl.load(out_kv_indptr + row)

    written = 0
    for tile in range(0, K, BLOCK_N):
        col = tile + tl.arange(0, BLOCK_N)
        col_valid = col < K
        keep = tl.load(keep_mask + row * km_stride0 + col, mask=col_valid, other=0) != 0
        idx = tl.load(local_idx + row * li_stride0 + col, mask=keep, other=0)
        jl = tl.maximum(idx - base, 0)
        phys = tl.load(
            block_table + req_id * bt_stride0 + (jl // BLOCK_SIZE), mask=keep, other=0
        )
        slot = phys * BLOCK_SIZE + (jl % BLOCK_SIZE)

        keep_i32 = keep.to(tl.int32)
        dst = written + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(out_kv_indices + out_start + dst, slot, mask=keep)
        written += tl.sum(keep_i32)

    # Persistent MLA's fast metadata builder cannot handle a zero-length row:
    # reserve one slot pointing at a valid dummy cache entry. The attention
    # caller uses `owned_counts == 0` to replace the row with the softmax
    # identity (O=0, LSE=-inf), so the dummy never reaches the DCP merge.
    tl.store(out_kv_indices + out_start, 0, mask=written == 0)


def emit_owned_slots(
    local_val: torch.Tensor,
    local_idx: torch.Tensor,
    thr: torch.Tensor,
    local_ks: torch.Tensor,
    batch_id_per_q_token: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    out_kv_indices: torch.Tensor,
    out_kv_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
    BLOCK_N: int = 128,
) -> None:
    """Write this rank's admitted KV slots, compacted, plus indptr and counts.

    Local index ``jl = local_idx - local_ks`` maps to the physical slot
    ``block_table[req, jl // block_size] * block_size + jl % block_size``: the
    local shard was gathered with ``dcp_indexer_local_cu_seqlens``, whose slot
    formula is exactly the round-robin WRITE layout, and a rank only ever holds
    its own tokens -- so the interleave-S virtual-block arithmetic the global
    filter needs does not appear here.

    A candidate is admitted when it is finite, at or above ``thr``, carries a
    real column (``top_k_per_row_prefill`` tail-pads with ``-1``), and belongs
    to a real request (``batch_id_per_q_token < 0`` marks a CUDAGraph pad
    token). That predicate is evaluated ONCE, here, and handed to the kernel --
    counts and writes read the same tensor by construction.
    """
    rows, k = local_val.shape
    assert local_idx.shape == (rows, k), local_idx.shape
    assert out_kv_indptr.shape[0] >= rows + 1, out_kv_indptr.shape
    assert owned_counts.shape[0] >= rows, owned_counts.shape

    keep = (
        torch.isfinite(local_val)
        & (local_val >= thr[:, None])
        & (local_idx >= 0)
        & (batch_id_per_q_token >= 0)[:, None]
    )
    counts = keep.sum(1, dtype=torch.int32)
    owned_counts[:rows].copy_(counts)
    out_kv_indptr[:1].zero_()
    torch.cumsum(
        counts.clamp_min(1), dim=0, dtype=torch.int32, out=out_kv_indptr[1 : rows + 1]
    )

    keep_i8 = keep.to(torch.int8).contiguous()
    local_idx_c = local_idx.contiguous()
    block_table_c = block_table.to(torch.int32).contiguous()
    _emit_owned_slots_kernel[(rows,)](
        keep_i8,
        local_idx_c,
        local_ks.contiguous(),
        batch_id_per_q_token.contiguous(),
        block_table_c,
        out_kv_indptr,
        out_kv_indices,
        block_size,
        k,
        BLOCK_N,
        keep_i8.stride(0),
        local_idx_c.stride(0),
        block_table_c.stride(0),
    )
