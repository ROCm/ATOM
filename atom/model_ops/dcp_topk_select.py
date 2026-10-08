# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Distributed top-k threshold selection for the DCP sparse-prefill indexer.

Each DCP rank scores only its own round-robin shard, so no rank can rank the
global candidates. What crosses the wire is therefore not the candidates but
the global K-th score: a token in the global top-K is in its own rank's local
top-K (fewer tokens outrank it locally than globally), so every rank already
holds every global winner it owns and only needs to know where to cut.

Two protocols agree on that cut, selected by ATOM_DCP_INDEXER_PREFILL_SELECT.

"histogram" (default) spends two collectives, neither scaling with topk or with
context length:

  * an all-gather of four per-row scalars, reduced to a bracket [lo, hi] that
    provably contains the global K-th score;
  * an all-reduce of a per-row linear histogram over that bracket.

The cut admits the whole threshold bin, so the emitted set is a SUPERSET of the
exact global top-K -- nothing the single-rank path would select is lost, but
some extra is gained.

"exact" instead all-gathers the candidate scores themselves and takes the true
global K-th, reproducing the dcp=1 selection exactly. The same superset argument
is what makes it correct: the union of the local top-K's contains the global
top-K, so ranking the union ranks the global set. Its payload is rows * topk
fp32 per rank per layer, which is why it is a reference rather than a default --
it is how the histogram's over-selection gets measured instead of estimated.

Both share the emit below, so the modes differ only in the value of one scalar
per row.

The decode twin is ``dcp_ops.dcp_decode_candidate_exchange_fused``, which
exchanges the candidates themselves. That does not transfer: decode has ~10^2
rows, prefill has up to ``max_num_batched_tokens``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from atom.distributed.dcp_utils import get_dcp_world_size
from atom.distributed.pcp_utils import pcp_is_enabled
from atom.utils import envs

NEG_INF = float("-inf")
POS_INF = float("inf")

# Column layout of the exchanged per-row bracket statistics.
_MAX, _KTH, _MIN, _CNT = 0, 1, 2, 3


def use_dcp_local_indexer_prefill(dcp_world_size: int | None = None) -> bool:
    """Does this forward score the indexer's LOCAL shard instead of the whole key set?

    PCP is excluded rather than refused. Under PCP the query side is round-robin
    sharded, so the per-token position ``dcp_indexer_local_ke`` is built from is
    not the chunk offset and the local causal window would be wrong -- but the
    gather path is fully intact and handles the combination exactly as before,
    so falling back costs nothing. Raising instead would take down a server that
    started fine and answered every request shorter than ``index_topk``, at the
    first long one, on every rank at once.

    ``dcp_world_size`` lets a caller that already knows it -- the attention
    metadata builder holds ``self.dcp_world_size`` -- skip the lookup, which
    needs a live ``AtomConfig`` and so cannot run in a plain unit test.
    """
    world = get_dcp_world_size() if dcp_world_size is None else dcp_world_size
    return world > 1 and envs.ATOM_DCP_INDEXER_PREFILL_LOCAL and not pcp_is_enabled()


SELECT_HISTOGRAM = "histogram"
SELECT_EXACT = "exact"
_SELECT_MODES = (SELECT_HISTOGRAM, SELECT_EXACT)


def dcp_prefill_select_mode() -> str:
    """Which cut-agreement protocol this forward runs. Validated, not defaulted.

    A typo in ``ATOM_DCP_INDEXER_PREFILL_SELECT`` must not silently resolve to the
    histogram: the whole reason the exact mode exists is to be trusted as the
    reference the histogram is scored against, and a reference that quietly
    becomes the thing under test measures a difference of zero.
    """
    mode = envs.ATOM_DCP_INDEXER_PREFILL_SELECT
    if mode not in _SELECT_MODES:
        raise ValueError(
            f"ATOM_DCP_INDEXER_PREFILL_SELECT={mode!r} is not one of {_SELECT_MODES}"
        )
    return mode


@triton.jit
def _row_bracket_stats_kernel(
    local_val,  # fp32 [rows, K]
    out,  # fp32 [rows, 4] -- max, kth, min_valid, valid_count
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lv_stride0: tl.int64,
    o_stride0: tl.int64,
):
    """All four bracket scalars in ONE pass over the row.

    The torch spelling was six passes over ``[rows, K]`` (isfinite, two
    ``where``, ``amax``, ``amin``, ``sum``) plus their temporaries -- 0.129 ms
    at rows=4096, K=2048, against a 0.092 ms scorer. Every reduction here is
    order-independent, so one fused pass is the same answer.
    """
    row = tl.program_id(0)
    vmax = float("-inf")
    vmin = float("inf")
    count = 0
    for tile in range(0, K, BLOCK_N):
        col = tile + tl.arange(0, BLOCK_N)
        col_valid = col < K
        val = tl.load(local_val + row * lv_stride0 + col, mask=col_valid, other=0.0)
        finite = (val == val) & (tl.abs(val) < float("inf"))  # noqa: PLR0124
        keep = col_valid & finite
        vmax = tl.maximum(vmax, tl.max(tl.where(keep, val, float("-inf")), axis=0))
        vmin = tl.minimum(vmin, tl.min(tl.where(keep, val, float("inf")), axis=0))
        count += tl.sum(keep.to(tl.int32), axis=0)
    # A full row's K entries ARE its top-K, so their minimum is the K-th
    # largest; a short row proves nothing about the global K-th, hence -inf.
    kth = tl.where(count == K, vmin, float("-inf"))
    tl.store(out + row * o_stride0 + 0, vmax)
    tl.store(out + row * o_stride0 + 1, kth)
    tl.store(out + row * o_stride0 + 2, vmin)
    tl.store(out + row * o_stride0 + 3, count.to(tl.float32))


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
    rows, k = local_val.shape
    out = local_val.new_empty((rows, 4))
    local_val_c = local_val.contiguous()
    _row_bracket_stats_kernel[(rows,)](
        local_val_c,
        out,
        k,
        min(1024, triton.next_power_of_2(k)),
        local_val_c.stride(0),
        out.stride(0),
    )
    return out


@triton.jit
def _reduce_bracket_kernel(
    gathered,  # fp32 [W, rows, 4]
    lo,  # fp32 [rows]
    hi,  # fp32 [rows]
    W,
    g_stride0: tl.int64,
    g_stride1: tl.int64,
    W_POW2: tl.constexpr,
):
    """``(lo, hi)`` for one row, in one launch.

    Three torch reductions over a 256 KB tensor cost 0.022 ms -- all launch
    overhead, and by then the second-largest step of the whole exchange. Max and
    min are exact and order-independent, so fusing them changes nothing about
    the bit-identical result every rank must agree on.
    """
    row = tl.program_id(0)
    # tl.arange needs a power-of-2 extent; the DCP world size need not be one
    # (and the unit tests deliberately use 3), so pad and mask.
    r = tl.arange(0, W_POW2)
    m = r < W
    base = gathered + r * g_stride0 + row * g_stride1
    row_hi = tl.max(tl.load(base + 0, mask=m, other=float("-inf")), axis=0)
    row_kth = tl.max(tl.load(base + 1, mask=m, other=float("-inf")), axis=0)
    row_min = tl.min(tl.load(base + 2, mask=m, other=float("inf")), axis=0)
    tl.store(hi + row, row_hi)
    tl.store(lo + row, tl.maximum(row_kth, row_min))


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
    world, rows, _ = gathered.shape
    lo = gathered.new_empty(rows)
    hi = gathered.new_empty(rows)
    g = gathered.contiguous()
    _reduce_bracket_kernel[(rows,)](
        g, lo, hi, world, g.stride(0), g.stride(1), triton.next_power_of_2(world)
    )
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


@triton.jit
def _local_histogram_kernel(
    local_val,  # fp32 [rows, K]
    lo,  # fp32 [rows]
    scale,  # fp32 [rows] -- nbins / (hi - lo), 0 where the bracket is degenerate
    hist,  # int32 [rows, nbins], zeroed by the caller
    K: tl.constexpr,
    NBINS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lv_stride0: tl.int64,
    h_stride0: tl.int64,
):
    """One program per row, accumulating that row's bins in registers.

    Two things this deliberately does NOT do:

    ``scatter_add_`` -- torch's scatter index must be int64, so the elementwise
    version materialized two int64 ``[rows, K]`` temporaries plus three fp32
    ones (~200 MiB for an 8 MiB histogram at rows=4096). This step runs OUTSIDE
    ``sparse_indexer_row_chunk``'s budget loop, which is the unbounded prefill
    allocation #1376 is about.

    ``tl.atomic_add`` per candidate -- that was the first version of this
    kernel, and it was the single most expensive kernel in the whole prefill:
    2048 candidates landing in 512 bins means most lanes of a wave collide on
    the same address, and it measured 0.59 ms for a 32 MiB read (57 GB/s, ~80x
    off HBM) against the 0.09 ms scorer this whole exchange exists to shrink.
    ``tl.histogram`` reduces within the tile first, so the only traffic is the
    read and one ``[NBINS]`` store per row.
    """
    row = tl.program_id(0)
    row_lo = tl.load(lo + row)
    row_scale = tl.load(scale + row)
    acc = tl.zeros([NBINS], dtype=tl.int32)
    for tile in range(0, K, BLOCK_N):
        col = tile + tl.arange(0, BLOCK_N)
        col_valid = col < K
        val = tl.load(local_val + row * lv_stride0 + col, mask=col_valid, other=0.0)
        # Valid == finite and at or above lo. `reduce_bracket` proved nothing
        # below lo can enter the global top-k, and the padding
        # top_k_per_row_prefill writes is -inf, so both fall out of one test.
        finite = (val == val) & (tl.abs(val) < float("inf"))  # noqa: PLR0124
        keep = col_valid & finite & (val >= row_lo)
        b = ((val - row_lo) * row_scale).to(tl.int32)
        b = tl.minimum(tl.maximum(b, 0), NBINS - 1)
        acc += tl.histogram(b, NBINS, mask=keep)
    tl.store(hist + row * h_stride0 + tl.arange(0, NBINS), acc)


def local_histogram(
    local_val: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor, nbins: int
) -> torch.Tensor:
    """``[rows, nbins]`` int32 counts of this rank's candidates over ``[lo, hi]``.

    Only finite candidates at or above ``lo`` are counted: ``reduce_bracket``
    proved nothing below ``lo`` can enter the global top-k, and the padding
    ``top_k_per_row_prefill`` writes is ``-inf``. Counts are int32 so the
    all-reduce that follows is an integer sum and therefore order-independent --
    every rank must scan a bit-identical histogram.

    The only allocation is the output: see the kernel's docstring for why that
    matters here and not in an ordinary elementwise op.
    """
    rows, k = local_val.shape
    scale = _bin_scale(lo, hi, nbins)
    hist = torch.zeros((rows, nbins), dtype=torch.int32, device=local_val.device)
    local_val_c = local_val.contiguous()
    _local_histogram_kernel[(rows,)](
        local_val_c,
        lo.contiguous(),
        scale.contiguous(),
        hist,
        k,
        nbins,
        min(1024, triton.next_power_of_2(k)),
        local_val_c.stride(0),
        hist.stride(0),
    )
    return hist


@triton.jit
def _threshold_kernel(
    hist,  # int32 [rows, NBINS] -- already all-reduced across the DCP group
    lo,  # fp32 [rows]
    hi,  # fp32 [rows]
    thr,  # fp32 [rows]
    TOPK: tl.constexpr,
    NBINS: tl.constexpr,
    h_stride0: tl.int64,
):
    """Per-row cut, in ONE pass over that row's bins.

    The torch spelling was a flip / cumsum / flip / compare / where / amax chain
    over ``[rows, NBINS]`` with a temporary per step -- 0.119 ms for an 8 MiB
    input (67 GB/s) at rows=4096. The whole row's bins fit in registers, so the
    reverse cumulative count is a single ``tl.cumsum`` on a reversed tile.
    """
    row = tl.program_id(0)
    b = tl.arange(0, NBINS)
    counts = tl.load(hist + row * h_stride0 + b)
    # Inclusive running count from the TOP bin down: reverse, scan, reverse.
    rev = tl.flip(counts, 0)
    from_top = tl.flip(tl.cumsum(rev, axis=0), 0)
    reached = from_top >= TOPK
    # `from_top` is non-increasing in the bin index, so the last True is the bin
    # the global k-th falls in. A row that never reaches TOPK takes bin 0, i.e.
    # `lo`, and therefore selects everything.
    b_star = tl.max(tl.where(reached, b, 0), axis=0)

    row_lo = tl.load(lo + row)
    row_hi = tl.load(hi + row)
    span = row_hi - row_lo
    ok = (span == span) & (tl.abs(span) < float("inf")) & (span > 0)  # noqa: PLR0124
    span = tl.where(ok, span, 0.0)
    cut = row_lo + b_star.to(tl.float32) * (span / NBINS)
    # An empty row has lo = +inf (no valid candidate on any rank); hand the emit
    # kernel a number rather than an inf so its `>=` compare is well defined.
    # Nothing is admitted either way -- every candidate in such a row is -inf.
    finite = (cut == cut) & (tl.abs(cut) < float("inf"))  # noqa: PLR0124
    tl.store(thr + row, tl.where(finite, cut, float("-inf")))


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
    rows, nbins = hist.shape
    thr = lo.new_empty(rows)
    hist_c = hist.contiguous()
    _threshold_kernel[(rows,)](
        hist_c,
        lo.contiguous(),
        hi.contiguous(),
        thr,
        topk,
        nbins,
        hist_c.stride(0),
    )
    return thr


@triton.jit
def _admit(val, idx, cut, col_valid, row_ok):
    """THE admit predicate. One definition, read by both emit passes.

    A candidate is admitted when it is finite, at or above the cut, carries a
    real column (``top_k_per_row_prefill`` tail-pads with ``-1``) and belongs to
    a real request (``batch_id_per_q_token < 0`` marks a CUDAGraph pad token).

    Shared rather than written twice on purpose: the counting pass builds
    ``out_kv_indptr`` and the writing pass fills the regions it describes, so any
    drift between them overflows a row into its neighbour -- silently. The
    earlier version of this code kept the two in lockstep by materializing the
    mask as an int8 ``[rows, K]`` tensor, which cost 224 MiB at the shipped
    default; a shared device function buys the same guarantee for nothing.
    """
    finite = (val == val) & (tl.abs(val) < float("inf"))  # noqa: PLR0124
    return col_valid & row_ok & finite & (idx >= 0) & (val >= cut)


@triton.jit
def _count_owned_slots_kernel(
    local_val,  # fp32 [rows, K]
    local_idx,  # int32 [rows, K]
    thr,  # fp32 [rows]
    batch_id_per_q_token,  # int32 [rows]
    out_counts,  # int32 [rows]
    out_metadata_counts,  # int32 [rows] -- max(count, 1), cumsummed by the caller
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lv_stride0: tl.int64,
    li_stride0: tl.int64,
):
    """Pass 1: how many candidates does this row admit?"""
    row = tl.program_id(0)
    cut = tl.load(thr + row)
    row_ok = tl.load(batch_id_per_q_token + row) >= 0
    count = 0
    for tile in range(0, K, BLOCK_N):
        col = tile + tl.arange(0, BLOCK_N)
        col_valid = col < K
        val = tl.load(local_val + row * lv_stride0 + col, mask=col_valid, other=0.0)
        idx = tl.load(local_idx + row * li_stride0 + col, mask=col_valid, other=-1)
        count += tl.sum(_admit(val, idx, cut, col_valid, row_ok).to(tl.int32))
    tl.store(out_counts + row, count)
    tl.store(out_metadata_counts + row, tl.maximum(count, 1))


@triton.jit
def _emit_owned_slots_kernel(
    local_val,  # fp32 [rows, K]
    local_idx,  # int32 [rows, K] absolute column in the LOCAL flat plane
    thr,  # fp32 [rows]
    local_ks,  # int32 [rows] this row's local-plane region start
    batch_id_per_q_token,  # int32 [rows]
    block_table,  # int32 [num_req, cols]
    out_kv_indptr,  # int32 [rows + 1] prefix offsets, already cumsummed
    out_kv_indices,  # int32 [>= out_kv_indptr[-1]]
    BLOCK_SIZE: tl.constexpr,  # runner (physical) block size
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    lv_stride0: tl.int64,
    li_stride0: tl.int64,
    bt_stride0: tl.int64,
):
    """Pass 2: pack this row's admitted slots to the front of its region.

    Mirrors ``dcp_ops._compact_filter_dcp_prefill_kernel``'s two invariants --
    no ``-1`` holes (they break aiter's lse path) and an order-preserving
    compaction so the fp accumulation order is deterministic -- but with the
    simple local slot formula (see ``emit_owned_slots``' docstring).
    """
    row = tl.program_id(0)
    req_id = tl.load(batch_id_per_q_token + row)
    row_ok = req_id >= 0
    req_id = tl.maximum(req_id, 0)  # a pad row admits nothing; keep the load legal
    cut = tl.load(thr + row)
    base = tl.load(local_ks + row)
    out_start = tl.load(out_kv_indptr + row)

    written = 0
    for tile in range(0, K, BLOCK_N):
        col = tile + tl.arange(0, BLOCK_N)
        col_valid = col < K
        val = tl.load(local_val + row * lv_stride0 + col, mask=col_valid, other=0.0)
        idx = tl.load(local_idx + row * li_stride0 + col, mask=col_valid, other=-1)
        keep = _admit(val, idx, cut, col_valid, row_ok)
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

    Two Triton passes over the same inputs, sharing ``_admit``; nothing of size
    ``[rows, K]`` is allocated. This step runs OUTSIDE the
    ``sparse_indexer_row_chunk`` budget loop, so a temporary here is unbounded
    prefill memory of exactly the kind #1376 is about.
    """
    rows, k = local_val.shape
    assert local_idx.shape == (rows, k), local_idx.shape
    assert out_kv_indptr.shape[0] >= rows + 1, out_kv_indptr.shape
    assert owned_counts.shape[0] >= rows, owned_counts.shape

    local_val_c = local_val.contiguous()
    local_idx_c = local_idx.contiguous()
    thr_c = thr.contiguous()
    batch_ids_c = batch_id_per_q_token.contiguous()
    block_table_c = block_table.to(torch.int32).contiguous()

    # The count kernel writes max(count, 1) straight into the indptr tail, so
    # the cumsum needs no allocation of its own (exact in/out overlap is
    # supported). Same shape as dcp_ops' prefill filter.
    metadata_counts = out_kv_indptr[1 : rows + 1]
    _count_owned_slots_kernel[(rows,)](
        local_val_c,
        local_idx_c,
        thr_c,
        batch_ids_c,
        owned_counts[:rows],
        metadata_counts,
        k,
        BLOCK_N,
        local_val_c.stride(0),
        local_idx_c.stride(0),
    )
    out_kv_indptr[:1].zero_()
    torch.cumsum(metadata_counts, dim=0, dtype=torch.int32, out=metadata_counts)

    _emit_owned_slots_kernel[(rows,)](
        local_val_c,
        local_idx_c,
        thr_c,
        local_ks.contiguous(),
        batch_ids_c,
        block_table_c,
        out_kv_indptr,
        out_kv_indices,
        block_size,
        k,
        BLOCK_N,
        local_val_c.stride(0),
        local_idx_c.stride(0),
        block_table_c.stride(0),
    )


def _all_gather_stats(cp_group, stats: torch.Tensor) -> torch.Tensor:
    """Gather every rank's ``[rows, 4]`` bracket statistics, concatenated on dim 0.

    ``GroupCoordinator.all_gather`` is fine here: a gather is a pure copy, and
    these are fp32 anyway.
    """
    return cp_group.all_gather(stats.contiguous(), dim=0)


def _all_reduce_counts(cp_group, counts: torch.Tensor) -> torch.Tensor:
    """Integer SUM of the per-row histograms, in place, on plain RCCL.

    Deliberately NOT ``GroupCoordinator.all_reduce``. That dispatches through
    quick-reduce, then custom all-reduce, then symmetric memory, and only then
    falls back to NCCL -- and ``CustomAllreduce.should_custom_ar`` gates on size
    and contiguity but NOT on dtype, so a correctly sized int32 tensor is routed
    into a reduction kernel whose C++ dispatch enum has no integer entry (see
    the ``_INT_TO_FP_VIEW`` note in ``custom_all_reduce.py``: that int-as-float
    view exists only for all-GATHER, which is a memcpy, never for the reduce).
    The failure would be wrong counts, not an error.

    Exactness is load-bearing twice over: the counts decide the cut, and every
    rank must scan bit-identical counts or their owned sets stop being a
    partition of one global selection. Integer SUM over NCCL is both.

    Matches ``dcp_ops``' own precedent of reaching for ``cp_group.device_group``
    when it wants a raw collective.
    """
    assert counts.dtype == torch.int32, counts.dtype
    torch.distributed.all_reduce(counts, group=cp_group.device_group)
    return counts


def _all_gather_candidates(cp_group, local_val: torch.Tensor) -> torch.Tensor:
    """Gather every rank's ``[rows, K]`` candidate SCORES, concatenated on dim 0.

    Scores only -- the indices stay home. A rank never needs to name another
    rank's winners, because it never emits them: the global cut is a scalar per
    row, and once a rank knows it, it filters its own candidates with it. Sending
    the indices too would double a payload that is already the heaviest thing
    this file moves, to carry numbers nobody reads.
    """
    return cp_group.all_gather(local_val.contiguous(), dim=0)


def exact_threshold_from_candidates(
    cp_group, local_val: torch.Tensor, topk: int, row_tile: int = 512
) -> torch.Tensor:
    """Per-row cut value: the TRUE global k-th score, with no bin quantization.

    The reference implementation of the same contract
    ``threshold_from_histogram`` approximates. It is exact because the union of
    the per-rank local top-K's provably contains the global top-K (see the module
    docstring), so a top-K over the gathered ``world * K`` scores IS the global
    top-K and its minimum IS the global K-th. Emitting ``>= cut`` then reproduces
    the ``dcp=1`` selection exactly, ties included.

    The price is the payload: ``rows * K`` fp32 per rank per full-index layer,
    which the histogram exists to avoid. Keep this for A/B and accuracy work.

    Non-finite scores are mapped to ``-inf`` rather than passed to ``topk``, to
    match ``_admit``, which refuses them. Letting a ``+inf`` through would make it
    outrank real candidates here and then be dropped at emit, costing the row a
    selection -- under-selection, the one failure mode the superset argument is
    supposed to rule out. Rows holding fewer than ``topk`` finite candidates get a
    ``-inf`` cut and select all of them, exactly as the histogram path does.

    Tiled over rows because the ``[tile, world * K]`` matrix ``topk`` wants is
    materialized: at ``world=8``, ``K=2048`` the untiled form is 1 GiB for a
    4096-row chunk. The gathered buffer itself is not tiled -- one collective per
    layer is the point.
    """
    rows, k = local_val.shape
    world = cp_group.world_size
    gathered = _all_gather_candidates(cp_group, local_val).reshape(world, rows, k)
    thr = local_val.new_empty(rows)
    for start in range(0, rows, row_tile):
        stop = min(start + row_tile, rows)
        tile = (
            gathered[:, start:stop, :].permute(1, 0, 2).reshape(stop - start, world * k)
        )
        tile = tile.masked_fill(~torch.isfinite(tile), NEG_INF)
        kth = torch.topk(tile, topk, dim=1, sorted=False).values.amin(dim=1)
        thr[start:stop] = kth
    return thr


def dcp_prefill_candidate_exchange(
    local_val: torch.Tensor,
    local_idx: torch.Tensor,
    local_ks: torch.Tensor,
    batch_id_per_q_token: torch.Tensor,
    block_table: torch.Tensor,
    cp_group,
    topk_tokens: int,
    block_size: int,
    nbins: int,
    out_kv_indices: torch.Tensor,
    out_kv_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
    select: str = SELECT_HISTOGRAM,
) -> None:
    """Agree on the global top-k cut, then emit this rank's owned slots.

    Only the cut-agreement step differs between the two ``select`` modes; the
    emit below is shared, so whatever a mode proves about the cut it also proves
    about the selection.

    ``"histogram"`` spends two collectives whose payloads scale with the prefill
    ROW count rather than with the context length: ``rows * 4`` fp32 gathered,
    then ``rows * nbins`` int32 reduced. The index-cache all-gather this replaces
    scaled with ``total_kv`` instead, and forced every rank to re-score the
    whole sequence. ``"exact"`` spends one collective of ``rows * topk`` fp32 per
    rank and returns the true global k-th -- a reference, not a shipping option.
    ``nbins`` is unread in that mode.

    Writes ``out_kv_indices`` / ``out_kv_indptr`` / ``owned_counts`` in place and
    returns nothing: like the decode twin, the ownership filter, the slot
    localize and the compaction all happen inside, so there is no global top-k
    left for the caller to convert.
    """
    rows, k = local_val.shape
    world = cp_group.world_size
    # The superset guarantee is conditional on this. `reduce_bracket` pins `lo`
    # at max_r(rank r's K-th) whenever some rank came back full; with K < topk a
    # rank can be full while the GLOBAL candidate count is still under topk, and
    # every candidate below that rank's K-th is then discarded from a row that
    # should have selected all of them. Nothing downstream can detect it -- only
    # the caller's shapes can, so check them here.
    assert k == topk_tokens, (
        f"local_val is {k} wide for topk_tokens={topk_tokens}; the local top-k "
        "must be taken at the full width or the cut can discard candidates from "
        "a row that holds fewer than topk globally"
    )

    if select == SELECT_EXACT:
        thr = exact_threshold_from_candidates(cp_group, local_val, topk_tokens)
    elif select == SELECT_HISTOGRAM:
        stats = row_bracket_stats(local_val)
        gathered = _all_gather_stats(cp_group, stats).reshape(world, rows, 4)
        lo, hi = reduce_bracket(gathered)

        hist = _all_reduce_counts(cp_group, local_histogram(local_val, lo, hi, nbins))
        thr = threshold_from_histogram(hist, lo, hi, topk_tokens)
    else:
        raise ValueError(f"unknown select mode {select!r}, want one of {_SELECT_MODES}")

    emit_owned_slots(
        local_val,
        local_idx,
        thr,
        local_ks,
        batch_id_per_q_token,
        block_table,
        block_size,
        out_kv_indices,
        out_kv_indptr,
        owned_counts,
    )
