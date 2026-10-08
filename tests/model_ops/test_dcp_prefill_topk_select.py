# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-process tests for the DCP sparse-prefill top-k threshold selection.

The prefill exchange replaces "every rank reconstructs the global top-k" with
"every rank scores its own shard and the ranks agree on a threshold". That is a
composition of four steps, and its failure mode is silent: wrong tokens, no
error. So each step is pinned here against a pure-torch reference, and the
composition is pinned against the exact global top-k it is supposed to be a
superset of.

WHY THIS NEEDS NO W GPUs. The two collectives are data movement, not math: the
all-gather delivers [W, rows, 4] and the all-reduce delivers an elementwise sum
of [rows, NBINS]. Running the W shards sequentially on one device and stacking
them produces exactly the tensors the collectives would, so everything
downstream of the transport is the shipped code under test.
"""

import pytest
import torch

pytest.importorskip("triton", reason="requires Triton")

from atom.model_ops.dcp_topk_select import (
    dcp_prefill_candidate_exchange,
    emit_owned_slots,
    local_histogram,
    reduce_bracket,
    row_bracket_stats,
    threshold_from_histogram,
)

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

NEG_INF = float("-inf")
POS_INF = float("inf")


def _padded_rows(rows, k, counts, seed=0):
    """[rows, k] fp32, row i holding counts[i] finite entries then -inf padding.

    Deliberately UNSORTED: top_k_per_row_prefill returns its winners ordered by
    column index, not by score, so anything here that depends on descending
    order is a bug this helper must be able to catch.
    """
    torch.manual_seed(seed)
    out = torch.full((rows, k), NEG_INF, dtype=torch.float32, device="cuda")
    for i, n in enumerate(counts):
        out[i, : int(n)] = torch.randn(int(n), dtype=torch.float32, device="cuda")
    return out


@needs_gpu
def test_bracket_stats_full_row():
    val = _padded_rows(1, 8, [8], seed=1)
    stats = row_bracket_stats(val)
    assert stats[0, 0].item() == pytest.approx(val[0].max().item())
    # All k entries valid -> they ARE the top-k, so their min is the k-th.
    assert stats[0, 1].item() == pytest.approx(val[0].min().item())
    assert stats[0, 2].item() == pytest.approx(val[0].min().item())
    assert stats[0, 3].item() == 8.0


@needs_gpu
def test_bracket_stats_short_row_has_no_kth():
    val = _padded_rows(1, 8, [3], seed=2)
    stats = row_bracket_stats(val)
    assert stats[0, 0].item() == pytest.approx(val[0, :3].max().item())
    assert stats[0, 1].item() == NEG_INF  # fewer than k valid -> no k-th
    assert stats[0, 2].item() == pytest.approx(val[0, :3].min().item())
    assert stats[0, 3].item() == 3.0


@needs_gpu
def test_bracket_stats_ignore_input_order():
    """The kernel orders its output by column index; nothing may depend on that."""
    val = _padded_rows(1, 8, [8], seed=1)
    shuffled = val[:, torch.randperm(8, device="cuda")]
    assert torch.equal(row_bracket_stats(val), row_bracket_stats(shuffled))


@needs_gpu
def test_bracket_stats_empty_row():
    val = torch.full((1, 8), NEG_INF, dtype=torch.float32, device="cuda")
    stats = row_bracket_stats(val)
    assert stats[0, 0].item() == NEG_INF
    assert stats[0, 1].item() == NEG_INF
    assert stats[0, 2].item() == POS_INF
    assert stats[0, 3].item() == 0.0


@needs_gpu
def test_reduce_bracket_takes_kth_when_some_rank_is_full():
    # rank 0 full (its k-th is a real bound), rank 1 short.
    r0 = row_bracket_stats(_padded_rows(1, 8, [8], seed=3))
    r1 = row_bracket_stats(_padded_rows(1, 8, [2], seed=4))
    lo, hi = reduce_bracket(torch.stack([r0, r1], dim=0))
    assert hi.item() == pytest.approx(max(r0[0, 0].item(), r1[0, 0].item()))
    assert lo.item() == pytest.approx(r0[0, 1].item())


@needs_gpu
def test_reduce_bracket_falls_back_to_global_min_when_no_rank_is_full():
    r0 = row_bracket_stats(_padded_rows(1, 8, [3], seed=5))
    r1 = row_bracket_stats(_padded_rows(1, 8, [2], seed=6))
    lo, hi = reduce_bracket(torch.stack([r0, r1], dim=0))
    assert lo.item() == pytest.approx(min(r0[0, 2].item(), r1[0, 2].item()))
    assert lo.item() > NEG_INF
    assert hi.item() == pytest.approx(max(r0[0, 0].item(), r1[0, 0].item()))


@needs_gpu
def test_reduce_bracket_is_rank_order_independent():
    """Review Focus #5: every rank must derive a bit-identical threshold."""
    stats = [
        row_bracket_stats(_padded_rows(4, 8, [8, 5, 0, 8], seed=s)) for s in range(3)
    ]
    lo_a, hi_a = reduce_bracket(torch.stack(stats, dim=0))
    lo_b, hi_b = reduce_bracket(torch.stack(stats[::-1], dim=0))
    assert torch.equal(lo_a, lo_b)
    assert torch.equal(hi_a, hi_b)


def _shards(rows, k, counts_per_rank, seed=0):
    """List of W [rows, k] padded value planes, one per simulated DCP rank."""
    return [
        _padded_rows(rows, k, counts, seed=seed + 100 * r)
        for r, counts in enumerate(counts_per_rank)
    ]


def _exact_global_kth(shards, topk):
    """Reference: the exact global topk-th value per row, or -inf if fewer."""
    allv = torch.cat(shards, dim=1)
    rows = allv.shape[0]
    out = allv.new_full((rows,), NEG_INF)
    for i in range(rows):
        finite = allv[i][torch.isfinite(allv[i])]
        if finite.numel() >= topk:
            out[i] = torch.sort(finite, descending=True).values[topk - 1]
    return out


@needs_gpu
def test_threshold_brackets_the_exact_kth():
    topk, k, nbins = 16, 32, 64
    shards = _shards(4, k, [[32, 32, 20, 32], [32, 32, 32, 32]], seed=7)
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    exact = _exact_global_kth(shards, topk)
    # Over-selection: the cut never sits above the exact k-th, so the admitted
    # set contains the whole exact top-k.
    assert torch.all(thr <= exact + 1e-6)


@needs_gpu
def test_threshold_over_selects_by_at_most_one_bin():
    topk, k, nbins = 16, 32, 64
    shards = _shards(4, k, [[32] * 4, [32] * 4], seed=8)
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    allv = torch.cat(shards, dim=1)
    admitted = (allv >= thr[:, None]).sum(1)
    assert torch.all(admitted >= topk)
    # One bin's worth of slack, plus the bin that straddles the cut.
    per_bin = allv.shape[1] / nbins
    assert torch.all(admitted <= topk + 2 * per_bin + 1)


@needs_gpu
def test_row_with_fewer_than_topk_candidates_selects_everything():
    """Review Focus #1.

    K == topk deliberately: that is the shipped wiring, and it is what makes the
    guarantee hold. With K < topk a rank CAN be full while the global candidate
    count is still under topk, which pins `lo` at that rank's K-th and discards
    everything below it -- see test_sub_topk_row_needs_k_equal_to_topk.
    """
    topk, k, nbins = 16, 16, 64
    shards = _shards(1, k, [[3], [4]], seed=9)
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    allv = torch.cat(shards, dim=1)
    admitted = (allv >= thr[:, None]).sum(1)
    assert admitted.item() == 7


@needs_gpu
def test_all_scores_equal_admits_the_whole_row():
    """Review Focus #2: hi == lo must not divide by zero."""
    topk, k, nbins = 16, 32, 64
    shards = [
        torch.full((1, k), 2.5, dtype=torch.float32, device="cuda") for _ in range(2)
    ]
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    assert torch.isfinite(thr).all()
    assert (shards[0] >= thr[:, None]).sum().item() == k


@needs_gpu
def test_empty_row_threshold_is_finite():
    topk, k, nbins = 16, 32, 64
    shards = [
        torch.full((1, k), NEG_INF, dtype=torch.float32, device="cuda")
        for _ in range(2)
    ]
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    # No candidate is finite, so nothing is admitted whatever the threshold is;
    # it only has to be a usable number for the emit kernel's comparison.
    assert not torch.isnan(thr).any()


def _emit_reference(
    local_val, local_idx, thr, local_ks, batch_ids, block_table, block_size
):
    """Pure-torch reference for emit_owned_slots. Returns (indices, indptr, counts)."""
    rows, _ = local_val.shape
    counts, per_row = [], []
    for t in range(rows):
        b = int(batch_ids[t].item())
        if b < 0:
            counts.append(0)
            per_row.append([])
            continue
        keep = (
            torch.isfinite(local_val[t])
            & (local_val[t] >= thr[t])
            & (local_idx[t] >= 0)
        )
        jl = (local_idx[t][keep] - int(local_ks[t].item())).tolist()
        slots = [
            int(block_table[b, j // block_size].item()) * block_size + j % block_size
            for j in jl
        ]
        counts.append(len(slots))
        per_row.append(slots)
    indptr = [0]
    for c in counts:
        indptr.append(indptr[-1] + max(c, 1))
    flat = torch.zeros(indptr[-1], dtype=torch.int32)
    for t, slots in enumerate(per_row):
        for i, s in enumerate(slots):
            flat[indptr[t] + i] = s
    return (
        flat,
        torch.tensor(indptr, dtype=torch.int32),
        torch.tensor(counts, dtype=torch.int32),
    )


def _emit_fixture(rows=5, k=16, num_req=2, cols=8, block_size=4, seed=11):
    torch.manual_seed(seed)
    local_val = _padded_rows(rows, k, [k, k, 0, 3, k], seed=seed)
    local_idx = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
    for t in range(rows):
        n = int(torch.isfinite(local_val[t]).sum().item())
        local_idx[t, :n] = torch.randperm(cols * block_size, device="cuda")[:n].to(
            torch.int32
        )
    local_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    batch_ids = torch.tensor([0, 1, 0, 1, -1], dtype=torch.int32, device="cuda")
    block_table = (
        torch.randperm(64, device="cuda")[: num_req * cols]
        .reshape(num_req, cols)
        .to(torch.int32)
    )
    return local_val, local_idx, local_ks, batch_ids, block_table, block_size


@needs_gpu
def test_emit_matches_reference():
    local_val, local_idx, local_ks, batch_ids, bt, bs = _emit_fixture()
    rows, _ = local_val.shape
    thr = torch.zeros(rows, dtype=torch.float32, device="cuda")
    ref_idx, ref_indptr, ref_counts = _emit_reference(
        local_val, local_idx, thr, local_ks, batch_ids, bt, bs
    )
    out_indices = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    out_indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    emit_owned_slots(
        local_val,
        local_idx,
        thr,
        local_ks,
        batch_ids,
        bt,
        bs,
        out_indices,
        out_indptr,
        counts,
    )
    assert torch.equal(out_indptr.cpu(), ref_indptr)
    assert torch.equal(counts.cpu(), ref_counts)
    n = int(ref_indptr[-1].item())
    assert torch.equal(out_indices[:n].cpu(), ref_idx)


@needs_gpu
def test_emit_pad_token_owns_nothing_and_indptr_stays_monotonic():
    """Review Focus #3."""
    local_val, local_idx, local_ks, batch_ids, bt, bs = _emit_fixture()
    rows, _ = local_val.shape
    thr = torch.full((rows,), NEG_INF, dtype=torch.float32, device="cuda")
    out_indices = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    out_indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    emit_owned_slots(
        local_val,
        local_idx,
        thr,
        local_ks,
        batch_ids,
        bt,
        bs,
        out_indices,
        out_indptr,
        counts,
    )
    assert counts[4].item() == 0  # batch_ids[4] == -1
    diffs = out_indptr[1:] - out_indptr[:-1]
    assert torch.all(diffs >= 1)


@needs_gpu
def test_emit_empty_row_gets_one_dummy_slot():
    """Review Focus #4."""
    local_val, local_idx, local_ks, batch_ids, bt, bs = _emit_fixture()
    rows, _ = local_val.shape
    thr = torch.full((rows,), POS_INF, dtype=torch.float32, device="cuda")
    out_indices = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    out_indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    emit_owned_slots(
        local_val,
        local_idx,
        thr,
        local_ks,
        batch_ids,
        bt,
        bs,
        out_indices,
        out_indptr,
        counts,
    )
    assert torch.all(counts == 0)
    for t in range(rows):
        assert out_indices[int(out_indptr[t].item())].item() == 0


@needs_gpu
def test_emit_preserves_candidate_order():
    """Compaction is order-preserving, so the fp accumulation order is fixed.

    The order it preserves is the one top_k_per_row_prefill emits, which is
    ascending column index -- NOT descending score. Either is deterministic;
    this pins which.
    """
    local_val, local_idx, local_ks, batch_ids, bt, bs = _emit_fixture()
    rows, _ = local_val.shape
    thr = torch.zeros(rows, dtype=torch.float32, device="cuda")
    out_indices = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    out_indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
    emit_owned_slots(
        local_val,
        local_idx,
        thr,
        local_ks,
        batch_ids,
        bt,
        bs,
        out_indices,
        out_indptr,
        counts,
    )
    t = 0
    n = int(counts[t].item())
    keep = torch.isfinite(local_val[t]) & (local_val[t] >= thr[t])
    jl = local_idx[t][keep] - local_ks[t]
    want = bt[0, (jl // bs).long()].to(torch.int32) * bs + (jl % bs).to(torch.int32)
    start = int(out_indptr[t].item())
    assert torch.equal(out_indices[start : start + n], want)


@needs_gpu
def test_exchange_emits_a_superset_of_the_exact_global_topk():
    """The whole point of the design: nothing the dcp=1 path selects is lost."""
    torch.manual_seed(21)
    W, rows, k, topk, nbins = 4, 6, 64, 16, 32
    block_size, cols, num_req = 4, 16, 2
    batch_ids = torch.tensor([0, 0, 1, 1, 0, -1], dtype=torch.int32, device="cuda")
    block_table = (
        torch.randperm(256, device="cuda")[: num_req * cols]
        .reshape(num_req, cols)
        .to(torch.int32)
    )
    local_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")

    shards_val, shards_idx = [], []
    for r in range(W):
        v = _padded_rows(rows, k, [k, k, 10, k, 0, k], seed=30 + r)
        i = torch.full((rows, k), -1, dtype=torch.int32, device="cuda")
        for t in range(rows):
            n = int(torch.isfinite(v[t]).sum().item())
            i[t, :n] = torch.randperm(cols * block_size, device="cuda")[:n].to(
                torch.int32
            )
        shards_val.append(v)
        shards_idx.append(i)

    stats = torch.stack([row_bracket_stats(s) for s in shards_val], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards_val)
    thr = threshold_from_histogram(hist, lo, hi, topk)

    # What the dcp=1 path would select: the exact global top-k per row.
    allv = torch.cat(shards_val, dim=1)
    for t in range(rows):
        if batch_ids[t].item() < 0:
            continue
        finite = allv[t][torch.isfinite(allv[t])]
        if finite.numel() == 0:
            continue
        want = torch.sort(finite, descending=True).values[: min(topk, finite.numel())]
        cut = want[-1].item()
        assert thr[t].item() <= cut + 1e-6, f"row {t} cut above the exact k-th"

    # And every rank's emit admits exactly its own candidates at or above the cut.
    for r in range(W):
        out_indices = torch.full((rows * 256,), -7, dtype=torch.int32, device="cuda")
        out_indptr = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
        counts = torch.zeros(rows, dtype=torch.int32, device="cuda")
        emit_owned_slots(
            shards_val[r],
            shards_idx[r],
            thr,
            local_ks,
            batch_ids,
            block_table,
            block_size,
            out_indices,
            out_indptr,
            counts,
        )
        for t in range(rows):
            if batch_ids[t].item() < 0:
                assert counts[t].item() == 0
                continue
            want_n = int(
                ((shards_val[r][t] >= thr[t]) & torch.isfinite(shards_val[r][t])).sum()
            )
            assert counts[t].item() == want_n


@needs_gpu
def test_exchange_orchestrator_matches_the_step_by_step_composition(monkeypatch):
    """The orchestrator's wiring, with the two collectives stubbed.

    Only the TRANSPORT is replaced: an all-gather delivers [W, rows, 4] and an
    all-reduce delivers the elementwise sum of [rows, NBINS], and running the W
    shards sequentially here produces exactly those tensors. Everything the
    orchestrator does with them is the shipped code.
    """
    from atom.model_ops import dcp_topk_select as mod

    class _Group:
        world_size = 0
        rank_in_group = 0
        device_group = None

    torch.manual_seed(22)
    W, rows, k, topk, nbins = 2, 3, 32, 32, 16
    block_size, cols, num_req = 4, 8, 1
    batch_ids = torch.zeros(rows, dtype=torch.int32, device="cuda")
    block_table = (
        torch.arange(num_req * cols, device="cuda")
        .reshape(num_req, cols)
        .to(torch.int32)
    )
    local_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    shards = [_padded_rows(rows, k, [k, k, k], seed=40 + r) for r in range(W)]
    idx0 = (
        torch.arange(k, dtype=torch.int32, device="cuda").expand(rows, k).contiguous()
    )

    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr_ref = threshold_from_histogram(hist, lo, hi, topk)

    ref_i = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    ref_p = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    ref_c = torch.zeros(rows, dtype=torch.int32, device="cuda")
    emit_owned_slots(
        shards[0],
        idx0,
        thr_ref,
        local_ks,
        batch_ids,
        block_table,
        block_size,
        ref_i,
        ref_p,
        ref_c,
    )

    got_i = torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda")
    got_p = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    got_c = torch.zeros(rows, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(
        mod, "_all_gather_stats", lambda g, s: stats.reshape(-1, s.shape[-1])
    )
    monkeypatch.setattr(mod, "_all_reduce_counts", lambda g, c: hist)
    group = _Group()
    group.world_size = W
    dcp_prefill_candidate_exchange(
        shards[0],
        idx0,
        local_ks,
        batch_ids,
        block_table,
        group,
        topk,
        block_size,
        nbins,
        got_i,
        got_p,
        got_c,
    )
    assert torch.equal(got_p, ref_p)
    assert torch.equal(got_c, ref_c)
    n = int(ref_p[-1].item())
    assert torch.equal(got_i[:n], ref_i[:n])


@needs_gpu
def test_top_k_per_row_prefill_index_convention():
    """Pins what emit_owned_slots assumes: indices are ABSOLUTE plane columns.

    If a future aiter switches to row-relative indices, `jl = idx - local_ks`
    silently reads the wrong keys instead of failing, so this is the tripwire.
    Also pins the tail padding the bracket stats treat as "not a candidate".
    """
    aiter_topk = pytest.importorskip("aiter.ops.topk", reason="requires AITER")
    rows, width, k = 2, 64, 8
    logits = torch.arange(rows * width, dtype=torch.float32, device="cuda").reshape(
        rows, width
    )
    starts = torch.tensor([16, 32], dtype=torch.int32, device="cuda")
    ends = torch.tensor([24, 35], dtype=torch.int32, device="cuda")  # 8 then 3 wide
    idx = torch.full((rows, k), -99, dtype=torch.int32, device="cuda")
    val = torch.full((rows, k), -99.0, dtype=torch.float32, device="cuda")
    aiter_topk.top_k_per_row_prefill(
        logits,
        starts,
        ends,
        idx,
        val,
        rows,
        logits.stride(0),
        logits.stride(1),
        k=k,
        stable=True,
    )
    # Absolute columns, emitted in ascending column order.
    assert idx[0].tolist() == [16, 17, 18, 19, 20, 21, 22, 23]
    assert val[0].tolist() == [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
    # Short row: winners first, then (-1, -inf) tail padding.
    assert idx[1].tolist() == [32, 33, 34, -1, -1, -1, -1, -1]
    assert val[1][:3].tolist() == [96.0, 97.0, 98.0]
    assert all(v == NEG_INF for v in val[1][3:].tolist())


def _peak_delta_mib(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - before
    del out
    return peak / (1024 * 1024)


@needs_gpu
def test_exchange_steps_do_not_allocate_row_by_k_temporaries():
    """The exchange runs OUTSIDE sparse_indexer_row_chunk's budget loop.

    `local_histogram` and `emit_owned_slots` both see the full [rows, K] plane,
    which at the shipped default (max_num_batched_tokens=16384, K=2048) is
    128 MiB of fp32. Anything that materializes even one more of those -- let
    alone the int64 ones a torch `scatter_add_` index forces -- re-opens the
    unbounded prefill allocation that ATOM_SPARSE_INDEXER_LOGITS_BUDGET_MB
    exists to bound (#1376), on a server whose startup memory profile never saw
    it.

    So the property, stated directly: neither step may allocate anything of the
    input plane's order.
    """
    rows, k, nbins = 4096, 2048, 512
    local_val = torch.randn(rows, k, dtype=torch.float32, device="cuda")
    plane_mib = rows * k * 4 / (1024 * 1024)
    lo = torch.full((rows,), -1.0, dtype=torch.float32, device="cuda")
    hi = torch.full((rows,), 1.0, dtype=torch.float32, device="cuda")

    peak = _peak_delta_mib(lambda: local_histogram(local_val, lo, hi, nbins))
    # The histogram itself is rows*nbins*4 = 8 MiB of the 32 MiB plane.
    assert peak < 0.5 * plane_mib, f"local_histogram peak {peak:.0f} MiB"

    # Materialized outside the measurement: this pins the op, not the fixture.
    local_idx = (
        torch.arange(k, dtype=torch.int32, device="cuda").expand(rows, k).contiguous()
    )
    thr = torch.zeros(rows, dtype=torch.float32, device="cuda")
    local_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    batch_ids = torch.zeros(rows, dtype=torch.int32, device="cuda")
    bt = torch.arange(64, dtype=torch.int32, device="cuda").expand(1, 64).contiguous()
    out_i = torch.zeros(rows * k, dtype=torch.int32, device="cuda")
    out_p = torch.zeros(rows + 1, dtype=torch.int32, device="cuda")
    cnt = torch.zeros(rows, dtype=torch.int32, device="cuda")

    peak = _peak_delta_mib(
        lambda: emit_owned_slots(
            local_val,
            local_idx,
            thr,
            local_ks,
            batch_ids,
            bt,
            1,
            out_i,
            out_p,
            cnt,
        )
    )
    assert peak < 0.5 * plane_mib, f"emit_owned_slots peak {peak:.0f} MiB"


def test_pcp_falls_back_to_the_gather_path_instead_of_raising(monkeypatch):
    """The spec says guard PCP *and keep the gather path* for that combination.

    A server already running GLM-5.2 at pcp>1 + dcp>1 upgrades, changes no
    flags, and starts fine because short requests never reach the indexer. The
    first request past index_topk would then raise out of a custom op inside the
    forward, on every rank at once, taking every co-scheduled request with it.
    Falling back costs nothing: the gather path is fully intact.
    """
    from atom.model_ops import dcp_topk_select as mod

    monkeypatch.setattr(mod, "get_dcp_world_size", lambda: 4)
    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_LOCAL", True)

    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    assert mod.use_dcp_local_indexer_prefill() is True

    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: True)
    assert mod.use_dcp_local_indexer_prefill() is False


def test_dcp_local_indexer_prefill_is_off_without_dcp(monkeypatch):
    from atom.model_ops import dcp_topk_select as mod

    monkeypatch.setattr(mod, "pcp_is_enabled", lambda: False)
    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_LOCAL", True)
    monkeypatch.setattr(mod, "get_dcp_world_size", lambda: 1)
    assert mod.use_dcp_local_indexer_prefill() is False
    monkeypatch.setattr(mod, "get_dcp_world_size", lambda: 4)
    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_LOCAL", False)
    assert mod.use_dcp_local_indexer_prefill() is False


@needs_gpu
def test_sub_topk_row_needs_k_equal_to_topk(monkeypatch):
    """The superset guarantee is conditional on K == topk_tokens; assert it.

    Counterexample it protects against: W=2, K=8, topk=16, rank 0 holding 8
    candidates (full) and rank 1 holding 3. Only 11 candidates exist globally,
    under the target of 16, so every one of them should be selected -- but
    rank 0 being full pins lo at its 8th and the three on rank 1 fall out.
    Nothing in the math can detect this; only the caller's shapes can.
    """
    rows, k, topk, nbins = 1, 8, 16, 16
    full = torch.arange(1.0, 9.0, dtype=torch.float32, device="cuda").reshape(1, k)
    short = torch.full((1, k), NEG_INF, dtype=torch.float32, device="cuda")
    short[0, :3] = torch.tensor([0.1, 0.2, 0.3], device="cuda")
    shards = [full, short]
    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr = threshold_from_histogram(hist, lo, hi, topk)
    allv = torch.cat(shards, dim=1)
    finite = int(torch.isfinite(allv).sum())
    admitted = int(((allv >= thr[:, None]) & torch.isfinite(allv)).sum())
    assert finite == 11
    assert admitted < finite  # the defect, reproduced

    class _Group:
        world_size = 2
        rank_in_group = 0
        device_group = None

    with pytest.raises(AssertionError, match="topk_tokens"):
        dcp_prefill_candidate_exchange(
            shards[0],
            torch.zeros(rows, k, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.zeros(1, 8, dtype=torch.int32, device="cuda"),
            _Group(),
            topk,
            1,
            nbins,
            torch.zeros(rows * k, dtype=torch.int32, device="cuda"),
            torch.zeros(rows + 1, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
        )


# --------------------------------------------------------------------------
# select="exact": the all-gather-candidates reference the histogram is scored
# against. Same emit, different scalar per row.
# --------------------------------------------------------------------------


def test_select_mode_rejects_a_typo(monkeypatch):
    """An unreadable mode name must raise, not resolve to the default.

    The exact path's only job is to be the thing the histogram is compared to.
    A typo that silently selects the histogram makes that comparison measure
    zero difference and report it as "no over-selection".
    """
    from atom.model_ops import dcp_topk_select as mod

    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_SELECT", "histogram")
    assert mod.dcp_prefill_select_mode() == "histogram"
    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_SELECT", "exact")
    assert mod.dcp_prefill_select_mode() == "exact"
    monkeypatch.setattr(mod.envs, "ATOM_DCP_INDEXER_PREFILL_SELECT", "Exact")
    with pytest.raises(ValueError, match="ATOM_DCP_INDEXER_PREFILL_SELECT"):
        mod.dcp_prefill_select_mode()


class _FakeGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.rank_in_group = 0
        self.device_group = None


def _exact_thr(monkeypatch, shards, topk, row_tile=512):
    """Run the shipped exact path with the all-gather replaced by a cat."""
    from atom.model_ops import dcp_topk_select as mod

    gathered = torch.cat(shards, dim=0)
    monkeypatch.setattr(mod, "_all_gather_candidates", lambda g, v: gathered)
    return mod.exact_threshold_from_candidates(
        _FakeGroup(len(shards)), shards[0], topk, row_tile=row_tile
    )


@needs_gpu
@pytest.mark.parametrize("row_tile", [512, 2])
def test_exact_mode_cut_is_the_true_global_kth(monkeypatch, row_tile):
    """thr is the k-th largest of the union, not the low edge of a bin.

    row_tile=2 runs the loop more than once over 5 rows, which is the only way
    an off-by-one in the tile slicing shows up -- with the default tile every
    test is a single iteration.
    """
    torch.manual_seed(31)
    W, rows, k = 3, 5, 24
    topk = k
    counts = [k, k, 7, 0, k]
    shards = [_padded_rows(rows, k, counts, seed=50 + r) for r in range(W)]
    thr = _exact_thr(monkeypatch, shards, topk, row_tile=row_tile)

    allv = torch.cat(shards, dim=1)
    for t in range(rows):
        finite = allv[t][torch.isfinite(allv[t])]
        if finite.numel() < topk:
            # Fewer candidates than the target: select all of them, same as the
            # histogram path's fall to bin 0.
            assert thr[t].item() == NEG_INF
            continue
        want = torch.sort(finite, descending=True).values[topk - 1].item()
        assert thr[t].item() == pytest.approx(want)


@needs_gpu
def test_exact_mode_reproduces_the_dcp1_selection_exactly(monkeypatch):
    """Union of the per-rank emits == the exact global top-k. No extras."""
    torch.manual_seed(32)
    W, rows, k = 4, 6, 32
    topk = k
    shards = [_padded_rows(rows, k, [k, k, 11, k, 0, k], seed=60 + r) for r in range(W)]
    thr = _exact_thr(monkeypatch, shards, topk)

    allv = torch.cat(shards, dim=1)
    for t in range(rows):
        finite = allv[t][torch.isfinite(allv[t])]
        admitted = int((finite >= thr[t]).sum())
        assert admitted == min(
            topk, finite.numel()
        ), f"row {t} admitted {admitted}, want {min(topk, finite.numel())}"


@needs_gpu
def test_exact_mode_is_a_subset_of_the_histogram_mode(monkeypatch):
    """The two modes bracket the answer: exact <= histogram, never the reverse.

    This is the measurement harness in test form -- the gap between the two
    admitted counts IS the over-selection the histogram pays, and it must never
    be negative, which would mean the histogram had lost a real winner.
    """
    torch.manual_seed(33)
    W, rows, k, nbins = 4, 8, 64, 16
    topk = k
    shards = [_padded_rows(rows, k, [k] * rows, seed=70 + r) for r in range(W)]
    thr_exact = _exact_thr(monkeypatch, shards, topk)

    stats = torch.stack([row_bracket_stats(s) for s in shards], dim=0)
    lo, hi = reduce_bracket(stats)
    hist = sum(local_histogram(s, lo, hi, nbins) for s in shards)
    thr_hist = threshold_from_histogram(hist, lo, hi, topk)

    assert torch.all(thr_hist <= thr_exact + 1e-6)
    allv = torch.cat(shards, dim=1)
    n_exact = ((allv >= thr_exact[:, None]) & torch.isfinite(allv)).sum(1)
    n_hist = ((allv >= thr_hist[:, None]) & torch.isfinite(allv)).sum(1)
    assert torch.all(n_hist >= n_exact)
    # A coarse 16 bins over W*K=256 candidates must actually over-select, or
    # this test would pass just as well against a broken histogram that
    # happened to return the exact cut.
    assert int((n_hist - n_exact).sum()) > 0


@needs_gpu
def test_exact_mode_treats_a_non_finite_score_as_absent(monkeypatch):
    """+inf must not outrank real candidates and then be dropped at emit.

    _admit refuses non-finite scores. If the cut were computed with a +inf
    counted as a winner, the row would admit topk-1 real tokens -- under-
    selection, which is exactly what the superset argument exists to rule out.
    """
    k = topk = 8
    a = torch.arange(1.0, 9.0, dtype=torch.float32, device="cuda").reshape(1, k)
    b = torch.arange(9.0, 17.0, dtype=torch.float32, device="cuda").reshape(1, k)
    clean = _exact_thr(monkeypatch, [a.clone(), b.clone()], topk)
    assert clean[0].item() == pytest.approx(9.0)  # 8th of 1..16

    # Poison a candidate ABOVE the cut: replacing 9.0 leaves 15 finite values, so
    # the 8th largest drops to 8.0. Counting the +inf as a winner would instead
    # push it UP to 10.0 and leave the row one real token short.
    poisoned = b.clone()
    poisoned[0, 0] = POS_INF
    got = _exact_thr(monkeypatch, [a.clone(), poisoned], topk)
    assert got[0].item() == pytest.approx(8.0)


@needs_gpu
def test_exchange_rejects_an_unknown_select_mode():
    rows, k, topk = 1, 4, 4
    z32 = torch.zeros(rows, k, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="unknown select mode"):
        dcp_prefill_candidate_exchange(
            torch.zeros(rows, k, device="cuda"),
            z32,
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            torch.zeros(1, 4, dtype=torch.int32, device="cuda"),
            _FakeGroup(2),
            topk,
            1,
            16,
            torch.zeros(rows * k, dtype=torch.int32, device="cuda"),
            torch.zeros(rows + 1, dtype=torch.int32, device="cuda"),
            torch.zeros(rows, dtype=torch.int32, device="cuda"),
            select="bisect",
        )


@needs_gpu
def test_exchange_in_exact_mode_uses_the_exact_cut_and_the_shared_emit(monkeypatch):
    """Orchestrator wiring for select="exact": one collective, same emit."""
    from atom.model_ops import dcp_topk_select as mod

    torch.manual_seed(34)
    W, rows, k, topk = 2, 3, 32, 32
    block_size, cols = 4, 8
    batch_ids = torch.zeros(rows, dtype=torch.int32, device="cuda")
    block_table = torch.arange(cols, device="cuda").reshape(1, cols).to(torch.int32)
    local_ks = torch.zeros(rows, dtype=torch.int32, device="cuda")
    shards = [_padded_rows(rows, k, [k, k, k], seed=80 + r) for r in range(W)]
    idx0 = (
        torch.arange(k, dtype=torch.int32, device="cuda").expand(rows, k).contiguous()
    )

    thr_ref = _exact_thr(monkeypatch, shards, topk)
    ref = [
        torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda"),
        torch.zeros(rows + 1, dtype=torch.int32, device="cuda"),
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
    ]
    emit_owned_slots(
        shards[0], idx0, thr_ref, local_ks, batch_ids, block_table, block_size, *ref
    )

    # Any use of the histogram transports in this mode is a wiring bug.
    def _boom(*_a, **_k):
        raise AssertionError("histogram collective reached in exact mode")

    monkeypatch.setattr(mod, "_all_gather_stats", _boom)
    monkeypatch.setattr(mod, "_all_reduce_counts", _boom)
    got = [
        torch.full((rows * 64,), -7, dtype=torch.int32, device="cuda"),
        torch.zeros(rows + 1, dtype=torch.int32, device="cuda"),
        torch.zeros(rows, dtype=torch.int32, device="cuda"),
    ]
    dcp_prefill_candidate_exchange(
        shards[0],
        idx0,
        local_ks,
        batch_ids,
        block_table,
        _FakeGroup(W),
        topk,
        block_size,
        16,
        *got,
        select="exact",
    )
    for g, r in zip(got, ref):
        assert torch.equal(g, r)
