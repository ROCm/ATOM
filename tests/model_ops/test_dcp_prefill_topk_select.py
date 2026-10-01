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
    """Review Focus #1."""
    topk, k, nbins = 16, 32, 64
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
