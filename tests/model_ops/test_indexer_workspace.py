# SPDX-License-Identifier: MIT
"""The FP8 prefill key workspace and pass plan: host arithmetic, runnable
without AITER or a GPU. Packed bands: tests/models/deepseek_v41/test_indexer.py."""

import itertools

import numpy as np
import pytest
import torch

from atom.model_ops.indexer.workspace import (
    ConcatKeyWorkspace,
    aligned_spans,
    concat_passes,
)


def test_key_workspace_hands_out_views_of_held_buffers_and_never_grows():
    ws = ConcatKeyWorkspace(64, 2, torch.uint8, "cpu", gathered_capacity=96)
    held = ws.keys(64)[0].data_ptr()
    for total in range(8, 65, 8):  # a long prefill's chunks
        values, scales = ws.keys(total)
        assert values.data_ptr() == held and scales.shape == (total, 1)
    with pytest.raises(ValueError, match="65 keys exceed the 64-key"):
        ws.keys(65)
    assert ws.gathered(96)[1].dtype == torch.float32
    with pytest.raises(ValueError, match="gathered-shard"):
        ws.gathered(97)
    assert ws.nbytes == 64 * (2 + 4) + 96 * (2 + 4)


@pytest.mark.parametrize("seed", range(4))
def test_key_workspace_gathers_any_pass_of_padded_shards(seed):
    """DCP8 pads each request's shard to its interleave super-block: any pass
    of requests within the key capacity still fits the gathered buffer."""
    world, interleave, requests = 8, 2, 6
    ws = ConcatKeyWorkspace.for_config(
        1000, 24, requests, 1, torch.uint8, "cpu", world, interleave
    )
    cuts = np.sort(np.random.default_rng(seed).integers(0, ws.capacity, requests - 1))
    lengths = np.diff([0, *cuts, ws.capacity])
    padded = -(-lengths // (interleave * world)) * interleave
    assert world * int(padded.sum()) <= ws.gathered_capacity


def _plan(rows, keys, key_capacity, plane_capacity, world):
    row_cu = np.array([0, *itertools.accumulate(rows)])
    key_cu = np.array([0, *itertools.accumulate(keys)])
    # Causal rows: a request's last row sees all its keys.
    ends = np.concatenate(
        [np.arange(k - r + 1, k + 1) for r, k in zip(rows, keys)]
    ).astype(np.int32)
    plans = []
    for rank in range(world):
        offsets = np.full(row_cu[-1], -1, np.int32)
        plan = (row_cu, key_cu, ends, offsets, key_capacity, plane_capacity)
        plans.append((concat_passes(*plan, rank, world), offsets))
    return row_cu, key_cu, ends, plans


@pytest.mark.parametrize("world", [1, 2, 8])
@pytest.mark.parametrize("key_capacity", [10**6, 700, 300])
def test_passes_gather_whole_requests_that_fit_and_split_each_over_the_ranks(
    world, key_capacity
):
    rows, keys, plane = [5, 0, 9, 3, 12], [300, 120, 260, 30, 290], 1024
    row_cu, key_cu, ends, plans = _plan(rows, keys, key_capacity, plane, world)
    passes = plans[0][0]
    assert passes[0].first == 0 and passes[-1].end == len(keys)
    for p, nxt in itertools.zip_longest(passes, passes[1:]):
        assert p.keys == key_cu[p.end] - key_cu[p.first] <= key_capacity
        assert p.key_base == key_cu[p.first] and p.row_base == row_cu[p.first]
        if nxt is not None:  # the next request would not have fit
            assert (
                nxt.first == p.end
                and key_cu[p.end + 1] - key_cu[p.first] > key_capacity
            )
    for index, p in enumerate(passes):
        shares = [plan[index] for plan, _ in plans]
        assert all(s[:6] == p[:6] for s in shares)  # same pass on every rank
        assert [s.rank for s in shares] == list(range(world))
        covered = [row for s in shares for row in range(s.lo, s.hi)]
        assert covered == list(range(row_cu[p.first], row_cu[p.end]))
        assert all(s.hi - s.lo <= p.stride for s in shares)
    for passes, offsets in plans:
        for p in passes:
            bands = list(p.bands)
            assert [b for b, _ in bands] == [p.lo, *[e for _, e in bands]][: len(bands)]
            assert (bands[-1][1] if bands else p.lo) == p.hi
            for first, end in bands:
                spans = aligned_spans(ends[first:end].astype(np.int64))
                assert np.array_equal(offsets[first:end], np.cumsum(spans) - spans)
                assert spans.sum() <= plane


def test_passes_refuse_a_request_past_the_key_workspace():
    cu, ends = np.array([0, 1, 2]), np.array([10, 400], np.int32)
    with pytest.raises(ValueError, match="request 1's 400 keys"):
        concat_passes(
            cu, np.array([0, 10, 410]), ends, np.zeros(2, np.int32), 300, 4096
        )
