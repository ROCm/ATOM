# SPDX-License-Identifier: MIT
"""The FP8 prefill key workspace and pass plan: host arithmetic, runnable
without AITER or a GPU. Packed bands: tests/models/deepseek_v41/test_indexer.py."""

import itertools

import numpy as np
import pytest
import torch

from atom.distributed.dcp_layout import dcp_local_index, dcp_owner_rank
from atom.model_ops.indexer.workspace import (
    ConcatKeyWorkspace,
    aligned_spans,
    concat_passes,
    dcp_gather_index,
)


@pytest.mark.parametrize("seed", range(4))
def test_key_workspace_holds_any_pass_and_never_grows(seed):
    """Views of held buffers, refused past capacity; under DCP8 (S=2) any pass
    within the key capacity fits the gathered buffer, shards padded."""
    ws = ConcatKeyWorkspace.for_config(1000, 24, 6, 2, torch.uint8, "cpu", 8, 2)
    held = ws.keys(ws.capacity)[0].data_ptr()
    assert all(ws.keys(n)[0].data_ptr() == held for n in range(8, ws.capacity, 64))
    with pytest.raises(ValueError, match="1025 keys exceed the 1024-key"):
        ws.keys(ws.capacity + 1)
    cuts = np.sort(np.random.default_rng(seed).integers(0, ws.capacity, 5))
    shards = -(-np.diff([0, *cuts, ws.capacity]) // 16) * 2
    assert ws.gathered(8 * int(shards.sum()))[1].dtype == torch.float32


@pytest.mark.parametrize("cuts", [[(0, 4)], [(0, 2), (2, 4)], [(0, 1), (1, 3), (3, 4)]])
def test_dcp_gathers_put_each_pass_of_shards_back_in_key_order(cuts):
    """Interleaved, padded shards (W=4, S=2): each pass's index takes its
    requests' keys out of their all-gather, every key once, in order."""
    world, interleave, lens = 4, 2, np.array([45, 17, 130, 3])
    g_cu = np.concatenate([[0], np.cumsum(lens)])
    cu_pad = np.concatenate([[0], np.cumsum(-(-lens // 8) * interleave)])
    for (first, end), g in zip(
        cuts, dcp_gather_index(cuts, g_cu, cu_pad, world, interleave)
    ):
        shards = np.full((world, g.local_total), -1)
        for r in range(first, end):
            pos = np.arange(lens[r])
            local = cu_pad[r] - cu_pad[first] + dcp_local_index(pos, world, interleave)
            shards[dcp_owner_rank(pos, world, interleave), local] = g_cu[r] + pos
        assert np.array_equal(g.local_cu, cu_pad[first : end + 1] - cu_pad[first])
        taken = shards.reshape(-1)[g.gather_index]
        assert np.array_equal(taken, np.arange(g_cu[first], g_cu[end]))


@pytest.mark.parametrize("world", [1, 2, 8])
@pytest.mark.parametrize("key_capacity", [10**6, 700, 300])
def test_passes_take_whole_requests_that_fit_split_over_every_rank(world, key_capacity):
    rows, keys, plane = [5, 0, 9, 3, 12], [300, 120, 260, 30, 290], 1024
    row_cu = np.array([0, *itertools.accumulate(rows)])
    key_cu = np.array([0, *itertools.accumulate(keys)])
    ends = np.concatenate([np.arange(k - r, k) + 1 for r, k in zip(rows, keys)])
    offsets = [np.full(row_cu[-1], -1, np.int32) for _ in range(world)]
    plans = [
        concat_passes(
            row_cu, key_cu, ends, offsets[rank], key_capacity, plane, rank, world
        )
        for rank in range(world)
    ]
    passes = plans[0]
    assert passes[0].first == 0 and passes[-1].end == len(keys)
    for p, nxt in zip(passes, [*passes[1:], None]):
        assert p.keys == key_cu[p.end] - key_cu[p.first] <= key_capacity
        assert p.key_base == key_cu[p.first] and p.row_base == row_cu[p.first]
        # the next request would not have fit
        assert nxt is None or key_cu[p.end + 1] - key_cu[p.first] > key_capacity
    for index, p in enumerate(passes):
        shares = [plan[index] for plan in plans]
        assert all(s[:6] == p[:6] and s.hi - s.lo <= p.stride for s in shares)
        covered = [row for s in shares for row in range(s.lo, s.hi)]
        assert covered == list(range(row_cu[p.first], row_cu[p.end]))
    for plan, rank_offsets in zip(plans, offsets):
        for p in plan:
            assert [r for b in p.bands for r in range(*b)] == list(range(p.lo, p.hi))
            for first, end in p.bands:
                spans = aligned_spans(ends[first:end].astype(np.int64))
                within = rank_offsets[first:end]
                assert np.array_equal(within, np.cumsum(spans) - spans)
                assert spans.sum() <= plane


def test_passes_refuse_a_request_past_the_key_workspace():
    cu, ends = np.array([0, 1, 2]), np.array([10, 400], np.int32)
    with pytest.raises(ValueError, match="request 1's 400 keys"):
        concat_passes(
            cu, np.array([0, 10, 410]), ends, np.zeros(2, np.int32), 300, 4096
        )
