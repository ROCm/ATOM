# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`dcp_concat_gather_plan`: the DCP FP8 prefill indexer reads each rank's
padded shard of some requests' index keys, all-gathers the shards rank-major
and de-interleaves them by the plan's `src`. Simulated here on host arrays:
every rank's shard laid out as the interleaved KV cache holds it."""

import numpy as np
import pytest

from atom.distributed.dcp_layout import (
    dcp_concat_gather_plan,
    dcp_local_index,
    dcp_owner_rank,
)


def _keys(g_cu):
    """A distinct value per committed key: request b's position p is
    b * 1_000_000 + p."""
    lens = np.diff(g_cu)
    return np.concatenate(
        [b * 1_000_000 + np.arange(n, dtype=np.int64) for b, n in enumerate(lens)]
    )


def _gather(g_cu, first, end, world, interleave):
    """Requests [first, end)'s keys as the indexer gathers them: each rank's
    padded shard -- request by request, `lpad` slots, -1 in the padding --
    all-gathered rank-major and de-interleaved by `src`."""
    _, lpad, cu_pad, local_total, src = dcp_concat_gather_plan(
        g_cu, first, end, world, interleave
    )
    shards = np.full((world, local_total), -1, dtype=np.int64)
    for b in range(first, end):
        pos = np.arange(g_cu[b + 1] - g_cu[b], dtype=np.int64)
        owner = dcp_owner_rank(pos, world, interleave)
        local = cu_pad[b - first] + dcp_local_index(pos, world, interleave)
        assert (dcp_local_index(pos, world, interleave) < lpad[b - first]).all()
        shards[owner, local] = b * 1_000_000 + pos
    return shards.reshape(-1)[src]


STEPS = [
    np.array([0, 1], dtype=np.int64),
    np.array([0, 7, 7 + 64, 7 + 64 + 129], dtype=np.int64),
    np.cumsum([0, 1000, 3, 17, 4096, 5], dtype=np.int64),
]


@pytest.mark.parametrize("g_cu", STEPS)
@pytest.mark.parametrize("world", [1, 2, 8])
@pytest.mark.parametrize("interleave", [1, 2])
def test_the_plan_de_interleaves_every_shard_back_to_the_keys(g_cu, world, interleave):
    requests = len(g_cu) - 1
    assert (_gather(g_cu, 0, requests, world, interleave) == _keys(g_cu)).all()


@pytest.mark.parametrize("world", [2, 8])
@pytest.mark.parametrize("interleave", [1, 2])
def test_a_group_s_plan_gathers_its_slice_of_the_step(world, interleave):
    """Requests gathered a group at a time come back as the step's run, cut
    at the groups: the slice from the group's first key."""
    g_cu = STEPS[2]
    step = _keys(g_cu)
    for first, end in [(0, 1), (1, 4), (4, 5), (2, 3), (0, 5)]:
        group = _gather(g_cu, first, end, world, interleave)
        assert (group == step[g_cu[first] : g_cu[end]]).all()


@pytest.mark.parametrize("world", [2, 8])
def test_every_rank_reads_the_same_padded_shard_length(world):
    """`local_total` is rank-independent, so the all-gather is a plain concat;
    W shards of it hold every key and at most a super-block of padding per
    request."""
    g_cu = STEPS[2]
    _, lpad, _, local_total, src = dcp_concat_gather_plan(g_cu, 0, 5, world, 1)
    assert local_total == lpad.sum()
    assert len(src) == g_cu[-1]
    assert world * local_total - g_cu[-1] < 5 * world
