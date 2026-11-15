# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV block counts agreed across a worker group, without a GPU or a process group.

Rank 0's count is the one every worker builds. These tests run the same
agreement twice, once for a rank with more free memory and once for the rank
with less, and a stand-in ``group_min`` returns the smaller of the two.
"""

import pytest

from atom.distributed import group_agreement
from atom.distributed.group_agreement import agree_kv_budget
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, plan_pools

_ENTRY_BYTES = 1024


class _Group:
    def __init__(self, world_size):
        self.world_size = world_size
        self.cpu_group = object()


def _install_peer_min(monkeypatch, peers):
    """``group_min`` returns the smaller of this rank's value and its peer's."""

    def fake_group_min(value, group):
        return min(value, peers[value])

    monkeypatch.setattr(group_agreement, "group_min", fake_group_min)


def _block_count(free, budget, groups):
    agreed = agree_kv_budget(free, budget, 0, groups)
    return plan_pools(
        [page_pool(_ENTRY_BYTES)], agreed.available_for_kv, max_num_seqs=1
    ).paged_entries


def test_larger_rank_builds_the_smaller_block_count(monkeypatch):
    """The rank with more free memory sizes from the group's minimum, so both
    ranks build the same, smaller block count."""
    low_free, low_budget = 5_000, 4_096
    high_free, high_budget = 9_000, 8_192
    _install_peer_min(
        monkeypatch,
        {
            low_free: high_free,
            low_budget: high_budget,
            high_free: low_free,
            high_budget: low_budget,
        },
    )
    groups = (_Group(2), _Group(2))

    low_blocks = _block_count(low_free, low_budget, groups)
    high_blocks = _block_count(high_free, high_budget, groups)

    assert low_blocks == high_blocks == 4
    assert (
        plan_pools(
            [page_pool(_ENTRY_BYTES)], min(high_budget, high_free), max_num_seqs=1
        ).paged_entries
        == 8
    )


def test_connector_reserve_raises_on_the_larger_rank_too(monkeypatch):
    """The connector check reads the agreed free memory. A rank whose own free
    memory would cover the reserve still raises when a peer's does not."""
    low_free, high_free = 1 << 30, 4 << 30
    low_budget, high_budget = 3 << 30, 5 << 30
    connector_bytes = 2 << 30
    _install_peer_min(
        monkeypatch,
        {
            low_free: high_free,
            high_free: low_free,
            low_budget: high_budget,
            high_budget: low_budget,
        },
    )
    groups = (_Group(4), _Group(1))

    with pytest.raises(RuntimeError, match="KV connector"):
        agree_kv_budget(low_free, low_budget, connector_bytes, groups)
    with pytest.raises(RuntimeError, match="KV connector"):
        agree_kv_budget(high_free, high_budget, connector_bytes, groups)

    # The same high rank, with no peer in the group, keeps its own free memory
    # and does not raise.
    agree_kv_budget(high_free, high_budget, connector_bytes, (_Group(1),))


def test_world_size_one_keeps_this_rank_budget(monkeypatch):
    """A group of one rank is not reduced. A later group larger than one is."""

    def fail_if_called(value, group):
        raise AssertionError(group)

    monkeypatch.setattr(group_agreement, "group_min", fail_if_called)
    agreed = agree_kv_budget(9_000, 8_000, 0, (_Group(1), _Group(1)))
    assert agreed.available_for_kv == 8_000

    reduced = []

    def take_a_slice(value, group):
        reduced.append(group)
        return value - 1_000

    monkeypatch.setattr(group_agreement, "group_min", take_a_slice)
    tp, pcp = _Group(1), _Group(2)
    agreed = agree_kv_budget(8_000, 7_000, 0, (tp, pcp))
    assert reduced == [pcp.cpu_group, pcp.cpu_group]
    assert agreed.available_for_kv == 6_000
