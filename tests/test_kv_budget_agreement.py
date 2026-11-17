# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""KV block counts agreed across a worker group, without a GPU or a process group.

Rank 0's count is the one every worker builds. These tests run the same
agreement twice, once for a rank with more free memory and once for the rank
with less, and a stand-in ``group_min`` returns the smaller of the two.
"""

from atom.distributed import group_agreement
from atom.distributed.group_agreement import agree_kv_budget
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, plan_pools

_ENTRY_BYTES = 1024


class _Group:
    def __init__(self, world_size):
        self.world_size = world_size
        self.cpu_group = object()


def _install_peer_min(monkeypatch, peers):
    """``group_min`` pops the next peer in call order, ignoring the value.

    ``agree_kv_budget`` calls it once for ``free`` and once for the budget, for
    each group whose ``world_size`` exceeds one. Keying the peer on the value
    would return the wrong one when a rank's free memory equals its budget.
    """
    pending = list(peers)

    def fake_group_min(value, group):
        return min(value, pending.pop(0))

    monkeypatch.setattr(group_agreement, "group_min", fake_group_min)


def _block_count(free, budget, groups):
    agreed_free, agreed_budget = agree_kv_budget(free, budget, groups)
    # The caller sizes from the budget clamped by free memory, with no connector.
    available_for_kv = min(agreed_budget, agreed_free)
    return plan_pools(
        [page_pool(_ENTRY_BYTES)], available_for_kv, max_num_seqs=1
    ).paged_entries


def test_larger_rank_builds_the_smaller_block_count(monkeypatch):
    """The rank with more free memory sizes from the group's minimum, so both
    ranks build the same, smaller block count."""
    low_free, low_budget = 5_000, 4_096
    # Equal on purpose: a fake that looks the peer up by value cannot tell
    # this rank's free memory from its budget.
    high_free = high_budget = 8_192
    _install_peer_min(
        monkeypatch,
        # Low rank first, then the high rank. Each rank is free, then budget.
        [high_free, high_budget, low_free, low_budget],
    )
    groups = (_Group(2),)

    low_blocks = _block_count(low_free, low_budget, groups)
    high_blocks = _block_count(high_free, high_budget, groups)

    assert low_blocks == high_blocks == 4
    assert (
        plan_pools(
            [page_pool(_ENTRY_BYTES)], min(high_budget, high_free), max_num_seqs=1
        ).paged_entries
        == 8
    )


def test_larger_rank_gets_the_peers_smaller_free_memory(monkeypatch):
    """The caller checks the connector reserve against the agreed free memory.
    A rank with more free memory must get the peer's smaller value, so its
    check agrees with the rank that is short."""
    low_free, low_budget = 1 << 30, 3 << 30
    high_free = high_budget = 4 << 30
    _install_peer_min(
        monkeypatch,
        # High rank first, then the low rank. Each rank is free, then budget.
        [low_free, low_budget, high_free, high_budget],
    )
    groups = (_Group(4), _Group(1))

    assert agree_kv_budget(high_free, high_budget, groups) == (low_free, low_budget)
    assert agree_kv_budget(low_free, low_budget, groups) == (low_free, low_budget)


def test_world_size_one_keeps_this_rank_budget(monkeypatch):
    """A group of one rank is not reduced. A later group larger than one is."""

    def fail_if_called(value, group):
        raise AssertionError(group)

    monkeypatch.setattr(group_agreement, "group_min", fail_if_called)
    assert agree_kv_budget(9_000, 8_000, (_Group(1), _Group(1))) == (9_000, 8_000)

    reduced = []

    def take_a_slice(value, group):
        reduced.append(group)
        return value - 1_000

    monkeypatch.setattr(group_agreement, "group_min", take_a_slice)
    tp, pcp = _Group(1), _Group(2)
    assert agree_kv_budget(8_000, 7_000, (tp, pcp)) == (7_000, 6_000)
    assert reduced == [pcp.cpu_group, pcp.cpu_group]
