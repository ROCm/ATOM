# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`UmbpBlockPool`: private blocks first, cold content demoted into the part
UMBP manages (atom/kv_transfer/l1/umbp_l1/pool.py)."""

import test_umbp_l1_pool as umbp_tests
from test_umbp_l1_pool import make_pool, settle, toks

# The L1 service fixture, the Python model and then the real one.
l1 = umbp_tests.l1

from atom.kv_transfer.l1.umbp_l1 import pool as umbp_pool
from atom.kv_transfer.l1.umbp_l1.pool import (
    UmbpBlockPool,
    make_umbp_block_pool,
    split_point,
)
from atom.kv_transfer.l1.umbp_l1.settings import UmbpL1Settings


def make_split(client, name, private_blocks, shared_blocks, evicted=None):
    """A pool built the way `make_umbp_block_pool` builds one."""
    original = umbp_pool.make_umbp_l1_pool

    def l1_pool(config, settings, num_blocks, **kwargs):
        return make_pool(client, name, num_blocks, **kwargs)

    umbp_pool.make_umbp_l1_pool = l1_pool
    settings = UmbpL1Settings(
        address="unix:///unused",
        share_ratio=shared_blocks / (private_blocks + shared_blocks),
    )
    try:
        return make_umbp_block_pool(
            None,
            settings,
            private_blocks + shared_blocks,
            on_evict=None if evicted is None else evicted.append,
        )
    finally:
        umbp_pool.make_umbp_l1_pool = original


def fill(pool, h):
    """Compute content `h` into a fresh block and release it, still indexed."""
    block_id = pool.pop()
    pool.allocate(block_id)
    pool.publish(block_id, h, toks(h))
    pool.free(block_id)
    return block_id


def test_split_point_rounds_the_shared_part():
    assert split_point(100, 0.3) == 70
    assert split_point(100, 1.0) == 0
    assert split_point(7, 0.5) == 3  # round(3.5) = 4 shared


def test_private_blocks_come_first(l1):
    pool = make_split(l1, "a", 3, 2)
    assert isinstance(pool, UmbpBlockPool)
    taken = []
    for _ in range(3):
        taken.append(pool.pop())
        pool.allocate(taken[-1])
    assert taken == [0, 1, 2]
    # Every private block held: fresh content goes to the shared part.
    block_id = pool.pop()
    assert block_id >= 3
    pool.allocate(block_id)
    assert pool.is_used(block_id) and pool.num_used == 4


def test_cold_private_content_is_demoted_not_dropped(l1):
    evicted = []
    pool = make_split(l1, "a", 2, 2, evicted)
    fill(pool, 11)
    fill(pool, 12)
    # Both private blocks hold cached content: the next one evicts 11 (LRU).
    block_id = pool.pop()
    pool.allocate(block_id)
    copies = pool.take_block_copies()
    assert len(copies) == 1
    src, dst = copies[0]
    assert src == block_id and dst >= 2
    # Not yet visible: the copy has not run. The hash did not leave the pool.
    assert pool.lookup(11) == -1
    assert evicted == []
    pool.settle()
    assert pool.lookup(11) == dst
    assert pool.blocks_demoted == 1


def test_nothing_is_published_before_its_batch_was_taken(l1):
    pool = make_split(l1, "a", 1, 2)
    fill(pool, 11)
    pool.allocate(pool.pop())  # demotes 11
    pool.settle()  # no batch took the copy yet
    assert pool.lookup(11) == -1
    pool.take_block_copies()
    pool.settle()
    assert pool.lookup(11) >= 1


def test_content_already_shared_is_not_copied_again(l1):
    pool = make_split(l1, "a", 1, 2)
    fill(pool, 11)
    newer = pool.pop()
    pool.allocate(newer)  # demotes 11
    pool.take_block_copies()
    pool.settle()
    shared_id = pool.lookup(11)
    pool.free(newer)
    # Computed privately again, then evicted again: the shared copy stays.
    block_id = pool.pop()
    assert block_id == 0
    pool.allocate(block_id)
    pool.publish(block_id, 11, toks(11))
    pool.free(block_id)
    assert pool.lookup(11) == 0  # the private copy wins a lookup
    pool.allocate(pool.pop())
    assert pool.take_block_copies() == ()
    assert pool.lookup(11) == shared_id


def test_a_full_shared_part_drops_the_demotion(l1):
    evicted = []
    pool = make_split(l1, "a", 1, 1, evicted)
    held = pool.pop()
    assert held == 0
    pool.allocate(held)
    shared_held = pool.pop()
    pool.allocate(shared_held)  # the only shared block, held
    pool.publish(held, 11, toks(11))
    pool.free(held)
    pool.allocate(pool.pop())  # evicts 11 with nowhere to keep it
    assert pool.take_block_copies() == ()
    assert pool.demotions_dropped == 1
    assert evicted == [11]


def test_another_rank_fetches_demoted_content(l1):
    a = make_split(l1, "a", 1, 2)
    b = make_pool(l1, "b", 4)
    fill(a, 11)
    a.allocate(a.pop())
    a.take_block_copies()
    a.settle()
    settle(a.umbp)  # the device ran the batch that carried the copy
    assert b.fetch([11], lambda i: toks(11)) == 1
    assert b.lookup(11) != -1


def test_private_content_is_not_visible_to_other_ranks(l1):
    a = make_split(l1, "a", 2, 2)
    b = make_pool(l1, "b", 4)
    fill(a, 11)  # private, never demoted
    settle(a.umbp)
    assert b.fetch([11], lambda i: toks(11)) == 0


def test_stats_carry_the_whole_pools_figures_under_the_usual_names(l1):
    pool = make_split(l1, "a", 3, 2)
    fill(pool, 11)
    stats = pool.eviction_stats()
    for key in ("blocks_total", "blocks_used", "blocks_free", "blocks_indexed"):
        assert key in stats
    assert stats["blocks_total"] == 5
    assert stats["blocks_indexed"] == 1
    assert stats["private_blocks_total"] == 3 and stats["umbp_blocks_total"] == 2


def test_a_share_ratio_of_one_sends_every_call_to_umbp(l1):
    pool = make_split(l1, "a", 0, 4)
    assert pool.offset == 0 and pool.private.num_blocks == 0
    block_id = fill(pool, 11)
    assert pool.lookup(11) == block_id
    assert pool.umbp.lookup(11) == block_id  # no private part to route around
    # Nothing to demote from: allocating never records a copy.
    pool.allocate(pool.pop())
    assert pool.take_block_copies() == ()
