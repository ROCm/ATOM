# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Eviction policy of the recurrent-state checkpoint pool.

`StateCachePool` used to rank victims by `(hit_count, last_access)` with hit
count primary, on the theory that write-once junk is always the most recent
thing in the pool. Replaying the SemiAnalysis agent traces through the pool
showed the premise inverted for that traffic: 89.8% of checkpoints are hit
exactly once (mean 1.00 over their lifetime), because prefixes *grow* -- turn N
anchors at its prompt end, turn N+1 resumes there and anchors deeper, and the
old anchor is dead forever. `hit_count > 0` therefore marked a checkpoint as
already spent, and ranking by it evicted the fresh anchor that was about to be
used. Cost: up to 30 points of hit rate at small pool sizes.

These drive the real `StateCachePool` through its public API. `test_a_hit_entry_
does_not_outrank_a_fresher_one` is the regression itself -- it fails on the old
rule and passes on LRU.
"""

import pytest

from atom.model_engine.state_cache import StateCachePool

G = 64  # granularity == block size here, so position N*G <-> block N


def make_pool(num_slots):
    return StateCachePool(num_slots=num_slots, granularity=G, block_size=G)


def add(pool, key, block_index):
    """Reserve + publish a checkpoint, the way the runner does across a step."""
    entry = pool.try_reserve(key, block_index * G, token_ids=(key,))
    if entry is not None:
        pool.publish(entry)
    return entry


def test_evicts_the_least_recently_used():
    pool = make_pool(3)
    for i, key in enumerate([10, 20, 30], start=1):
        add(pool, key, i)

    add(pool, 40, 4)  # one over capacity

    assert pool.lookup(10) is None, "oldest entry should have been evicted"
    assert pool.lookup(20) is not None
    assert pool.lookup(30) is not None
    assert pool.lookup(40) is not None


def test_a_lookup_refreshes_recency():
    pool = make_pool(3)
    for i, key in enumerate([10, 20, 30], start=1):
        add(pool, key, i)

    pool.lookup(10)  # 10 is now the most recently used
    add(pool, 40, 4)

    assert pool.lookup(10) is not None, "just-read entry must survive"
    assert pool.lookup(20) is None, "20 is now the LRU"


def test_a_spent_entry_does_not_outrank_a_fresher_unhit_one():
    """The regression, in the one arrangement where the two rules disagree.

    `spent` is old and has been hit; `anchor` is newer and has not. That is the
    steady state of a growing-prefix conversation: the previous turn's anchor
    has been consumed and will never be read again, while the new one is about
    to be. Hit-count-primary reads it exactly backwards -- it evicts the
    hit_count==0 anchor and keeps the corpse.

    Both rules agree whenever the hit entry is also the most recent, so a test
    built that way passes under the old rule and guards nothing.
    """
    pool = make_pool(2)
    add(pool, spent := 1, 1)
    pool.lookup(spent)  # consumed: hit_count 1, and now the older of the two
    add(pool, anchor := 2, 2)  # fresh, hit_count 0, never read

    add(pool, 3, 3)  # capacity exceeded -> exactly one of them must go

    assert pool.lookup(anchor) is not None, (
        "the fresh anchor is what the next turn resumes from; evicting it for a "
        "spent entry is the bug this rule change fixes"
    )
    assert pool.lookup(spent) is None, "the consumed entry is the LRU victim"


def test_a_pinned_entry_is_never_evicted():
    """A pin covers an in-flight DMA; evicting under it would free a slot that
    is actively being read."""
    pool = make_pool(2)
    pinned = add(pool, 1, 1)
    add(pool, 2, 2)
    pool.pin(pinned)

    add(pool, 3, 3)

    assert pool.lookup(1) is not None, "pinned entry must survive"
    assert pool.lookup(2) is None, "the unpinned LRU is the victim"


def test_an_unevictable_pool_degrades_to_no_checkpoint():
    """Checkpointing is an optimization: exhaustion returns None rather than
    raising or blocking admission."""
    pool = make_pool(1)
    only = add(pool, 1, 1)
    pool.pin(only)

    assert pool.try_reserve(2, 2 * G, token_ids=(2,)) is None
    assert pool.stats.skipped_full == 1


def test_eviction_leaks_no_slots():
    pool = make_pool(4)
    for key in range(40):
        add(pool, key, (key % 4) + 1)

    assert len(pool._allocated) + pool.num_free == pool.num_slots


@pytest.mark.parametrize("num_slots", [1, 2, 8])
def test_pool_never_exceeds_capacity(num_slots):
    pool = make_pool(num_slots)
    for key in range(100):
        add(pool, key, (key % 8) + 1)
        assert len(pool._allocated) <= num_slots
