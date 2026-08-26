# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Free-list order and resizing for the paged block pool.

Two properties are worth a test here and neither is visible from
`BlockManager`: which free block gets handed out — the pool has to spend a
block that holds nothing before one that still holds reusable content — and
what it costs to take the highest block id away, which is what a moving
compress/state boundary does.
"""

import array

import pytest

from atom.model_engine.block_pool import BlockPool


def toks(*ids: int) -> array.array:
    """What the block manager publishes: a slice of `Sequence.token_ids`.

    A list here is what `Block.update` refuses, so the tests have to hold the
    production type or they stop covering the comparison the pool relies on.
    """
    return array.array("i", ids)


def published(pool: BlockPool, block_id: int, h: int) -> int:
    """Allocate, publish under `h`, and release — a cached, free block."""
    pool.allocate(block_id)
    pool.publish(block_id, h, toks(h))
    pool.free(block_id)
    return block_id


class TestHandOutOrder:
    def test_a_block_holding_nothing_goes_before_a_cached_one(self):
        pool = BlockPool(num_blocks=4)
        # 0 and 1 become cached, in that order; 2 and 3 were never used.
        published(pool, pool.pop(), h=100)
        published(pool, pool.pop(), h=200)
        # Release order alone would hand out the cached blocks first.
        assert [pool.pop(), pool.pop()] == [2, 3]
        assert pool.lookup(100) == 0 and pool.lookup(200) == 1

    def test_cached_blocks_go_least_recently_freed_first(self):
        pool = BlockPool(num_blocks=3)
        for h in (100, 200, 300):
            published(pool, pool.pop(), h=h)
        assert [pool.pop(), pool.pop(), pool.pop()] == [0, 1, 2]

    def test_reuse_refreshes_nothing_but_release_does(self):
        # Claiming a cached block and releasing it puts it back at the end of
        # the queue, so reuse is what keeps content alive.
        pool = BlockPool(num_blocks=3)
        for h in (100, 200, 300):
            published(pool, pool.pop(), h=h)
        pool.claim(pool.lookup(100))
        pool.free(0)
        assert [pool.pop(), pool.pop(), pool.pop()] == [1, 2, 0]

    def test_vacant_blocks_go_lowest_id_first(self):
        # Order within the vacant half is free to choose, and low-first is
        # what drains the top of the pool for a shrinking boundary.
        pool = BlockPool(num_blocks=4)
        for block_id in (3, 1, 2, 0):
            pool.allocate(block_id)
        for block_id in (3, 1, 2, 0):
            pool.free(block_id)
        assert [pool.pop() for _ in range(4)] == [0, 1, 2, 3]


class TestRetiringTheTopBlock:
    def test_a_vacant_top_costs_nothing(self):
        pool = BlockPool(num_blocks=3)
        retirement = pool.retire_top()
        assert (retirement.retired, retirement.moved_to) == (2, -1)
        assert pool.num_blocks == 2
        assert pool.num_free == 2
        assert 2 not in {pool.pop(), pool.pop()}

    def test_a_cached_top_loses_its_content_and_says_so(self):
        evicted = []
        pool = BlockPool(num_blocks=3, on_evict=evicted.append)
        published(pool, 2, h=100)
        retirement = pool.retire_top()
        assert (retirement.retired, retirement.moved_to) == (2, -1)
        assert evicted == [100]
        assert pool.lookup(100) == -1

    def test_a_held_top_moves_and_keeps_its_identity(self):
        pool = BlockPool(num_blocks=3)
        pool.allocate(2)
        pool.publish(2, 100, toks(7, 8))
        pool.claim(2)  # a second holder, so the ref count has to travel too

        retirement = pool.retire_top()
        assert retirement.retired == 2
        assert 0 <= retirement.moved_to < 2

        moved = pool.block(retirement.moved_to)
        assert (moved.hash, moved.token_ids, moved.ref_count) == (100, toks(7, 8), 2)
        assert pool.lookup(100) == retirement.moved_to
        assert pool.is_used(retirement.moved_to)
        assert not pool.is_used(2)
        assert pool.num_blocks == 2

    def test_moving_evicts_whatever_the_destination_held(self):
        evicted = []
        pool = BlockPool(num_blocks=3, on_evict=evicted.append)
        published(pool, 0, h=100)
        published(pool, 1, h=200)
        pool.allocate(2)
        pool.publish(2, 300, toks(3))

        retirement = pool.retire_top()
        # The destination is a cached block, so its content is the price.
        assert retirement.moved_to in (0, 1)
        assert evicted == [100 if retirement.moved_to == 0 else 200]
        assert pool.lookup(300) == retirement.moved_to

    def test_a_held_top_with_nothing_free_refuses(self):
        pool = BlockPool(num_blocks=2)
        pool.allocate(0)
        pool.allocate(1)
        assert pool.retire_top() is None
        assert pool.num_blocks == 2

    def test_an_empty_pool_has_nothing_to_retire(self):
        assert BlockPool(num_blocks=0).retire_top() is None


class TestGrowing:
    def test_extend_stops_at_the_maximum(self):
        pool = BlockPool(num_blocks=2, max_blocks=4)
        assert pool.extend(1) == 1
        assert pool.num_blocks == 3
        assert pool.extend(5) == 1
        assert pool.num_blocks == 4
        assert pool.extend(1) == 0

    def test_a_pinned_pool_cannot_grow(self):
        assert BlockPool(num_blocks=2).extend(1) == 0

    def test_a_regrown_block_is_empty_again(self):
        # Every block cached, so the one that comes back is the only vacant
        # one and hand-out order says which half it is in.
        pool = BlockPool(num_blocks=3, max_blocks=3)
        for block_id, h in ((0, 200), (1, 300), (2, 100)):
            published(pool, block_id, h)
        pool.retire_top()
        pool.extend(1)
        assert pool.num_blocks == 3
        assert pool.block(2).hash == -1
        assert pool.block(2).ref_count == 0
        assert pool.pop() == 2

    def test_growing_beyond_the_allocation_is_refused(self):
        with pytest.raises(ValueError, match="outside"):
            BlockPool(num_blocks=5, max_blocks=4)


class TestRawPageUnits:
    def test_reservation_uses_arbitrary_non_contiguous_free_ids(self):
        pool = BlockPool(num_blocks=8)
        for _ in range(8):
            pool.allocate(pool.pop())
        for block_id in (0, 2, 5, 7):
            pool.free(block_id)

        units = pool.reserve_units(4, owner=("checkpoint", 1))

        assert units == [0, 2, 5, 7]
        assert all(pool.is_used(i) for i in units)
        assert pool.num_free == 0
        with pytest.raises(AssertionError, match="belongs"):
            pool.release_units(reversed(units), owner=("checkpoint", 1))
        pool.release_units(units, owner=("checkpoint", 1))
        assert pool.num_free == 4

    def test_failed_reservation_is_atomic(self):
        pool = BlockPool(num_blocks=2)
        pool.allocate(pool.pop())
        assert pool.reserve_units(2, owner=("checkpoint", 1)) is None
        assert pool.num_free == 1

    def test_only_the_whole_owner_can_release_units(self):
        pool = BlockPool(num_blocks=3)
        units = pool.reserve_units(2, owner=("checkpoint", 1))
        with pytest.raises(AssertionError, match="belongs"):
            pool.release_units(units, owner=("checkpoint", 2))
        assert pool.num_free == 1
        pool.release_units(units, owner=("checkpoint", 1))
        assert pool.num_free == 3

    def test_retirement_refuses_a_fragment_without_relocation_protocol(self):
        pool = BlockPool(num_blocks=3)
        # Reserve the highest id specifically by occupying the lower two.
        pool.allocate(0)
        pool.allocate(1)
        units = pool.reserve_units(1, owner=("checkpoint", 1))
        assert units == [2]
        assert pool.retire_top() is None
        assert pool.num_blocks == 3


class TestEvictionAccounting:
    """The counters a benchmark reads to tell lost reuse from absent reuse.

    A hit rate cannot distinguish them, so these have to be exact in both
    directions: every destroyed cache entry counted once, and nothing counted
    that did not destroy one.
    """

    def test_spending_a_cached_block_counts_an_eviction(self):
        pool = BlockPool(num_blocks=2)
        published(pool, pool.pop(), h=100)
        published(pool, pool.pop(), h=200)
        assert pool.blocks_evicted == 0  # freeing destroys nothing
        pool.allocate(pool.pop())
        assert pool.blocks_evicted == 1
        assert pool.lookup(100) == -1

    def test_reusing_a_vacant_block_is_not_an_eviction(self):
        pool = BlockPool(num_blocks=2)
        pool.allocate(pool.pop())
        assert pool.blocks_evicted == 0

    def test_reclaiming_a_block_by_hash_is_not_an_eviction(self):
        # A prefix hit takes a named cached block off the free list. Its
        # content is being *used*, so counting it would report the cache
        # working as the cache failing.
        pool = BlockPool(num_blocks=2)
        published(pool, pool.pop(), h=100)
        pool.claim(pool.lookup(100))
        assert pool.blocks_evicted == 0

    def test_the_boundary_is_counted_apart_from_ordinary_eviction(self):
        # Same lost content, opposite fixes: `evicted` says the pool is too
        # small, `retired` says the split is wrong.
        pool = BlockPool(num_blocks=2, max_blocks=2)
        published(pool, 0, h=100)
        published(pool, 1, h=200)
        pool.retire_top()
        assert (pool.blocks_retired, pool.blocks_evicted) == (1, 0)

    def test_retiring_a_vacant_block_costs_nothing(self):
        pool = BlockPool(num_blocks=2, max_blocks=2)
        pool.retire_top()
        assert pool.blocks_retired == 0

    def test_relocating_a_live_block_is_not_an_eviction_of_itself(self):
        # `_adopt` unindexes the destination, then re-points the source's hash
        # at it. The surviving hash must not be counted as destroyed.
        pool = BlockPool(num_blocks=2, max_blocks=2)
        pool.allocate(0)  # vacant destination
        pool.free(0)
        pool.allocate(1)
        pool.publish(1, 300, toks(300))  # live, holds a hash
        out = pool.retire_top()
        assert out is not None and out.moved_to == 0
        assert pool.lookup(300) == 0  # content survived the move
        assert (pool.blocks_retired, pool.blocks_evicted) == (0, 0)

    def test_vacant_is_the_headroom_before_eviction_starts(self):
        pool = BlockPool(num_blocks=3)
        published(pool, pool.pop(), h=100)
        stats = pool.eviction_stats()
        assert stats["blocks_free_reusable"] == 1
        assert stats["blocks_free"] - stats["blocks_free_reusable"] == 2
        assert stats["blocks_indexed"] == 1
