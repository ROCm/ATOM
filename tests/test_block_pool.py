# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Free-list order and resizing for the paged block pool.

Two properties are worth a test here and neither is visible from
`BlockManager`: which free block gets handed out — the pool has to spend a
block that holds nothing before one that still holds reusable content — and
what it costs to take the highest block id away, which is what a moving
compress/state boundary does.
"""

import pytest

from atom.model_engine.block_pool import BlockPool


def published(pool: BlockPool, block_id: int, h: int) -> int:
    """Allocate, publish under `h`, and release — a cached, free block."""
    pool.allocate(block_id)
    pool.publish(block_id, h, [h])
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
        pool.publish(2, 100, [7, 8])
        pool.claim(2)  # a second holder, so the ref count has to travel too

        retirement = pool.retire_top()
        assert retirement.retired == 2
        assert 0 <= retirement.moved_to < 2

        moved = pool.block(retirement.moved_to)
        assert (moved.hash, moved.token_ids, moved.ref_count) == (100, [7, 8], 2)
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
        pool.publish(2, 300, [3])

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
