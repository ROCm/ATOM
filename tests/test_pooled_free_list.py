"""Unit tests for shared logical-ID pool mechanics."""

import pytest

from atom.kv_cache.pools.pooled_free_list import PooledFreeList


def _allocate(pool):
    logical_id = pool.pop()
    pool.mark_used(logical_id)
    return logical_id


def test_stale_queue_entry_is_skipped():
    pool = PooledFreeList(3)
    pool.backed_ids.appendleft(99)
    pool.backed_ids.appendleft(0)
    pool.backed_set.discard(0)
    assert pool.pop() == 1


def test_backed_ids_are_preferred_over_unbacked():
    pool = PooledFreeList(3)
    first = _allocate(pool)
    pool.move_to_unbacked(first)
    assert first == 0
    assert pool.pop() == 1


def test_unbacked_fallback_and_exhaustion():
    pool = PooledFreeList(2, initially_backed=False)
    assert [pool.pop(), pool.pop()] == [0, 1]
    with pytest.raises(AssertionError, match="No free"):
        pool.pop()


def test_double_deallocate_does_not_duplicate_id():
    pool = PooledFreeList(1)
    logical_id = _allocate(pool)
    assert pool.deallocate(logical_id)
    assert not pool.deallocate(logical_id)
    assert list(pool.backed_ids).count(logical_id) == 1
    assert pool.ids_conserved()


def test_id_conservation_across_backing_transitions():
    pool = PooledFreeList(8, initially_backed=False)
    used = [_allocate(pool) for _ in range(4)]
    for logical_id in used:
        pool.deallocate(logical_id)
    for logical_id in used[:2]:
        pool.move_to_unbacked(logical_id)
    assert pool.ids_conserved()
    assert len(pool.used_ids) == 0
    assert len(pool.backed_set) == 2
    assert len(pool.unbacked_set) == 6
