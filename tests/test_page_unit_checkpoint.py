# SPDX-License-Identifier: MIT

"""Control-plane invariants for PAGE-backed state checkpoint images."""

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    COPYING,
    EVICTING,
    READY,
    PageUnitCheckpointStore,
)


def make_store(num_units=20, unit_bytes=10, slot_bytes=25):
    pool = BlockPool(num_units)
    return pool, PageUnitCheckpointStore(
        pool, unit_bytes=unit_bytes, slot_bytes=slot_bytes, layout_id="layout-v1"
    )


def ready(store, prefix_hash, src_slot=0):
    op = store.begin_store(prefix_hash, boundary_blocks=7, src_slot=src_slot)
    assert op is not None
    assert store.record(op.checkpoint_id).state == COPYING
    store.complete_inflight()
    assert store.record(op.checkpoint_id).state == READY
    return op


def test_copying_is_not_hash_visible_and_ready_is():
    pool, store = make_store()
    op = store.begin_store(101, boundary_blocks=7, src_slot=3)

    assert op is not None
    assert len(op.unit_ids) == 3
    assert op.last_unit_valid_bytes == 5
    assert store.lookup(101) == -1
    assert pool.num_free == 17

    store.complete_inflight()
    assert store.lookup(101) == op.checkpoint_id


def test_multiple_restore_readers_pin_the_whole_record():
    pool, store = make_store()
    op = ready(store, 101)
    assert store.begin_restore(101, dst_slot=4) is not None
    assert store.begin_restore(101, dst_slot=8) is not None
    assert store.record(op.checkpoint_id).pin_count == 2

    store.unindex(101)
    assert store.lookup(101) == -1
    assert store.record(op.checkpoint_id).state == EVICTING
    assert pool.num_free == 17

    # Both readers rode the same batch; no fragment is returned early.
    store.complete_inflight()
    assert op.checkpoint_id not in store.records
    assert pool.num_free == 20


def test_lru_eviction_releases_one_complete_image():
    pool, store = make_store(num_units=7)
    first = ready(store, 101)
    second = ready(store, 202)
    assert pool.num_free == 1

    third = store.begin_store(303, boundary_blocks=9, src_slot=2)
    assert third is not None
    assert store.lookup(101) == -1
    assert store.lookup(202) == second.checkpoint_id
    assert first.checkpoint_id not in store.records
    assert store.evictions == 1
    assert len(third.unit_ids) == 3
    assert pool.num_free == 1


def test_unindex_during_copy_waits_for_the_queued_writer():
    pool, store = make_store()
    op = store.begin_store(101, boundary_blocks=7, src_slot=3)
    store.unindex(101)
    assert store.record(op.checkpoint_id).state == EVICTING
    assert pool.num_free == 17

    store.complete_inflight()
    assert op.checkpoint_id not in store.records
    assert pool.num_free == 20


def test_protected_hit_is_excluded_from_admission_reclaim():
    pool, store = make_store(num_units=6)
    ready(store, 101)
    assert pool.num_free == 3
    assert store.has_available_units(6)
    assert not store.has_available_units(6, protected_hash=101)


def test_clear_releases_ready_images_but_defers_a_pinned_reader():
    pool, store = make_store()
    first = ready(store, 101)
    second = ready(store, 202)
    store.begin_restore(202, dst_slot=4)

    store.clear()
    assert store.lookup(101) == store.lookup(202) == -1
    assert first.checkpoint_id not in store.records
    assert second.checkpoint_id in store.records

    store.complete_inflight()
    assert not store.records
    assert pool.num_free == 20
