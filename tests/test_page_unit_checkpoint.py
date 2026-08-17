# SPDX-License-Identifier: MIT

"""Control-plane invariants for PAGE-backed state checkpoint images."""

import pickle
from dataclasses import FrozenInstanceError

import pytest

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    COPYING,
    EVICTING,
    READY,
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
)


def make_store(num_units=20, unit_bytes=10, slot_bytes=25):
    pool = BlockPool(num_units)
    return pool, PageUnitCheckpointStore(
        pool,
        PagedStateCheckpointSpec(
            page_unit_bytes=unit_bytes,
            slot_bytes=slot_bytes,
            layout_id="layout-v1",
            image_bytes=slot_bytes,
        ),
    )


def ready(store, prefix_hash, src_slot=0):
    op = store.begin_store(prefix_hash, src_slot=src_slot)
    assert op is not None
    checkpoint_id = next(
        cid
        for cid, record in store.records.items()
        if record.prefix_hash == prefix_hash
    )
    assert store.records[checkpoint_id].state == COPYING
    store.complete_inflight()
    assert store.records[checkpoint_id].state == READY
    return checkpoint_id, op


def test_runtime_spec_derives_units_and_has_a_minimal_wire_form():
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)

    assert spec.units_per_checkpoint == 3
    assert spec.to_wire() == {
        "page_unit_bytes": 10,
        "slot_bytes": 25,
        "image_bytes": 25,
        "layout_id": "layout-v1",
    }
    assert "units_per_checkpoint" not in spec.to_wire()
    assert (
        PagedStateCheckpointSpec.from_wire(pickle.loads(pickle.dumps(spec.to_wire())))
        == spec
    )
    with pytest.raises(FrozenInstanceError):
        spec.slot_bytes = 30


def test_units_are_priced_off_the_image_not_the_whole_slot():
    """An image holds part of a slot, so that part is what has to fit."""
    whole = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    narrowed = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=11)

    assert whole.units_per_checkpoint == 3
    assert narrowed.units_per_checkpoint == 2


@pytest.mark.parametrize(
    "args",
    [
        (0, 25, "layout-v1", 25),
        (10, -1, "layout-v1", 25),
        (10, 25, "", 25),
        (10, 25, "layout-v1", 0),
        # An image cannot hold more than the slot it was taken from.
        (10, 25, "layout-v1", 26),
    ],
)
def test_runtime_spec_rejects_invalid_geometry(args):
    with pytest.raises(ValueError):
        PagedStateCheckpointSpec(*args)


def test_runtime_spec_rejects_a_drifted_wire_shape():
    with pytest.raises(ValueError, match="fields"):
        PagedStateCheckpointSpec.from_wire(
            {
                "page_unit_bytes": 10,
                "slot_bytes": 25,
                "units_per_checkpoint": 3,
                "layout_id": "layout-v1",
            }
        )


def test_copying_is_not_hash_visible_and_ready_is():
    pool, store = make_store()
    op = store.begin_store(101, src_slot=3)

    assert op is not None
    assert len(op.unit_ids) == 3
    assert op.total_bytes == 25
    assert store.lookup(101) == -1
    assert pool.num_free == 17

    store.complete_inflight()
    assert store.lookup(101) >= 0


def test_multiple_restore_readers_pin_the_whole_record():
    pool, store = make_store()
    checkpoint_id, _ = ready(store, 101)
    assert store.begin_restore(101, dst_slot=4) is not None
    assert store.begin_restore(101, dst_slot=8) is not None
    assert store.records[checkpoint_id].pin_count == 2

    store.unindex(101)
    assert store.lookup(101) == -1
    assert store.records[checkpoint_id].state == EVICTING
    assert pool.num_free == 17

    restores = store.take_restore_ops()
    assert {op.dst_slot for op in restores} == {4, 8}
    store.complete_inflight()
    assert checkpoint_id not in store.records
    assert pool.num_free == 20


def test_empty_batch_does_not_complete_a_queued_restore():
    pool = BlockPool(20)
    coordinator = PagedStateCheckpointCoordinator(
        pool,
        PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25),
        enabled=True,
    )
    checkpoint_id, _ = ready(coordinator.store, 101)
    assert coordinator.begin_restore(101, dst_slot=4)

    coordinator.complete_previous_batch()
    assert coordinator.store.records[checkpoint_id].pin_count == 1

    _, restores = coordinator.take_checkpoint_ops()
    assert len(restores) == 1
    coordinator.complete_previous_batch()
    assert coordinator.store.records[checkpoint_id].pin_count == 0


def test_cancel_queued_restore_drops_its_op_and_pin():
    pool, store = make_store()
    checkpoint_id, _ = ready(store, 101)
    assert store.begin_restore(101, dst_slot=4) is not None
    store.unindex(101)

    store.cancel_queued_restore(4)

    assert store.take_restore_ops() == ()
    assert checkpoint_id not in store.records
    assert pool.num_free == 20


def test_lru_eviction_releases_one_complete_image():
    pool, store = make_store(num_units=7)
    first_id, _ = ready(store, 101)
    second_id, _ = ready(store, 202)
    assert pool.num_free == 1

    third = store.begin_store(303, src_slot=2)
    assert third is not None
    assert store.lookup(101) == -1
    assert store.lookup(202) == second_id
    assert first_id not in store.records
    assert store.evictions == 1
    assert len(third.unit_ids) == 3
    assert pool.num_free == 1


def test_unindex_during_copy_waits_for_the_queued_writer():
    pool, store = make_store()
    assert store.begin_store(101, src_slot=3) is not None
    checkpoint_id = next(iter(store.records))
    store.unindex(101)
    assert store.records[checkpoint_id].state == EVICTING
    assert pool.num_free == 17

    store.complete_inflight()
    assert checkpoint_id not in store.records
    assert pool.num_free == 20


def test_protected_hit_is_excluded_from_admission_reclaim():
    pool, store = make_store(num_units=6)
    ready(store, 101)
    assert pool.num_free == 3
    assert store.has_available_units(6)
    assert not store.has_available_units(6, protected_hash=101)


def test_clear_releases_ready_images_but_defers_a_pinned_reader():
    pool, store = make_store()
    first_id, _ = ready(store, 101)
    second_id, _ = ready(store, 202)
    store.begin_restore(202, dst_slot=4)

    store.clear()
    assert store.lookup(101) == store.lookup(202) == -1
    assert first_id not in store.records
    assert second_id in store.records

    assert len(store.take_restore_ops()) == 1
    store.complete_inflight()
    assert not store.records
    assert pool.num_free == 20


def test_a_store_leaves_the_reserve_for_live_kv():
    """A checkpoint is best-effort; the KV block after it is not.

    `_fresh_block` raises when the pool is dry and nothing is evictable, so a
    store that took the last units would turn "no checkpoint this time" into a
    crash. Dropping it is the same answer the pool already gives when nothing
    can be evicted, one step earlier — and the reason `checkpoints_dropped`
    exists.
    """
    reserve = 12
    pool = BlockPool(20)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=reserve)

    stored = 0
    for prefix_hash in range(100, 110):
        if store.begin_store(prefix_hash, src_slot=0) is None:
            break
        stored += 1
        assert pool.num_free >= reserve, "a store dipped under the floor"
    else:
        raise AssertionError("the floor never stopped a store")

    assert stored, "the floor stopped the very first store"
    # Every one of them is COPYING, so nothing above can be reclaimed either:
    # the floor is all live KV has, and it is still there.
    assert pool.num_free >= reserve
    assert store.ensure_free_units(reserve)


def test_an_unevictable_cache_empties_the_pool_without_a_reserve():
    """The shape the floor exists to keep the pool out of.

    A checkpoint is unevictable while `COPYING` or while a restore pins it.
    Either way a full free list plus a fully unevictable cache leaves live KV
    with nothing, which is what `_fresh_block` raises on. Stores are the
    quicker way to write it down; the reachable one is pins, below.
    """
    pool = BlockPool(6)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=0)

    assert store.begin_store(101, src_slot=0) is not None
    assert store.begin_store(202, src_slot=1) is not None

    # Both are COPYING, so `ensure_free_units` has no victim and a live block
    # request now fails where a reserve would have kept one in hand.
    assert pool.num_free == 0
    assert not store.ensure_free_units(1)


def test_a_dropped_store_evicts_nothing():
    """The floor is a gate, not a target to evict towards.

    `ensure_free_units` gives up only after it has evicted every checkpoint
    it can, so asking it for `needed + reserve` when live KV holds the rest of
    the pool used to empty the entire cache and still refuse — every batch,
    for as long as the pressure lasted. The units it freed did go to live KV,
    but `_fresh_block` already takes those on demand, one at a time, when they
    are actually needed.
    """
    pool = BlockPool(100)
    spec = PagedStateCheckpointSpec(10, 10, "layout-v1", image_bytes=10)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=50)
    for prefix_hash in range(50):
        assert store.begin_store(prefix_hash, src_slot=0) is not None
    store.complete_inflight()
    # Live KV takes everything the checkpoints left, which is the state the
    # floor can no longer be reached from.
    pool.reserve_units(pool.num_free, ("live-kv", 0))

    assert store.begin_store(999, src_slot=0) is None
    assert store.evictions == 0, "a dropped store evicted"
    assert len(store.records) == 50, "a dropped store cost the cache"


def test_a_store_still_recycles_the_oldest_checkpoint():
    """The floor gates a store; it does not stop the LRU doing its job."""
    reserve = 10
    pool = BlockPool(100)
    spec = PagedStateCheckpointSpec(10, 10, "layout-v1", image_bytes=10)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=reserve)
    for prefix_hash in range(100 - reserve):
        assert store.begin_store(prefix_hash, src_slot=0) is not None
    store.complete_inflight()
    assert pool.num_free == reserve

    assert store.begin_store(999, src_slot=0) is not None

    assert store.evictions == 1, "the oldest checkpoint was not recycled"
    assert store.lookup(0) < 0, "the victim was not the oldest"
    assert pool.num_free == reserve, "recycling dipped under the floor"


def test_the_floor_survives_a_pass_that_pins_the_whole_cache():
    """What the floor is actually for.

    `BlockManager.allocate` pins a restore and then asks for fresh blocks in
    the same pass, and the pin holds until the next `complete_inflight`. So a
    pass of prefix hits can pin every checkpoint it resumes from and find
    nothing left to evict. A floor of one pass's worth of blocks is what stops
    that pass from ever needing to evict — which is why it is sized against
    the pass, not against the unevictable set.
    """
    reserve = 12
    pool = BlockPool(40)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=reserve)

    stored = []
    for prefix_hash in range(100, 120):
        if store.begin_store(prefix_hash, src_slot=0) is None:
            break
        stored.append(prefix_hash)
    store.complete_inflight()
    assert stored, "nothing was stored, so nothing is being tested"

    # A pass of prefix hits: every checkpoint resumed from, none released
    # until the next `complete_inflight`.
    for slot, prefix_hash in enumerate(stored):
        assert store.begin_restore(prefix_hash, dst_slot=slot) is not None
    assert store.reclaimable_units() == 0, "the cache is meant to be pinned here"

    # The pass can still take its blocks, which is the whole point.
    assert store.ensure_free_units(reserve), "live KV was starved by its own hits"
    assert pool.num_free >= reserve


def test_the_reserve_does_not_block_a_restore():
    """Only new images are rationed; reading one back takes no units."""
    pool = BlockPool(20)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    store = PageUnitCheckpointStore(pool, spec, reserve_units=12)
    ready(store, 101)

    assert store.begin_restore(101, dst_slot=4) is not None


def test_a_negative_reserve_is_refused():
    pool = BlockPool(20)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)

    with pytest.raises(ValueError, match="reserve_units"):
        PageUnitCheckpointStore(pool, spec, reserve_units=-1)
