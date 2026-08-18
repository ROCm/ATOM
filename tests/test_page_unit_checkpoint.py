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


def _filled(num_units, unit_bytes, image_bytes, count):
    """A store holding `count` READY checkpoints, oldest first."""
    pool = BlockPool(num_units)
    store = PageUnitCheckpointStore(
        pool,
        PagedStateCheckpointSpec(
            page_unit_bytes=unit_bytes,
            slot_bytes=image_bytes,
            layout_id="layout-v1",
            image_bytes=image_bytes,
        ),
    )
    for prefix_hash in range(count):
        assert store.begin_store(prefix_hash, src_slot=0) is not None
    store.complete_inflight()
    return pool, store


def test_a_store_with_free_units_spends_no_checkpoint():
    """Free units first. A store asking for what is already there evicts nothing.

    The cache is not a reservoir a store drains to a level -- it takes its own
    image's worth. This used to be `needed + reserve_units`, which meant an
    accepted store spent tens of checkpoints to build a cushion for live KV
    that live KV never needed.
    """
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=100, count=3)
    assert pool.num_free == 70

    assert store.begin_store(999, src_slot=0) is not None

    assert store.evictions == 0, "a store with 70 free units spent a checkpoint"
    assert len(store.records) == 4, "the cache lost an entry it did not have to"


def test_a_store_spends_only_the_shortfall():
    """Short by half an image: one checkpoint covers it, and only one goes."""
    pool, store = _filled(num_units=35, unit_bytes=10, image_bytes=100, count=3)
    assert pool.num_free == 5, "the pool is meant to be short by half an image"

    assert store.begin_store(999, src_slot=0) is not None

    assert store.evictions == 1, "the shortfall cost more than one checkpoint"
    assert store.lookup(0) < 0, "the victim was not the oldest"
    assert store.lookup(1) >= 0 and store.lookup(2) >= 0


def test_a_dropped_store_evicts_nothing():
    """A store that cannot get its units has to cost nothing.

    `ensure_free_units` gives up only after it has evicted everything it can,
    so asking it for units that are not there would destroy the cache on the
    way to refusing. `begin_store` asks whether they are reachable first.
    """
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=10, count=50)
    # Live KV takes every unit the checkpoints left.
    pool.reserve_units(pool.num_free, ("live-kv", 0))
    for record in store.records.values():
        record.pin_count = 1  # every checkpoint is being read, so none is spendable

    assert store.begin_store(999, src_slot=0) is None

    assert store.evictions == 0, "a dropped store evicted"
    assert len(store.records) == 50, "a dropped store cost the cache"


def test_the_eviction_policy_cannot_move_the_gate():
    """Eligibility is shared; order is policy. Only the second one may change.

    `has_available_units` asks whether the eligible set reaches a count, which
    the order it is walked in cannot change -- only how soon the loop gets
    there. Swapping the policy here has to leave every gate answer identical.
    """
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=10, count=6)
    lru_pick = store._next_victim()
    available = [store.has_available_units(n) for n in range(0, 101, 10)]

    def newest_first(protected=-1):
        return next(
            (
                cid
                for cid in reversed(store._lru)
                if store._is_evictable(cid, protected)
            ),
            -1,
        )

    store._next_victim = newest_first

    assert [store.has_available_units(n) for n in range(0, 101, 10)] == available
    assert store._next_victim() != lru_pick, "the policy swap did not take"

    pool.reserve_units(pool.num_free, ("live-kv", 0))
    assert store.ensure_free_units(1)
    assert store.lookup(5) < 0, "the new policy's victim was not spent"
    assert store.lookup(0) >= 0, "the LRU victim was spent under another policy"


def test_a_store_still_recycles_the_oldest_checkpoint():
    """The gate refuses a store; it does not stop the policy doing its job."""
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=10, count=100)
    assert pool.num_free == 0

    assert store.begin_store(999, src_slot=0) is not None

    assert store.evictions == 1
    assert store.lookup(0) < 0, "the victim was not the oldest"


def test_a_restore_takes_no_units():
    """Only new images need units; reading one back does not."""
    pool = BlockPool(20)
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25)
    store = PageUnitCheckpointStore(pool, spec)
    ready(store, 101)
    pool.reserve_units(pool.num_free, ("live-kv", 0))

    assert store.begin_restore(101, dst_slot=4) is not None


def test_an_unreachable_count_evicts_nothing_whoever_asks():
    """The refusal lives in `ensure_free_units`, not in one of its callers.

    `begin_store` used to carry the reachability test itself, which left
    `BlockManager._ensure_page_units` calling the raw loop -- harmless only
    because its single caller passes 1, where there is nothing to spend before
    giving up. Ask for more than the cache can reach and the bare loop empties
    it and refuses anyway, which is the behaviour 0c46f4ed3 removed from one
    call site and left available at the other.
    """
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=10, count=50)
    pool.reserve_units(pool.num_free, ("live-kv", 0))
    assert pool.num_free == 0 and len(store.records) == 50

    # 50 spendable units against a request for 60: unreachable, and reachable
    # only after spending every one of them.
    assert not store.ensure_free_units(60)

    assert store.evictions == 0, "a refused request emptied the cache"
    assert len(store.records) == 50


def test_a_store_refuses_when_the_policy_leaves_the_loop_short():
    """`begin_store` reads the answer rather than assuming it.

    `_next_victim` exists to be replaced. A policy that passes over an
    eligible checkpoint makes the loop end short of `count`, and a
    `begin_store` that assumed success would take an identity for a store that
    cannot happen -- the record is safe only because `pool.reserve_units`
    happens to refuse second.
    """
    pool, store = _filled(num_units=100, unit_bytes=10, image_bytes=10, count=100)
    assert pool.num_free == 0
    store._next_victim = lambda protected=-1: -1  # a policy that spends nothing
    before = store._next_checkpoint_id

    assert store.begin_store(999, src_slot=0) is None

    assert store._next_checkpoint_id == before, "a refused store took an identity"
    assert len(store.records) == 100
