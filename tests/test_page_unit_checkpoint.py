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
    spec = PagedStateCheckpointSpec(10, 25, "layout-v1")

    assert spec.units_per_checkpoint == 3
    assert spec.to_wire() == {
        "page_unit_bytes": 10,
        "slot_bytes": 25,
        "layout_id": "layout-v1",
    }
    assert "units_per_checkpoint" not in spec.to_wire()
    assert (
        PagedStateCheckpointSpec.from_wire(pickle.loads(pickle.dumps(spec.to_wire())))
        == spec
    )
    with pytest.raises(FrozenInstanceError):
        spec.slot_bytes = 30


@pytest.mark.parametrize(
    "args",
    [
        (0, 25, "layout-v1"),
        (10, -1, "layout-v1"),
        (10, 25, ""),
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

    # Both readers rode the same batch; no fragment is returned early.
    store.complete_inflight()
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

    store.complete_inflight()
    assert not store.records
    assert pool.num_free == 20
