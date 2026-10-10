# SPDX-License-Identifier: MIT
"""CPU control-plane tests for K3's optional borrowed STATE checkpoint tier."""

import pickle
from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import StateSlotSource
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    COPYING,
    READY,
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
)
from atom.model_engine.state_pool import StateSlotPool


def make_store(n=6, reserve=1, units=20, offload=False):
    slots = StateSlotPool(n)
    producer = slots.pop()
    pages = BlockPool(units)
    spec = PagedStateCheckpointSpec(10, 25, "k3-test", image_bytes=25)
    store = PageUnitCheckpointStore(pages, spec, offload_sink=offload)
    store.attach_slots(slots, reserve)
    return slots, pages, store, producer


def save(store, h, producer):
    op = store.begin_store(h, producer)
    assert op is not None
    store.complete_inflight()
    return op


def test_c48_actual20_uses_68_slots_without_changing_page_budget():
    slots, pages, store, producer = make_store(n=96, reserve=8)
    active = [producer] + slots.pop_many(19)
    for h in range(68):
        op = save(store, h, producer)
        assert op.dst_slot not in active and not op.unit_ids
    assert pages.num_free == 20
    assert slots.occupancy() == {
        "slots_total": 96,
        "slots_used": 20,
        "slots_held": 68,
        "slots_vacant": 8,
    }
    generation = store.generation
    pages.reserve_units(pages.num_free, "live-kv")
    save(store, 1000, producer)
    assert not store.contains(0) and store.contains(1000)
    assert store.changed_since(generation) == {0, 1000}
    # Admission spends the eight vacant slots, then an old READY checkpoint.
    new_active = slots.pop_many(9)
    assert not set(active) & set(new_active)
    assert store.slot_evictions == 2
    assert len(store.records) == 67
    assert len(slots._free) == 67


def test_copy_is_invisible_and_destination_unavailable_until_completion():
    slots, pages, store, producer = make_store(n=4, reserve=1)
    op = store.begin_store(11, producer)
    assert op.slot_pair == (producer, op.dst_slot)
    assert not store.contains(11)
    assert not slots.is_free(op.dst_slot)
    assert next(iter(store.records.values())).state == COPYING
    other = slots.pop_many(slots.num_free())
    assert op.dst_slot not in other
    store.complete_inflight()
    assert store.contains(11) and slots.is_free(op.dst_slot)
    assert pages.num_free == 20
    assert pickle.loads(pickle.dumps(op)) == op


def test_reserve_and_pending_copies_fall_back_to_page():
    slots, pages, store, producer = make_store(n=4, reserve=1)
    a = store.begin_store(11, producer)
    b = store.begin_store(22, producer)
    c = store.begin_store(33, producer)
    assert a.dst_slot >= 0 and b.dst_slot >= 0
    assert c.dst_slot == -1 and len(c.unit_ids) == 3
    assert slots.num_free() == 1 and pages.num_free == 17
    store.complete_inflight()
    assert all(store.contains(h) for h in (11, 22, 33))
    # PAGE pressure may spend the PAGE image, never a useless zero-unit slot.
    pages.reserve_units(pages.num_free, "live")
    assert store.ensure_free_units(3)
    assert not store.contains(33)
    assert store.contains(11) and store.contains(22)


def test_multiple_readers_and_cancel_keep_the_snapshot_until_last_reader():
    slots, _, store, producer = make_store()
    op = save(store, 11, producer)
    a = store.allocate_slot_restore(11)
    b = store.allocate_slot_restore(11)
    cid = store.lookup(11)
    assert len({a, b, op.dst_slot, producer}) == 4
    assert store.records[cid].pin_count == 2
    assert not slots.is_free(op.dst_slot)
    store.cancel_queued_restore(a)
    slots.release(a)
    assert store.records[cid].pin_count == 1
    restores = store.take_restore_ops()
    assert len(restores) == 1
    assert restores[0].slot_pair == (op.dst_slot, b)
    assert pickle.loads(pickle.dumps(restores)) == restores
    store.complete_inflight()
    assert store.records[cid].pin_count == 0
    assert slots.is_free(op.dst_slot)
    assert store.contains(11)


def test_suspended_restore_retains_its_source_across_other_batches():
    slots, _, store, producer = make_store()
    op = save(store, 11, producer)
    dst = store.allocate_slot_restore(11)
    suspended = store.suspend_queued_restore(dst)
    assert suspended is not None
    store.complete_inflight()
    assert not slots.is_free(op.dst_slot)
    assert not store.restore_queued_for(dst)
    store.resume_suspended_restore(suspended)
    assert store.restore_queued_for(dst)
    store.take_restore_ops()
    store.complete_inflight()
    assert slots.is_free(op.dst_slot)


def test_last_free_source_is_adopted_without_copy_or_losing_admission():
    slots, _, store, producer = make_store(n=2, reserve=0)
    op = save(store, 11, producer)
    dst = store.allocate_slot_restore(11)
    assert dst == op.dst_slot
    assert not store.contains(11) and not store.records
    assert not slots.has_free()
    assert not store.take_restore_ops()
    assert store.slot_adoptions == 1
    slots.release(dst)
    assert slots.occupancy()["slots_held"] == 0


@pytest.mark.parametrize("invalidate", ["clear", "unindex"])
def test_reset_or_orphan_during_copy_cannot_publish_or_reuse_old_bytes(invalidate):
    slots, _, store, producer = make_store()
    old = store.begin_store(11, producer)
    if invalidate == "clear":
        store.clear()
    else:
        store.unindex(11)
    assert not slots.is_free(old.dst_slot)
    new = store.begin_store(11, producer)
    assert new.dst_slot != old.dst_slot
    store.complete_inflight()
    assert slots.is_free(old.dst_slot)
    assert store.records[store.lookup(11)].slot_id == new.dst_slot
    assert len(store.records) == 1


def test_orphaned_source_stays_pinned_until_both_restore_and_cpu_read_finish():
    slots, _, store, producer = make_store(offload=True)
    snap = save(store, 11, producer)
    dst = store.allocate_slot_restore(11)
    ((operation, source),) = store.take_offload_stores(8)
    assert source == StateSlotSource(snap.dst_slot)
    store.unindex(11)
    assert not slots.is_free(snap.dst_slot)
    store.take_restore_ops()
    store.complete_inflight()
    assert not slots.is_free(snap.dst_slot)
    store.release_offload_store_source(operation)
    assert slots.is_free(snap.dst_slot)
    assert store.has_offload_pins()  # CPU put still has to report its result.
    store.settle_offload_store(operation)
    assert not store.has_offload_pins() and not store.records
    assert not slots.is_free(dst)


def test_timeout_never_recycles_a_slot_that_cpu_may_still_read():
    slots, _, store, producer = make_store(offload=True)
    snap = save(store, 11, producer)
    ((operation, _),) = store.take_offload_stores(8)
    store._offload_pins[operation].pinned_at -= 1000
    assert store.reclaim_stale_offload_pins(1) == 0
    assert not slots.is_free(snap.dst_slot)
    store.settle_offload_store(operation)  # terminal failure also unpins
    assert slots.is_free(snap.dst_slot)


def test_cpu_slot_pin_budget_defers_instead_of_dropping_nominations():
    slots, _, store, producer = make_store(n=8, reserve=1, offload=True)
    save(store, 11, producer)
    save(store, 22, producer)
    first = store.take_offload_stores(8)
    assert len(first) == 1  # slot pin limit follows reserve, independently of PAGE
    assert len(store.store_backlog) == 1
    store.release_offload_store_source(first[0][0])
    second = store.take_offload_stores(8)
    assert len(second) == 1 and not store.store_backlog
    for operation, _ in first + second:
        store.settle_offload_store(operation)
    assert slots.occupancy()["slots_used"] == 1


def test_late_cpu_report_cannot_unpin_new_generation_of_same_hash():
    slots, _, store, producer = make_store(offload=True)
    save(store, 11, producer)
    ((old, _),) = store.take_offload_stores(8)
    store.release_offload_store_source(old)
    store.unindex(11)
    snap = save(store, 11, producer)
    assert not store.take_offload_stores(8)  # old put still outstanding
    store.settle_offload_store(old)
    ((new, _),) = store.take_offload_stores(8)
    assert old != new
    store.settle_offload_store(old)
    store.release_offload_store_source(old)
    assert not slots.is_free(snap.dst_slot)
    store.settle_offload_store(new)
    assert slots.is_free(snap.dst_slot)


def test_slot_page_cpu_are_alternative_sources_for_one_state_gate():
    _slots, pages, store, producer = make_store(n=3, reserve=1)
    slot_op = store.begin_store(11, producer)
    page_op = store.begin_store(22, producer)
    assert slot_op.dst_slot >= 0 and page_op.unit_ids
    store.complete_inflight()
    coordinator = PagedStateCheckpointCoordinator(pages, store.spec, enabled=True)
    coordinator.store = store
    coordinator.offload = SimpleNamespace(could_serve=lambda h: h == 33)
    seq = SimpleNamespace(has_per_req_cache=True)
    assert coordinator.resumable_hit(seq, 3, [11, 22, 33]) == 3
    assert coordinator.resumable_hit(seq, 2, [11, 22, 33]) == 2
    assert coordinator.resumable_hit(seq, 1, [11, 22, 33]) == 1
    store.unindex(22)
    assert coordinator.resumable_hit(seq, 2, [11, 22, 33]) == 1
    assert coordinator.acquire_checkpoint_source(11) is None  # PAGE-only P/D API


def test_restoring_a_cold_slot_refreshes_its_admission_lru_order():
    slots, _, store, producer = make_store(n=4, reserve=1)
    save(store, 11, producer)
    save(store, 22, producer)
    dst = store.allocate_slot_restore(11)
    store.take_restore_ops()
    store.complete_inflight()
    # All vacant slots are now owned by producer and reader.
    assert slots.pop() != dst
    assert store.contains(11) and not store.contains(22)


def test_slot_source_is_typed_and_survives_worker_serialization():
    source = StateSlotSource(2)
    assert pickle.loads(pickle.dumps(source)) == source
    with pytest.raises(ValueError):
        StateSlotSource(-1)
    with pytest.raises(TypeError):
        StateSlotSource(True)


def test_full_slot_tier_uses_available_page_space_before_evicting_images():
    _slots, pages, store, producer = make_store(n=3, reserve=1)
    save(store, 11, producer)
    second = save(store, 22, producer)
    assert second.unit_ids and second.dst_slot == -1
    assert store.contains(11) and store.contains(22)
    assert store.evictions == 0 and pages.num_free == 17


def make_manager(monkeypatch, **overrides):
    from conftest import MockConfig

    from atom.model_engine.block_manager import BlockManager
    from atom.model_engine.state_runtime import StateRuntime, StateTransfer

    monkeypatch.setenv("ATOM_KDA_SPARE_STATE_CHECKPOINTS", "1")
    monkeypatch.setenv("ATOM_ENABLE_REPLAYSSM", "0")
    monkeypatch.setenv("ATOM_KDA_SPARE_STATE_RESERVE", "1")
    config = {
        "hf_config": SimpleNamespace(model_type="kimi_linear"),
        "enable_prefix_caching": True,
        "pool_entries": {"state": 6},
        "num_kvcache_blocks": 100,
        "state_checkpoint_interval_tokens": -1,
        "state_checkpoint_demand": True,
    }
    config.update(overrides)
    spec = PagedStateCheckpointSpec(10, 25, "k3-test", image_bytes=25)
    return BlockManager(
        MockConfig(**config),
        state_runtime=StateRuntime(StateTransfer.copy(spec.layout_id), spec),
    )


def publish_anchor(bm):
    from atom.model_engine.sequence import Sequence

    seq = Sequence(list(range(40)), 4, has_per_req_cache=True)
    assert bm.allocate(seq, bm.can_allocate(seq))
    boundary = seq.checkpoint_end_pos
    assert boundary > 0
    bm.hash_blocks(seq, boundary)
    h = bm.kv.block(seq.block_table[boundary // bm.hash_block_size - 1]).hash
    ops = bm.take_state_maintenance_ops()
    assert len(ops.checkpoint_stores) == 1
    assert ops.checkpoint_stores[0].dst_slot >= 0
    bm.complete_previous_state_batch()
    return seq, h, boundary


def test_block_manager_admission_preserves_exact_prefix_and_shared_slot(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_manager(monkeypatch)
    producer, h, boundary = publish_anchor(bm)
    bm.deallocate(producer)
    readers = [Sequence(list(range(40)), 4, has_per_req_cache=True) for _ in range(2)]
    for reader in readers:
        hit = bm.can_allocate(reader)
        assert hit == boundary // bm.hash_block_size
        assert bm.allocate(reader, hit)
        assert reader.num_cached_tokens == boundary
    copies = bm.take_state_maintenance_ops().checkpoint_restores
    assert len(copies) == 2 and copies[0].src_slot == copies[1].src_slot
    assert copies[0].dst_slot != copies[1].dst_slot
    bm.complete_previous_state_batch()
    assert bm.paged_state_checkpoints.store.contains(h)
    assert len(bm.state_caches) == 1
    bm._record_evicted(h)
    assert not bm.paged_state_checkpoints.store.contains(h)
    assert all(
        bm.can_allocate(Sequence(list(range(40)), 4, has_per_req_cache=True)) == 0
        for _ in range(2)
    )


def test_last_slot_adoption_secures_joint_cpu_kv_boundary(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_manager(monkeypatch, pool_entries={"state": 3})
    _producer, h, boundary = publish_anchor(bm)
    bm.state.pop()  # only the READY checkpoint remains free
    reader = Sequence(list(range(40)), 4, has_per_req_cache=True)
    hit = bm.can_allocate(reader)
    reader.offload_joint.boundary_hash = h
    reader.offload_joint.boundary_tokens = boundary
    assert bm.allocate(reader, hit)
    assert reader.num_cached_tokens == boundary
    assert reader.offload_joint.boundary_hash == h
    assert bm._state_leg_secured(reader)
    assert bm.paged_state_checkpoints.store.slot_adoptions == 1
    assert not bm.take_state_maintenance_ops().checkpoint_restores


def test_spare_slot_room_accepts_demand_without_spending_live_kv(monkeypatch):
    bm = make_manager(monkeypatch, num_kvcache_blocks=1)
    assert bm._checkpoint_has_room(live_blocks=1)
    assert not bm._checkpoint_has_room(live_blocks=2)


@pytest.mark.parametrize(
    "overrides,env",
    [
        ({"speculative_config": SimpleNamespace(num_speculative_tokens=1)}, {}),
        ({}, {"ATOM_ENABLE_REPLAYSSM": "1"}),
        ({"pipeline_parallel_size": 2}, {}),
        ({"enable_prefix_caching": False}, {}),
        ({"hf_config": SimpleNamespace(model_type="deepseek_v4")}, {}),
        (
            {
                "kv_transfer_config": {
                    "kv_connector": "mooncake",
                    "kv_role": "kv_producer",
                }
            },
            {},
        ),
    ],
)
def test_unsupported_configurations_fail_explicitly(monkeypatch, overrides, env):
    # Constructor helper sets baseline env first; overriding env in envs itself
    # keeps this check independent of module-level caching behavior.
    from atom.utils import envs

    for key, value in env.items():
        monkeypatch.setattr(envs, key, value == "1")
    with pytest.raises(ValueError, match="spare STATE checkpoints require"):
        make_manager(monkeypatch, **overrides)


def test_negative_reserve_is_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        make_store(reserve=-1)


def test_pool_ownership_invariants_under_cache_churn_and_admission():
    import random

    rng = random.Random(42)
    slots, pages, store, producer = make_store(n=9, reserve=2, units=9, offload=True)
    active = {producer}
    receipts = []
    for _ in range(800):
        action = rng.randrange(8)
        if action == 0:
            store.begin_store(rng.randrange(12), producer)
        elif action == 1:
            store.take_restore_ops()
            store.complete_inflight()
        elif action == 2 and slots.has_free():
            active.add(slots.pop())
        elif action == 3 and len(active) > 1:
            # A destination is not released while a dispatched copy can write it.
            store.take_restore_ops()
            store.complete_inflight()
            slot = rng.choice(sorted(active - {producer}))
            store.cancel_queued_restore(slot)
            active.remove(slot)
            slots.release(slot)
        elif action == 4:
            store.unindex(rng.randrange(12))
        elif action == 5:
            receipts.extend(store.take_offload_stores(4))
        elif action == 6 and receipts:
            operation, _ = receipts.pop(rng.randrange(len(receipts)))
            store.release_offload_store_source(operation)
            store.settle_offload_store(operation)
        elif slots.has_free():
            slot = store.allocate_slot_restore(rng.randrange(12))
            if slot >= 0:
                active.add(slot)

        images = {r.slot_id for r in store.records.values() if r.slot_id >= 0}
        assert not images & active
        assert len(images) == sum(r.slot_id >= 0 for r in store.records.values())
        reusable = {
            r.slot_id
            for r in store.records.values()
            if r.slot_id >= 0 and r.state == READY and r.pin_count == 0
        }
        assert set(slots._borrowed) == reusable
        assert slots._free == set(range(slots.num_slots)) - active - (images - reusable)
        assert all(
            store.records[cid].state == READY and store.records[cid].prefix_hash == h
            for h, cid in store.hash_to_checkpoint.items()
        )
        assert (
            pages.num_free + sum(len(r.unit_ids) for r in store.records.values()) == 9
        )
    store.clear()
    store.take_restore_ops()
    store.complete_inflight()
    for operation, _ in receipts:
        store.settle_offload_store(operation)
    assert not store.records
    assert slots.num_free() + len(active) == slots.num_slots


def test_demand_creates_a_missing_branch_checkpoint_in_a_spare_slot(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_manager(monkeypatch)
    producer, _, _ = publish_anchor(bm)
    bm.deallocate(producer)
    sibling = Sequence(
        list(range(16)) + list(range(100, 124)), 4, has_per_req_cache=True
    )
    hit = bm.can_allocate(sibling)
    assert hit == 0  # MLA shares 16 tokens, but the old state is only at 36.
    assert bm.allocate(sibling, hit)
    assert sibling.checkpoint_demand_pos == 16
    assert bm.checkpoint_cut(sibling, 0, 32) == 16
    bm.hash_blocks(sibling, 16)
    ops = bm.take_state_maintenance_ops()
    assert len(ops.checkpoint_stores) == 1
    assert ops.checkpoint_stores[0].dst_slot >= 0
    bm.complete_previous_state_batch()
    next_sibling = Sequence(
        list(range(16)) + list(range(200, 224)), 4, has_per_req_cache=True
    )
    assert bm.can_allocate(next_sibling) == 4
    assert bm.allocate(next_sibling, 4)
    assert next_sibling.num_cached_tokens == 16
    assert bm.take_state_maintenance_ops().checkpoint_restores[0].src_slot >= 0


def make_dspark_manager(monkeypatch, **overrides):
    config = {
        "speculative_config": SimpleNamespace(
            num_speculative_tokens=3, use_dspark=lambda: True
        ),
        "pool_entries": {"state": 20},
        "pool_entries_per_req": {"state": 4},
    }
    config.update(overrides)
    return make_manager(monkeypatch, **config)


@pytest.mark.parametrize("role", ["offload", "kv_both"])
def test_dspark_prefill_restore_allocates_private_rollback_sets(monkeypatch, role):
    from atom.model_engine.sequence import Sequence

    bm = make_dspark_manager(
        monkeypatch,
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": role},
    )
    producer, h, boundary = publish_anchor(bm)
    assert len(producer.state_slots) == 4
    bm.deallocate(producer)
    readers = [Sequence(list(range(40)), 4, has_per_req_cache=True) for _ in range(2)]
    for reader in readers:
        assert bm.allocate(reader, bm.can_allocate(reader))
        assert reader.num_cached_tokens == boundary
        assert len(set(reader.state_slots)) == 4
    assert not set(readers[0].state_slots) & set(readers[1].state_slots)
    copies = bm.take_state_maintenance_ops().checkpoint_restores
    assert len(copies) == 2 and copies[0].src_slot == copies[1].src_slot
    assert copies[0].src_slot not in set().union(*(r.state_slots for r in readers))
    assert [c.dst_slot for c in copies] == [r.state_slot for r in readers]
    bm.complete_previous_state_batch()
    for reader in readers:
        bm.deallocate(reader)
    assert bm.state.num_free() == 20  # the immutable image is reclaimable too
    assert bm.paged_state_checkpoints.store.contains(h)


def test_dspark_adopts_source_when_only_one_full_request_set_fits(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_dspark_manager(monkeypatch, pool_entries={"state": 8})
    producer, h, boundary = publish_anchor(bm)
    assert bm.state.num_free() == 4  # includes the single-slot checkpoint
    reader = Sequence(list(range(40)), 4, has_per_req_cache=True)
    hit = bm.can_allocate(reader)
    reader.offload_joint.boundary_hash = h
    reader.offload_joint.boundary_tokens = boundary
    assert bm.allocate(reader, hit)
    assert reader.num_cached_tokens == boundary
    assert reader.offload_joint.boundary_hash == h
    assert len(reader.state_slots) == 4
    assert not set(reader.state_slots) & set(producer.state_slots)
    assert bm._state_leg_secured(reader)
    assert not bm.take_state_maintenance_ops().checkpoint_restores
    assert bm.paged_state_checkpoints.store.slot_adoptions == 1
    assert bm.state.num_free() == 0
    assert bm.can_allocate(Sequence(list(range(40)), 4, has_per_req_cache=True)) == -1
    bm.deallocate(reader)
    bm.deallocate(producer)
    assert bm.state.num_free() == 8


def test_dspark_cancel_reader_releases_four_slots_and_only_its_source_pin(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_dspark_manager(monkeypatch)
    producer, h, _ = publish_anchor(bm)
    readers = [Sequence(list(range(40)), 4, has_per_req_cache=True) for _ in range(2)]
    for reader in readers:
        assert bm.allocate(reader, bm.can_allocate(reader))
    store = bm.paged_state_checkpoints.store
    record = store.records[store.lookup(h)]
    assert record.pin_count == 2
    free = bm.state.num_free()
    bm.deallocate(readers[0])
    assert bm.state.num_free() == free + 4
    assert record.pin_count == 1 and not bm.state.is_free(record.slot_id)
    assert len(bm.take_state_maintenance_ops().checkpoint_restores) == 1
    bm.complete_previous_state_batch()
    assert record.pin_count == 0 and store.contains(h)
    bm.deallocate(readers[1])
    bm.deallocate(producer)
    assert bm.state.num_free() == 20


@pytest.mark.parametrize("source", ["page", "cpu"])
def test_dspark_other_sources_still_allocate_full_request_set(monkeypatch, source):
    from atom.model_engine.sequence import Sequence

    bm = make_dspark_manager(monkeypatch)
    store = bm.paged_state_checkpoints.store
    h = 123
    if source == "page":
        store.slot_reserve = bm.num_state_slots
        op = store.begin_store(h, bm.state.pop())
        assert op.unit_ids and op.dst_slot == -1
        store.complete_inflight()
    else:
        from atom.model_engine.state_offload import StateOffloadIndex

        bm.state_offload = StateOffloadIndex()
        bm.state_offload.note_stored(h)
    reader = Sequence(list(range(40)), 4, has_per_req_cache=True)
    assert bm._attach_state_slots(reader, h)
    assert len(set(reader.state_slots)) == 4
    if source == "page":
        assert bm.take_state_maintenance_ops().checkpoint_restores[0].src_slot == -1
    else:
        assert bm._state_loads == [(reader.id, h, reader.state_slot)]


def test_dspark_checkpoint_budget_accounts_for_all_four_future_slots(monkeypatch):
    bm = make_dspark_manager(
        monkeypatch, pool_entries={"state": 5}, num_kvcache_blocks=1
    )
    # Five vacant slots fit one four-slot request plus reserve=1, but no image.
    assert not bm._checkpoint_has_room(live_blocks=1)
    bm = make_dspark_manager(
        monkeypatch, pool_entries={"state": 6}, num_kvcache_blocks=1
    )
    assert bm._checkpoint_has_room(live_blocks=1)


def test_dspark_prefill_only_keeps_decode_kv_hashing_without_state_images(monkeypatch):
    from atom.model_engine.sequence import SequenceType

    bm = make_dspark_manager(monkeypatch)
    seq, _, boundary = publish_anchor(bm)
    store = bm.paged_state_checkpoints.store
    images = len(store.records)
    seq.type = SequenceType.DECODE
    assert not bm.checkpointers_at(seq, boundary, 4, aimed=False)
    # Even a future erroneous interval change cannot publish decode states.
    bm.state_checkpoint_interval_tokens = 4
    assert not bm.checkpointers_at(seq, 44, 4, aimed=False)
    assert not bm.checkpointers_at(seq, 44, 4, aimed=True)
    bm.hash_decode_blocks(seq, 40, next_forward_tokens=4)
    assert seq.num_hashed_tokens == 40
    assert not bm.take_state_maintenance_ops().checkpoint_stores
    assert len(store.records) == images


@pytest.mark.parametrize(
    "overrides",
    [
        {"state_checkpoint_interval_tokens": 8192},
        {"state_checkpoint_interval_tokens": 0},
        {"pool_entries_per_req": {"state": 1}},
        {
            "speculative_config": SimpleNamespace(
                num_speculative_tokens=7, use_dspark=lambda: True
            )
        },
        {
            "speculative_config": SimpleNamespace(
                num_speculative_tokens=3, use_dspark=lambda: False
            )
        },
    ],
)
def test_dspark_rejects_unadapted_checkpoint_modes(monkeypatch, overrides):
    with pytest.raises(ValueError, match="prefill checkpoints only"):
        make_dspark_manager(monkeypatch, **overrides)


def test_dspark_c48_capacity_counts_physical_slots(monkeypatch):
    slots, pages, store, producer = make_store(n=384, reserve=8)
    active = {producer, *slots.pop_many(20 * 4 - 1)}
    for h in range(296):
        op = save(store, h, producer)
        assert op.dst_slot not in active and not op.unit_ids
    assert slots.occupancy() == {
        "slots_total": 384,
        "slots_used": 80,
        "slots_held": 296,
        "slots_vacant": 8,
    }
    assert pages.num_free == 20
    # Another full request consumes four vacant slots; the next uses the rest.
    slots.pop_many(4)
    slots.pop_many(4)
    free = slots.num_free()
    dst = store.allocate_slot_restore(0, request_slots=4)
    rollback = slots.pop_many(3)
    assert len({dst, *rollback}) == 4
    assert slots.num_free() == free - 5  # request plus pinned immutable source


def test_dspark_cpu_pinned_source_serves_two_full_requests_at_capacity(monkeypatch):
    from atom.model_engine.sequence import Sequence

    bm = make_dspark_manager(monkeypatch, pool_entries={"state": 13})
    store = bm.paged_state_checkpoints.store
    from atom.model_engine.state_offload import StateOffloadIndex

    bm.paged_state_checkpoints.attach_offload(StateOffloadIndex())
    producer, h, boundary = publish_anchor(bm)
    ((operation, source),) = store.take_offload_stores(8)
    assert isinstance(source, StateSlotSource)
    record = store.records[store.lookup(h)]
    assert record.pin_count == 1 and bm.state.num_free() == 8
    readers = [Sequence(list(range(40)), 4, has_per_req_cache=True) for _ in range(2)]
    for reader in readers:
        assert bm.allocate(reader, bm.can_allocate(reader))
        assert reader.num_cached_tokens == boundary
        assert len(set(reader.state_slots)) == 4
        assert source.slot_id not in reader.state_slots
    assert (
        len(set(producer.state_slots + readers[0].state_slots + readers[1].state_slots))
        == 12
    )
    assert record.pin_count == 3 and bm.state.num_free() == 0
    assert bm.can_allocate(Sequence(list(range(40)), 4, has_per_req_cache=True)) == -1
    bm.take_state_maintenance_ops()
    bm.complete_previous_state_batch()
    assert record.pin_count == 1 and not bm.state.is_free(source.slot_id)
    assert store.slot_adoptions == 0
    store.release_offload_store_source(operation)
    assert bm.state.is_free(source.slot_id) and store.contains(h)
    store.settle_offload_store(operation)
    for seq in [producer, *readers]:
        bm.deallocate(seq)
    assert bm.state.num_free() == 13
