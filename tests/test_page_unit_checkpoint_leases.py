# SPDX-License-Identifier: MIT

"""Native PAGE image leases used by external checkpoint transfers."""

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)


def coordinator(num_units=9):
    return PagedStateCheckpointCoordinator(
        BlockPool(num_units),
        PagedStateCheckpointSpec(10, 50, "lease-layout", image_bytes=25),
        enabled=True,
    )


def ready(c, prefix_hash):
    op = c.store.begin_store(prefix_hash, src_slot=7)
    assert op is not None
    c.complete_previous_batch()
    return op.unit_ids


def test_source_lease_exposes_only_an_existing_exact_ready_image():
    c = coordinator()
    assert c.acquire_checkpoint_source(101) is None
    original = c.store.begin_store(101, src_slot=7)
    assert original is not None
    assert c.acquire_checkpoint_source(101) is None

    c.complete_previous_batch()
    leased = c.acquire_checkpoint_source(101)
    assert leased is not None
    operation, units = leased
    assert operation.prefix_hash == 101
    assert units == original.unit_ids
    assert c.acquire_checkpoint_source(102) is None
    assert c.take_checkpoint_ops() == ((), ())
    assert c.store.pool.num_used == 3
    assert len(c.store.records) == 1
    assert c.checkpoints_kept == 0


def test_source_lease_blocks_eviction_until_reader_release():
    c = coordinator(num_units=3)
    units = ready(c, 101)
    operation, leased_units = c.acquire_checkpoint_source(101)
    assert leased_units == units
    assert not c.has_available_units(1)
    assert c.store.begin_store(202, src_slot=8) is None

    c.unindex(101)
    assert not c.contains(101)
    assert c.store.pool.num_free == 0
    c.release_offload_store_source(operation)
    assert c.store.pool.num_free == 3
    assert c.has_offload_pins()
    c.release_offload_store_source(operation)
    c.settle_offload_store(operation)
    c.settle_offload_store(operation)
    assert c.store.pool.num_free == 3
    assert not c.has_offload_pins()


def test_source_release_keeps_duplicate_guard_until_settlement():
    c = coordinator(num_units=3)
    ready(c, 101)
    first, _ = c.acquire_checkpoint_source(101)
    assert c.acquire_checkpoint_source(101) is None
    c.release_offload_store_source(first)
    assert c.acquire_checkpoint_source(101) is None

    c.settle_offload_store(first)
    second, _ = c.acquire_checkpoint_source(101)
    assert second.prefix_hash == first.prefix_hash
    assert second.generation > first.generation
    c.release_offload_store_source(first)
    c.settle_offload_store(first)
    assert not c.has_available_units(1)
    assert c.has_offload_pins()
    c.settle_offload_store(second)
    assert c.has_available_units(3)


def test_operation_limit_includes_source_released_results_still_pending():
    c = coordinator()
    ready(c, 101)
    ready(c, 202)
    assert c.acquire_checkpoint_source(101, max_inflight=0) is None
    first, _ = c.acquire_checkpoint_source(101, max_inflight=1)
    assert c.acquire_checkpoint_source(202, max_inflight=1) is None
    c.release_offload_store_source(first)
    assert c.acquire_checkpoint_source(202, max_inflight=1) is None
    c.settle_offload_store(first)
    assert c.acquire_checkpoint_source(202, max_inflight=1) is not None


def test_external_source_never_times_out_while_legacy_policy_is_preserved(
    monkeypatch,
):
    monkeypatch.setattr("atom.model_engine.page_unit_checkpoint.monotonic", lambda: 1.0)
    c = coordinator(num_units=6)
    c.store._offload_sink = True
    ready(c, 101)
    ready(c, 202)
    external, _ = c.acquire_checkpoint_source(101)
    [(legacy, _)] = c.take_offload_stores(max_inflight=2)
    assert legacy.prefix_hash == 202
    assert legacy.generation > external.generation

    monkeypatch.setattr(
        "atom.model_engine.page_unit_checkpoint.monotonic", lambda: 1000.0
    )
    assert c.reclaim_stale_offload_pins(timeout_s=1) == 1
    assert c.was_reclaimed(legacy)
    assert not c.was_reclaimed(external)
    c.unindex(101)
    assert c.store.pool.num_free == 0
    assert c.reclaim_stale_offload_pins(timeout_s=1) == 0
    assert c.has_offload_pins()
    c.release_offload_store_source(external)
    assert c.store.pool.num_free == 3
    assert c.reclaim_stale_offload_pins(timeout_s=1) == 0
    assert c.has_offload_pins()
    c.settle_offload_store(external)
    assert not c.has_offload_pins()


def test_source_lease_survives_cache_reset_until_terminal_completion():
    c = coordinator(num_units=3)
    ready(c, 101)
    operation, _ = c.acquire_checkpoint_source(101)
    c.clear_index()
    assert c.store.pool.num_free == 0
    assert c.acquire_checkpoint_source(101) is None
    # A terminal cancellation before DMA starts may settle in one phase.
    c.settle_offload_store(operation)
    assert c.store.pool.num_free == 3


def test_external_load_reserves_raw_image_units_without_publishing_checkpoint():
    c = coordinator(num_units=3)
    owner = ("request-a", 1)
    units = c.reserve_transfer_units(owner)
    assert units is not None and len(units) == 3
    assert c.store.pool.num_free == 0
    assert c.reserve_transfer_units(owner) is None
    assert c.reserve_transfer_units(("request-b", 1)) is None
    assert c.store.records == {}
    assert c.take_checkpoint_ops() == ((), ())

    c.clear_index()
    c.complete_previous_batch()
    assert c.store.pool.num_free == 0
    c.release_transfer_units(("request-a", 0))
    assert c.store.pool.num_free == 0
    c.release_transfer_units(owner)
    c.release_transfer_units(owner)
    assert c.store.pool.num_free == 3

    new_owner = ("request-a", 2)
    assert c.reserve_transfer_units(new_owner) is not None
    c.release_transfer_units(owner)
    assert c.store.pool.num_free == 0
    c.release_transfer_units(new_owner)
    assert c.store.pool.num_free == 3


def test_completed_external_load_is_adopted_as_ready_without_free_list_window():
    c = coordinator(num_units=3)
    owner = ("request-a", 1)
    units = c.reserve_transfer_units(owner)
    assert units is not None
    assert c.store.pool.num_free == 0

    assert c.adopt_transfer_units(owner, 101)
    assert c.contains(101)
    checkpoint_id = c.store.lookup(101)
    record = c.store.records[checkpoint_id]
    assert record.unit_ids == units
    assert record.pin_count == 0
    assert c.store.pool.num_free == 0
    assert not c.store._offload_ready

    c.unindex(101)
    assert c.store.pool.num_free == 3


def test_duplicate_external_load_adoption_releases_incoming_units():
    c = coordinator(num_units=6)
    canonical = ready(c, 101)
    owner = ("request-a", 1)
    incoming = c.reserve_transfer_units(owner)
    assert incoming is not None and incoming != canonical
    assert c.store.pool.num_free == 0

    assert not c.adopt_transfer_units(owner, 101)
    assert c.store.records[c.store.lookup(101)].unit_ids == canonical
    assert c.store.pool.num_free == 3


def test_suspended_restore_can_resume_or_release_its_source_pin():
    c = coordinator(num_units=3)
    ready(c, 101)
    checkpoint_id = c.store.lookup(101)

    assert c.begin_restore(101, dst_slot=9)
    suspended = c.suspend_queued_restore(9)
    assert suspended is not None
    assert c.take_checkpoint_ops()[1] == ()
    assert c.store.records[checkpoint_id].pin_count == 1

    c.resume_suspended_restore(suspended)
    [restore] = c.take_checkpoint_ops()[1]
    assert restore.dst_slot == 9
    c.complete_previous_batch()
    assert c.store.records[checkpoint_id].pin_count == 0

    assert c.begin_restore(101, dst_slot=10)
    suspended = c.suspend_queued_restore(10)
    assert suspended is not None
    c.release_suspended_restore(suspended)
    assert c.store.records[checkpoint_id].pin_count == 0
    assert c.take_checkpoint_ops()[1] == ()


def test_external_load_evicts_only_available_checkpoint_sources():
    c = coordinator(num_units=6)
    ready(c, 101)
    ready(c, 202)
    operation, source_units = c.acquire_checkpoint_source(101)
    incoming = c.reserve_transfer_units(("request-a", 1))
    assert incoming is not None
    assert set(incoming).isdisjoint(source_units)
    assert c.contains(101)
    assert not c.contains(202)
    assert c.store.evictions == 1
    assert c.reserve_transfer_units(("request-b", 1)) is None
    assert c.store.evictions == 1

    c.release_transfer_units(("request-a", 1))
    c.settle_offload_store(operation)
    assert c.has_available_units(6)
