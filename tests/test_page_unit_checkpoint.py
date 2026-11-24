# SPDX-License-Identifier: MIT

"""Control-plane invariants for PAGE-backed state checkpoint images."""

import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from atom.model_engine.block_pool import BlockPool
from atom.model_engine.kv_block import STATE_CHECKPOINT_CLASS, STATE_SLOT_CLASS
from atom.model_engine.page_unit_checkpoint import (
    COPYING,
    EVICTING,
    READY,
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
    PageUnitCheckpointStore,
    PipelineCheckpointStage,
    checkpoint_store_entry_bytes,
    pipeline_kv_blocks,
    published_page_unit_bytes,
    size_pipeline_checkpoint_store,
)
from atom.model_engine.state_cache import KEEP_SETTLED, KEEP_SNAPSHOT
from atom.model_engine.state_pool import StateSlotPool
from atom.model_ops.attentions.pool_layout.sub_pool_spec import (
    page_pool,
    plan_pools,
    state_pool,
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


# ── offloading a checkpoint to the CPU tier ───────────────────────────────


def offload_store(**kw):
    pool, store = make_store(**kw)
    store._offload_sink = True
    return pool, store


class TestOffloadNomination:
    """READY nominates; `take_offload_stores` is what pins.

    Splitting the two is the whole design. A pin lives in the engine process
    while the D2H runs in the worker, so it spans several scheduler passes;
    pinning at READY would make EVERY checkpoint un-evictable for that window
    and break the one thing #2045's admission argument rests on -- that a READY
    unpinned checkpoint counts as available to live KV.
    """

    def test_reaching_ready_nominates_without_pinning(self):
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        assert store.records[cid].pin_count == 0
        assert store._is_evictable(cid), "a nominee must still be spendable"

    def test_handing_it_over_is_what_pins(self):
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        stores = store.take_offload_stores(max_inflight=4)
        [(op, unit_ids)] = stores
        assert (op.prefix_hash, unit_ids) == (11, store.records[cid].unit_ids)
        assert store.records[cid].pin_count == 1
        assert not store._is_evictable(cid)

    def test_a_nominee_the_pool_needed_more_is_simply_gone(self):
        """Nomination deliberately leaves it spendable, so `_next_victim` may
        take it. Losing a CPU copy is the right price for never making the
        pool wait on the tier."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        store._evict(cid)
        assert store.take_offload_stores(max_inflight=4) == []

    def test_no_sink_nominates_nothing(self):
        """A pin nobody can release holds a whole image out of the pool
        forever, so a deployment with no tier must take none."""
        _pool, store = make_store()  # _offload_sink defaults False
        ready(store, 11)
        assert store.take_offload_stores(max_inflight=4) == []

    def test_it_is_offered_once(self):
        _pool, store = offload_store()
        ready(store, 11)
        assert len(store.take_offload_stores(max_inflight=4)) == 1
        assert store.take_offload_stores(max_inflight=4) == []


class TestOffloadInflightCap:
    """The cap is what bounds how much of the pool a slow backend can hold."""

    def test_over_the_cap_the_rest_wait_rather_than_drop(self):
        _pool, store = offload_store(num_units=40)
        for h in (11, 12, 13):
            ready(store, h)
        first = store.take_offload_stores(max_inflight=2)
        assert len(first) == 2
        # The third is still nominated, unpinned, and offered again next pass.
        assert store.take_offload_stores(max_inflight=2) == []
        store.settle_offload_store(first[0][0])
        assert len(store.take_offload_stores(max_inflight=2)) == 1

    def test_a_waiting_nominee_stays_evictable(self):
        _pool, store = offload_store(num_units=40)
        for h in (11, 12):
            ready(store, h)
        store.take_offload_stores(max_inflight=1)
        waiting = store.lookup(12)
        assert store._is_evictable(waiting)


class TestOffloadPinRelease:
    def test_success_and_failure_release_identically(self):
        """The pin keeps the bytes still during the copy, and the copy is over
        either way. Whether the hash becomes reachable is the index's business."""
        for _ in range(2):
            _pool, store = offload_store()
            cid, _op = ready(store, 11)
            [(sent, _units)] = store.take_offload_stores(max_inflight=4)
            store.settle_offload_store(sent)
            assert store.records[cid].pin_count == 0

    def test_a_report_for_a_hash_never_sent_is_a_no_op(self):
        from atom.kv_transfer.disaggregation.types import StateStoreOperationId

        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        store.settle_offload_store(StateStoreOperationId(11, 1))
        assert store.records[cid].pin_count == 0

    def test_source_release_frees_units_but_keeps_the_pin(self):
        """Phase one: the gather drained, so the units go back -- but the store
        is still dispatched-but-unreported, so the engine must keep polling for
        its report. Retiring the pin here (as it once did) let liveness read
        idle while the report still sat undrained, so it was never votable."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        [(sent, _u)] = store.take_offload_stores(max_inflight=4)

        store.release_offload_store_source(sent)
        assert store.records[cid].pin_count == 0, "no longer pinned for the copy"
        assert store._is_evictable(cid), "the units are the pool's again"
        assert store.has_offload_pins(), "still unreported: keep polling"
        assert store._hash_in_flight(11), "no second store of 11 yet"

        store.settle_offload_store(sent)
        assert not store.has_offload_pins()
        assert not store._hash_in_flight(11)

    def test_source_release_is_idempotent_and_does_not_double_free(self):
        """A second release -- or a release after the report -- must not
        decrement the record a second time and underflow the pin count."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        [(sent, _u)] = store.take_offload_stores(max_inflight=4)
        store.release_offload_store_source(sent)
        store.release_offload_store_source(sent)  # idempotent
        store.settle_offload_store(sent)
        store.release_offload_store_source(sent)  # after the report: no pin left
        assert store.records[cid].pin_count == 0

    def test_a_source_released_store_that_never_reports_is_reclaimed_but_kept(self):
        """Its units are already back and the reader provably stopped, so a lost
        report is not the taken-back-source case: nothing is released again (no
        underflow) and the image is NOT forfeited -- a late report may index."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        [(sent, _u)] = store.take_offload_stores(max_inflight=4)
        store.release_offload_store_source(sent)
        assert store.records[cid].pin_count == 0

        assert store.reclaim_stale_offload_pins(timeout_s=1e-9) == 1
        assert store.records[cid].pin_count == 0, "not released twice"
        assert not store.was_reclaimed(sent), "source-released bytes are trusted"
        assert store.offload_pins_reclaimed == 1

    def test_unindex_during_the_copy_holds_the_units_to_the_end(self):
        """The tier's whole point: the CPU copy outliving the HBM one. So
        `unindex` must not pull the source out from under a running D2H."""
        pool, store = offload_store()
        cid, _op = ready(store, 11)
        units = store.records[cid].unit_ids
        [(sent, _units)] = store.take_offload_stores(max_inflight=4)

        store.unindex(11)
        assert cid in store.records, "the units are still being read"
        assert all(pool.is_used(u) for u in units)

        store.settle_offload_store(sent)
        assert cid not in store.records
        assert not any(pool.is_used(u) for u in units)

    def test_two_generations_of_one_hash_each_settle_their_own_pin(self):
        """The same prefix is stored again after an eviction or a load miss.

        Keyed by bare hash, the aggregator tombstoned the first store and the
        second was dropped before quorum -- its pin waited for the stale
        reclaimer and its bytes were never re-indexed. Each hand-out gets its
        own generation, so both attempts settle.
        """
        _pool, store = offload_store()
        cid1, _op = ready(store, 11)
        [(first, _u)] = store.take_offload_stores(max_inflight=4)
        store.settle_offload_store(first)
        assert store.records[cid1].pin_count == 0

        store.unindex(11)
        cid2, _op = ready(store, 11)
        [(second, _u)] = store.take_offload_stores(max_inflight=4)
        assert second != first, "a re-store is its own operation"
        assert second.prefix_hash == first.prefix_hash
        store.settle_offload_store(second)
        assert store.records[cid2].pin_count == 0

    def test_a_late_report_from_a_superseded_attempt_settles_nothing(self):
        """A reclaimed pin's report can still be in flight. It must not release
        the pin a newer attempt at the same hash is holding."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        [(first, _u)] = store.take_offload_stores(max_inflight=4)
        assert store.reclaim_stale_offload_pins(timeout_s=1e-9) == 1

        [(second, _u)] = store.take_offload_stores(max_inflight=4) or [(None, None)]
        if second is None:  # the nomination was drained by the first hand-out
            store._queue_offload_store(cid, store.records[cid])
            [(second, _u)] = store.take_offload_stores(max_inflight=4)
        assert store.records[cid].pin_count == 1

        store.settle_offload_store(first)  # the late one
        assert store.records[cid].pin_count == 1, "the newer pin still stands"
        store.settle_offload_store(second)
        assert store.records[cid].pin_count == 0

    def test_one_hash_may_not_have_two_stores_in_flight(self):
        """The generation tells sequential attempts apart; it does not license
        concurrent ones, which would copy the same bytes twice and have the
        first report unpin what the second is holding."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        assert len(store.take_offload_stores(max_inflight=4)) == 1
        store._queue_offload_store(cid, store.records[cid])
        assert store.take_offload_stores(max_inflight=4) == []

    def test_a_reclaimed_store_is_remembered_so_it_cannot_be_indexed(self):
        """The reclaimer cannot tell a lost report from a worker still inside
        the gather, so the units come back but the image is forfeited."""
        _pool, store = offload_store()
        _cid, _op = ready(store, 11)
        [(op, _u)] = store.take_offload_stores(max_inflight=4)
        assert not store.was_reclaimed(op)
        assert store.reclaim_stale_offload_pins(timeout_s=1e-9) == 1
        assert store.was_reclaimed(op)

    def test_a_store_that_reported_in_time_is_not_marked_reclaimed(self):
        _pool, store = offload_store()
        _cid, _op = ready(store, 11)
        [(op, _u)] = store.take_offload_stores(max_inflight=4)
        store.settle_offload_store(op)
        assert store.reclaim_stale_offload_pins(timeout_s=1e-9) == 0
        assert not store.was_reclaimed(op)

    def test_a_lost_report_is_reclaimed_and_counted(self):
        """The pin is in this process and the copy is in the worker, so a
        crashed worker would otherwise hold an image out of the pool forever."""
        _pool, store = offload_store()
        cid, _op = ready(store, 11)
        store.take_offload_stores(max_inflight=4)

        assert store.reclaim_stale_offload_pins(timeout_s=3600) == 0
        assert store.reclaim_stale_offload_pins(timeout_s=-1) == 0, "disabled"
        assert store.reclaim_stale_offload_pins(timeout_s=0.0) == 0, "disabled"

        assert store.reclaim_stale_offload_pins(timeout_s=1e-9) == 1
        assert store.records[cid].pin_count == 0
        assert store.offload_pins_reclaimed == 1
        # And recovery is total: the record is spendable again, unbroken.
        assert store._is_evictable(cid)
        assert store.lookup(11) == cid


class TestTheTierVotes:
    """`resumable_hit` accepts a boundary the CPU tier holds, not just HBM.

    This is what makes the whole tier reachable: without it a hash whose image
    went to LMCache is invisible to `can_allocate`, and the bytes are written
    and never read.
    """

    class _Index:
        def __init__(self, *hashes):
            self.hashes = set(hashes)

        def could_serve(self, h):
            return h in self.hashes

        def want_touch(self, h):
            pass

    @staticmethod
    def coordinator(num_units=40, offload=None):
        from atom.model_engine.page_unit_checkpoint import (
            PagedStateCheckpointCoordinator,
            PagedStateCheckpointSpec,
        )

        c = PagedStateCheckpointCoordinator(
            BlockPool(num_units),
            PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25),
            enabled=True,
        )
        if offload is not None:
            c.attach_offload(offload)
        return c

    @staticmethod
    def seq():
        from types import SimpleNamespace

        return SimpleNamespace(has_per_req_cache=True)

    def test_a_hash_only_the_tier_holds_is_accepted(self):
        c = self.coordinator(offload=self._Index(77))
        assert c.resumable_hit(self.seq(), 3, [11, 77, 99]) == 2

    def test_without_a_tier_it_is_not(self):
        c = self.coordinator()
        assert c.resumable_hit(self.seq(), 3, [11, 77, 99]) == 0

    def test_the_scan_still_takes_the_rightmost_boundary(self):
        """Both tiers are keyed by the same content hash, so no preference rule
        is needed -- and `_attach_state_slots` tries HBM first regardless."""
        c = self.coordinator(offload=self._Index(11, 99))
        assert c.resumable_hit(self.seq(), 3, [11, 77, 99]) == 3

    def test_attaching_turns_on_the_vote_and_the_sink_together(self):
        """Half-attached is worse than off: a vote with no sink accepts hashes
        nothing ever stores; a sink with no vote pins units nothing reads."""
        c = self.coordinator()
        assert c.offload is None and c.store._offload_sink is False
        c.attach_offload(self._Index())
        assert c.offload is not None and c.store._offload_sink is True

    def test_the_tier_half_is_optimistic_on_purpose(self):
        """`hashes` means "was stored once", never "is still there" -- LMCache's
        own LRU can drop bytes under it. A false positive costs one park plus a
        recompute and retracts itself; being certain would cost a synchronous
        cross-process lookup on the admission path."""
        index = self._Index(77)
        c = self.coordinator(offload=index)
        assert c.resumable_hit(self.seq(), 3, [11, 77, 99]) == 2
        index.hashes.discard(77)  # what `fail_load` -> `forget` does
        assert c.resumable_hit(self.seq(), 3, [11, 77, 99]) == 0


class _Builder:
    def __init__(self, specs, published="absent"):
        self._specs = specs
        if published != "absent":
            self.checkpoint_page_unit_bytes = lambda: published

    def sub_pool_specs(self):
        return self._specs


def test_page_unit_is_the_target_share_not_the_merged_class():
    target = _Builder(
        [page_pool(1000), state_pool(STATE_SLOT_CLASS, 64, entries_per_req=2)]
    )
    draft = _Builder([page_pool(300)])
    plan = plan_pools(target.sub_pool_specs() + draft.sub_pool_specs(), 10**6, 4)

    assert plan.entry_bytes[plan.paged_class] == 1300
    assert published_page_unit_bytes(target) == 1000


def test_page_unit_defaults_to_the_target_share_without_a_draft():
    target = _Builder([page_pool(1000), state_pool("s", 64, entries_per_req=1)])
    plan = plan_pools(target.sub_pool_specs(), 10**6, 4)

    assert published_page_unit_bytes(target) == plan.entry_bytes[plan.paged_class]
    assert published_page_unit_bytes(_Builder(target._specs, None)) == 1000


def test_backend_published_unit_overrides_the_share():
    specs = [page_pool(1024), state_pool("s", 64, entries_per_req=1)]

    assert published_page_unit_bytes(_Builder(specs, 1000)) == 1000
    for bad in (0, 1025):
        with pytest.raises(RuntimeError, match="checkpoint PAGE unit"):
            published_page_unit_bytes(_Builder(specs, bad))


def test_page_unit_needs_target_page_bytes():
    with pytest.raises(RuntimeError, match="declares no PAGE bytes"):
        published_page_unit_bytes(_Builder([state_pool("s", 64, entries_per_req=1)]))


def _snapshot_coordinator(free_slots, decode_keep=KEEP_SNAPSHOT):
    c = PagedStateCheckpointCoordinator(
        BlockPool(20),
        PagedStateCheckpointSpec(10, 25, "layout-v1", image_bytes=25),
        enabled=True,
        decode_keep=decode_keep,
    )
    pool = StateSlotPool(free_slots)
    c.attach_midstep_scratch(pool, 8)
    return c, pool


def _seq(slot, spec_slots=()):
    return SimpleNamespace(
        has_per_req_cache=True,
        state_slot=slot,
        state_slots=[slot, *spec_slots],
        num_forwards=0,
    )


@pytest.mark.parametrize(
    "staged, forwards, kept", [(False, 0, 1), (False, 1, 0), (True, 1, 1), (True, 2, 0)]
)
def test_a_boundary_is_stored_only_while_the_slot_holds_it(staged, forwards, kept):
    """One filed after its forward expects no other before the drain; one staged
    with its forward in the batch being built expects that one."""
    c, _ = _snapshot_coordinator(1, KEEP_SETTLED)
    seq = _seq(5)
    if staged:
        c.checkpoint_when_scheduled(seq, 101)
        assert c.take_checkpoint_ops()[0] == ()  # held back from its own batch
    else:
        c.checkpoint(seq, 8, 101)
    seq.num_forwards += forwards
    stores, _ = c.take_checkpoint_ops()
    assert (len(stores), c.checkpoints_dropped) == (kept, 1 - kept)


@pytest.mark.parametrize("spec_slots, step_slot", [((), 5), ((6, 7, 8), 7)])
def test_a_decode_image_names_the_slot_its_step_is_in(spec_slots, step_slot):
    """A slot per verify step, or one for all of them (ReplaySSM)."""
    c, _ = _snapshot_coordinator(1)
    c.checkpoint(_seq(5, spec_slots), 8, 101, step=2)
    (op,), _ = c.take_checkpoint_ops()
    assert (op.src_slot, op.step, op.scratch_slot, op.step_slot) == (5, 2, 0, step_slot)


def test_a_decode_boundary_with_no_scratch_free_is_dropped():
    c, _ = _snapshot_coordinator(0)
    c.checkpoint(_seq(5), 8, 101, step=0)
    stores, _ = c.take_checkpoint_ops()
    assert stores == ()
    assert c.checkpoints_dropped == 1


@pytest.mark.parametrize("decode_keep, step", [(KEEP_SETTLED, 2), (KEEP_SNAPSHOT, -1)])
def test_a_slot_copied_as_is_borrows_nothing(decode_keep, step):
    """A backend keeping its slot as it stands, or a slot a prefill wrote whole."""
    c, pool = _snapshot_coordinator(1, decode_keep)
    c.checkpoint(_seq(5), 8, 101, step=step)
    stores, _ = c.take_checkpoint_ops()
    assert [(op.src_slot, op.step, op.scratch_slot) for op in stores] == [(5, -1, -1)]
    assert pool.has_free()


# ── PP: a checkpoint store per stage ───────────────────────────────────────

GIB = 1 << 30
# Kimi-K3 at PP8 x TP1 on MI455X, as the measured runs logged it: an MLA row is
# 128 tokens x 576 B a block, a KDA layer's image (conv + SSM) 6,807,552 B and
# its slot 8,380,416 B (with ReplaySSM's records), 64 slots a stage.
MLA_ROW = 128 * 576
KDA_IMAGE = 6_807_552
KDA_SLOT = 8_380_416
# `VLLM_PP_LAYER_PARTITION` and each stage's logged `available_for_kv` (GiB).
SPLIT = (13, 13, 13, 13, 13, 12, 12, 4)
AVAILABLE_GIB = (137.32, 137.37, 157.57, 138.30, 138.54, 152.46, 153.11, 276.36)
DEFAULT_SPLIT = (11, 11, 12, 12, 12, 12, 12, 11)
DEFAULT_AVAILABLE_GIB = (188.09, 169.20, 154.19, 153.47, 153.42, 154.48, 152.54, 169.45)


def k3_stages(split, available_gib, slots=None, kv_share=0.5):
    """Each stage's numbers: layers 4, 8, ..., 92 and 93 (1-indexed) are MLA,
    the other 69 KDA, and the last stage also holds the drafter's 5 MLA rows."""
    mla = set(range(4, 93, 4)) | {93}
    stages, first = [], 1
    for i, (n, gib) in enumerate(zip(split, available_gib)):
        rows = sum(layer in mla for layer in range(first, first + n))
        kda = n - rows
        rows += 5 if i == len(split) - 1 else 0
        first += n
        stages.append(
            PipelineCheckpointStage(
                budget_bytes=int(gib * GIB) - 64 * kda * KDA_SLOT,
                page_bytes=rows * MLA_ROW,
                image_bytes=kda * KDA_IMAGE,
                slots=slots,
                kv_share=kv_share,
            )
        )
    return stages


def own_blocks(stage):
    return -(-stage.image_bytes // stage.page_bytes)


def agreed(stages, unit_bytes):
    """Each stage's spec after `ModelRunner._agree_pipeline_checkpoint_spec`,
    its unit `unit_bytes(stage)`: a KV block, or a checkpoint-store entry."""
    specs = [
        PagedStateCheckpointSpec(unit_bytes(s), s.image_bytes, "kda", s.image_bytes)
        for s in stages
    ]
    units = max(spec.units_per_checkpoint for spec in specs)
    op_bytes = sum(s.image_bytes for s in stages)
    return [spec.for_pipeline("pp8-joint:test", units, op_bytes) for spec in specs]


class TestAPipelineCheckpointStore:
    """Under PP every stage writes its own part of an image under the one unit
    id the head names. In KV blocks an id is a block on every stage, so each
    image took the largest stage's block count everywhere; a store entry is a
    whole image at each stage's own size."""

    def test_the_stages_need_different_block_counts(self):
        """The per-stage counts the measured runs logged."""
        split = k3_stages(SPLIT, AVAILABLE_GIB)
        default = k3_stages(DEFAULT_SPLIT, DEFAULT_AVAILABLE_GIB)
        assert [own_blocks(s) for s in split] == [308] * 3 + [208, 308, 277, 277, 27]
        assert [own_blocks(s) for s in default] == [416, 247] + [277] * 5 + [72]

    def test_a_stage_pays_its_own_image_not_the_largest_stages_blocks(self):
        stages = k3_stages(SPLIT, AVAILABLE_GIB)
        in_blocks = agreed(stages, lambda s: s.page_bytes)
        in_store = agreed(stages, lambda s: checkpoint_store_entry_bytes(s.image_bytes))

        for stage, kv, store in zip(stages, in_blocks, in_store):
            assert kv.units_per_checkpoint == 308
            assert store.units_per_checkpoint == 1
            paid = store.units_per_checkpoint * store.page_unit_bytes
            # Its own image, rounded to its own entry, never more than its own
            # block count would have cost it.
            assert paid == checkpoint_store_entry_bytes(stage.image_bytes)
            assert stage.image_bytes <= paid <= own_blocks(stage) * stage.page_bytes
            assert store.op_bytes == kv.op_bytes == sum(s.image_bytes for s in stages)
        # The last stage: a 13.6 MB image used to take 308 of its 516 KB blocks.
        last = stages[-1]
        assert in_blocks[-1].units_per_checkpoint * last.page_bytes == 158_957_568
        assert in_store[-1].page_unit_bytes == last.image_bytes == 13_615_104

    @pytest.mark.parametrize(
        "split, available, images, before, after",
        [
            (SPLIT, AVAILABLE_GIB, 1000, 179_166, 279_416),
            (SPLIT, AVAILABLE_GIB, 1500, 25_166, 175_541),
            # 416 blocks an image: 1,000 of them did not fit at all.
            (DEFAULT_SPLIT, DEFAULT_AVAILABLE_GIB, 1000, -147_459, 196_726),
            (DEFAULT_SPLIT, DEFAULT_AVAILABLE_GIB, 500, 60_541, 232_633),
        ],
    )
    def test_the_kv_blocks_left_for_a_number_of_images(
        self, split, available, images, before, after
    ):
        stages = k3_stages(split, available)
        largest = max(own_blocks(s) for s in stages)
        assert pipeline_kv_blocks(stages, 0) - images * largest == before
        assert pipeline_kv_blocks(stages, images) == after
        # Every stage still backs that many blocks beside its own images.
        for s in stages:
            assert (
                after * s.page_bytes
                + images * checkpoint_store_entry_bytes(s.image_bytes)
                <= s.budget_bytes
            )

    @pytest.mark.parametrize(
        "split, available, images, kv_blocks",
        [
            (SPLIT, AVAILABLE_GIB, 1172, 243_683),
            (DEFAULT_SPLIT, DEFAULT_AVAILABLE_GIB, 1869, 134_319),
        ],
    )
    def test_the_share_keeps_its_blocks_and_no_more_images_fit(
        self, split, available, images, kv_blocks
    ):
        stages = k3_stages(split, available, kv_share=0.5)
        promised = -(-pipeline_kv_blocks(stages, 0) // 2)
        assert size_pipeline_checkpoint_store(stages) == images
        assert pipeline_kv_blocks(stages, images) == kv_blocks >= promised
        assert pipeline_kv_blocks(stages, images + 1) < promised

    @pytest.mark.parametrize(
        "split, available, images, own_half",
        [
            (SPLIT, AVAILABLE_GIB, 1172, 1043),
            (DEFAULT_SPLIT, DEFAULT_AVAILABLE_GIB, 1869, 1297),
        ],
    )
    def test_the_share_is_of_the_blocks_every_stage_backs(
        self, split, available, images, own_half
    ):
        """Not of each stage's own memory: a stage whose memory backs more
        blocks than the binding one puts images in that surplus first, so the
        store holds more than every stage giving up half its memory would."""
        stages = k3_stages(split, available, kv_share=0.5)
        assert size_pipeline_checkpoint_store(stages) == images
        assert min(s.budget_bytes // 2 // s.image_bytes for s in stages) == own_half

    def test_an_explicit_count_is_taken_as_given_or_refused(self):
        """Stages 0 and 1 hold 2,087 images and one KV block at most; a
        negative count keeps the images in KV blocks."""
        for slots in (0, 1000, 2087):
            stages = k3_stages(SPLIT, AVAILABLE_GIB, slots)
            assert size_pipeline_checkpoint_store(stages) == slots
        with pytest.raises(ValueError, match=r"stages \[0, 1\].*at most 2087"):
            size_pipeline_checkpoint_store(k3_stages(SPLIT, AVAILABLE_GIB, 2088))
        assert (
            size_pipeline_checkpoint_store(k3_stages(SPLIT, AVAILABLE_GIB, -1)) is None
        )

    @pytest.mark.parametrize(
        "share, stage_5_share, match",
        [(0.5, 0.25, "launched with different"), (1.0, 1.0, "KV_SHARE")],
    )
    def test_a_bad_or_mixed_share_is_refused(self, share, stage_5_share, match):
        stages = k3_stages(SPLIT, AVAILABLE_GIB, kv_share=share)
        stages[5] = stages[5]._replace(kv_share=stage_5_share)
        with pytest.raises(ValueError, match=match):
            size_pipeline_checkpoint_store(stages)

    def test_an_entry_is_the_image_rounded_to_the_alignment(self):
        assert checkpoint_store_entry_bytes(KDA_IMAGE) == KDA_IMAGE
        assert checkpoint_store_entry_bytes(1000) == 1024
        with pytest.raises(ValueError):
            checkpoint_store_entry_bytes(0)


def test_a_plan_carries_the_store_as_a_class_of_its_own():
    """Planned with the store's bytes taken out, the plan stays the one place
    every count and reserved byte is read from, the KV resize included."""
    specs = [page_pool(1000), state_pool(STATE_SLOT_CLASS, 64, entries_per_req=1)]
    plan = plan_pools(specs, 100_000 - 18_000, 4).with_entry_class(
        STATE_CHECKPOINT_CLASS, 30, 600
    )
    assert plan.paged_entries == (100_000 - 18_000 - 4 * 64) // 1000
    for p in (plan, plan.with_paged_entries(10)):
        assert p.entries[STATE_CHECKPOINT_CLASS] == 30
        assert p.reserved_bytes[STATE_CHECKPOINT_CLASS] == 18_000
    assert plan.with_entry_class("empty", 0, 600).entries["empty"] == 0
    with pytest.raises(ValueError, match="already planned"):
        plan.with_entry_class(STATE_CHECKPOINT_CLASS, 1, 600)
    with pytest.raises(ValueError, match="not a pool"):
        plan.with_entry_class("bad", 1, 0)
