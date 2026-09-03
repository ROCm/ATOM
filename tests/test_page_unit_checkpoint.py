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
