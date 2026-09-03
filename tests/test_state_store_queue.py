# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The state store queue's contract, asserted rather than assumed.

The store leg of the offload tier has two ends -- the source release and the
CPU report -- and the previous implementation spread them over two channels and
two owners, so no object could state ``nominated == settled + inflight +
queued`` and no test could assert it. Every defect below is one that shipped:
an in-flight-hash nomination silently dropped, a pin released twice, a lost
report holding an image out of the pool forever, a reclaimed store indexed by
its own late report.

Pure Python: no torch, no aiter, no GPU. It runs on the CPU CI runner.
"""

import time

import pytest

from atom.model_engine.state_offload import StateOffloadIndex
from atom.model_engine.state_store_queue import StateStoreQueue


class FakeCheckpointStore:
    """The pin half of `PageUnitCheckpointStore`, and nothing else."""

    def __init__(self):
        # prefix hash -> checkpoint id, for the images that are READY.
        self.ready: dict[int, int] = {}
        self.pins: list[int] = []
        self.unpins: list[int] = []
        self._next_id = 0

    def add(self, prefix_hash: int) -> int:
        """Make an image READY, as `complete_inflight` would."""
        self._next_id += 1
        self.ready[prefix_hash] = self._next_id
        return self._next_id

    def evict(self, prefix_hash: int) -> None:
        self.ready.pop(prefix_hash, None)

    def units(self, prefix_hash: int) -> tuple[int, ...]:
        checkpoint_id = self.ready[prefix_hash]
        return (checkpoint_id * 10, checkpoint_id * 10 + 1)

    def pin_checkpoint(self, prefix_hash: int):
        checkpoint_id = self.ready.get(prefix_hash)
        if checkpoint_id is None:
            return None
        units = self.units(prefix_hash)
        self.pins.append(checkpoint_id)
        return checkpoint_id, units

    def unpin_checkpoint(self, checkpoint_id: int) -> None:
        self.unpins.append(checkpoint_id)

    @property
    def held(self) -> int:
        return len(self.pins) - len(self.unpins)


def make_queue(*, max_inflight=4, can_store=True, can_load=True):
    store = FakeCheckpointStore()
    index = StateOffloadIndex(
        can_store=can_store,
        can_load=can_load,
        chunk_tokens=1024,
        release_slot=lambda slot: None,
    )
    queue = StateStoreQueue(store=store, index=index, max_inflight=max_inflight)
    return store, index, queue


def nominate(store, queue, prefix_hash):
    """Nominate exactly as the READY transition does."""
    store.add(prefix_hash)
    queue.nominate(prefix_hash, store.units(prefix_hash))


class TestTheHappyPath:
    def test_one_store_pins_once_releases_once_and_indexes_once(self):
        store, index, queue = make_queue()
        nominate(store, queue, 0xA1)
        queue.check_invariant()
        assert queue.queued == 1

        specs = queue.take(8)
        queue.check_invariant()
        assert [spec.prefix_hash for spec in specs] == [0xA1]
        assert specs[0].unit_ids == store.units(0xA1)
        assert store.held == 1
        assert queue.inflight == 1
        assert index.stores_attempted == 1

        queue.settle_source(specs[0].op_id)
        queue.check_invariant()
        # The units go back the instant the gather drains, but the store is
        # still owed a report.
        assert store.held == 0
        assert queue.inflight == 1
        assert queue.has_pending()

        queue.settle_stored(specs[0].op_id, ok=True)
        queue.check_invariant()
        assert index.hashes == {0xA1}
        assert index.stores_completed == 1
        assert store.unpins == [store.ready[0xA1]]
        assert not queue.has_pending()
        assert queue.settled == 1

    def test_op_id_is_hashable_and_unique(self):
        # It becomes `ConnectorCompletion.operation_id`, which must be hashable.
        store, _index, queue = make_queue()
        for h in (1, 2, 3):
            nominate(store, queue, h)
        specs = queue.take(8)
        assert len({spec.op_id for spec in specs}) == 3

    def test_a_load_only_role_never_nominates(self):
        store, _index, queue = make_queue(can_store=False)
        nominate(store, queue, 0xB1)
        queue.check_invariant()
        # Nothing queued and nothing owed: a nomination nobody drains would
        # read as a tier falling behind rather than one never granted.
        assert queue.queued == 0
        assert queue.nominated == 0
        assert not queue.has_pending()


class TestBounds:
    def test_take_is_bounded_by_max_inflight_and_leaves_the_rest_queued(self):
        store, _index, queue = make_queue(max_inflight=2)
        for h in (1, 2, 3, 4):
            nominate(store, queue, h)

        specs = queue.take(8)
        queue.check_invariant()
        assert len(specs) == 2
        assert queue.queued == 2
        assert store.held == 2

        # Still full: nothing has reported.
        assert queue.take(8) == []
        queue.check_invariant()

        queue.settle_source(specs[0].op_id)
        queue.settle_stored(specs[0].op_id, ok=True)
        more = queue.take(8)
        queue.check_invariant()
        assert len(more) == 1
        assert queue.queued == 1

    def test_take_is_also_bounded_by_its_own_limit(self):
        store, _index, queue = make_queue(max_inflight=8)
        for h in (1, 2, 3):
            nominate(store, queue, h)
        assert len(queue.take(1)) == 1
        assert queue.queued == 2
        queue.check_invariant()

    def test_the_backlog_is_capped(self):
        store, _index, queue = make_queue()
        queue._backlog_cap = 3
        for h in range(6):
            nominate(store, queue, h)
        queue.check_invariant()
        assert queue.queued == 3
        assert queue.nominations_dropped == 3
        # The oldest go, so the freshest images are the ones still offered.
        assert list(queue._queued) == [3, 4, 5]


class TestNominationsAreNeverLost:
    def test_a_nomination_for_an_inflight_hash_is_requeued_not_dropped(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xC1)
        first = queue.take(8)[0]

        # The image was evicted and stored again while the first attempt was
        # still on the wire, so the same hash reaches READY a second time.
        queue.nominate(0xC1, store.units(0xC1))
        assert queue.nominated == 2

        assert queue.take(8) == []
        queue.check_invariant()
        assert queue.nominations_requeued == 1
        assert queue.queued == 1

        # The in-flight attempt fails. The re-queued nomination is what makes
        # the image storable at all -- dropping it here, as the previous
        # implementation did, lost it for good, since nomination only ever
        # happens on the READY transition.
        queue.settle_source(first.op_id)
        queue.settle_stored(first.op_id, ok=False)
        queue.check_invariant()

        retry = queue.take(8)
        assert [spec.prefix_hash for spec in retry] == [0xC1]
        queue.settle_source(retry[0].op_id)
        queue.settle_stored(retry[0].op_id, ok=True)
        queue.check_invariant()
        assert queue.settled == queue.nominated == 2

    def test_a_requeued_nomination_whose_store_succeeded_is_not_stored_twice(self):
        store, index, queue = make_queue()
        nominate(store, queue, 0xC2)
        first = queue.take(8)[0]
        queue.nominate(0xC2, store.units(0xC2))
        assert queue.take(8) == []

        queue.settle_source(first.op_id)
        queue.settle_stored(first.op_id, ok=True)
        assert 0xC2 in index.hashes

        assert queue.take(8) == []
        queue.check_invariant()
        assert queue.nominations_satisfied == 1
        assert queue.settled == queue.nominated == 2
        assert index.stores_attempted == 1

    def test_a_store_only_role_still_dedups_its_own_stores(self):
        # `could_serve` is False for `can_load=False`, so routing the duplicate
        # check through it would re-store every image.
        store, index, queue = make_queue(can_load=False)
        nominate(store, queue, 0xC3)
        first = queue.take(8)[0]
        queue.nominate(0xC3, store.units(0xC3))
        queue.settle_source(first.op_id)
        queue.settle_stored(first.op_id, ok=True)
        assert not index.could_serve(0xC3)

        assert queue.take(8) == []
        assert queue.nominations_satisfied == 1
        queue.check_invariant()

    def test_a_second_nomination_of_a_queued_hash_collapses(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xC4)
        queue.nominate(0xC4, store.units(0xC4))
        queue.check_invariant()
        assert queue.queued == 1
        assert queue.nominated == 1
        assert queue.nominations_collapsed == 1

    def test_a_nomination_whose_image_was_spent_settles_as_stale(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xC5)
        # Nomination takes no pin, so the pool may spend the checkpoint.
        store.evict(0xC5)
        assert queue.take(8) == []
        queue.check_invariant()
        assert queue.nominations_stale == 1
        assert queue.settled == queue.nominated == 1
        assert store.pins == []


class TestCompletions:
    def test_settle_source_twice_unpins_once(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xD1)
        spec = queue.take(8)[0]
        queue.settle_source(spec.op_id)
        queue.settle_source(spec.op_id)
        queue.check_invariant()
        assert len(store.unpins) == 1
        assert queue.sources_released == 1

    def test_a_report_that_beats_the_source_release_still_unpins_once(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xD2)
        spec = queue.take(8)[0]
        queue.settle_stored(spec.op_id, ok=True)
        queue.check_invariant()
        assert len(store.unpins) == 1
        # And the release that arrives afterwards is inert.
        queue.settle_source(spec.op_id)
        assert len(store.unpins) == 1
        queue.check_invariant()

    def test_a_failed_store_is_not_indexed(self):
        store, index, queue = make_queue()
        nominate(store, queue, 0xD3)
        spec = queue.take(8)[0]
        queue.settle_source(spec.op_id)
        queue.settle_stored(spec.op_id, ok=False)
        queue.check_invariant()
        assert index.hashes == set()
        assert index.stores_failed == 1
        assert index.stores_completed == 0

    def test_a_report_for_an_unknown_operation_is_inert(self):
        store, index, queue = make_queue()
        nominate(store, queue, 0xD4)
        spec = queue.take(8)[0]
        queue.settle_stored(spec.op_id, ok=True)
        queue.settle_stored(spec.op_id, ok=True)
        queue.settle_stored(9999, ok=True)
        queue.check_invariant()
        assert index.stores_completed == 1
        assert queue.settled == 1


class TestReclaim:
    def test_reclaim_frees_a_lost_stores_pins_and_forfeits_its_image(self):
        store, index, queue = make_queue()
        nominate(store, queue, 0xE1)
        spec = queue.take(8)[0]
        assert store.held == 1

        time.sleep(0.01)
        assert queue.reclaim(0.001) == 1
        queue.check_invariant()
        # Without this the image is out of the KV pool for the process lifetime.
        assert store.held == 0
        assert queue.stores_reclaimed == 1
        assert queue.settled == queue.nominated == 1
        assert not queue.has_pending()

        # A late report cannot be counted: the reclaimer took the source back,
        # so nothing proves the worker had stopped reading those units, and the
        # CPU image may mix two prefixes under one hash.
        queue.settle_stored(spec.op_id, ok=True)
        queue.check_invariant()
        assert index.hashes == set()
        assert index.stores_completed == 0
        assert queue.stores_untrusted == 1
        assert queue.settled == 1

    def test_reclaim_does_not_unpin_a_released_source_twice(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xE2)
        spec = queue.take(8)[0]
        queue.settle_source(spec.op_id)

        time.sleep(0.01)
        assert queue.reclaim(0.001) == 1
        queue.check_invariant()
        assert len(store.unpins) == 1

    def test_reclaim_leaves_young_stores_alone(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xE3)
        queue.take(8)
        assert queue.reclaim(60.0) == 0
        assert queue.reclaim(0.0) == 0
        assert queue.inflight == 1
        queue.check_invariant()


class TestTheInvariantCanFail:
    """A check that cannot fail proves nothing."""

    def test_a_lost_nomination_is_caught(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xF1)
        queue.take(8)
        queue.check_invariant()
        queue._records.clear()  # a record dropped without a terminal transition
        with pytest.raises(AssertionError, match="nominated !="):
            queue.check_invariant()

    def test_a_pin_left_behind_is_caught(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xF2)
        spec = queue.take(8)[0]
        queue._records[spec.op_id].source_released = True  # unpinned by nobody
        with pytest.raises(AssertionError, match="pins held !="):
            queue.check_invariant()

    def test_a_diverged_nomination_set_is_caught(self):
        store, _index, queue = make_queue()
        nominate(store, queue, 0xF3)
        queue._queued_set.clear()
        with pytest.raises(AssertionError, match="diverged"):
            queue.check_invariant()


class TestStats:
    def test_every_counter_is_reachable(self):
        store, _index, queue = make_queue(max_inflight=2)
        queue._backlog_cap = 2

        # dispatched / sources_released / settled, then satisfied on the retry.
        nominate(store, queue, 1)
        spec = queue.take(8)[0]
        queue.nominate(1, store.units(1))
        queue.take(8)  # requeued
        queue.settle_source(spec.op_id)
        queue.settle_stored(spec.op_id, ok=True)
        queue.take(8)  # satisfied

        # stale
        nominate(store, queue, 2)
        store.evict(2)
        queue.take(8)

        # collapsed and dropped
        nominate(store, queue, 3)
        queue.nominate(3, store.units(3))
        nominate(store, queue, 4)
        nominate(store, queue, 5)

        # reclaimed and untrusted
        nominate(store, queue, 6)
        specs = queue.take(8)
        time.sleep(0.01)
        queue.reclaim(0.001)
        for reclaimed in specs:
            queue.settle_stored(reclaimed.op_id, ok=True)

        # queued, the one gauge that has to be read while something waits.
        nominate(store, queue, 7)

        queue.check_invariant()
        stats = queue.stats()
        assert set(stats) == {
            "store_nominated",
            "store_settled",
            "store_queued",
            "store_inflight",
            "store_dispatched",
            "store_sources_released",
            "nominations_collapsed",
            "nominations_dropped",
            "nominations_requeued",
            "nominations_satisfied",
            "nominations_stale",
            "stores_reclaimed",
            "stores_untrusted",
        }
        # Every counter but one moved; `store_inflight` is a gauge and the
        # queue is drained here.
        zero = [name for name, value in stats.items() if value == 0]
        assert zero == ["store_inflight"], zero
        assert stats["store_nominated"] == (
            stats["store_settled"] + stats["store_inflight"] + stats["store_queued"]
        )
