# SPDX-License-Identifier: MIT
"""`StateOffloadIndex` is the sole engine-side owner of a state load's life.

That ownership is the point of the object, and these tests are what it buys:
the invariant `dispatched == settled + outstanding` can be *stated*, so it can
be asserted. While the three facts lived in three objects across two processes,
no test could express the property that every load-path defect in this
subsystem was a violation of.
"""

import pytest

from atom.model_engine.state_offload import StateOffloadIndex


def _index(*hashes, **kw):
    released: list[int] = []
    index = StateOffloadIndex(release_slot=released.append, **kw)
    for h in hashes:
        index.note_stored(h)
    index.released = released
    return index


class TestTheInvariantHolds:
    def test_three_dispatched_two_settled_leaves_one_outstanding(self):
        index = _index(1, 2, 3)
        for i, h in enumerate((1, 2, 3)):
            assert index.request_load(f"r{i}", h, slot=i) is True
        index.complete_load("r0")
        index.fail_load("r1")

        assert index.dispatched == 3
        assert index.settled == 2
        assert index.outstanding == 1
        index.check_invariant()

    def test_every_terminal_settles_exactly_once(self):
        """A second report for the same request must not double-count: the KV
        completion channel offers every id here, including replays."""
        index = _index(1, 2, 3)
        index.request_load("a", 1, slot=0)
        index.request_load("b", 2, slot=1)
        index.request_load("c", 3, slot=2)
        index.complete_load("a")
        index.complete_load("a")  # replay
        index.fail_load("b")
        index.abandon_load("c")
        index.abandon_load("c")  # replay

        assert (index.dispatched, index.settled, index.outstanding) == (3, 3, 0)
        index.check_invariant()

    def test_an_id_that_never_dispatched_is_a_no_op(self):
        """Every KV load report is offered to this object, and only a hybrid's
        carries a state leg, so the common case is an unknown id."""
        index = _index()
        index.complete_load("nobody")
        index.fail_load("nobody")
        index.abandon_load("nobody")
        assert (index.dispatched, index.settled) == (0, 0)
        index.check_invariant()


class TestTheInvariantCanFail:
    """A check that cannot fail proves nothing about the checks that pass."""

    def test_a_lost_settlement_is_caught(self):
        index = _index(1)
        index.request_load("a", 1, slot=0)
        index.dispatched += 1  # a dispatch nothing will ever settle
        with pytest.raises(AssertionError, match="dispatched != settled"):
            index.check_invariant()

    def test_a_divergence_between_hashes_and_the_lru_is_caught(self):
        """The cap trims `_hash_lru` and mirrors the drop into `hashes`; if the
        two drift the cap becomes either a leak or a false-positive generator."""
        index = _index(1, 2)
        index.hashes.add(999)  # added without its LRU entry
        with pytest.raises(AssertionError, match="diverged"):
            index.check_invariant()


class TestOnlyARealMissRetractsAHash:
    def test_a_fused_failure_keeps_a_hash_whose_bytes_are_present(self):
        """The report covers BOTH legs, so a failure may be the KV leg alone.
        Forgetting on that permanently denies state that is still there."""
        index = _index(77)
        index.request_load("a", 77, slot=0)
        index.fail_load("a")

        assert 77 in index.hashes
        assert index.could_serve(77) is True
        index.check_invariant()

    def test_a_state_get_miss_retracts_it(self):
        index = _index(77)
        index.request_load("a", 77, slot=0)
        index.fail_load("a", missing=True)

        assert 77 not in index.hashes
        assert index.could_serve(77) is False
        index.check_invariant()

    def test_an_abandon_is_not_a_miss(self):
        """Nothing was attempted, so it says nothing about the bytes and must
        not count against the index's false-positive rate."""
        index = _index(77)
        index.request_load("a", 77, slot=0)
        index.abandon_load("a")

        assert 77 in index.hashes
        assert index.loads_failed == 0
        assert index.loads_abandoned == 1
        index.check_invariant()


class TestSlotOwnership:
    def test_a_live_requests_slot_is_never_handed_back(self):
        """The worker may still be scattering into it."""
        index = _index(1)
        index.request_load("a", 1, slot=5)
        index.complete_load("a")
        assert index.released == []

    def test_an_orphaned_slot_comes_back_when_the_load_settles(self):
        index = _index(1)
        index.request_load("a", 1, slot=5)
        assert index.orphan("a") is True
        index.fail_load("a")
        assert index.released == [5]
        index.check_invariant()

    def test_orphaning_an_id_with_nothing_in_flight_says_so(self):
        """False means the caller keeps the slot -- the ordinary teardown."""
        assert _index().orphan("a") is False

    def test_reclaim_frees_only_an_orphan_whose_report_never_came(self):
        index = _index(1, 2)
        index.request_load("live", 1, slot=5)
        index.request_load("gone", 2, slot=6)
        index.orphan("gone")
        for entry in index._outstanding.values():
            entry.at -= 3600.0

        assert index.reclaim(timeout_s=1.0) == 1
        assert index.released == [6], "a live request's slot must not be yanked"
        assert index.orphan_load_slots_reclaimed == 1
        assert index.outstanding == 1
        index.check_invariant()

    def test_reclaim_spares_an_orphan_still_inside_its_window(self):
        index = _index(1)
        index.request_load("gone", 1, slot=6)
        index.orphan("gone")
        assert index.reclaim(timeout_s=3600.0) == 0
        assert index.released == []

    def test_reclaim_is_disabled_by_a_nonpositive_window(self):
        index = _index(1)
        index.request_load("gone", 1, slot=6)
        index.orphan("gone")
        index._outstanding["gone"].at -= 3600.0
        assert index.reclaim(timeout_s=0.0) == 0

    def test_a_late_report_after_reclaim_releases_nothing_twice(self):
        index = _index(1)
        index.request_load("gone", 1, slot=6)
        index.orphan("gone")
        index._outstanding["gone"].at -= 3600.0
        index.reclaim(timeout_s=1.0)
        index.fail_load("gone")  # the report finally arrives
        assert index.released == [6]
        index.check_invariant()


class TestDispatchIsRefusedRatherThanRisked:
    def test_a_hash_never_stored_is_refused(self):
        assert _index().request_load("a", 77, slot=0) is False

    def test_a_store_only_role_refuses_to_load(self):
        """`hashes` is populated from its own stores, so membership alone would
        have it vote a hit it will not serve."""
        index = _index(77, can_load=False)
        assert index.could_serve(77) is False
        assert index.request_load("a", 77, slot=0) is False

    def test_a_second_load_for_one_request_is_refused(self):
        """Reports are keyed by request, so the first completion would settle
        the second load's slot -- and lose an orphaned first slot entirely."""
        index = _index(1, 2)
        assert index.request_load("a", 1, slot=0) is True
        assert index.request_load("a", 2, slot=1) is False
        assert index.dispatched == 1
        index.check_invariant()


def test_the_stats_line_carries_the_invariants_three_terms():
    """Every counter here is reachable from `checkpoint_funnel`; a number that
    cannot be read is a defect that can only be found by reading the source."""
    index = _index(1, 2)
    index.request_load("a", 1, slot=0)
    index.request_load("b", 2, slot=1)
    index.complete_load("a")
    stats = index.stats()
    assert stats["loads_attempted"] == 2
    assert stats["loads_settled"] == 1
    assert stats["loads_outstanding"] == 1
    assert stats["indexed"] == 2
    assert "orphan_load_slots_reclaimed" in stats


class TestTheInvariantIsActuallyWiredUp:
    """An assertion nothing runs in production is not an assertion.

    `check_invariant` raises so a test can prove it fails; the serving path
    needs the opposite -- a bookkeeping fault must surface without taking the
    engine down. `stats()` is where the two meet, because it is the periodic
    path `checkpoint_funnel` already pulls.
    """

    def test_stats_audits_on_every_read(self):
        index = _index()
        assert index.stats()["invariant_violations"] == 0

        index.dispatched = 7  # corrupt the accounting behind the object's back

        assert index.stats()["invariant_violations"] == 1, "stats must audit"

    def test_a_violation_does_not_take_the_engine_down(self):
        index = _index()
        index.dispatched = 3
        index.stats()
        index.stats()
        assert index.invariant_violations == 2, "counted every read"

    def test_a_healthy_index_never_reports_a_violation(self):
        index = _index(1)
        index.request_load("r", 1, slot=0)
        index.stats()
        index.complete_load("r")
        index.stats()
        assert index.invariant_violations == 0
        assert index.released == []


class TestTheInvariantReachesAnOperator:
    """The audit is only worth having if its result leaves the process.

    The chain is `stats()` -> `PagedStateCheckpointCoordinator.checkpoint_fates`
    -> `BlockManager.state_checkpoint_fates` -> `checkpoint_funnel`, and the
    middle link is a `getattr(self.offload, "stats", None)` rather than a direct
    call. That indirection is deliberate (a test double need not carry
    counters), but it means the chain cannot be seen by grep -- so it is
    asserted here instead of left to be re-derived.
    """

    def _bm_with_index(self):
        from test_state_checkpoint import (
            PAGED_COPY_RUNTIME,
            make_block_manager,
            paged_copy_config,
        )

        bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
        index = _index()
        bm.state_offload = index
        bm.paged_state_checkpoints.attach_offload(index)
        return bm, index

    def test_a_violation_reaches_the_checkpoint_funnel(self):
        bm, index = self._bm_with_index()
        assert bm.checkpoint_funnel()["state_offload_invariant_violations"] == 0

        index.dispatched = 9  # corrupt the accounting behind the object's back

        assert bm.checkpoint_funnel()["state_offload_invariant_violations"] == 1

    def test_the_outstanding_count_reaches_the_funnel_too(self):
        """`loads_outstanding` is what says a request is parked on a report that
        cannot come, so an operator has to be able to see it."""
        bm, index = self._bm_with_index()
        index.note_stored(7)
        index.request_load("r", 7, slot=0)

        assert bm.checkpoint_funnel()["state_offload_loads_outstanding"] == 1


class TestARefusedKvLegSettlesItsStateLoad:
    """The path a review asked about: armed, then the KV leg refused.

    `request_load` has already counted a dispatch by the time
    `build_connector_meta` clears `_state_load_seqs`. If the refusal path did
    not settle it, the entry would stay outstanding for the life of the process
    and its request would be parked on a report nothing can produce. The
    terminal transition is `abandon_load` -- not `fail_load`, because nothing
    was attempted, so the hash must stay loadable for the next request.
    """

    def _scheduler_with(self, index, *, disown_ok=True):
        from types import SimpleNamespace

        from atom.model_engine.scheduler import Scheduler

        sched = object.__new__(Scheduler)
        sched.block_manager = SimpleNamespace(
            state_offload=index, disown_claimed_prefix=lambda seq: disown_ok
        )
        return sched

    def _seq(self, h):
        from types import SimpleNamespace

        return SimpleNamespace(
            id="r1", offload_joint=SimpleNamespace(load_hash=h, boundary_tokens=512)
        )

    def test_the_refusal_settles_the_dispatch(self):
        index = _index(7)
        index.request_load("r1", 7, slot=0)
        assert index.outstanding == 1

        sched = self._scheduler_with(index)
        assert sched._drop_state_load(self._seq(7)) is True

        assert index.outstanding == 0, "a dropped load must not stay in flight"
        index.check_invariant()

    def test_an_abandon_is_not_a_miss(self):
        """Nothing was attempted, so the hash stays loadable and no failure is
        counted -- forgetting it would send the next request over that prefix
        to a full recompute for no reason."""
        index = _index(7)
        index.request_load("r1", 7, slot=0)

        self._scheduler_with(index)._drop_state_load(self._seq(7))

        assert index.could_serve(7), "an abandon must not retract the hash"
        assert index.loads_failed == 0
        index.check_invariant()

    def test_a_seq_with_no_armed_load_settles_nothing(self):
        index = _index(7)
        self._scheduler_with(index)._drop_state_load(self._seq(-1))
        assert index.dispatched == 0
        index.check_invariant()
