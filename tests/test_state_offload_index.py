# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The state offload index's contract, asserted rather than assumed.

The previous implementation spread ``dispatched == settled + outstanding`` over
three owners in two processes, so no object could state it and no test could
assert it. Every load-path defect found across six review rounds was that one
gap surfacing again. This file exists so a seventh cannot.

Pure Python: no torch, no aiter, no GPU. It runs on the CPU CI runner.
"""

import pytest

from atom.model_engine.state_offload import StateOffloadIndex


def make_index(*, can_store=True, can_load=True, chunk_tokens=1024):
    released: list[int] = []
    index = StateOffloadIndex(
        can_store=can_store,
        can_load=can_load,
        chunk_tokens=chunk_tokens,
        release_slot=released.append,
    )
    return index, released


class TestTheInvariant:
    """dispatched == settled + outstanding, on one object."""

    def test_hand_out_three_settle_two_leaves_one_outstanding(self):
        index, _ = make_index()
        for h in (11, 22, 33):
            index.note_stored(h)

        assert index.dispatch("r1", 11, slot=0)
        assert index.dispatch("r2", 22, slot=1)
        assert index.dispatch("r3", 33, slot=2)
        index.check_invariant()
        assert (index.dispatched, index.settled, index.outstanding) == (3, 0, 3)

        index.settle("r1", ok=True)
        index.settle("r2", ok=False, missing=True)

        assert index.dispatched == 3
        assert index.settled == 2
        assert index.outstanding == 1
        index.check_invariant()

    def test_every_terminal_transition_counts_exactly_once(self):
        index, _ = make_index()
        for h in (1, 2, 3, 4):
            index.note_stored(h)
        index.dispatch("a", 1, slot=0)
        index.dispatch("b", 2, slot=1)
        index.dispatch("c", 3, slot=2)
        index.dispatch("d", 4, slot=3)

        index.settle("a", ok=True)
        index.settle("b", ok=False)
        index.abandon("c")
        index.orphan("d")
        index.reclaim(timeout_s=0.0000001)

        assert index.outstanding == 0
        assert index.settled == 4
        index.check_invariant()

    def test_a_duplicate_settle_is_not_double_counted(self):
        index, _ = make_index()
        index.note_stored(7)
        index.dispatch("r", 7, slot=0)
        index.settle("r", ok=True)
        index.settle("r", ok=True)
        index.settle("r", ok=False)
        assert index.settled == 1
        index.check_invariant()

    def test_settling_a_load_that_was_never_dispatched_is_inert(self):
        index, released = make_index()
        index.settle("ghost", ok=True)
        index.abandon("ghost")
        assert (index.dispatched, index.settled, index.outstanding) == (0, 0, 0)
        assert released == []
        index.check_invariant()

    def test_the_invariant_can_actually_fail(self):
        """A test that cannot fail proves nothing about the one that can."""
        index, _ = make_index()
        index.dispatched = 5
        with pytest.raises(AssertionError, match="dispatched != settled"):
            index.check_invariant()


class TestCouldServeIsTheOnlyPredicate:
    def test_a_store_only_role_never_votes_for_its_own_stores(self):
        """The kv_producer trap: `hashes` fills from this rank's own stores, so
        a bare `h in hashes` votes a tier hit the load path then refuses."""
        index, _ = make_index(can_load=False)
        index.note_stored(42)
        assert 42 in index.hashes
        assert not index.could_serve(42)
        assert not index.dispatch("r", 42, slot=0)
        assert index.dispatched == 0
        index.check_invariant()

    def test_dispatch_and_could_serve_cannot_disagree(self):
        index, _ = make_index()
        index.note_stored(5)
        for h in (5, 6):
            assert index.dispatch(f"req-{h}", h, slot=h) is index.could_serve(h)
        index.check_invariant()

    def test_a_missing_load_un_advertises_but_a_fused_failure_does_not(self):
        """`ok=False` is the verdict for BOTH legs. Only the state leg missing
        may retract the hash; retracting on a failed KV leg would permanently
        deny state bytes that are still present."""
        index, _ = make_index()
        index.note_stored(100)
        index.note_stored(200)

        index.dispatch("kv-failed", 100, slot=0)
        index.settle("kv-failed", ok=False)
        assert index.could_serve(100), "a failed KV leg must not retract state"

        index.dispatch("state-missed", 200, slot=1)
        index.settle("state-missed", ok=False, missing=True)
        assert not index.could_serve(200)
        index.check_invariant()


class TestOneLoadPerRequest:
    def test_a_second_load_is_refused_while_the_first_is_in_flight(self):
        """The orphan-slot overwrite: with a bare re-registration the second
        teardown overwrites the first parked slot, which then vanishes from the
        reclaimer's view (it iterates the dict) and leaks permanently."""
        index, released = make_index()
        index.note_stored(9)
        assert index.dispatch("r", 9, slot=3)
        assert not index.dispatch("r", 9, slot=4)
        assert index.loads_refused_inflight == 1
        assert index.outstanding == 1
        index.check_invariant()

        index.orphan("r")
        index.settle("r", ok=True)
        assert released == [3], "the first slot must be the one released"
        index.check_invariant()

    def test_a_request_may_load_again_once_its_first_load_settled(self):
        index, _ = make_index()
        index.note_stored(9)
        index.dispatch("r", 9, slot=3)
        index.settle("r", ok=True)
        assert index.dispatch("r", 9, slot=4)
        index.check_invariant()


class TestOrphanedSlots:
    def test_a_live_request_keeps_its_own_slot(self):
        index, released = make_index()
        index.note_stored(1)
        index.dispatch("r", 1, slot=8)
        index.settle("r", ok=True)
        assert released == [], "a live request releases its own slots"
        index.check_invariant()

    def test_orphan_on_a_request_with_no_load_leaves_the_slot_with_the_caller(self):
        index, released = make_index()
        assert index.orphan("r") is False
        assert released == []

    def test_an_orphaned_slot_comes_back_when_the_report_lands(self):
        index, released = make_index()
        index.note_stored(1)
        index.dispatch("r", 1, slot=8)
        assert index.orphan("r") is True
        index.settle("r", ok=False, missing=True)
        assert released == [8]
        index.check_invariant()

    def test_reclaim_never_yanks_a_live_requests_slot(self):
        """Releasing a slot a worker may still be scattering into writes the
        loaded image over the next request's live recurrent state."""
        index, released = make_index()
        index.note_stored(1)
        index.dispatch("live", 1, slot=8)
        assert index.reclaim(timeout_s=0.0000001) == 0
        assert released == []
        assert index.outstanding == 1
        index.check_invariant()

    def test_reclaim_frees_only_the_timed_out_orphans(self):
        index, released = make_index()
        for h in (1, 2):
            index.note_stored(h)
        index.dispatch("old", 1, slot=8)
        index.orphan("old")
        index.dispatch("new", 2, slot=9)
        index.orphan("new")
        # `new` is younger than the window, `old` is not.
        index._outstanding["old"].at -= 100.0

        assert index.reclaim(timeout_s=10.0) == 1
        assert released == [8]
        assert index.outstanding == 1
        index.check_invariant()

    def test_a_disabled_reclaimer_frees_nothing(self):
        index, released = make_index()
        index.note_stored(1)
        index.dispatch("r", 1, slot=8)
        index.orphan("r")
        assert index.reclaim(timeout_s=0.0) == 0
        assert released == []
        index.check_invariant()


class TestTheMembershipIndexStaysBounded:
    def test_hashes_and_the_lru_move_in_lockstep(self):
        index, _ = make_index()
        for h in range(50):
            index.note_stored(h)
        for h in range(0, 50, 2):
            index.forget(h)
        assert len(index.hashes) == 25
        index.check_invariant()

    def test_the_coldest_hash_is_dropped_on_overflow(self):
        index, _ = make_index()
        index._hash_cap = 4
        for h in (1, 2, 3, 4):
            index.note_stored(h)
        index.note_stored(5)
        assert 1 not in index.hashes
        assert index.hashes == {2, 3, 4, 5}
        assert index.hashes_evicted == 1
        index.check_invariant()

    def test_restoring_a_hash_makes_it_young_again(self):
        index, _ = make_index()
        index._hash_cap = 3
        for h in (1, 2, 3):
            index.note_stored(h)
        index.note_stored(1)
        index.note_stored(4)
        assert 1 in index.hashes, "re-stored hash must not be the eviction victim"
        assert 2 not in index.hashes
        index.check_invariant()

    def test_forgetting_an_absent_hash_is_inert(self):
        index, _ = make_index()
        index.forget(999)
        index.check_invariant()


class TestStatsAreReachable:
    def test_every_counter_the_tester_asserts_on_is_exported(self):
        """Pass 2 of the accuracy run asserts on these by name. In the previous
        implementation every observability hook added for this tier was itself
        unreachable, which is why six defects were silent."""
        index, _ = make_index()
        exported = set(index.stats())
        for name in (
            "stores_attempted",
            "stores_completed",
            "loads_dispatched",
            "loads_completed",
            "loads_outstanding",
            "indexed",
        ):
            assert name in exported

    def test_the_chunk_grid_is_reported_not_re_derived(self):
        index, _ = make_index(chunk_tokens=512)
        assert index.chunk_tokens == 512
