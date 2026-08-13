# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for KVOutputAggregator."""

import pytest

from atom.kv_transfer.disaggregation import KVConnectorOutput, KVOutputAggregator


class TestKVOutputAggregatorInit:
    def test_positive_world_size(self):
        agg = KVOutputAggregator(world_size=4)
        assert agg.world_size == 4

    def test_zero_world_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            KVOutputAggregator(world_size=0)

    def test_negative_world_size_raises(self):
        with pytest.raises(ValueError, match="positive"):
            KVOutputAggregator(world_size=-1)


class TestAggregateBasic:
    def test_empty_worker_outputs(self):
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate([])
        assert result.finished_sending == set()
        assert result.finished_recving == set()

    def test_all_empty(self):
        agg = KVOutputAggregator(world_size=3)
        result = agg.aggregate([KVConnectorOutput() for _ in range(3)])
        assert result.is_empty()

    def test_all_workers_report_same_sending(self):
        agg = KVOutputAggregator(world_size=3)
        outputs = [KVConnectorOutput(finished_sending={"r1"}) for _ in range(3)]
        result = agg.aggregate(outputs)
        assert result.finished_sending == {"r1"}
        assert result.finished_recving == set()

    def test_all_workers_report_same_recving(self):
        agg = KVOutputAggregator(world_size=2)
        outputs = [KVConnectorOutput(finished_recving={"r1"}) for _ in range(2)]
        result = agg.aggregate(outputs)
        assert result.finished_recving == {"r1"}

    def test_all_workers_report_same_loading(self):
        agg = KVOutputAggregator(world_size=2)
        outputs = [KVConnectorOutput(finished_loading={"r1"}) for _ in range(2)]
        result = agg.aggregate(outputs)
        assert result.finished_loading == {"r1"}
        assert result.finished_recving == set()

    def test_loading_failure_wins_after_all_workers_report(self):
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate(
            [
                KVConnectorOutput(finished_loading={"r1"}),
                KVConnectorOutput(failed_loading={"r1"}),
            ]
        )
        assert result.finished_loading == set()
        assert result.failed_loading == {"r1"}

    def test_partial_workers_not_emitted(self):
        agg = KVOutputAggregator(world_size=3)
        outputs = [
            KVConnectorOutput(finished_sending={"r1"}),
            KVConnectorOutput(finished_sending={"r1"}),
            KVConnectorOutput(),
        ]
        result = agg.aggregate(outputs)
        assert result.finished_sending == set()

    def test_counter_cleared_after_emission(self):
        """Once emitted, the request ID should not leak in internal state."""
        agg = KVOutputAggregator(world_size=2)
        outputs = [KVConnectorOutput(finished_sending={"r1"}) for _ in range(2)]
        agg.aggregate(outputs)
        assert agg.pending_count == (0, 0)


class TestAggregateMultiRound:
    def test_progressive_completion(self):
        agg = KVOutputAggregator(world_size=3)

        # Round 1: 2 of 3 workers done
        result = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={"r1"}),
                KVConnectorOutput(finished_sending={"r1"}),
                KVConnectorOutput(),
            ]
        )
        assert result.finished_sending == set()
        assert agg.pending_count == (1, 0)

        # Round 2: last worker reports
        result = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(),
                KVConnectorOutput(finished_sending={"r1"}),
            ]
        )
        assert result.finished_sending == {"r1"}
        assert agg.pending_count == (0, 0)

    def test_interleaved_send_recv(self):
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={"s1"}, finished_recving={"r1"}),
                KVConnectorOutput(finished_sending={"s1"}, finished_recving={"r1"}),
            ]
        )
        assert result.finished_sending == {"s1"}
        assert result.finished_recving == {"r1"}

    def test_multiple_requests_mixed_progress(self):
        agg = KVOutputAggregator(world_size=2)

        result = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={"a", "b"}),
                KVConnectorOutput(finished_sending={"a"}),
            ]
        )
        assert result.finished_sending == {"a"}
        assert "b" not in result.finished_sending

        result = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(finished_sending={"b"}),
            ]
        )
        assert result.finished_sending == {"b"}


class TestReset:
    def test_reset_clears_pending(self):
        agg = KVOutputAggregator(world_size=3)
        agg.aggregate(
            [
                KVConnectorOutput(finished_sending={"r1"}),
                KVConnectorOutput(),
                KVConnectorOutput(),
            ]
        )
        assert agg.pending_count == (1, 0)
        agg.reset()
        assert agg.pending_count == (0, 0)


class TestStateIndexedAggregation:
    """Tests for the state-indexed / state-index-failed union-quorum logic.

    I3 finding from Task 9b, promoted to Important: when one rank's codec.put
    fails and another's succeeds the hash sat in _seen_state_indexed forever
    because the failure was never reported, so quorum could never be reached.
    """

    def test_partial_store_emits_nothing_and_leaves_no_stale_key(self):
        """Leak test: rank 0 stored, rank 1 failed → nothing emitted, both
        internal dicts empty.

        Non-vacuousness: before the fix, _seen_state_indexed retained {9: {0}}
        forever. Remove the _seen_state_index_failed tracking from aggregate()
        and this test fails because the dicts are not empty.
        """
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate(
            [
                KVConnectorOutput(state_indexed={9}),
                KVConnectorOutput(state_index_failed={9}),
            ]
        )
        # Nothing must be emitted for a partially-stored hash (unloadable).
        assert result.state_indexed == set()
        # Both internal tracking dicts must be cleared — no stale key pinned.
        assert agg._seen_state_indexed == {}
        assert agg._seen_state_index_failed == {}

    def test_two_step_straddle_still_emits(self):
        """Guard against the wrong fix: cross-step spills must still resolve.

        Spills run on per-rank ThreadPoolExecutors; rank 0's completion can
        land in step 1 while rank 1's lands in step 2. Dropping non-quorum
        keys at end-of-step would destroy this valid pending spill.

        Non-vacuousness: an 'end-of-step prune' fix would empty _seen_state_indexed
        after step 1, so step 2 would see an empty dict and emit nothing. This
        test catches that regression by asserting the hash IS emitted at step 2.
        """
        agg = KVOutputAggregator(world_size=2)

        # Step 1: only rank 0 reports.
        result1 = agg.aggregate(
            [
                KVConnectorOutput(state_indexed={7}),
                KVConnectorOutput(),
            ]
        )
        assert result1.state_indexed == set(), "must not emit after one rank"
        assert 7 in agg._seen_state_indexed, "key must be retained for step 2"

        # Step 2: rank 1 reports.
        result2 = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(state_indexed={7}),
            ]
        )
        assert result2.state_indexed == {7}, "must emit once both ranks reported"
        assert agg._seen_state_indexed == {}

    def test_all_ranks_stored_emits(self):
        """Happy path: all ranks succeed → hash emitted on state_indexed."""
        agg = KVOutputAggregator(world_size=3)
        result = agg.aggregate(
            [KVConnectorOutput(state_indexed={42}) for _ in range(3)]
        )
        assert result.state_indexed == {42}
        assert agg._seen_state_indexed == {}
        assert agg._seen_state_index_failed == {}

    def test_all_ranks_failed_emits_nothing_and_clears(self):
        """All ranks failed → hash dropped, nothing emitted, dicts cleared."""
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate(
            [KVConnectorOutput(state_index_failed={5}) for _ in range(2)]
        )
        assert result.state_indexed == set()
        assert agg._seen_state_indexed == {}
        assert agg._seen_state_index_failed == {}

    def test_state_index_failed_included_in_is_empty(self):
        """is_empty() must reflect the new field."""
        assert KVConnectorOutput().is_empty()
        assert not KVConnectorOutput(state_index_failed={1}).is_empty()

    def test_state_index_failed_included_in_repr(self):
        """__repr__ must include the new field so log messages are complete."""
        r = repr(KVConnectorOutput(state_index_failed={3}))
        assert "state_index_failed" in r

    def test_reset_clears_state_index_failed(self):
        """reset() must clear _seen_state_index_failed alongside the others."""
        agg = KVOutputAggregator(world_size=2)
        agg.aggregate(
            [
                KVConnectorOutput(state_index_failed={99}),
                KVConnectorOutput(),
            ]
        )
        assert agg._seen_state_index_failed  # non-empty before reset
        agg.reset()
        assert agg._seen_state_index_failed == {}


class TestKVConnectorOutput:
    def test_defaults(self):
        out = KVConnectorOutput()
        assert out.finished_sending == set()
        assert out.finished_recving == set()
        assert out.finished_loading == set()
        assert out.expected_finished_count == 0

    def test_is_empty(self):
        assert KVConnectorOutput().is_empty()
        assert not KVConnectorOutput(finished_sending={"x"}).is_empty()
        assert not KVConnectorOutput(finished_recving={"x"}).is_empty()
        assert not KVConnectorOutput(finished_loading={"x"}).is_empty()

    def test_repr(self):
        out = KVConnectorOutput(finished_sending={"a"}, finished_recving={"b"})
        r = repr(out)
        assert "sending" in r
        assert "recving" in r
