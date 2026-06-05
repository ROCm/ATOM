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


class TestKVConnectorOutput:
    def test_defaults(self):
        out = KVConnectorOutput()
        assert out.finished_sending == set()
        assert out.finished_recving == set()
        assert out.expected_finished_count == 0

    def test_is_empty(self):
        assert KVConnectorOutput().is_empty()
        assert not KVConnectorOutput(finished_sending={"x"}).is_empty()
        assert not KVConnectorOutput(finished_recving={"x"}).is_empty()

    def test_repr(self):
        out = KVConnectorOutput(finished_sending={"a"}, finished_recving={"b"})
        r = repr(out)
        assert "sending" in r
        assert "recving" in r
