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


class TestPromotedGpuPages:
    """SparseKV promote-done signal: cold tiers are per-rank and the scheduler's
    page budget mirrors ONE rank, so a request's credit is the minimum across
    ranks and is only released once every rank has reported."""

    def test_min_across_workers(self):
        agg = KVOutputAggregator(world_size=3)
        result = agg.aggregate(
            [
                KVConnectorOutput(promoted_gpu_pages={"r1": 5}),
                KVConnectorOutput(promoted_gpu_pages={"r1": 3}),
                KVConnectorOutput(promoted_gpu_pages={"r1": 4}),
            ]
        )
        assert result.promoted_gpu_pages == {"r1": 3}

    def test_withheld_until_all_workers_report(self):
        agg = KVOutputAggregator(world_size=2)
        first = agg.aggregate(
            [KVConnectorOutput(promoted_gpu_pages={"r1": 4}), KVConnectorOutput()]
        )
        assert first.promoted_gpu_pages == {}
        second = agg.aggregate(
            [KVConnectorOutput(), KVConnectorOutput(promoted_gpu_pages={"r1": 2})]
        )
        assert second.promoted_gpu_pages == {"r1": 2}

    def test_zero_report_yields_no_credit(self):
        # A rank whose GPU tier was full promotes nothing; the request then gets
        # no host-budget relief, but the signal still resolves.
        agg = KVOutputAggregator(world_size=2)
        result = agg.aggregate(
            [
                KVConnectorOutput(promoted_gpu_pages={"r1": 4}),
                KVConnectorOutput(promoted_gpu_pages={"r1": 0}),
            ]
        )
        assert result.promoted_gpu_pages == {"r1": 0}

    def test_repeat_report_from_same_worker_accumulates(self):
        agg = KVOutputAggregator(world_size=2)
        agg.aggregate(
            [KVConnectorOutput(promoted_gpu_pages={"r1": 2}), KVConnectorOutput()]
        )
        result = agg.aggregate(
            [
                KVConnectorOutput(promoted_gpu_pages={"r1": 1}),
                KVConnectorOutput(promoted_gpu_pages={"r1": 3}),
            ]
        )
        assert result.promoted_gpu_pages == {"r1": 3}

    def test_failed_recv_drops_partial_fan_in(self):
        # A rank whose recv failed never enqueues a promote, so the fan-in for
        # this request can never complete; retaining it would leak forever.
        agg = KVOutputAggregator(world_size=2)
        agg.aggregate(
            [KVConnectorOutput(promoted_gpu_pages={"r1": 4}), KVConnectorOutput()]
        )
        agg.aggregate(
            [
                KVConnectorOutput(finished_recving={"r1"}),
                KVConnectorOutput(failed_recving={"r1"}),
            ]
        )
        assert agg._seen_promoted == {}

    def test_emitted_once(self):
        agg = KVOutputAggregator(world_size=1)
        agg.aggregate([KVConnectorOutput(promoted_gpu_pages={"r1": 4})])
        assert agg.aggregate([KVConnectorOutput()]).promoted_gpu_pages == {}


class TestPartialFanIn:
    """A dropped per-worker report is unrecoverable: the connector clears its
    finished set once handed over, so the request would never complete on every
    rank again. Partial rounds must therefore accumulate, not be discarded."""

    def test_partial_round_completes_on_a_later_call(self):
        agg = KVOutputAggregator(world_size=4)
        # Round 1: only workers 0 and 1 reported before the collector timed out.
        partial = agg.aggregate([KVConnectorOutput(finished_recving={"r1"})] * 2)
        assert partial.finished_recving == set()

        # Round 2: the late workers report. Combined with what round 1 retained,
        # the fan-in now covers all four and the request is released — which is
        # exactly what discarding the partial round used to make impossible.
        rest = agg.aggregate(
            [KVConnectorOutput() for _ in range(2)]
            + [KVConnectorOutput(finished_recving={"r1"})] * 2
        )
        assert rest.finished_recving == {"r1"}

    def test_same_worker_indices_do_not_double_count(self):
        agg = KVOutputAggregator(world_size=4)
        for _ in range(5):
            out = agg.aggregate([KVConnectorOutput(finished_recving={"r1"})] * 2)
            assert out.finished_recving == set()
        assert agg.aggregate(
            [KVConnectorOutput(finished_recving={"r1"}) for _ in range(4)]
        ).finished_recving == {"r1"}


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
