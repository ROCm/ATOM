# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for KVOutputAggregator."""

from dataclasses import FrozenInstanceError

import pytest

from atom.kv_transfer.disaggregation import (
    ConnectorCompletion,
    KVConnectorOutput,
    KVOutputAggregator,
)
from atom.kv_transfer.disaggregation.types import (
    STATE_CHECKPOINT_STAGING_CHANNEL,
    LoadOperationId,
    SaveOperationId,
    SendOperationId,
)
from atom.kv_transfer.offload.hybrid.dsv4.connector import (
    DSV4_CHECKPOINT_SAVE_CHANNEL,
)


def _completion(channel, operation_id, *, succeeded=True):
    return ConnectorCompletion(
        channel=channel,
        operation_id=operation_id,
        succeeded=succeeded,
    )


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

    def test_send_generations_never_cross_complete(self):
        agg = KVOutputAggregator(world_size=2)
        gen0 = SendOperationId("r1", 0)
        gen1 = SendOperationId("r1", 1)

        mixed = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={gen0}),
                KVConnectorOutput(finished_sending={gen1}),
            ]
        )
        assert mixed.finished_sending == set()

        matched = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={gen1}),
                KVConnectorOutput(finished_sending={gen0}),
            ]
        )
        assert matched.finished_sending == {gen0, gen1}

    def test_late_send_generation_duplicate_is_tombstoned(self):
        agg = KVOutputAggregator(world_size=2)
        operation = SendOperationId("r1", 3)

        terminal = agg.aggregate(
            [
                KVConnectorOutput(finished_sending={operation}),
                KVConnectorOutput(finished_sending={operation}),
            ]
        )
        assert terminal.finished_sending == {operation}

        late = agg.aggregate(
            [KVConnectorOutput(finished_sending={operation}), KVConnectorOutput()]
        )
        assert late.finished_sending == set()
        assert agg.pending_count == (0, 0)
        assert agg.terminal_send_tombstone_count == 1

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

    def test_load_generations_never_cross_complete(self):
        agg = KVOutputAggregator(world_size=2)
        gen0 = LoadOperationId("r1", 0)
        gen1 = LoadOperationId("r1", 1)

        mixed = agg.aggregate(
            [
                KVConnectorOutput(finished_loading={gen0}),
                KVConnectorOutput(finished_loading={gen1}),
            ]
        )
        assert mixed.finished_loading == set()

        matched = agg.aggregate(
            [
                KVConnectorOutput(failed_loading={gen1}),
                KVConnectorOutput(finished_loading={gen0}),
            ]
        )
        assert matched.finished_loading == {gen0}
        assert matched.failed_loading == {gen1}

    def test_late_load_generation_duplicate_is_tombstoned(self):
        agg = KVOutputAggregator(world_size=2, terminal_tombstone_limit=2)
        operations = [LoadOperationId("r1", nonce) for nonce in range(3)]
        for operation in operations:
            result = agg.aggregate(
                [
                    KVConnectorOutput(finished_loading={operation}),
                    KVConnectorOutput(finished_loading={operation}),
                ]
            )
            assert result.finished_loading == {operation}

        late = agg.aggregate(
            [KVConnectorOutput(finished_loading={operations[-1]}), KVConnectorOutput()]
        )
        assert late.finished_loading == set()
        assert agg.pending_count == (0, 0)
        assert agg.terminal_load_tombstone_count == 2

    def test_all_workers_report_sidecar_save_success(self):
        agg = KVOutputAggregator(world_size=2)
        completion = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")
        result = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={completion}),
                KVConnectorOutput(connector_completions={completion}),
            ]
        )
        assert result.connector_completions == {completion}
        assert agg.pending_count == (0, 0)

    def test_sidecar_save_failure_waits_for_every_terminal_report(self):
        agg = KVOutputAggregator(world_size=2)
        failed = _completion(
            DSV4_CHECKPOINT_SAVE_CHANNEL,
            "r1",
            succeeded=False,
        )
        succeeded = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")

        partial = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={failed}),
                KVConnectorOutput(),
            ]
        )

        assert partial.connector_completions == set()
        assert agg.pending_count == (0, 1)

        terminal = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(connector_completions={succeeded}),
            ]
        )

        assert terminal.connector_completions == {failed}
        assert agg.pending_count == (0, 0)

    def test_same_worker_contradictory_completion_keeps_failure(self):
        agg = KVOutputAggregator(world_size=2)
        succeeded = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")
        failed = _completion(
            DSV4_CHECKPOINT_SAVE_CHANNEL,
            "r1",
            succeeded=False,
        )

        terminal = agg.aggregate(
            [
                KVConnectorOutput(
                    connector_completions={succeeded, failed},
                ),
                KVConnectorOutput(connector_completions={succeeded}),
            ]
        )

        assert terminal.connector_completions == {failed}

    def test_partial_sidecar_save_success_is_not_emitted(self):
        agg = KVOutputAggregator(world_size=2)
        completion = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")
        result = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={completion}),
                KVConnectorOutput(),
            ]
        )
        assert result.connector_completions == set()
        assert agg.pending_count == (0, 1)

    def test_connector_completion_channels_never_cross_complete(self):
        agg = KVOutputAggregator(world_size=2)
        sidecar = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, 42)
        checkpoint = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 42)

        mixed = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={sidecar}),
                KVConnectorOutput(connector_completions={checkpoint}),
            ]
        )
        assert mixed.connector_completions == set()
        assert agg.pending_count == (0, 2)

        matched = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={checkpoint}),
                KVConnectorOutput(connector_completions={sidecar}),
            ]
        )
        assert matched.connector_completions == {sidecar, checkpoint}
        assert agg.pending_count == (0, 0)

    def test_checkpoint_staging_waits_for_all_workers(self):
        agg = KVOutputAggregator(world_size=2)
        completion = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 42)

        partial = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={completion}),
                KVConnectorOutput(),
            ]
        )
        assert partial.connector_completions == set()
        assert agg.pending_count == (0, 1)

        terminal = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(connector_completions={completion}),
            ]
        )
        assert terminal.connector_completions == {completion}
        assert agg.pending_count == (0, 0)

    def test_any_checkpoint_abort_invalidates_after_all_workers_terminal(self):
        agg = KVOutputAggregator(world_size=3)
        succeeded = _completion(STATE_CHECKPOINT_STAGING_CHANNEL, 9)
        aborted = _completion(
            STATE_CHECKPOINT_STAGING_CHANNEL,
            9,
            succeeded=False,
        )

        partial = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={succeeded}),
                KVConnectorOutput(connector_completions={aborted}),
                KVConnectorOutput(),
            ]
        )
        assert partial.connector_completions == set()

        terminal = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(),
                KVConnectorOutput(connector_completions={succeeded}),
            ]
        )
        assert terminal.connector_completions == {aborted}

        late = agg.aggregate(
            [KVConnectorOutput(connector_completions={succeeded})]
            + [
                KVConnectorOutput(),
                KVConnectorOutput(),
            ]
        )
        assert late.is_empty()
        assert agg.pending_count == (0, 0)

    def test_partial_workers_not_emitted(self):
        agg = KVOutputAggregator(world_size=3)
        outputs = [
            KVConnectorOutput(finished_sending={"r1"}),
            KVConnectorOutput(finished_sending={"r1"}),
            KVConnectorOutput(),
        ]
        result = agg.aggregate(outputs)
        assert result.finished_sending == set()

    def test_save_generations_never_cross_complete(self):
        agg = KVOutputAggregator(world_size=2)
        gen0 = SaveOperationId("r1", 0)
        gen1 = SaveOperationId("r1", 1)

        mixed = agg.aggregate(
            [
                KVConnectorOutput(finished_saving={gen0}),
                KVConnectorOutput(finished_saving={gen1}),
            ]
        )

        assert mixed.finished_saving == set()
        assert agg.pending_count == (0, 2)

        matched = agg.aggregate(
            [
                KVConnectorOutput(finished_saving={gen1}),
                KVConnectorOutput(finished_saving={gen0}),
            ]
        )

        assert matched.finished_saving == {gen0, gen1}
        assert agg.pending_count == (0, 0)

    def test_sidecar_failure_is_scoped_to_generation(self):
        agg = KVOutputAggregator(world_size=2)
        gen0 = SaveOperationId(7, 0)
        gen1 = SaveOperationId(7, 1)
        failed_gen0 = _completion(
            DSV4_CHECKPOINT_SAVE_CHANNEL,
            gen0,
            succeeded=False,
        )
        succeeded_gen0 = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, gen0)
        succeeded_gen1 = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, gen1)

        mixed = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={failed_gen0}),
                KVConnectorOutput(connector_completions={succeeded_gen1}),
            ]
        )
        assert mixed.connector_completions == set()

        matched = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={succeeded_gen1}),
                KVConnectorOutput(connector_completions={succeeded_gen0}),
            ]
        )

        assert matched.connector_completions == {succeeded_gen1, failed_gen0}
        assert agg.pending_count == (0, 0)

    def test_counter_cleared_after_emission(self):
        """Once emitted, the request ID should not leak in internal state."""
        agg = KVOutputAggregator(world_size=2)
        outputs = [KVConnectorOutput(finished_sending={"r1"}) for _ in range(2)]
        agg.aggregate(outputs)
        assert agg.pending_count == (0, 0)

    def test_late_generation_save_duplicate_does_not_recreate_pending_state(self):
        agg = KVOutputAggregator(world_size=2)
        operation = SaveOperationId("r1", 10)
        terminal = [
            KVConnectorOutput(finished_saving={operation}),
            KVConnectorOutput(finished_saving={operation}),
        ]

        assert agg.aggregate(terminal).finished_saving == {operation}
        late = agg.aggregate(
            [KVConnectorOutput(finished_saving={operation}), KVConnectorOutput()]
        )

        assert late.finished_saving == set()
        assert agg.pending_count == (0, 0)

    @pytest.mark.parametrize("succeeded", [True, False])
    def test_late_generation_sidecar_duplicate_is_ignored(self, succeeded):
        agg = KVOutputAggregator(world_size=2)
        operation = SaveOperationId("r1", 11)
        completion = _completion(
            DSV4_CHECKPOINT_SAVE_CHANNEL,
            operation,
            succeeded=succeeded,
        )
        outputs = [
            KVConnectorOutput(connector_completions={completion}),
            KVConnectorOutput(connector_completions={completion}),
        ]

        assert agg.aggregate(outputs).connector_completions == {completion}
        late = agg.aggregate(
            [
                KVConnectorOutput(connector_completions={completion}),
                KVConnectorOutput(),
            ]
        )

        assert late.is_empty()
        assert agg.pending_count == (0, 0)

    def test_terminal_tombstones_are_bounded_and_reset(self):
        agg = KVOutputAggregator(world_size=1, terminal_tombstone_limit=2)
        operations = [SaveOperationId("r", generation) for generation in range(3)]

        for operation in operations:
            agg.aggregate([KVConnectorOutput(finished_saving={operation})])

        assert agg.terminal_tombstone_count == (2, 0)
        # The oldest identity was evicted and may be treated as a new operation.
        assert agg.aggregate(
            [KVConnectorOutput(finished_saving={operations[0]})]
        ).finished_saving == {operations[0]}

        agg.reset()
        assert agg.terminal_tombstone_count == (0, 0)

    def test_legacy_raw_save_ids_are_not_tombstoned_for_reuse(self):
        agg = KVOutputAggregator(world_size=2)
        terminal = [
            KVConnectorOutput(finished_saving={"r1"}),
            KVConnectorOutput(finished_saving={"r1"}),
        ]
        assert agg.aggregate(terminal).finished_saving == {"r1"}

        reused = agg.aggregate(
            [KVConnectorOutput(finished_saving={"r1"}), KVConnectorOutput()]
        )

        assert reused.finished_saving == set()
        assert agg.pending_count == (0, 1)


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

    def test_reset_clears_partial_sidecar_reports(self):
        agg = KVOutputAggregator(world_size=2)
        failed = _completion(
            DSV4_CHECKPOINT_SAVE_CHANNEL,
            "r1",
            succeeded=False,
        )
        succeeded = _completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "r1")
        agg.aggregate(
            [
                KVConnectorOutput(connector_completions={failed}),
                KVConnectorOutput(),
            ]
        )
        assert agg.pending_count == (0, 1)

        agg.reset()

        assert agg.pending_count == (0, 0)
        result = agg.aggregate(
            [
                KVConnectorOutput(),
                KVConnectorOutput(connector_completions={succeeded}),
            ]
        )
        assert result.is_empty()


class TestKVConnectorOutput:
    def test_save_operation_id_is_immutable_hashable_and_validated(self):
        operation = SaveOperationId("r1", 0)

        assert {operation} == {SaveOperationId("r1", 0)}
        with pytest.raises(FrozenInstanceError):
            operation.generation = 1
        with pytest.raises(ValueError, match="nonnegative"):
            SaveOperationId("r1", -1)

    def test_defaults(self):
        out = KVConnectorOutput()
        assert out.finished_sending == set()
        assert out.finished_recving == set()
        assert out.finished_loading == set()
        assert out.connector_completions == set()
        assert out.expected_finished_count == 0

    def test_is_empty(self):
        assert KVConnectorOutput().is_empty()
        assert not KVConnectorOutput(finished_sending={"x"}).is_empty()
        assert not KVConnectorOutput(finished_recving={"x"}).is_empty()
        assert not KVConnectorOutput(finished_loading={"x"}).is_empty()
        assert not KVConnectorOutput(
            connector_completions={_completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "x")}
        ).is_empty()
        assert not KVConnectorOutput(
            connector_completions={
                _completion(
                    STATE_CHECKPOINT_STAGING_CHANNEL,
                    1,
                    succeeded=False,
                )
            }
        ).is_empty()

    def test_repr(self):
        out = KVConnectorOutput(
            finished_sending={"a"},
            finished_recving={"b"},
            connector_completions={_completion(DSV4_CHECKPOINT_SAVE_CHANNEL, "c")},
        )
        r = repr(out)
        assert "sending" in r
        assert "recving" in r
        assert "connector_completions" in r
        assert DSV4_CHECKPOINT_SAVE_CHANNEL in r
