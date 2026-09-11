# SPDX-License-Identifier: MIT
# PP-stage offload KV status aggregation (GPU-free).

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
    from atom.kv_transfer.disaggregation.types import (
        ConnectorCompletion,
        KVConnectorOutput,
        SaveOperationId,
        SaveSourceGroupId,
    )
    from atom.model_engine.pp_engine_core import PPEngineCoreProc


class FakeScheduler:
    def __init__(self):
        self.outputs = []

    def _update_from_kv_xfer_finished(self, out):
        self.outputs.append(out)

    def released_sending(self):
        rel = set()
        for out in self.outputs:
            rel |= set(out.finished_sending or ())
        return rel

    def released_saving(self):
        rel = set()
        for out in self.outputs:
            rel |= set(out.finished_saving or ())
        return rel


class FakeRunnerMgr:
    """Returns one queued worker-side KVConnectorOutput per poll."""

    def __init__(self, outputs):
        self._outputs = list(outputs)

    def call_func_with_aggregation(self, name):
        assert name == "async_proc_aggregation"
        return self._outputs.pop(0) if self._outputs else KVConnectorOutput()


class FakePPTransport:
    """Returns one queued list of (pp_rank, output) per poll."""

    def __init__(self, messages):
        self._messages = list(messages)

    def recv_kv_status(self, timeout_ms=0):
        return self._messages.pop(0) if self._messages else []


def _head(pp_size, local_outputs, downstream_messages=()):
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.kv_transfer_enabled = True
    proc.pp_size = pp_size
    proc._pp_kv_aggregator = None
    proc._held_sending = {}
    proc.scheduler = FakeScheduler()
    proc.runner_mgr = FakeRunnerMgr(local_outputs)
    proc.pp_transport = FakePPTransport(downstream_messages)
    return proc


def test_send_waits_for_every_pp_stage_save():
    proc = _head(
        pp_size=3,
        local_outputs=[
            KVConnectorOutput(finished_sending={"a"}, finished_saving={"a"}),
            KVConnectorOutput(),
        ],
        downstream_messages=[
            [(1, KVConnectorOutput(finished_saving={"a"}))],
            [(2, KVConnectorOutput(finished_saving={"a"}))],
        ],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == set()  # stage 2 still saving
    assert proc._held_sending == {"a": ("a", {"a"})}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}
    assert proc.scheduler.released_saving() == {"a"}
    assert proc._held_sending == {}


def test_send_pairs_with_a_save_operation_id():
    # The offload connector reports a SaveOperationId(req_id, generation) once
    # it tracks save generations, while mooncake reports a bare request id.
    # Both have to collapse onto the request before they can be paired; keying
    # the two sides differently releases every send unheld and lets the head
    # free blocks a downstream stage is still saving from.
    op = SaveOperationId(9, 2)
    proc = _head(
        pp_size=2,
        local_outputs=[
            KVConnectorOutput(finished_sending={9}, finished_saving={op}),
            KVConnectorOutput(),
        ],
        downstream_messages=[
            [],
            [(1, KVConnectorOutput(finished_saving={op}))],
        ],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == set()  # stage 1 still saving
    assert proc._held_sending == {"9": (9, {op})}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == {op}
    assert proc._held_sending == {}


def test_send_waits_for_every_save_generation():
    # A chunked prefill saves once per chunk, so the pairing rank flushes the
    # send together with every generation it accumulated. The head must hold
    # the send until each of those generations has reached PP quorum, not just
    # the first one — the stages lag each other, and a stage still short of
    # quorum is still reading the blocks the send would free.
    g2, g3 = SaveOperationId(9, 2), SaveOperationId(9, 3)
    proc = _head(
        pp_size=2,
        local_outputs=[
            KVConnectorOutput(finished_sending={9}, finished_saving={g2, g3}),
            KVConnectorOutput(),
            KVConnectorOutput(),
        ],
        downstream_messages=[
            [],
            [(1, KVConnectorOutput(finished_saving={g2}))],
            [(1, KVConnectorOutput(finished_saving={g3}))],
        ],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == set()
    assert proc._held_sending == {"9": (9, {g2, g3})}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == set()  # generation 3 pending
    assert proc.scheduler.released_saving() == {g2}
    assert proc._held_sending == {"9": (9, {g3})}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == {g2, g3}
    assert proc._held_sending == {}


def test_send_without_a_save_is_not_held():
    # Once the aggregator exists, a later send-only request (prompt shorter
    # than the offload chunk, or already persisted) must still pass straight
    # through — no finished_saving is ever coming for it.
    proc = _head(
        pp_size=2,
        local_outputs=[
            KVConnectorOutput(finished_sending={"a"}, finished_saving={"a"}),
            KVConnectorOutput(finished_sending={"b"}),
        ],
        downstream_messages=[[(1, KVConnectorOutput(finished_saving={"a"}))], []],
    )

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a", "b"}
    assert proc._held_sending == {}


def test_send_passes_through_before_any_offload_activity():
    proc = _head(pp_size=2, local_outputs=[KVConnectorOutput(finished_sending={"a"})])
    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}
    assert proc._pp_kv_aggregator is None


def test_recv_bypasses_the_aggregator():
    proc = _head(
        pp_size=2,
        local_outputs=[KVConnectorOutput(finished_recving={"a"}, failed_recving={"b"})],
    )
    proc._poll_kv_transfer_progress()
    assert proc.scheduler.outputs[0].finished_recving == {"a"}
    assert proc.scheduler.outputs[0].failed_recving == {"b"}


def test_aggregator_requires_all_stages():
    agg = PPKVAggregator(3)
    assert agg.ingest(0, KVConnectorOutput(finished_saving={"a"})).is_empty()
    assert agg.ingest(1, KVConnectorOutput(finished_saving={"a"})).is_empty()
    assert agg.ingest(2, KVConnectorOutput(finished_saving={"a"})).finished_saving == {
        "a"
    }


def test_load_failure_waits_for_every_stage():
    # Reporting at the first failing stage wakes the request for recompute
    # into blocks the other stages are still loading into.
    agg = PPKVAggregator(3)
    assert agg.ingest(0, KVConnectorOutput(failed_loading={"a"})).is_empty()
    assert agg.has_pending() is True

    assert agg.ingest(1, KVConnectorOutput(finished_loading={"a"})).is_empty()
    assert agg.has_pending() is True

    out = agg.ingest(2, KVConnectorOutput(finished_loading={"a"}))
    assert out.failed_loading == {"a"}
    assert out.finished_loading == set()


def test_terminal_load_failure_leaves_no_residue():
    # The tally is dropped only once no stage can still report, so the verdict
    # is emitted exactly once and nothing is left to spin the engine's idle
    # KV drain forever.
    agg = PPKVAggregator(2)
    assert agg.ingest(0, KVConnectorOutput(failed_loading={"a"})).is_empty()

    out = agg.ingest(1, KVConnectorOutput(failed_loading={"a"}))
    assert out.failed_loading == {"a"}
    assert agg.has_pending() is False

    assert agg.ingest(0, KVConnectorOutput()).is_empty()


def test_load_failure_does_not_block_another_request():
    agg = PPKVAggregator(2)
    agg.ingest(0, KVConnectorOutput(failed_loading={"a"}, finished_loading={"b"}))
    out = agg.ingest(1, KVConnectorOutput(finished_loading={"a", "b"}))
    assert out.finished_loading == {"b"}
    assert out.failed_loading == {"a"}
    assert agg.has_pending() is False


def test_aggregator_rejects_bad_pp_size():
    with pytest.raises(ValueError):
        PPKVAggregator(0)


def _output(*completions):
    return KVConnectorOutput(connector_completions=set(completions))


def _source(operation, start=0, succeeded=True):
    return ConnectorCompletion(
        "dense.page.source_safe",
        SaveSourceGroupId(operation, ((start, start + 8),)),
        succeeded,
    )


def _retired(operation, succeeded=True):
    return ConnectorCompletion("dense.page.retired", operation, succeeded)


@pytest.mark.parametrize("group_succeeded", [True, False])
def test_never_started_stage_groups_wait_for_all_stage_safe_retirement(group_succeeded):
    agg = PPKVAggregator(3)
    operation = SaveOperationId("request", 1)
    first = _source(operation)
    second = _source(operation, 8, succeeded=group_succeeded)
    rejected = ConnectorCompletion("dense.page.store", operation, False)

    assert agg.ingest(0, _output(first, second)).is_empty()
    assert agg.ingest(1, _output(first)).is_empty()
    assert agg.ingest(2, _output(rejected, _retired(operation))).is_empty()
    assert agg.ingest(0, _output(_retired(operation))).is_empty()
    # Repeated reports from one stage cannot stand in for the missing stage.
    assert agg.ingest(0, _output(_retired(operation))).is_empty()
    assert first.key in agg._connector
    assert second.key in agg._connector

    result = agg.ingest(1, _output(_retired(operation)))
    assert result.connector_completions == {_retired(operation)}
    assert agg._connector == {rejected.key: {2}}
    assert agg._connector_failed == {rejected.key}
    assert agg.has_pending() is True

    stored = ConnectorCompletion("dense.page.store", operation, True)
    assert agg.ingest(0, _output(stored)).is_empty()
    result = agg.ingest(1, _output(stored))
    assert result.connector_completions == {rejected}
    assert agg.has_pending() is False
    assert agg._connector_failed == set()

    for stage in range(3):
        assert agg.ingest(stage, _output(first, second, _retired(operation))).is_empty()
        assert agg.has_pending() is False


def test_failed_retirement_preserves_source_groups_and_late_reports():
    agg = PPKVAggregator(2)
    operation = SaveOperationId("request", 1)
    group = _source(operation)
    assert agg.ingest(0, _output(group, _retired(operation, False))).is_empty()
    # A later success from that stage must not erase its failure verdict.
    assert agg.ingest(0, _output(_retired(operation))).is_empty()
    result = agg.ingest(1, _output(_retired(operation)))
    assert result.connector_completions == {_retired(operation, False)}
    assert agg.has_pending() is True

    result = agg.ingest(1, _output(group))
    assert result.connector_completions == {group}
    assert agg.has_pending() is False

    later_group = _source(operation, 8)
    assert agg.ingest(0, _output(later_group)).is_empty()
    result = agg.ingest(1, _output(later_group))
    assert result.connector_completions == {later_group}
    assert agg.has_pending() is False


def test_newer_retirement_preserves_an_older_running_generation():
    agg = PPKVAggregator(2)
    older, newer = SaveOperationId("old", 1), SaveOperationId("new", 2)
    older_group, newer_group = _source(older), _source(newer)
    assert agg.ingest(0, _output(older_group, newer_group)).is_empty()
    assert agg.ingest(0, _output(_retired(newer))).is_empty()
    result = agg.ingest(1, _output(_retired(newer)))
    assert result.connector_completions == {_retired(newer)}
    assert agg.has_pending() is True

    result = agg.ingest(1, _output(older_group))
    assert result.connector_completions == {older_group}
    assert agg.has_pending() is False
    assert agg.ingest(0, _output(_retired(older))).is_empty()
    result = agg.ingest(1, _output(_retired(older)))
    assert result.connector_completions == {_retired(older)}
    assert agg._retired_dense_generations._intervals == [(1, 2)]


def test_old_retired_generation_cannot_recreate_pending_groups():
    agg = PPKVAggregator(2)
    for generation in range(1, 101):
        operation = SaveOperationId(str(generation), generation)
        assert agg.ingest(
            0, _output(_source(operation), _retired(operation))
        ).is_empty()
        result = agg.ingest(1, _output(_retired(operation)))
        assert result.connector_completions == {_retired(operation)}
        assert agg.has_pending() is False
    assert agg._retired_dense_generations._intervals == [(1, 100)]

    oldest = SaveOperationId("1", 1)
    for stage in range(2):
        assert agg.ingest(stage, _output(_source(oldest), _retired(oldest))).is_empty()
        assert agg.has_pending() is False


def test_retirement_preserves_store_outcomes_and_other_channels():
    agg = PPKVAggregator(2)
    operation = SaveOperationId("request", 1)
    unrelated = ConnectorCompletion(
        "other.page.source_safe", _source(operation).operation_id, True
    )
    assert agg.ingest(0, _output(unrelated, _retired(operation))).is_empty()
    result = agg.ingest(1, _output(_retired(operation)))
    assert result.connector_completions == {_retired(operation)}
    assert agg.has_pending() is True

    # No store report arrived before retirement; its entire quorum is late.
    stored = ConnectorCompletion("dense.page.store", operation, True)
    failed = ConnectorCompletion("dense.page.store", operation, False)
    assert agg.ingest(0, _output(stored)).is_empty()
    result = agg.ingest(1, _output(unrelated, failed))
    assert result.connector_completions == {unrelated, failed}
    assert agg.has_pending() is False


def test_reset_allows_generation_reuse_in_a_new_scheduler_lifetime():
    agg = PPKVAggregator(2)
    operation = SaveOperationId("request", 1)
    assert agg.ingest(0, _output(_retired(operation))).is_empty()
    agg.ingest(1, _output(_retired(operation)))
    agg.reset()

    group = _source(operation)
    assert agg.ingest(0, _output(group)).is_empty()
    result = agg.ingest(1, _output(group))
    assert result.connector_completions == {group}
    assert agg.has_pending() is False
