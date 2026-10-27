# SPDX-License-Identifier: MIT
# PP-stage offload KV status aggregation (GPU-free).

import pytest
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.kv_transfer.disaggregation.pp_kv_aggregator import PPKVAggregator
    from atom.kv_transfer.disaggregation.types import (
        ConnectorCompletion,
        KVConnectorOutput,
        LoadOperationId,
        SaveOperationId,
        SaveSourceGroupId,
        StateStoreOperationId,
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
    proc.scheduler = FakeScheduler()
    proc.runner_mgr = FakeRunnerMgr(local_outputs)
    proc.pp_transport = FakePPTransport(downstream_messages)
    return proc


def test_send_is_reported_while_save_waits_for_every_pp_stage():
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
    assert proc.scheduler.released_sending() == {"a"}
    assert proc.scheduler.released_saving() == set()  # stage 2 still saving

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {"a"}
    assert proc.scheduler.released_saving() == {"a"}


def test_save_operation_id_waits_for_quorum_independently_of_send():
    # Send is request-scoped; each save generation still needs PP quorum.
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
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == set()  # stage 1 still saving

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == {op}


def test_each_save_generation_needs_its_own_quorum():
    # Completing one save must neither complete nor delay another generation.
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
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == set()

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == {g2}

    proc._poll_kv_transfer_progress()
    assert proc.scheduler.released_sending() == {9}
    assert proc.scheduler.released_saving() == {g2, g3}


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


def test_an_abandoned_save_releases_its_partial_quorum():
    """`forget`: the aggregator's terminal for a report that is not coming.

    A tally only drains on full quorum, so one lost stage report would pin
    `has_pending()` -- and with it `has_pending_kv_work()` -- for the life of
    the process: the head wakes every drain interval with nothing to do and
    every shutdown burns the full drain timeout. Bounded, but permanent, and
    it accumulates per lost report.

    Keyed by what the worker reported (a `SaveOperationId` here), while the
    scheduler abandons by `seq.id`, so `forget` has to collapse the two.
    """
    agg = PPKVAggregator(2)
    op = SaveOperationId(req_id="7", generation=1)
    assert agg.ingest(0, KVConnectorOutput(finished_saving={op})).is_empty()
    assert agg.ingest(0, KVConnectorOutput(finished_saving={"8"})).is_empty()
    assert agg.has_pending() is True

    agg.forget(7)  # the scheduler counts in ints; the connector in strings

    assert agg.has_pending() is True, "only request 7 is abandoned"
    agg.forget("8")
    assert agg.has_pending() is False

    # The late report from the missing stage cannot resurrect the tally into a
    # quorum of one.
    assert agg.ingest(1, KVConnectorOutput(finished_saving={op})).is_empty()
    assert agg.has_pending() is False


@pytest.mark.parametrize("seen_before_abandon", [False, True])
@pytest.mark.parametrize("succeeded", [False, True])
def test_abandoned_save_drops_late_store_and_source_reports(
    seen_before_abandon, succeeded
):
    agg = PPKVAggregator(2)
    operation = SaveOperationId("7", 1)
    source = SaveSourceGroupId(operation, ((0, 8),))
    report = KVConnectorOutput(
        finished_saving={operation},
        connector_completions={
            ConnectorCompletion("store", operation, succeeded),
            ConnectorCompletion("source_safe", source, True),
        },
    )
    if seen_before_abandon:
        assert agg.ingest(0, report).is_empty()

    agg.forget(7)

    # Save reports/channels need not have appeared before abandonment.
    for rank in (1, 0):
        assert agg.ingest(rank, report).is_empty()
        assert not agg.has_pending()


@pytest.mark.parametrize("failed_load", [False, True])
def test_abandoning_save_preserves_same_request_load_quorum(failed_load):
    agg = PPKVAggregator(2)
    save = SaveOperationId("7", 1)
    load = LoadOperationId("7", 2)
    disposition = ConnectorCompletion("load_disposition", load, True)
    state = ConnectorCompletion("state_store", StateStoreOperationId(7, 3), True)
    agg.ingest(
        0,
        KVConnectorOutput(
            finished_saving={save},
            finished_loading=set() if failed_load else {load},
            failed_loading={load} if failed_load else set(),
            connector_completions={disposition, state},
        ),
    )

    agg.forget(7)

    assert agg.has_pending()
    output = agg.ingest(
        1,
        KVConnectorOutput(
            finished_loading={load},
            connector_completions={disposition, state},
        ),
    )
    assert output.finished_loading == (set() if failed_load else {load})
    assert output.failed_loading == ({load} if failed_load else set())
    assert output.connector_completions == {disposition, state}
    assert not output.finished_saving
    assert not agg.has_pending()


def test_abandoning_save_preserves_untyped_connector_events():
    # A raw channel identity does not say it is a request's save. In
    # particular an independent state hash may equal a native request ID.
    agg = PPKVAggregator(2)
    event = ConnectorCompletion("custom_state", 7, True)
    agg.ingest(0, KVConnectorOutput(connector_completions={event}))
    agg.forget(7)
    output = agg.ingest(1, KVConnectorOutput(connector_completions={event}))
    assert output.connector_completions == {event}
    assert not agg.has_pending()


def test_save_tombstones_are_bounded_and_do_not_block_unrelated_requests():
    agg = PPKVAggregator(2, terminal_tombstone_limit=2)
    for request_id in range(3):
        agg.forget(request_id)
    assert len(agg._abandoned_saves) == 2
    assert agg.ingest(1, KVConnectorOutput(finished_saving={"1", 2})).is_empty()
    assert not agg.has_pending()

    other = SaveOperationId(3, 0)
    assert agg.ingest(0, KVConnectorOutput(finished_saving={other})).is_empty()
    assert agg.ingest(
        1, KVConnectorOutput(finished_saving={other})
    ).finished_saving == {other}
    assert not agg.has_pending()


def test_reset_clears_abandoned_save_tombstones():
    agg = PPKVAggregator(2)
    operation = SaveOperationId(7, 1)
    agg.forget(7)
    assert agg.ingest(0, KVConnectorOutput(finished_saving={operation})).is_empty()
    assert not agg.has_pending()
    agg.reset()
    assert agg.ingest(0, KVConnectorOutput(finished_saving={operation})).is_empty()
    assert agg.ingest(
        1, KVConnectorOutput(finished_saving={operation})
    ).finished_saving == {operation}
    assert not agg.has_pending()


def test_aggregator_rejects_nonpositive_tombstone_limit():
    with pytest.raises(ValueError, match="terminal_tombstone_limit"):
        PPKVAggregator(2, terminal_tombstone_limit=0)


@pytest.mark.parametrize("local_report", [False, True])
def test_pp_head_remembers_abandonment_before_first_stage_report(local_report):
    operation = SaveOperationId("7", 1)
    report = KVConnectorOutput(finished_saving={operation})
    proc = _head(
        pp_size=2,
        local_outputs=[report] if local_report else [],
        downstream_messages=[] if local_report else [[(1, report)]],
    )
    proc.scheduler.deferred_free_blocks = {}
    proc.scheduler.kv_connector = None
    proc.scheduler.on_save_abandoned = proc._forget_pp_save_quorum

    proc.scheduler.on_save_abandoned(7)
    assert not proc.has_pending_kv_work()
    proc._poll_kv_transfer_progress()

    assert not proc.scheduler.outputs
    assert not proc.has_pending_kv_work()


def test_the_pp_head_gives_the_aggregator_the_scheduler_s_verdict():
    """The two terminals share one trigger, so they cannot drift.

    The scheduler decides a save is beyond hope; it has no handle on the
    aggregator, so the head registers the only route between them.
    """
    proc = _head(pp_size=2, local_outputs=[])
    proc._pp_kv_aggregator = PPKVAggregator(2)
    # The wiring `__init__` does; asserted through `has_pending_kv_work`, which
    # is the predicate that keeps the head awake and stretches every shutdown.
    proc.scheduler.on_save_abandoned = proc._forget_pp_save_quorum
    proc.scheduler.deferred_free_blocks = {}
    proc.scheduler.is_finished = lambda: True
    proc.scheduler.kv_connector = None

    proc._pp_kv_aggregator.ingest(0, KVConnectorOutput(finished_saving={"3"}))
    assert proc.has_pending_kv_work() is True

    proc.scheduler.on_save_abandoned(3)
    assert proc.has_pending_kv_work() is False
