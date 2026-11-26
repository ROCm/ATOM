# SPDX-License-Identifier: MIT
# PP-stage offload KV status aggregation (GPU-free).

from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import zmq
from aiter_stub import stubbed_aiter

with stubbed_aiter():
    from atom.distributed.pp_transport import (
        PPKVStatusPusher,
        PPStageTransport,
        stages_push_load_terminals,
    )
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


@pytest.mark.parametrize(
    "kv_enabled,is_producer,chunked,is_offload,max_wait_ms",
    [
        (True, True, True, False, 1),
        (True, True, True, True, 1),  # Chunked Mooncake + LMCache.
        (True, True, False, False, 100),  # Legacy producer.
        (True, False, True, False, 100),  # Consumer.
        (True, False, False, True, 100),  # Offload only.
        (False, False, False, False, 100),
    ],
)
def test_downstream_reports_ready_without_waiting_for_a_forward(
    kv_enabled, is_producer, chunked, is_offload, max_wait_ms
):
    """D-ready arriving during metadata receive must promptly reach the head."""
    proc = PPEngineCoreProc.__new__(PPEngineCoreProc)
    proc.kv_transfer_enabled = kv_enabled
    proc.scheduler = MagicMock()
    proc.scheduler.kv_connector = SimpleNamespace(
        is_producer=is_producer,
        enable_chunked_transfer=chunked,
        is_offload=is_offload,
    )
    proc.utility_handler = MagicMock()
    proc.utility_queue = MagicMock()
    proc.pull_and_process_input_queue = MagicMock(side_effect=[False, True])
    proc._is_engine_held = lambda: False
    proc.runner_mgr = MagicMock()
    proc.pp_transport = MagicMock()
    report = KVConnectorOutput(
        connector_completions={ConnectorCompletion("pd_admission_ready", 7, True)}
    )
    elapsed_ms = 0
    reports = []

    def recv_metadata(timeout_ms):
        nonlocal elapsed_ms
        # A worker becomes ready just after the blocking receive begins.
        # Advance a virtual clock so the regression does not depend on load
        # on the host running the test. No forward arrives to wake the stage.
        proc.runner_mgr.call_func_with_aggregation.side_effect = [
            report,
            KVConnectorOutput(),  # Final shutdown poll.
        ]
        elapsed_ms += timeout_ms

    def send_kv_status(output):
        reports.append(output)
        assert elapsed_ms <= max_wait_ms
        assert proc.pull_and_process_input_queue.call_count == 1

    proc.pp_transport.recv_metadata.side_effect = recv_metadata
    proc.pp_transport.send_kv_status.side_effect = send_kv_status
    proc._downstream_busy_loop()

    assert elapsed_ms == max_wait_ms
    assert reports == ([report] if kv_enabled else [])
    assert all(c.args[0] != "forward" for c in proc.runner_mgr.call_func.call_args_list)
    proc.pp_transport.send_tokens.assert_not_called()


# --- load terminals pushed by the stage runners (`stages_push_load_terminals`) ---


def test_a_pushed_load_terminal_still_needs_every_stage():
    """A runner's push lands on the head's kv_status socket as its stage's
    report, the head's own stage (0) included: the quorum is unchanged."""
    op = LoadOperationId(5, 1)
    proc = _head(
        pp_size=3,
        local_outputs=[KVConnectorOutput()] * 3,
        downstream_messages=[
            [(0, KVConnectorOutput(finished_loading={op}))],
            [(2, KVConnectorOutput(finished_loading={op}))],
            [(1, KVConnectorOutput(finished_loading={op}))],
        ],
    )
    proc._poll_kv_transfer_progress()
    proc._poll_kv_transfer_progress()
    assert not any(out.finished_loading for out in proc.scheduler.outputs)
    proc._poll_kv_transfer_progress()
    assert [out.finished_loading for out in proc.scheduler.outputs] == [{op}]


def test_a_pushed_load_failure_fails_closed_once_every_stage_reported():
    op = LoadOperationId(6, 1)
    proc = _head(
        pp_size=2,
        local_outputs=[KVConnectorOutput()] * 2,
        downstream_messages=[
            [(1, KVConnectorOutput(failed_loading={op}))],
            [(0, KVConnectorOutput(finished_loading={op}))],
        ],
    )
    proc._poll_kv_transfer_progress()
    assert proc.scheduler.outputs == []  # stage 0 is still loading into the blocks
    proc._poll_kv_transfer_progress()
    [out] = proc.scheduler.outputs
    assert out.failed_loading == {op} and out.finished_loading == set()


def test_a_runner_push_reaches_the_head_socket(tmp_path):
    addrs = [f"ipc://{tmp_path}/meta{i}" for i in range(2)]
    ctx = zmq.Context()  # the head's close() terms it, not the shared instance
    head = PPStageTransport(
        0,
        2,
        addrs,
        f"ipc://{tmp_path}/tok",
        kv_status_addr=f"ipc://{tmp_path}/kv",
        ctx=ctx,
    )
    pushers = [PPKVStatusPusher(rank, f"ipc://{tmp_path}/kv") for rank in (0, 1)]
    try:
        for rank, pusher in enumerate(pushers):
            assert pusher.try_send(
                KVConnectorOutput(finished_loading={LoadOperationId(rank, 0)})
            )
        got = []
        for _ in range(50):
            got += head.recv_kv_status(timeout_ms=100)
            if len(got) == 2:
                break
        assert sorted((r, o.finished_loading) for r, o in got) == [
            (0, {LoadOperationId(0, 0)}),
            (1, {LoadOperationId(1, 0)}),
        ]
    finally:
        for pusher in pushers:
            pusher.close()
        head.close()
        ctx.term()


def test_a_push_the_head_does_not_take_is_refused_not_waited_on(tmp_path):
    pusher = PPKVStatusPusher(1, f"ipc://{tmp_path}/kv")  # no head binds it
    try:
        queued = 0
        while queued < 100_000 and pusher.try_send(KVConnectorOutput()):
            queued += 1
        assert 0 < queued < 100_000  # zmq's high-water mark, then False
    finally:
        pusher.close()


@pytest.mark.parametrize(
    ("pp", "tp", "pcp", "connectors", "pushes"),
    [
        (4, 1, 1, ["mooncake_store"], True),
        (4, 1, 1, ["mooncake", "mooncake_store"], True),  # P/D beside the Store
        (1, 1, 1, ["mooncake_store"], False),
        (4, 2, 1, ["mooncake_store"], False),
        (4, 1, 2, ["mooncake_store"], False),
        (4, 1, 1, ["lmcache_offload"], False),  # other offloads keep the poll
    ],
)
def test_which_deployments_push_load_terminals(pp, tp, pcp, connectors, pushes):
    subs = [{"kv_connector": name, "kv_role": "kv_both"} for name in connectors]
    config = SimpleNamespace(
        pipeline_parallel_size=pp,
        tensor_parallel_size=tp,
        prefill_context_parallel_size=pcp,
        kv_transfer_config=(
            subs[0] if len(subs) == 1 else {"kv_connector": "multi", "connectors": subs}
        ),
    )
    assert stages_push_load_terminals(config) is pushes


@pytest.mark.parametrize(
    ("tp", "connector", "pushes"),
    [(1, "mooncake_store", True), (2, "mooncake_store", False), (1, None, False)],
)
def test_a_pp_engine_core_takes_the_push_choice_from_its_config(
    tp, connector, pushes, monkeypatch
):
    def engine_core_init(self, config, input_address, output_address):
        self.label = "pp-head"
        self.scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(enable_prefix_caching=False)
        )

    # The class's own module and base: modules imported under the stub are dropped.
    monkeypatch.setitem(
        PPEngineCoreProc.__init__.__globals__, "PPStageTransport", MagicMock()
    )
    monkeypatch.setattr(PPEngineCoreProc.__bases__[0], "__init__", engine_core_init)
    config = SimpleNamespace(
        pipeline_parallel_size=2,
        tensor_parallel_size=tp,
        kv_transfer_config=(
            {"kv_connector": connector, "kv_role": "kv_both"} if connector else {}
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_rank=0, pp_meta_addrs=[], pp_token_addr=""
        ),
    )
    head = PPEngineCoreProc(config, "in", "out")
    assert head._stages_push_load_terminals is pushes


@pytest.mark.parametrize("pushes", [True, False])
def test_the_head_takes_kv_reports_between_the_launches_of_a_step(pushes):
    """Where stages push, a report that landed during one forward RPC reaches
    the scheduler before the next schedule() of the same step."""
    order = []
    batch = SimpleNamespace(
        req_ids=[1],
        connector_meta_output=None,
        produces_output=lambda: False,
    )
    head = SimpleNamespace(
        _in_flight=deque(),
        _stages_push_load_terminals=pushes,
        pp_max_inflight=3,
        kv_transfer_enabled=True,
        output_queue=MagicMock(),
        runner_mgr=MagicMock(),
        pp_transport=MagicMock(),
        scheduler=MagicMock(),
        _poll_kv_transfer_progress=lambda: order.append("poll"),
        _poll_dynamic_chunking_calibration=lambda: None,
        _collect_pp_completions=lambda poll_ms: None,
    )
    head.scheduler.take_rejected.return_value = None

    def schedule():
        order.append("schedule")
        return batch, {}

    head.scheduler.schedule.side_effect = schedule
    PPEngineCoreProc._pp_head_step(head)
    if pushes:
        assert order == ["schedule", "poll", "schedule", "poll", "schedule", "poll"]
    else:
        assert order == ["schedule", "schedule", "schedule", "poll"]
