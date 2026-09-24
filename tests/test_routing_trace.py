"""Observation contract tests. These do not assert any routing cost model."""

import json
from types import SimpleNamespace as NS

import pytest
from test_gpu_metrics import Event, batch, complete_event

from atom.metrics import routing_trace as trace
from atom.metrics.gpu import GPUForwardMetrics


class Sink:
    stream_id = "engine-instance"

    def __init__(self):
        self.events = []

    def emit(self, kind, **data):
        self.events.append(dict(kind=kind, **data))


def seq(number=1, request_id="attempt-a", cached=0):
    return NS(
        id=number,
        external_request_id=request_id,
        parent_request_id=request_id,
        sibling_index=0,
        num_prompt_tokens=100,
        num_tokens=100,
        num_cached_tokens=cached,
        prefix_cache_hit_tokens=0,
        num_compressed_hit_blocks=0,
        num_wanted_hit_blocks=0,
        state_slot=-1,
        state_fork_src=-1,
        checkpoint_demand_pos=0,
        status=NS(name="WAITING"),
        type=NS(name="PREFILL"),
        is_partial_prefill=False,
    )


def scheduler(*seqs):
    return NS(waiting=list(seqs), running=[], advance_on_schedule=False)


def engine(sink):
    return trace.EngineTrace(
        NS(
            pipeline_parallel_size=1,
            enable_rapidserve=False,
            parallel_config=NS(data_parallel_rank=3),
        ),
        sink,
    )


def test_snapshot_has_no_speculative_progress_and_does_not_refresh_completion():
    sink = Sink()
    observer = engine(sink)
    request = seq(cached=20)
    sched = scheduler(request)
    observer.begin(sched)
    b = NS(
        req_ids=[1],
        num_scheduled_tokens=[30],
        num_cached_tokens=[20],
        total_seqs_num_decode=0,
        is_final_chunk=[False],
    )
    observer.batch(b, {1: request})
    assert sink.events[0]["scheduler"]["waiting"][0]["scheduler_cached_tokens"] == 20
    observer.snapshot(sched, {})
    assert sink.events[-1]["last_completed_step"] == 0
    assert sink.events[-1]["last_completed_monotonic_ns"] is None
    request.num_cached_tokens = 50  # Postprocess, not a prediction.
    observer.end(sched, True)
    observer.snapshot(sched, {})
    completion_time = sink.events[-1]["last_completed_monotonic_ns"]
    observer.snapshot(sched, {})
    assert sink.events[-1]["last_completed_monotonic_ns"] == completion_time
    assert sink.events[1]["requests"][0]["cached_before"] == 20
    assert sink.events[1]["requests"][0]["query_tokens"] == 30
    assert sink.events[1]["requests"][0]["kv_end"] == 50
    assert sink.events[0]["scheduler"]["waiting"][0]["scheduler_cached_tokens"] == 20
    # Lack of admission is explicit: zero in WAITING is not proof of a miss.
    assert sink.events[0]["scheduler"]["waiting"][0]["status"] == "WAITING"


def test_attempts_and_siblings_remain_distinct_even_with_same_client_id(monkeypatch):
    sink = Sink()
    monkeypatch.setattr(trace, "get_writer", lambda: sink)
    headers = {"x-request-id": "same-client-id", "x-session-id": "session"}
    trace.trace_api_request("attempt-a", headers, 3)
    trace.trace_api_request("attempt-b", headers, 3)
    assert [e["request_id"] for e in sink.events] == ["attempt-a", "attempt-b"]
    a = seq(request_id=None)
    a.parent_request_id = "attempt-a"
    b = seq(2, request_id=None)
    b.parent_request_id, b.sibling_index = "attempt-a", 1
    records = trace.scheduler_snapshot(scheduler(a, b))["waiting"]
    assert [
        (r["parent_request_id"], r["sibling_index"], r["seq_id"]) for r in records
    ] == [
        ("attempt-a", 0, 1),
        ("attempt-a", 1, 2),
    ]


def test_pp_schedule_advance_cannot_be_reported_as_completed_work():
    sched = scheduler(seq())
    sched.advance_on_schedule = True
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        trace.scheduler_snapshot(sched)
    with pytest.raises(ValueError, match="PP=1"):
        trace.EngineTrace(NS(pipeline_parallel_size=2), Sink())


def test_disabled_trace_does_not_read_request_or_create_files(monkeypatch):
    monkeypatch.delenv("ATOM_ROUTING_TRACE_DIR", raising=False)
    assert trace.get_writer() is None
    trace.trace_enqueue(None, None)
    trace.trace_api_request(None, None, None)


def test_writer_caps_disk_and_accounts_for_drops(tmp_path):
    writer = trace.TraceWriter(tmp_path, max_pending=2, max_bytes=2048)
    for i in range(2000):
        writer.emit("test", index=i, marker="汉字" * 20)
    writer.close()
    status = json.loads(
        writer.path.with_suffix(".status.json").read_text(encoding="utf-8")
    )
    assert status["error"] is None and status["stopped"]
    assert status["attempted"] == status["written"] + status["dropped"] == 2000
    assert status["dropped"] > 0
    assert writer.path.stat().st_size <= 2048
    records = [
        json.loads(line)
        for line in writer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == status["written"]
    assert [r["event_seq"] for r in records] == sorted(
        {r["event_seq"] for r in records}
    )


def test_gpu_batch_duration_is_emitted_once_not_once_per_request():
    sink = Sink()
    metrics = GPUForwardMetrics(Event)
    metrics.trace_writer = sink
    b = batch(prefill=2, decode=0)
    b.req_ids = [1, 2]
    b.routing_trace_id = "batch-1"
    with metrics.measure(b):
        pass
    b.routing_trace_id = "mutated-later"
    metrics.poll()
    assert sink.events == []  # No GPU synchronization or premature completion.
    complete_event(metrics, milliseconds=10)
    metrics.poll()
    assert len(sink.events) == 1
    assert sink.events[0]["batch_id"] == "batch-1"
    assert sink.events[0]["seconds"] == 0.01
    metrics.poll()
    assert len(sink.events) == 1
    assert not metrics.trace_pending


def test_gpu_overflow_and_invalid_timing_are_not_silently_missing():
    sink = Sink()
    metrics = GPUForwardMetrics(Event, max_pending=1)
    metrics.trace_writer = sink
    b = batch(prefill=1, decode=0)
    b.routing_trace_id = "first"
    with metrics.measure(b):
        pass
    b.routing_trace_id = "overflow"
    with metrics.measure(b):
        pass
    assert sink.events[-1]["reason"] == "pending_limit"
    assert sink.events[-1]["batch_id"] == "overflow"
    complete_event(metrics, milliseconds=float("nan"))
    metrics.poll()
    assert sink.events[-1]["reason"] == "invalid_duration"
    assert sink.events[-1]["batch_id"] == "first"
    assert not metrics.trace_pending


def test_failed_gpu_forward_does_not_leave_pending_identity():
    metrics = GPUForwardMetrics(Event)
    metrics.trace_writer = Sink()
    b = batch(prefill=1, decode=0)
    b.routing_trace_id = "failed-attempt"
    with pytest.raises(RuntimeError), metrics.measure(b):
        raise RuntimeError("forward failed")
    assert not metrics.trace_pending and not metrics.pending


def test_real_sequence_and_batch_preserve_chunk_features():
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence, SequenceType

    request = Sequence(list(range(100)), block_size=4, request_id="real-attempt")
    request.type = SequenceType.PREFILL
    request.num_cached_tokens = request.prefix_cache_hit_tokens = 40
    batch = ScheduledBatch(
        {request.id: request},
        [20],
        20,
        total_tokens_num_prefill=20,
        total_seqs_num=1,
        total_seqs_num_prefill=1,
    )
    sink = Sink()
    observer = engine(sink)
    observer.begin(scheduler(request))
    observer.batch(batch, {request.id: request})
    row = sink.events[-1]["requests"][0]
    assert (
        row["request_id"],
        row["cached_before"],
        row["query_tokens"],
        row["kv_end"],
    ) == (
        "real-attempt",
        40,
        20,
        60,
    )
    json.dumps(sink.events, allow_nan=False)


def test_single_output_api_bridge_works_without_external_sequence_id(monkeypatch):
    import pickle

    from atom.model_engine.sequence import Sequence

    sink = Sink()
    monkeypatch.setattr(trace, "get_writer", lambda: sink)
    request = Sequence([1, 2, 3], block_size=4)
    assert request.external_request_id is None
    trace.trace_api_sequence("api-attempt", request)
    received = pickle.loads(pickle.dumps(request))
    bridge = sink.events[-1]
    assert bridge["request_id"] == "api-attempt"
    assert bridge["seq_id"] == received.id
    assert received.external_request_id is None


def test_real_engine_step_does_not_publish_completion_on_forward_error():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import EngineCore

    sink = Sink()
    observer = engine(sink)
    sched = scheduler(seq())
    sched.publish_kv_events = lambda: None

    def fail():
        raise RuntimeError("GPU failed")

    proc = NS(routing_trace=observer, scheduler=sched, _process_engine_step_inner=fail)
    with pytest.raises(RuntimeError, match="GPU failed"):
        EngineCore._process_engine_step(proc)
    assert [e["kind"] for e in sink.events] == ["step_start"]
    assert observer.last_completed_step == 0
