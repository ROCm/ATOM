"""Device event lifecycle and worker telemetry routing, without a GPU."""

from queue import Queue
from types import SimpleNamespace

import pytest
from metrics_helpers import histogram_values, histogram_values_by_name

from atom.metrics.gpu import GPUForwardMetrics, record_gpu_forward
from atom.model_engine.engine_utility import EngineUtilityHandler


class Event:
    def __init__(self):
        self.ready = False
        self.recorded = 0
        self.queries = 0
        self.duration_ms = 8.0

    def record(self):
        self.recorded += 1
        self.ready = False

    def query(self):
        self.queries += 1
        return self.ready

    def elapsed_time(self, end):
        assert self.ready and end.ready
        return self.duration_ms

    def synchronize(self):
        raise AssertionError("Telemetry must never synchronize the GPU")


def batch(prefill=0, decode=1, dummy=False):
    return SimpleNamespace(
        req_ids=[1],
        total_seqs_num_prefill=prefill,
        total_seqs_num_decode=decode,
        is_dummy_run=dummy,
    )


def test_events_are_polled_without_waiting_and_reused_only_after_completion():
    metrics = GPUForwardMetrics(Event, max_pending=2)
    with metrics.measure(batch()):
        pass
    with metrics.measure(batch(prefill=1, decode=0)):
        pass
    metrics.poll()
    assert len(metrics.pending) == 2
    with metrics.measure(batch()):
        pass
    assert len(metrics.pending) == 2
    # A different stream may complete the second pair before the first.
    for event in metrics.pending[1][:2]:
        event.ready = True
    snapshot = histogram_values_by_name(metrics)
    assert len(metrics.pending) == 2
    assert not metrics.free
    assert snapshot["steps"]["sum"] == 0
    complete_event(metrics)
    snapshot = histogram_values_by_name(metrics)
    assert not metrics.pending
    assert snapshot["steps"]["sum"] == 0.016
    assert snapshot["steps"]["buckets"][-1][1] == 2
    reused = tuple(metrics.free[-1])
    with metrics.measure(batch(prefill=1, decode=1)):
        pass
    assert tuple(metrics.pending[-1][:2]) == reused
    for start, end, _ in metrics.pending:
        start.ready = end.ready = True
    snapshot = histogram_values_by_name(metrics)
    assert len(metrics.pending) == 0
    assert snapshot["steps"]["sum"] == 0.024
    assert snapshot["steps"]["buckets"][-1][1] == 3
    assert histogram_values_by_name(metrics)["steps"] == snapshot["steps"]


def test_poll_checks_only_the_unfinished_head_of_a_full_queue():
    metrics = GPUForwardMetrics(Event)
    for _ in range(256):
        with metrics.measure(batch()):
            pass
    for index, (start, end, _) in enumerate(metrics.pending):
        start.ready = end.ready = index > 0
        end.queries = 0
    metrics.poll()
    assert [end.queries for _, end, _ in metrics.pending] == [1] + [0] * 255
    assert not metrics.free
    assert histogram_values(metrics.steps)["sum"] == 0


def test_warmup_dummy_failure_and_decorator_do_not_create_spurious_samples():
    metrics = GPUForwardMetrics(Event)
    for b in (None, batch(dummy=True)):
        with metrics.measure(b):
            pass
    with pytest.raises(RuntimeError), metrics.measure(batch()):
        raise RuntimeError("model failed")
    metrics.poll()
    assert len(metrics.pending) == 0

    @record_gpu_forward
    def model(self, inputs, batch=None):
        return inputs + 1

    assert model(SimpleNamespace(), 4, batch()) == 5
    assert model(SimpleNamespace(gpu_forward_metrics=None), 4, object()) == 5
    assert model(SimpleNamespace(gpu_forward_metrics=metrics), 4, batch()) == 5
    metrics.poll()
    assert len(metrics.pending) == 1


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_failed_forwards_recycle_events_and_successful_retry_is_measured(error_type):
    created = []

    def event_factory():
        event = Event()
        created.append(event)
        return event

    metrics = GPUForwardMetrics(event_factory)
    empty = histogram_values_by_name(metrics)
    for req_id in range(16):
        error = error_type("forward failed")
        with (
            pytest.raises(error_type) as caught,
            metrics.measure(prefill_batch((req_id, 1, True))),
        ):
            raise error
        assert caught.value is error
        assert len(created) == 2
        assert metrics.free == [tuple(created)]
        assert not metrics.pending and not metrics.requests
        assert histogram_values_by_name(metrics) == empty

    start, end = created
    assert (start.recorded, end.recorded) == (16, 0)
    with metrics.measure(prefill_batch((16, 1, True))):
        pass
    assert len(created) == 2
    assert (start.recorded, end.recorded) == (17, 1)
    assert not metrics.free
    assert (
        histogram_values_by_name(metrics) == empty
    )  # The successful forward is not ready yet.
    complete_event(metrics, milliseconds=12)
    snapshot = histogram_values_by_name(metrics)
    for histogram in (snapshot["steps"], snapshot["prefill_requests"]):
        assert histogram["buckets"][-1][1] == 1
        assert histogram["sum"] == pytest.approx(0.012)
    assert metrics.free == [(start, end)]
    assert not metrics.pending and not metrics.requests


@pytest.mark.parametrize("enabled", [False, True])
def test_idle_device_poll_has_no_snapshot_or_response(monkeypatch, enabled):
    monkeypatch.setenv("ATOM_ENABLE_METRICS_DEVICE_TIMER", "1" if enabled else "0")
    calls = []
    manager = SimpleNamespace(
        call_func=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    output = Queue()
    utility = EngineUtilityHandler(manager, output)
    utility.push_metrics(scheduler_metrics=False)
    assert calls == ([(("poll_forward_metrics",), {})] if enabled else [])
    assert output.empty()


def test_worker_telemetry_does_not_enter_forward_or_kv_result_queues():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProc
    worker = AsyncIOProc.__new__(AsyncIOProc)
    worker.label = "test"
    worker.runners = [
        SimpleNamespace(poll_forward_metrics=lambda: None, exit=lambda: None)
    ]
    worker.io_addrs = [None, "primary"]
    worker.io_queues = [Queue(), Queue()]
    worker.kv_queue = Queue()
    worker.all_ranks_barrier = None
    calls = iter([("poll_forward_metrics", []), ("exit", [])])
    worker.get_func = lambda: next(calls)
    worker.busy_loop()
    assert worker.io_queues[1].empty()
    assert worker.kv_queue.empty()


def test_device_events_measure_graph_replay_on_a_nondefault_stream():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA/HIP device")
    metrics = GPUForwardMetrics(lambda: torch.cuda.Event(enable_timing=True))
    stream = torch.cuda.Stream()
    x = torch.ones((64, 64), device="cuda")
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.mm(x, x)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = torch.mm(x, x)
    with torch.cuda.stream(stream), metrics.measure(batch()):
        graph.replay()
    # Synchronization is only in this test, never in the telemetry path.
    stream.synchronize()
    snapshot = histogram_values_by_name(metrics)
    assert len(metrics.pending) == 0
    assert snapshot["steps"]["buckets"][-1][1] == 1
    assert snapshot["steps"]["sum"] > 0
    assert output[0, 0].item() == 64


def prefill_batch(*chunks, decode=0):
    """Immutable (request ID, chunk ordinal, final) scheduler metadata."""
    b = batch(prefill=len(chunks), decode=decode)
    b.req_ids = [-1] * decode + [chunk[0] for chunk in chunks]
    b.prefill_gpu_requests = list(chunks)
    return b


def complete_event(metrics, index=0, milliseconds=8):
    start, end, _ = metrics.pending[index]
    start.duration_ms = milliseconds
    start.ready = end.ready = True


def test_step_buckets_pool_prefill_decode_and_mixed_batches():
    metrics = GPUForwardMetrics(Event)
    for prefill, decode, milliseconds in (
        (1, 0, 0.5),
        (0, 1, 8),
        (1, 1, 12),
        (1, 0, 800_000),
    ):
        with metrics.measure(batch(prefill=prefill, decode=decode)):
            pass
        complete_event(metrics, milliseconds=milliseconds)
        metrics.poll()
    values = histogram_values(metrics.steps)
    buckets = dict(values["buckets"])
    assert [buckets[b] for b in (0.001, 0.01, 0.02, 600, float("inf"))] == [
        1,
        2,
        3,
        3,
        4,
    ]
    assert values["sum"] == pytest.approx(800.0205)


def test_request_sum_waits_for_every_chunk_even_if_last_event_finishes_first():
    metrics = GPUForwardMetrics(Event)
    for chunk in range(1, 4):
        with metrics.measure(prefill_batch((7, chunk, chunk == 3))):
            pass
    complete_event(metrics, 2, 8)
    complete_event(metrics, 0, 10)
    first = histogram_values_by_name(metrics)
    assert first["prefill_requests"]["buckets"][-1][1] == 0
    assert first["steps"]["sum"] == pytest.approx(0.010)
    complete_event(metrics, 0, 12)
    final = histogram_values_by_name(metrics)
    assert final["prefill_requests"]["sum"] == pytest.approx(0.030)
    assert final["prefill_requests"]["buckets"][-1][1] == 1
    assert final["steps"]["buckets"][-1][1] == 3
    assert not metrics.requests
    for _ in range(2):
        assert (
            histogram_values_by_name(metrics)["prefill_requests"]
            == final["prefill_requests"]
        )
    assert first["prefill_requests"]["sum"] == 0  # published copies stay immutable


def test_shared_mixed_batch_time_counts_in_full_for_each_prefill_request():
    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((1, 1, True), (2, 1, False), decode=1)):
        pass
    complete_event(metrics, milliseconds=10)
    first = histogram_values_by_name(metrics)
    assert first["prefill_requests"]["sum"] == pytest.approx(0.010)
    assert first["prefill_requests"]["buckets"][-1][1] == 1
    assert first["steps"]["buckets"][-1][1] == 1
    with metrics.measure(prefill_batch((2, 2, True))):
        pass
    complete_event(metrics, milliseconds=5)
    final = histogram_values_by_name(metrics)
    # Request 1 = 10 ms; request 2 = 10 + 5 ms. Decode row gets no sample.
    assert final["prefill_requests"]["sum"] == pytest.approx(0.025)
    assert final["prefill_requests"]["buckets"][-1][1] == 2


@pytest.mark.parametrize("failure", ["queue_full", "exception", "missing_chunk"])
def test_incomplete_request_timing_is_never_published(failure):
    metrics = GPUForwardMetrics(Event, max_pending=1)
    with metrics.measure(prefill_batch((1, 1, False))):
        pass
    if failure != "queue_full":
        complete_event(metrics)
        metrics.poll()
    if failure == "exception":
        with pytest.raises(RuntimeError), metrics.measure(prefill_batch((1, 2, False))):
            raise RuntimeError("failed chunk")
    elif failure == "queue_full":
        with metrics.measure(prefill_batch((1, 2, False))):
            pass
        complete_event(metrics)
        metrics.poll()
    # For missing_chunk, chunk 2 was never sent to this worker.
    with metrics.measure(prefill_batch((1, 3, True))):
        pass
    complete_event(metrics)
    final = histogram_values_by_name(metrics)
    assert final["prefill_requests"]["buckets"][-1][1] == 0
    assert not metrics.requests


def test_abandoned_partial_requests_are_bounded_and_evicted_tails_are_not_samples():
    metrics = GPUForwardMetrics(Event, max_requests=2)
    for req_id in range(5):
        with metrics.measure(prefill_batch((req_id, 1, False))):
            pass
        complete_event(metrics)
        metrics.poll()
        assert len(metrics.requests) <= 2
    assert set(metrics.requests) == {3, 4}
    # A late final chunk cannot recreate an evicted accumulator.
    with metrics.measure(prefill_batch((0, 2, True))):
        pass
    complete_event(metrics)
    assert histogram_values_by_name(metrics)["prefill_requests"]["buckets"][-1][1] == 0


def test_reused_request_id_cannot_be_finished_by_an_old_pending_event():
    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((7, 1, True))):
        pass
    with metrics.measure(prefill_batch((7, 1, True))):
        pass
    complete_event(metrics, 1, 20)
    assert histogram_values_by_name(metrics)["prefill_requests"]["sum"] == 0
    complete_event(metrics, 0, 100)
    final = histogram_values_by_name(metrics)
    assert final["prefill_requests"]["sum"] == pytest.approx(0.020)
    assert final["prefill_requests"]["buckets"][-1][1] == 1


def test_scheduler_freezes_chunk_boundaries_and_excludes_later_recomputation(
    monkeypatch,
):
    import pickle

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence, SequenceType

    monkeypatch.setenv("ATOM_ENABLE_METRICS_DEVICE_TIMER", "1")
    seq = Sequence(list(range(10)), block_size=4)
    seq.type = SequenceType.PREFILL
    seq.num_cached_tokens = 4  # Cached prefix is not a forward or a zero-time chunk.

    def schedule(n, *, dummy=False, final=None):
        return ScheduledBatch(
            {seq.id: seq},
            [n],
            n,
            total_tokens_num_prefill=n,
            total_seqs_num=1,
            total_seqs_num_prefill=1,
            is_final_chunk=final,
            is_dummy_run=dummy,
        )

    assert schedule(2, dummy=True).prefill_gpu_requests == []
    first = pickle.loads(pickle.dumps(schedule(2, final=[False])))
    seq.num_cached_tokens += 2
    # Shared-GPU prefill does not supply is_final_chunk; infer the frozen end.
    last = schedule(4)
    assert first.prefill_gpu_requests == [(seq.id, 1, False)]
    assert last.prefill_gpu_requests == [(seq.id, 2, True)]
    seq.num_cached_tokens = 0
    assert schedule(10).prefill_gpu_requests == []
    assert first.prefill_gpu_requests == [(seq.id, 1, False)]

    metrics = GPUForwardMetrics(Event)
    for b in (first, last):
        with metrics.measure(b):
            pass
        complete_event(metrics)
        metrics.poll()
    assert histogram_values_by_name(metrics)["prefill_requests"][
        "sum"
    ] == pytest.approx(0.016)


def test_scheduler_does_not_track_gpu_chunks_by_default(monkeypatch):
    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence, SequenceType

    monkeypatch.delenv("ATOM_ENABLE_METRICS_DEVICE_TIMER", raising=False)
    seq = Sequence([1, 2, 3, 4], block_size=4)
    seq.type = SequenceType.PREFILL
    scheduled = ScheduledBatch(
        {seq.id: seq},
        [4],
        4,
        total_tokens_num_prefill=4,
        total_seqs_num=1,
        total_seqs_num_prefill=1,
    )
    assert scheduled.prefill_gpu_requests == []
    assert seq.prefill_gpu_chunks == 0
    assert not seq.prefill_gpu_complete


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1])
def test_invalid_device_duration_is_logged_and_does_not_poison_histograms(
    caplog, duration
):
    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((1, 1, True))):
        pass
    complete_event(metrics, milliseconds=duration)
    metrics.poll()
    assert "Invalid GPU forward duration" in caplog.text
    assert not metrics.pending and not metrics.requests
    assert len(metrics.free) == 1
    assert histogram_values(metrics.steps)["sum"] == 0
    with metrics.measure(prefill_batch((2, 1, True))):
        pass
    complete_event(metrics)
    values = histogram_values_by_name(metrics)
    assert values["steps"]["sum"] == values["prefill_requests"]["sum"] == 0.008
