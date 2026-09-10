"""Device event lifecycle and worker telemetry routing, without a GPU."""

from queue import Queue
from types import SimpleNamespace

import pytest

from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.gpu_metrics import GPUForwardMetrics, record_gpu_forward


class Event:
    def __init__(self):
        self.ready = False
        self.recorded = 0
        self.duration_ms = 8.0

    def record(self):
        self.recorded += 1
        self.ready = False

    def query(self):
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
    for event in metrics.pending[1][1:3]:
        event.ready = True
    snapshot = metrics.snapshot()
    assert len(metrics.pending) == 1
    assert snapshot["phases"]["prefill"]["sum"] == 0.008
    assert snapshot["phases"]["decode"]["sum"] == 0
    reused = tuple(metrics.free[-1])
    with metrics.measure(batch(prefill=1, decode=1)):
        pass
    assert tuple(metrics.pending[-1][1:3]) == reused
    for _, start, end, _ in metrics.pending:
        start.ready = end.ready = True
    snapshot = metrics.snapshot()
    assert len(metrics.pending) == 0
    assert snapshot["phases"]["decode"]["sum"] == 0.008
    assert snapshot["phases"]["mixed"]["sum"] == 0.008
    assert metrics.snapshot()["phases"] == snapshot["phases"]


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
    assert model(SimpleNamespace(gpu_forward_metrics=metrics), 4, batch()) == 5
    metrics.poll()
    assert len(metrics.pending) == 1


def test_device_snapshot_push_never_waits_or_duplicates_downstream_scheduler():
    calls = []
    manager = SimpleNamespace(
        latest_forward_metrics={0: {"tp_rank": 0}},
        call_func=lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    output = Queue()
    utility = EngineUtilityHandler(manager, output)
    utility.push_metrics(scheduler_metrics=False)
    assert calls == [(("collect_forward_metrics",), {})]
    tag, data = output.get_nowait()
    assert tag == "METRICS" and data["enabled"] is False
    assert data["forward_metrics"] == [{"tp_rank": 0}]


def test_worker_telemetry_does_not_enter_forward_or_kv_result_queues():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProc, AsyncIOProcManager
    manager = AsyncIOProcManager.__new__(AsyncIOProcManager)
    manager.latest_forward_metrics = {}
    manager.kv_outputs_queues = [Queue()]
    snapshot = {"tp_rank": 0}
    manager._receive_worker_output(0, ("FORWARD_METRICS", snapshot))
    assert manager.latest_forward_metrics == {0: snapshot}
    assert manager.kv_outputs_queues[0].empty()
    kv = object()
    manager._receive_worker_output(0, kv)
    assert manager.kv_outputs_queues[0].get_nowait() is kv
    worker = AsyncIOProc.__new__(AsyncIOProc)
    worker.label = "test"
    worker.runners = [
        SimpleNamespace(collect_forward_metrics=lambda: snapshot, exit=lambda: None)
    ]
    worker.io_addrs = [None, "primary"]
    worker.io_queues = [Queue(), Queue()]
    worker.kv_queue = Queue()
    worker.all_ranks_barrier = None
    calls = iter([("collect_forward_metrics", []), ("exit", [])])
    worker.get_func = lambda: next(calls)
    worker.busy_loop()
    assert worker.io_queues[1].empty()
    assert worker.kv_queue.get_nowait() == ("FORWARD_METRICS", snapshot)


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
    snapshot = metrics.snapshot()
    assert len(metrics.pending) == 0
    assert snapshot["phases"]["decode"]["buckets"][-1][1] == 1
    assert snapshot["phases"]["decode"]["sum"] > 0
    assert output[0, 0].item() == 64


def prefill_batch(*chunks, decode=0):
    """Immutable (request ID, chunk ordinal, final) scheduler metadata."""
    b = batch(prefill=len(chunks), decode=decode)
    b.req_ids = [-1] * decode + [chunk[0] for chunk in chunks]
    b.prefill_gpu_requests = list(chunks)
    return b


def complete_event(metrics, index=0, milliseconds=8):
    _, start, end, _ = metrics.pending[index]
    start.duration_ms = milliseconds
    start.ready = end.ready = True


def test_request_sum_waits_for_every_chunk_even_if_last_event_finishes_first():
    metrics = GPUForwardMetrics(Event)
    for chunk in range(1, 4):
        with metrics.measure(prefill_batch((7, chunk, chunk == 3))):
            pass
    complete_event(metrics, 2, 8)
    complete_event(metrics, 0, 10)
    first = metrics.snapshot()
    assert first["prefill_requests"]["buckets"][-1][1] == 0
    assert first["phases"]["prefill"]["sum"] == pytest.approx(0.018)
    complete_event(metrics, 0, 12)
    final = metrics.snapshot()
    assert final["prefill_requests"]["sum"] == pytest.approx(0.030)
    assert final["prefill_requests"]["buckets"][-1][1] == 1
    assert final["phases"]["prefill"]["buckets"][-1][1] == 3
    assert not metrics.requests
    for _ in range(2):
        assert metrics.snapshot()["prefill_requests"] == final["prefill_requests"]
    assert first["prefill_requests"]["sum"] == 0  # published copies stay immutable


def test_shared_mixed_batch_time_counts_in_full_for_each_prefill_request():
    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((1, 1, True), (2, 1, False), decode=1)):
        pass
    complete_event(metrics, milliseconds=10)
    first = metrics.snapshot()
    assert first["prefill_requests"]["sum"] == pytest.approx(0.010)
    assert first["prefill_requests"]["buckets"][-1][1] == 1
    assert first["phases"]["mixed"]["buckets"][-1][1] == 1
    with metrics.measure(prefill_batch((2, 2, True))):
        pass
    complete_event(metrics, milliseconds=5)
    final = metrics.snapshot()
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
    final = metrics.snapshot()
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
    assert metrics.snapshot()["prefill_requests"]["buckets"][-1][1] == 0


def test_reused_request_id_cannot_be_finished_by_an_old_pending_event():
    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((7, 1, True))):
        pass
    with metrics.measure(prefill_batch((7, 1, True))):
        pass
    complete_event(metrics, 1, 20)
    assert metrics.snapshot()["prefill_requests"]["sum"] == pytest.approx(0.020)
    complete_event(metrics, 0, 100)
    final = metrics.snapshot()
    assert final["prefill_requests"]["sum"] == pytest.approx(0.020)
    assert final["prefill_requests"]["buckets"][-1][1] == 1


def test_scheduler_freezes_chunk_boundaries_and_excludes_later_recomputation():
    import pickle

    from atom.model_engine.scheduler import ScheduledBatch
    from atom.model_engine.sequence import Sequence, SequenceType

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
    assert metrics.snapshot()["prefill_requests"]["sum"] == pytest.approx(0.016)


def test_request_gpu_histogram_export_is_per_worker_and_backward_compatible():
    from prometheus_client.parser import text_string_to_metric_families

    from atom.entrypoints.openai.metrics_setup import create_metrics_exporter

    metrics = GPUForwardMetrics(Event)
    with metrics.measure(prefill_batch((1, 1, True))):
        pass
    complete_event(metrics)
    snapshot = metrics.snapshot()
    workers = [
        dict(snapshot, dp_rank=0, pp_rank=pp, tp_rank=tp, engine_role="prefill")
        for pp, tp in ((0, 0), (0, 1), (1, 0), (1, 1))
    ]
    legacy = {
        k: v for k, v in workers[0].items() if not k.startswith("prefill_requests")
    }
    legacy["dp_rank"] = 1
    exporter, _, _ = create_metrics_exporter()
    exporter.update({"forward_metrics": [*workers, legacy]})
    for _ in range(2):
        samples = [
            s
            for family in text_string_to_metric_families(exporter.render().decode())
            for s in family.samples
            if s.name == "atom:prefill_request_gpu_forward_seconds_sum"
        ]
        assert len(samples) == 4  # Old workers remain unknown, not fabricated zero.
        assert all(s.value == pytest.approx(0.008) for s in samples)
        assert all(
            "req_id" not in s.labels and "phase" not in s.labels for s in samples
        )
