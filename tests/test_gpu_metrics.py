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

    def record(self):
        self.recorded += 1
        self.ready = False

    def query(self):
        return self.ready

    def elapsed_time(self, end):
        assert self.ready and end.ready
        return 8.0

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
    assert metrics.snapshot()["pending"] == 2
    with metrics.measure(batch()):
        pass
    assert metrics.snapshot()["dropped"] == 1
    # A different stream may complete the second pair before the first.
    for event in metrics.pending[1][1:]:
        event.ready = True
    snapshot = metrics.snapshot()
    assert snapshot["pending"] == 1
    assert snapshot["phases"]["prefill"]["sum"] == 0.008
    assert snapshot["phases"]["decode"]["sum"] == 0
    reused = tuple(metrics.free[-1])
    with metrics.measure(batch(prefill=1, decode=1)):
        pass
    assert tuple(metrics.pending[-1][1:]) == reused
    for _, start, end in metrics.pending:
        start.ready = end.ready = True
    snapshot = metrics.snapshot()
    assert snapshot["pending"] == 0
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
    assert metrics.snapshot()["pending"] == 0

    @record_gpu_forward
    def model(self, inputs, batch=None):
        return inputs + 1

    assert model(SimpleNamespace(), 4, batch()) == 5
    assert model(SimpleNamespace(gpu_forward_metrics=metrics), 4, batch()) == 5
    assert metrics.snapshot()["pending"] == 1


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
    assert snapshot["pending"] == 0
    assert snapshot["phases"]["decode"]["buckets"][-1][1] == 1
    assert snapshot["phases"]["decode"]["sum"] > 0
    assert output[0, 0].item() == 64
