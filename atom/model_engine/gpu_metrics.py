"""Nonblocking, bounded device-event timing for real target-model forwards."""

from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

from atom.utils.histogram import (
    LATENCY_BUCKETS,
    CumulativeHistogram,
    prometheus_buckets,
)


@dataclass
class _RequestTiming:
    req_id: int
    chunks: int = 0
    pending: int = 0
    final: bool = False
    valid: bool = True
    total: float = 0.0


class GPUForwardMetrics:
    def __init__(self, event_factory, max_pending=256, max_requests=4096):
        self.event_factory = event_factory
        self.max_pending = max_pending
        self.pending = deque()
        self.free = []
        self.histograms = {
            phase: CumulativeHistogram(LATENCY_BUCKETS)
            for phase in ("prefill", "decode", "mixed")
        }
        self.prefill_requests = CumulativeHistogram(LATENCY_BUCKETS)
        self.max_requests = max_requests
        self.requests = OrderedDict()

    def _discard_request(self, req_id):
        state = self.requests.pop(req_id, None)
        if state is not None:
            state.valid = False

    def _request_chunks(self, batch):
        states = []
        for req_id, chunk, final in getattr(batch, "prefill_gpu_requests", ()):
            if chunk == 1:
                # Request IDs may be reused; pending events keep the old state
                # object and must never finish a newly admitted request.
                self._discard_request(req_id)
                if len(self.requests) >= self.max_requests:
                    self._discard_request(next(iter(self.requests)))
                self.requests[req_id] = _RequestTiming(req_id)
            state = self.requests.get(req_id)
            if state is None:
                continue  # Missing/evicted first chunk: never publish a partial sum.
            if chunk != state.chunks + 1 or state.final:
                self._discard_request(req_id)
                continue
            state.chunks = chunk
            state.pending += 1
            state.final = final
            states.append(state)
            self.requests.move_to_end(req_id)
        return states

    def poll(self):
        # query() never waits for the GPU. Different streams can complete out
        # of order; retain unready pairs without blocking completed samples.
        for _ in range(len(self.pending)):
            phase, start, end, requests = self.pending.popleft()
            if end.query():
                seconds = start.elapsed_time(end) / 1000
                self.histograms[phase].observe(seconds)
                for state in requests:
                    state.pending -= 1
                    if not state.valid:
                        continue
                    state.total += seconds
                    # Last chunk can finish on a different stream before earlier
                    # events are ready. Publish only once every chunk is measured.
                    if state.final and state.pending == 0:
                        self.prefill_requests.observe(state.total)
                        del self.requests[state.req_id]
                self.free.append((start, end))
            else:
                self.pending.append((phase, start, end, requests))

    @contextmanager
    def measure(self, batch):
        self.poll()
        if batch is None or batch.is_dummy_run or not batch.req_ids:
            yield
            return
        requests = self._request_chunks(batch)
        if len(self.pending) >= self.max_pending:
            for state in requests:
                self._discard_request(state.req_id)
            yield
            return
        p = batch.total_seqs_num_prefill > 0
        d = batch.total_seqs_num_decode > 0
        phase = "mixed" if p and d else "prefill" if p else "decode"
        start, end = (
            self.free.pop()
            if self.free
            else (self.event_factory(), self.event_factory())
        )
        start.record()
        try:
            yield
            end.record()
        except BaseException:
            for state in requests:
                self._discard_request(state.req_id)
            raise
        self.pending.append((phase, start, end, requests))

    def snapshot(self):
        self.poll()
        return {
            "phases": {k: h.snapshot() for k, h in self.histograms.items()},
            "prefill_requests": self.prefill_requests.snapshot(),
        }


def record_gpu_forward(func):
    @wraps(func)
    def wrapped(self, input_ids, batch=None):
        metrics = getattr(self, "gpu_forward_metrics", None)
        if metrics is None:
            return func(self, input_ids, batch)
        with metrics.measure(batch):
            return func(self, input_ids, batch)

    return wrapped


def collect_gpu_metrics(snapshot):
    """Export worker snapshots without synchronizing devices or re-observing."""
    from prometheus_client.core import HistogramMetricFamily

    workers = (snapshot or {}).get("forward_metrics", [])
    labels = ["dp_rank", "pp_rank", "tp_rank", "engine_role"]
    duration = HistogramMetricFamily(
        "atom:gpu_forward_seconds",
        "Per-worker target forward device-event duration, including stream communication/waits; excludes input preparation, sampling and drafting.",
        labels=[*labels, "phase"],
    )
    request_duration = HistogramMetricFamily(
        "atom:prefill_request_gpu_forward_seconds",
        "Per-worker sum of participating batch device durations across a request's initial local prefill chunks; once after all chunks complete, not exclusive request compute time.",
        labels=labels,
    )
    for worker in workers:
        values = [str(worker[k]) for k in labels]
        for phase, hist in worker["phases"].items():
            duration.add_metric(
                [*values, phase],
                buckets=prometheus_buckets(hist),
                sum_value=hist["sum"],
            )
        if "prefill_requests" in worker:
            hist = worker["prefill_requests"]
            request_duration.add_metric(
                values,
                buckets=prometheus_buckets(hist),
                sum_value=hist["sum"],
            )
    yield duration
    yield request_duration
