"""Nonblocking, bounded device-event timing for real target-model forwards."""

import logging
import math
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

from prometheus_client import Histogram

from atom.metrics.histogram import LATENCY_BUCKETS

logger = logging.getLogger("atom")


@dataclass
class _RequestTiming:
    req_id: int
    chunks: int = 0
    pending: int = 0
    final: bool = False
    valid: bool = True
    total: float = 0.0


class GPUForwardMetrics:
    def __init__(
        self,
        event_factory,
        max_pending=256,
        max_requests=4096,
        *,
        dp_rank=0,
        pp_rank=0,
        tp_rank=0,
        engine_role="default",
        registry=None,
    ):
        self.event_factory = event_factory
        self.max_pending = max_pending
        self.pending = deque()
        self.free = []
        labels = {
            "dp_rank": str(dp_rank),
            "pp_rank": str(pp_rank),
            "tp_rank": str(tp_rank),
            "engine_role": engine_role,
        }
        self.steps = Histogram(
            "atom:gpu_forward_seconds",
            "Per-worker target forward step device-event duration, including stream communication/waits; excludes input preparation, sampling and drafting.",
            labels,
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.prefill_requests = Histogram(
            "atom:prefill_request_gpu_forward_seconds",
            "Per-worker sum of participating batch device durations across a request's initial local prefill chunks; once after all chunks complete, not exclusive request compute time.",
            labels,
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ).labels(**labels)
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
        # Stop at the first unfinished event instead of scanning the backlog.
        # Other streams may finish sooner; their samples wait for the head.
        while self.pending:
            start, end, requests = self.pending[0]
            if not end.query():
                break
            self.pending.popleft()
            seconds = start.elapsed_time(end) / 1000
            if not math.isfinite(seconds) or seconds < 0:
                logger.warning(
                    "Invalid GPU forward duration %r; discarding %d request timings",
                    seconds,
                    len(requests),
                )
                for state in requests:
                    if self.requests.get(state.req_id) is state:
                        self._discard_request(state.req_id)
                self.free.append((start, end))
                continue
            self.steps.observe(seconds)
            for state in requests:
                state.pending -= 1
                if not state.valid:
                    continue
                state.total += seconds
                # Publish only once every chunk has been measured.
                if state.final and state.pending == 0:
                    self.prefill_requests.observe(state.total)
                    del self.requests[state.req_id]
            self.free.append((start, end))

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
            # No pending sample owns this pair; record() replaces its state
            # when the next forward reuses it.
            self.free.append((start, end))
            raise
        self.pending.append((start, end, requests))


def record_gpu_forward(func):
    @wraps(func)
    def wrapped(self, input_ids, batch=None):
        metrics = getattr(self, "gpu_forward_metrics", None)
        if metrics is None:
            return func(self, input_ids, batch)
        with metrics.measure(batch):
            return func(self, input_ids, batch)

    return wrapped
