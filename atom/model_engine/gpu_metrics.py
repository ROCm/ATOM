"""Nonblocking, bounded device-event timing for real target-model forwards."""

import time
from collections import deque
from contextlib import contextmanager
from functools import wraps

from atom.model_engine.scheduler_metrics import LATENCY_BUCKETS, CumulativeHistogram


class GPUForwardMetrics:
    def __init__(self, event_factory, max_pending=256):
        self.event_factory = event_factory
        self.max_pending = max_pending
        self.pending = deque()
        self.free = []
        self.histograms = {
            phase: CumulativeHistogram(LATENCY_BUCKETS)
            for phase in ("prefill", "decode", "mixed")
        }
        self.dropped = 0

    def poll(self):
        # query() never waits for the GPU. Different streams can complete out
        # of order; retain unready pairs without blocking completed samples.
        for _ in range(len(self.pending)):
            phase, start, end = self.pending.popleft()
            if end.query():
                self.histograms[phase].observe(start.elapsed_time(end) / 1000)
                self.free.append((start, end))
            else:
                self.pending.append((phase, start, end))

    @contextmanager
    def measure(self, batch):
        self.poll()
        if batch is None or batch.is_dummy_run or not batch.req_ids:
            yield
            return
        if len(self.pending) >= self.max_pending:
            self.dropped += 1
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
        # An exception thrown through yield skips publication of this sample.
        yield
        end.record()
        self.pending.append((phase, start, end))

    def snapshot(self):
        self.poll()
        return {
            "phases": {k: h.snapshot() for k, h in self.histograms.items()},
            "pending": len(self.pending),
            "dropped": self.dropped,
            "timestamp": time.time(),
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
