"""Nonblocking, bounded device-event timing for real target-model forwards."""

import os
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

from atom.utils.histogram import (
    LATENCY_BUCKETS,
    CumulativeHistogram,
    prometheus_buckets,
)

# Column order of the per-batch trace. Fixed schema, so a header beats
# re-spelling the field names on every one of the ~10^5 rows an hour produces.
TRACE_COLUMNS = (
    "seq",
    "phase",
    "enqueue_ns",
    "n_seqs",
    "n_prefill_seqs",
    "n_decode_seqs",
    "tokens",
    "tokens_prefill",
    "tokens_decode",
    "ctx_sum",
    "ctx_max",
    "spec_steps",
    "retire_ns",
    "device_seconds",
)
_TRACE_HEADER = ",".join(TRACE_COLUMNS) + "\n"
_TRACE_ROW = "%d,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.6f\n"


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
        trace_path=None,
        trace_capacity=1 << 16,
    ):
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
        # Per-batch trace, off unless a path is given. The histograms above
        # answer "how is the fleet distributed"; they cannot answer "how long
        # did THIS batch take, at what shape", because `observe` keeps only the
        # bucket the sample fell in. `poll` already computes the duration, so
        # the trace costs an append, not a measurement.
        self.trace_path = trace_path
        # Bounded: a drain that never comes must cost memory, not the run. The
        # sequence number makes the resulting hole visible offline rather than
        # silently shortening the series.
        self.trace = deque(maxlen=trace_capacity) if trace_path else None
        self.trace_seq = 0
        self.trace_dropped = 0
        self._trace_file = None

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
            phase, start, end, requests, sample = self.pending[0]
            if not end.query():
                break
            self.pending.popleft()
            seconds = start.elapsed_time(end) / 1000
            self.histograms[phase].observe(seconds)
            if sample is not None:
                # `retire_ns` is when the host *observed* completion, not when
                # the device finished: polling is at forward granularity, so it
                # is an upper bound, late by at most one forward. Paired with
                # the enqueue stamp frozen in `sample` it still separates the
                # two questions the single device span cannot -- how much wall
                # clock the batch occupied, versus how much of that the device
                # was actually inside the measured region.
                if len(self.trace) == self.trace.maxlen:
                    self.trace_dropped += 1
                self.trace.append((*sample, time.monotonic_ns(), seconds))
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
        p = batch.total_seqs_num_prefill > 0
        d = batch.total_seqs_num_decode > 0
        phase = "mixed" if p and d else "prefill" if p else "decode"
        sample = self._trace_sample(batch, phase) if self.trace is not None else None
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
        self.pending.append((phase, start, end, requests, sample))

    def _trace_sample(self, batch, phase):
        """Freeze this batch's shape, as scalars, at enqueue time.

        Scalars rather than a reference to `batch`: the batch owns the block
        tables and token arrays, and keeping up to `max_pending` of them alive
        behind the device would cost far more than reducing over at most
        `max_num_seqs` int32s once.
        """
        ctx = batch.context_lens
        self.trace_seq += 1
        return (
            self.trace_seq,
            phase,
            time.monotonic_ns(),
            len(batch.req_ids),
            batch.total_seqs_num_prefill,
            batch.total_seqs_num_decode,
            batch.total_tokens_num,
            batch.total_tokens_num_prefill,
            batch.total_tokens_num_decode,
            int(ctx.sum()) if len(ctx) else 0,
            int(ctx.max()) if len(ctx) else 0,
            batch.num_spec_step,
        )

    def drain_trace(self):
        """Append retired rows to the trace file.

        Reached from `snapshot`, which the push-based telemetry path runs in
        the worker's own loop between forwards and never waits on -- so this
        write displaces forward *launch* time, not forward time, and only once
        per scrape. A row costs a memcpy into the stdio buffer; the flush that
        follows is one `write(2)` of a scrape's worth of rows, tens of KB. It
        is there because a run that is killed -- Slurm deadline, OOM, an
        operator ctrl-C -- must still leave every row it had already retired on
        disk, and because it lets the file be read while the run is live. No
        fsync: the page cache is as far as this needs to get.
        """
        if not self.trace:
            return
        if self._trace_file is None:
            fresh = not os.path.exists(self.trace_path)
            self._trace_file = open(self.trace_path, "a", buffering=1 << 20)
            if fresh:
                self._trace_file.write(_TRACE_HEADER)
        # Swap rather than drain in place: O(1), and `poll` keeps appending to
        # the new deque if anything retires while this write is in flight.
        rows, self.trace = self.trace, deque(maxlen=self.trace.maxlen)
        self._trace_file.write("".join(_TRACE_ROW % row for row in rows))
        self._trace_file.flush()

    def close_trace(self):
        if self._trace_file is None:
            return
        self.drain_trace()
        self._trace_file.close()
        self._trace_file = None

    def snapshot(self):
        self.poll()
        self.drain_trace()
        return {
            "phases": {k: h.snapshot() for k, h in self.histograms.items()},
            "prefill_requests": self.prefill_requests.snapshot(),
            "trace_dropped": self.trace_dropped,
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
