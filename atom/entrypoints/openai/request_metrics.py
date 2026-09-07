"""Per-sequence timing before streaming chunks are coalesced or detokenized.

These are engine-output timings, not client SSE timings. Only completed
sequences are observed; fanout contributes one observation per sequence.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from pathlib import Path

from prometheus_client import Counter, Histogram

logger = logging.getLogger(__name__)

# Include long prefill tails for the 1M-context agentic workload.
TTFT_BUCKETS = (
    0.02,
    0.04,
    0.07,
    0.12,
    0.2,
    0.32,
    0.5,
    0.8,
    1.25,
    2,
    3.2,
    5,
    8,
    15,
    30,
    60,
    120,
    240,
    480,
    900,
    1800,
    3600,
)
TPOT_BUCKETS = (
    0.003,
    0.005,
    0.008,
    0.012,
    0.018,
    0.025,
    0.035,
    0.05,
    0.075,
    0.11,
    0.16,
    0.25,
    0.5,
    1,
    2,
    5,
    10,
    30,
    60,
)
E2E_BUCKETS = (
    0.1,
    0.25,
    0.5,
    1,
    2,
    4,
    8,
    15,
    30,
    60,
    120,
    240,
    480,
    900,
    1800,
    3600,
    7200,
)


class RequestMetrics:
    def __init__(self, registry):
        self.histograms = {
            name: Histogram(
                f"atom:{name}_seconds",
                description,
                ("model", "role"),
                buckets=buckets,
                registry=registry,
            )
            for name, description, buckets in (
                (
                    "ttft",
                    "Engine time to first output token per completed sequence.",
                    TTFT_BUCKETS,
                ),
                (
                    "tpot",
                    "Mean time per output token after the first per completed sequence.",
                    TPOT_BUCKETS,
                ),
                ("e2e_latency", "Engine latency per completed sequence.", E2E_BUCKETS),
                (
                    "output_chunk_interval",
                    "Time between engine output batches; a batch may contain multiple speculative tokens.",
                    TPOT_BUCKETS,
                ),
            )
        }
        self.dropped = Counter(
            "atom:request_events_dropped",
            "Request events dropped by the bounded writer.",
            registry=registry,
        )
        self.errors = Counter(
            "atom:request_events_write_errors",
            "Request event file write errors.",
            registry=registry,
        )
        self.writer = None
        self.configure(model="unknown", role="hybrid")

    def configure(self, *, model, role, events_path=None, run_id=""):
        self.close()
        self.model, self.role, self.run_id = model, role, run_id
        # Resolve labels once, outside the per-token and per-request paths.
        self.children = {k: v.labels(model, role) for k, v in self.histograms.items()}
        if events_path:
            self.writer = EventWriter(events_path, self.dropped, self.errors)

    def observe(self, *, ttft, tpot, latency, **event):
        for name, value in (("ttft", ttft), ("tpot", tpot), ("e2e_latency", latency)):
            if value is not None and math.isfinite(value) and value >= 0:
                self.children[name].observe(value)
        if self.writer is not None:
            self.writer.put(
                {
                    "schema_version": 1,
                    "event": "request_finished",
                    "run_id": self.run_id,
                    "model": self.model,
                    "role": self.role,
                    "ttft": ttft,
                    "tpot": tpot,
                    "latency": latency,
                    **event,
                }
            )

    def output_chunk(self, *, interval, **event):
        if interval is not None:
            self.children["output_chunk_interval"].observe(interval)
        if self.writer is not None:
            self.writer.put(
                {
                    "schema_version": 1,
                    "event": "output_chunk",
                    "run_id": self.run_id,
                    "model": self.model,
                    "role": self.role,
                    "interval": interval,
                    **event,
                }
            )

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class EventWriter:
    """Bounded, non-blocking producer; JSON serialization and IO run off-thread."""

    def __init__(self, path, dropped, errors, capacity=8192):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # Fail during startup if the configured event path is not writable.
        self.file = open(  # noqa: SIM115 - closed by the writer thread
            path, "a", encoding="utf-8", buffering=65536
        )
        self.queue = queue.Queue(maxsize=capacity)
        self.stopping = threading.Event()
        self.dropped, self.errors = dropped, errors
        self.thread = threading.Thread(
            target=self._run, daemon=True, name="atom-events"
        )
        self.thread.start()

    def put(self, event):
        if self.stopping.is_set():
            self.dropped.inc()
            return
        try:
            self.queue.put_nowait(event)
        except queue.Full:
            self.dropped.inc()

    def _run(self):
        last_flush = time.monotonic()
        try:
            while not self.stopping.is_set() or not self.queue.empty():
                try:
                    event = self.queue.get(timeout=0.2)
                except queue.Empty:
                    event = None
                if event is not None:
                    try:
                        self.file.write(json.dumps(event, allow_nan=False) + "\n")
                    except (OSError, ValueError):
                        self.errors.inc()
                        self.dropped.inc()
                if time.monotonic() - last_flush >= 1:
                    self.file.flush()
                    last_flush = time.monotonic()
        except OSError:
            self.errors.inc()
            self.stopping.set()
            logger.exception("Request event writer failed")
        finally:
            try:
                self.file.close()
            except OSError:
                self.errors.inc()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.errors.inc()
            logger.warning("Request event writer did not drain within 5 seconds")


class RequestObservation:
    """One callback-owned observation, shared by streaming and non-streaming."""

    def __init__(self, metrics, request_id, dp_rank=None, choice_index=0):
        self.metrics = metrics
        self.request_id, self.dp_rank, self.choice_index = (
            request_id,
            dp_rank,
            choice_index,
        )
        self.started = time.monotonic()
        self.first = self.last = None
        self.num_prompt_tokens = self.tokens = self.cached = 0
        self.finished = False

    def on_output(self, output):
        if self.finished:
            return
        now = time.monotonic()
        count = len(output.output_tokens or ())
        if count:
            interval = None if self.last is None else now - self.last
            if self.first is None:
                self.first = now
            self.last = now
            self.tokens += count
            self.metrics.output_chunk(
                interval=interval,
                ts=time.time(),
                elapsed=now - self.started,
                request_id=self.request_id,
                choice_index=self.choice_index,
                token_count=count,
                cumulative_tokens=self.tokens,
            )
        self.cached = max(self.cached, getattr(output, "num_cached_tokens", 0) or 0)
        if output.finished:
            self.finished = True
            self.metrics.observe(
                ttft=None if self.first is None else self.first - self.started,
                tpot=(
                    (self.last - self.first) / (self.tokens - 1)
                    if self.tokens > 1
                    else None
                ),
                latency=now - self.started,
                ts=time.time(),
                request_id=self.request_id,
                choice_index=self.choice_index,
                dp_rank=self.dp_rank,
                isl=self.num_prompt_tokens,
                osl=self.tokens,
                cached=self.cached,
                finish=output.finish_reason,
            )
