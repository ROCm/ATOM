"""Scheduler-owned Prometheus observations.

Only the scheduler owner updates these metrics. Per-request contexts are kept
for a bounded window and published in the metrics snapshot. No GPU
synchronization is needed, and scrapes never consume observations.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import numpy as np
from prometheus_client import Histogram

from atom.metrics.histogram import LATENCY_BUCKETS
from atom.utils import envs

BATCH_BUCKETS = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512, 1024)
TOKEN_BUCKETS = (
    0,
    16,
    64,
    256,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
    2097152,
    4194304,
    8388608,
)
# A batch sums many contexts. Keep fixed bounds across schedulers so their
# histograms can be pooled, covering up to 1024 rows of 8M tokens each.
BATCH_CONTEXT_BUCKETS = (*TOKEN_BUCKETS, *(2**power for power in range(24, 34)))
# Bounds the retained contexts at high request rates. Each one costs every
# scrape about 13 us of the API server's Python (copied, then rendered), so
# the cap holds that to ~14 ms; 2.4 requests/s over the default 120 s window
# keeps about 300, and 20 requests/s still keeps each one for ~50 s.
REQUEST_CONTEXT_CAP = 1024


@dataclass
class RequestQueueTiming:
    received_at: float
    observed: bool = False
    is_pd: bool = False
    prefill_observed: bool = False
    decode_context_observed: bool = False


class SchedulerMetrics:
    def __init__(self, dp_rank=0, engine_role="default", *, registry=None):
        labels = {"dp_rank": str(dp_rank), "engine_role": engine_role}
        self._labels = labels
        self.queue_time = Histogram(
            "atom:request_queue_time_seconds",
            "Time from engine receipt to first real forward dispatch, including KV loading waits.",
            labels,
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.decode_batch_size = Histogram(
            "atom:decode_batch_size",
            "Real decode request rows per forward; excludes dummy work and graph padding.",
            labels,
            buckets=BATCH_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.pd_transfer = Histogram(
            "atom:pd_kv_transfer_seconds",
            "Decode-side PD KV load wait until all workers complete; includes dispatch, handshake and notification.",
            labels,
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.prefill_request_tokens = Histogram(
            "atom:prefill_request_tokens",
            "Prompt tokens remaining at first local prefill dispatch, once per request.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.prefill_batch_tokens = Histogram(
            "atom:prefill_batch_tokens",
            "Real prefill tokens scheduled per forward, excluding cached prefix and padding.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.prefill_context_tokens = Histogram(
            "atom:prefill_context_tokens",
            "Sum of logical prefill context lengths at the current chunk end per real forward, including cached prefixes and excluding decode rows and padding.",
            labels,
            buckets=BATCH_CONTEXT_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.decode_context_tokens = Histogram(
            "atom:decode_context_tokens",
            "Sum of logical decode sequence lengths per real forward, without padding or TP multiplication.",
            labels,
            buckets=BATCH_CONTEXT_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self._request_context_retention_s = (
            envs.ATOM_METRICS_REQUEST_CONTEXT_RETENTION_S
        )
        # (monotonic time, phase, request_id, sequence_id, started_at, tokens),
        # oldest first. Not multiprocess gauges: mmap storage can never drop a
        # label set, so a series per request served piles up there and every
        # scrape of the API server re-reads all of them.
        self._request_contexts: deque[tuple[float, str, str, str, str, int]] = deque(
            maxlen=REQUEST_CONTEXT_CAP
        )
        # Only in-flight external loads are retained; removed on every terminal
        # path, including abort and fallback. Sequence timing dies with the seq.
        self._loads: dict[str, tuple[object, float]] = {}

    @staticmethod
    def enqueue(seq, *, received_at: float | None = None) -> None:
        # The input thread stamps receipt before buffering the request. Keep
        # that timestamp when the scheduler drains the input queue later.
        # Direct scheduler users fall back to their admission time.
        if received_at is None:
            if getattr(seq, "queue_timing", None) is not None:
                return
            received_at = time.perf_counter()
        seq.queue_timing = RequestQueueTiming(
            received_at=received_at,
            is_pd=bool(
                (getattr(seq, "kv_transfer_params", None) or {}).get(
                    "do_remote_prefill"
                )
            ),
        )

    def start_kv_wait(self, seq) -> None:
        key = str(seq.id)
        if key in self._loads:
            return
        self._loads[key] = (seq, time.perf_counter())

    def finish_kv_wait(self, req_id, *, succeeded: bool) -> None:
        pending = self._loads.pop(str(req_id), None)
        if pending is None:
            return
        seq, started = pending
        now = time.perf_counter()
        timing = getattr(seq, "queue_timing", None)
        if timing is not None and succeeded and timing.is_pd:
            self.pd_transfer.observe(now - started)

    def _record_request_context(self, phase, req_id, seq, tokens):
        self._request_contexts.append(
            (
                time.monotonic(),
                phase,
                getattr(seq, "external_request_id", None)
                or getattr(seq, "parent_request_id", None)
                or str(req_id),
                str(req_id),
                str(time.time()),
                int(tokens),
            )
        )

    def request_contexts(self) -> list[tuple[str, str, str, str, str, str, int]]:
        """First-dispatch contexts recorded within the retention window, as
        (phase, dp_rank, engine_role, request_id, sequence_id, started_at,
        tokens). Completed requests stay for the whole window, so a scrape
        every few seconds still sees even short ones."""
        horizon = time.monotonic() - self._request_context_retention_s
        contexts = self._request_contexts
        while contexts and contexts[0][0] < horizon:
            contexts.popleft()
        rank, role = self._labels["dp_rank"], self._labels["engine_role"]
        return [(phase, rank, role, *rest) for _, phase, *rest in contexts]

    def record_forward(self, batch, seqs) -> None:
        if batch.is_dummy_run or not batch.req_ids:
            return
        now = time.perf_counter()
        context_lens = getattr(batch, "context_lens", None)
        for i, req_id in enumerate(batch.req_ids):
            seq = seqs[req_id]
            timing = getattr(seq, "queue_timing", None)
            if timing is None:
                continue
            if not timing.observed:
                self.queue_time.observe(now - timing.received_at)
                timing.observed = True
            if (
                i < batch.total_seqs_num_decode
                and context_lens is not None
                and not timing.decode_context_observed
            ):
                # The marker survives preemption.
                timing.decode_context_observed = True
                self._record_request_context("decode", req_id, seq, context_lens[i])
        # Count real request rows, not MTP tokens or a padded graph size.
        if batch.total_seqs_num_decode > 0:
            self.decode_batch_size.observe(batch.total_seqs_num_decode)
            if context_lens is not None:
                self.decode_context_tokens.observe(
                    int(
                        np.sum(
                            context_lens[: batch.total_seqs_num_decode], dtype=np.int64
                        )
                    )
                )
        if getattr(batch, "total_seqs_num_prefill", 0) > 0:
            self.prefill_batch_tokens.observe(batch.total_tokens_num_prefill)
            # ScheduledBatch packs decode rows before prefill rows.
            for i in range(batch.total_seqs_num_decode, len(batch.req_ids)):
                req_id = batch.req_ids[i]
                seq = seqs[req_id]
                timing = getattr(seq, "queue_timing", None)
                if timing is not None and not timing.prefill_observed:
                    self.prefill_request_tokens.observe(
                        max(0, seq.num_prompt_tokens - batch.num_cached_tokens[i])
                    )
                    # Full input length, independent of chunk size or prefix
                    # reuse. The same marker survives chunks and preemption.
                    self._record_request_context(
                        "prefill", req_id, seq, seq.num_prompt_tokens
                    )
                    timing.prefill_observed = True
            if context_lens is not None:
                # Batch context still uses immutable chunk ends, including
                # cached prefixes and excluding decode rows and graph padding.
                self.prefill_context_tokens.observe(
                    int(
                        np.sum(
                            context_lens[
                                batch.total_seqs_num_decode : len(batch.req_ids)
                            ],
                            dtype=np.int64,
                        )
                    )
                )
