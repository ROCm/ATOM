"""Scheduler-owned Prometheus observations.

Only the scheduler owner updates these metrics. Request context gauges carry
per-request labels. No GPU synchronization is needed, and scrapes never consume
observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from prometheus_client import Gauge, Histogram

from atom.metrics.histogram import LATENCY_BUCKETS

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
        self.prefill_request_context_tokens = Gauge(
            "atom:prefill_request_context_tokens",
            "Full prompt length including cached prefixes at first real prefill dispatch, once per request sequence; retained until metrics storage is cleaned.",
            [*labels, "request_id", "sequence_id", "started_at"],
            multiprocess_mode="max",
            registry=registry,
        )
        self.decode_context_tokens = Histogram(
            "atom:decode_context_tokens",
            "Sum of logical decode sequence lengths per real forward, without padding or TP multiplication.",
            labels,
            buckets=BATCH_CONTEXT_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.decode_request_context_tokens = Gauge(
            "atom:decode_request_context_tokens",
            "Logical context length at first real decode dispatch, once per request sequence; retained until metrics storage is cleaned.",
            [*labels, "request_id", "sequence_id", "started_at"],
            multiprocess_mode="max",
            registry=registry,
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

    def _record_request_context(self, metric, req_id, seq, tokens):
        # Retain completed requests so a scrape can still see short requests.
        metric.labels(
            **self._labels,
            request_id=(
                getattr(seq, "external_request_id", None)
                or getattr(seq, "parent_request_id", None)
                or str(req_id)
            ),
            sequence_id=str(req_id),
            started_at=str(time.time()),
        ).set(int(tokens))

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
                self._record_request_context(
                    self.decode_request_context_tokens, req_id, seq, context_lens[i]
                )
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
                        self.prefill_request_context_tokens,
                        req_id,
                        seq,
                        seq.num_prompt_tokens,
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
