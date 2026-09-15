"""Scheduler-owned Prometheus observations.

Only the scheduler owner updates these counters. No GPU synchronization or
per-request labels are needed, and a scrape never consumes observations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from prometheus_client import Histogram

from atom.utils.histogram import LATENCY_BUCKETS

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


class SchedulerMetrics:
    def __init__(self, dp_rank=0, engine_role="default", *, registry=None):
        labels = {"dp_rank": str(dp_rank), "engine_role": engine_role}
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
        self.prefill_request_context_tokens = Histogram(
            "atom:prefill_request_context_tokens",
            "Logical context length per real prefill request row on each forward, including cached prefixes through the current chunk; request-forward weighted, without padding or TP multiplication.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.decode_context_tokens = Histogram(
            "atom:decode_context_tokens",
            "Sum of logical decode sequence lengths per real forward, without padding or TP multiplication.",
            labels,
            buckets=BATCH_CONTEXT_BUCKETS,
            registry=registry,
        ).labels(**labels)
        self.decode_request_context_tokens = Histogram(
            "atom:decode_request_context_tokens",
            "Logical context length per real decode request row on each forward; request-forward weighted, without padding or TP multiplication.",
            labels,
            buckets=TOKEN_BUCKETS,
            registry=registry,
        ).labels(**labels)
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

    def record_forward(self, batch, seqs) -> None:
        if batch.is_dummy_run or not batch.req_ids:
            return
        now = time.perf_counter()
        for req_id in batch.req_ids:
            timing = getattr(seqs[req_id], "queue_timing", None)
            if timing is None or timing.observed:
                continue
            self.queue_time.observe(now - timing.received_at)
            timing.observed = True
        # Count real request rows, not MTP tokens or a padded graph size.
        if batch.total_seqs_num_decode > 0:
            self.decode_batch_size.observe(batch.total_seqs_num_decode)
            context_lens = getattr(batch, "context_lens", None)
            if context_lens is not None:
                total_context = 0
                for length in context_lens[: batch.total_seqs_num_decode]:
                    tokens = int(length)
                    self.decode_request_context_tokens.observe(tokens)
                    total_context += tokens
                self.decode_context_tokens.observe(total_context)
        if getattr(batch, "total_seqs_num_prefill", 0) > 0:
            self.prefill_batch_tokens.observe(batch.total_tokens_num_prefill)
            context_lens = getattr(batch, "context_lens", None)
            total_context = 0
            # ScheduledBatch packs decode rows before prefill rows. Use its
            # immutable offsets: scheduling may already have advanced the seq.
            for i in range(batch.total_seqs_num_decode, len(batch.req_ids)):
                if context_lens is not None:
                    tokens = int(context_lens[i])
                    self.prefill_request_context_tokens.observe(tokens)
                    total_context += tokens
                seq = seqs[batch.req_ids[i]]
                timing = getattr(seq, "queue_timing", None)
                if timing is not None and not timing.prefill_observed:
                    self.prefill_request_tokens.observe(
                        max(0, seq.num_prompt_tokens - batch.num_cached_tokens[i])
                    )
                    timing.prefill_observed = True
            if context_lens is not None:
                self.prefill_context_tokens.observe(total_context)
