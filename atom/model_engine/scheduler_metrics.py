"""Cumulative scheduler observations, transported by the existing metrics snapshot.

Only the scheduler owner updates these counters. No GPU synchronization or
per-request labels are needed, and a scrape never consumes observations.
"""

from __future__ import annotations

import math
import time
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate

LATENCY_BUCKETS = (
    0.001,
    0.002,
    0.005,
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.5,
    1,
    2,
    5,
    10,
    20,
    30,
    60,
    120,
    300,
    600,
)
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


class CumulativeHistogram:
    def __init__(self, bounds):
        self.bounds = (*bounds, math.inf)
        self.counts = [0] * len(self.bounds)
        self.total = 0.0

    def observe(self, value: float) -> None:
        if not math.isfinite(value) or value < 0:
            return
        self.counts[bisect_left(self.bounds, value)] += 1
        self.total += value

    def snapshot(self) -> dict:
        return {
            "buckets": list(zip(self.bounds, accumulate(self.counts))),
            "sum": self.total,
        }


@dataclass
class RequestQueueTiming:
    received_at: float
    observed: bool = False
    is_pd: bool = False
    prefill_observed: bool = False


class SchedulerMetrics:
    def __init__(self):
        self.queue_time = CumulativeHistogram(LATENCY_BUCKETS)
        self.decode_batch_size = CumulativeHistogram(BATCH_BUCKETS)
        self.pd_transfer = CumulativeHistogram(LATENCY_BUCKETS)
        self.prefill_request_tokens = CumulativeHistogram(TOKEN_BUCKETS)
        self.prefill_batch_tokens = CumulativeHistogram(TOKEN_BUCKETS)
        self.decode_context_tokens = CumulativeHistogram(TOKEN_BUCKETS)
        self.decode_request_context_tokens = CumulativeHistogram(TOKEN_BUCKETS)
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
            # ScheduledBatch packs decode rows before prefill rows. Use its
            # immutable offsets: scheduling may already have advanced the seq.
            for i in range(batch.total_seqs_num_decode, len(batch.req_ids)):
                seq = seqs[batch.req_ids[i]]
                timing = getattr(seq, "queue_timing", None)
                if timing is not None and not timing.prefill_observed:
                    self.prefill_request_tokens.observe(
                        max(0, seq.num_prompt_tokens - batch.num_cached_tokens[i])
                    )
                    timing.prefill_observed = True

    def snapshot(self) -> dict:
        return {
            "queue_time": self.queue_time.snapshot(),
            "decode_batch_size": self.decode_batch_size.snapshot(),
            "pd_kv_transfer": self.pd_transfer.snapshot(),
            "prefill_request_tokens": self.prefill_request_tokens.snapshot(),
            "prefill_batch_tokens": self.prefill_batch_tokens.snapshot(),
            "decode_context_tokens": self.decode_context_tokens.snapshot(),
            "decode_request_context_tokens": self.decode_request_context_tokens.snapshot(),
        }
