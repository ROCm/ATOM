"""CPU-only, cumulative scheduler and executed-batch statistics.

Updated on the engine thread and copied into its existing metrics snapshot.
No CUDA operations, per-step IO, or unbounded per-request maps.
"""

import bisect
import time

LAST_BATCH_FIELDS = (
    "batch_size",
    "query_tokens",
    "query_tokens_min",
    "query_tokens_max",
    "query_tokens_mean",
    "context_tokens_min",
    "context_tokens_max",
    "context_tokens_mean",
)


class CumulativeHistogram:
    def __init__(self, bounds):
        self.bounds = tuple(bounds) + (float("inf"),)
        self.counts = [0] * len(self.bounds)
        self.total = 0.0

    def observe(self, value):
        self.counts[bisect.bisect_left(self.bounds, value)] += 1
        self.total += value

    def snapshot(self):
        count = 0
        buckets = []
        for bound, observations in zip(self.bounds, self.counts):
            count += observations
            buckets.append(("+Inf" if bound == float("inf") else str(bound), count))
        return {"buckets": buckets, "sum": self.total}


class SchedulingMetrics:
    def __init__(self):
        self.duration = CumulativeHistogram(
            (
                0.00001,
                0.00005,
                0.0001,
                0.0005,
                0.001,
                0.005,
                0.01,
                0.05,
                0.1,
                0.5,
                1,
                5,
                30,
            )
        )
        self.queue = CumulativeHistogram(
            (0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 900, 1800, 3600)
        )
        self.stages = {}
        for stage in ("prefill", "decode"):
            self.stages[stage] = {
                "batch_size": CumulativeHistogram(
                    (
                        1,
                        2,
                        4,
                        8,
                        16,
                        24,
                        32,
                        40,
                        48,
                        64,
                        96,
                        128,
                        256,
                        512,
                        1024,
                        2048,
                        4096,
                        8192,
                    )
                ),
                "query_tokens": CumulativeHistogram(
                    (
                        1,
                        2,
                        4,
                        8,
                        16,
                        32,
                        64,
                        128,
                        256,
                        512,
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
                    )
                ),
                "context_tokens": CumulativeHistogram(
                    (
                        1,
                        16,
                        128,
                        512,
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
                    )
                ),
            }
        self.last = {
            stage: dict.fromkeys(LAST_BATCH_FIELDS, 0) for stage in self.stages
        }
        self.last_execution = 0.0

    @staticmethod
    def enqueue(seq):
        if not hasattr(seq, "_metrics_enqueued_at"):
            seq._metrics_enqueued_at = time.monotonic()
            seq._metrics_first_execution = False

    def execute(self, batch, seqs):
        if getattr(batch, "is_dummy_run", False) or not batch.req_ids:
            return
        now = time.monotonic()
        self.last_execution = time.time()
        # ScheduledBatch stores prefill rows first, followed by decode rows.
        prefill_count = batch.total_seqs_num_prefill
        for stage, begin, end in (
            ("prefill", 0, prefill_count),
            ("decode", prefill_count, len(batch.req_ids)),
        ):
            size = end - begin
            last = self.last[stage]
            for key in LAST_BATCH_FIELDS:
                last[key] = 0
            last["batch_size"] = size
            if not size:
                continue
            stats = self.stages[stage]
            stats["batch_size"].observe(size)
            for i in range(begin, end):
                query, context = int(batch.num_scheduled_tokens[i]), int(
                    batch.context_lens[i]
                )
                stats["query_tokens"].observe(query)
                stats["context_tokens"].observe(context)
                last["query_tokens"] += query
                last["query_tokens_min"] = (
                    query if i == begin else min(last["query_tokens_min"], query)
                )
                last["query_tokens_max"] = max(last["query_tokens_max"], query)
                last["context_tokens_min"] = (
                    context if i == begin else min(last["context_tokens_min"], context)
                )
                last["context_tokens_max"] = max(last["context_tokens_max"], context)
                last["context_tokens_mean"] += context
                seq = seqs[batch.req_ids[i]]
                if (
                    hasattr(seq, "_metrics_enqueued_at")
                    and not seq._metrics_first_execution
                ):
                    self.queue.observe(max(0, now - seq._metrics_enqueued_at))
                    seq._metrics_first_execution = True
            last["query_tokens_mean"] = last["query_tokens"] / size
            last["context_tokens_mean"] /= size

    def snapshot(self):
        return {
            "duration": self.duration.snapshot(),
            "queue": self.queue.snapshot(),
            "stages": {
                stage: {key: hist.snapshot() for key, hist in stats.items()}
                for stage, stats in self.stages.items()
            },
            "last": {stage: dict(values) for stage, values in self.last.items()},
            "last_execution_timestamp_seconds": self.last_execution,
        }
