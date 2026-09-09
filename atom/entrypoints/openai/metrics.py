"""Prometheus exposition for the standalone ATOM OpenAI server."""

from __future__ import annotations

import copy
import gc
import threading
import time
from bisect import bisect_left
from collections.abc import Iterable
from typing import Any

from prometheus_client import CollectorRegistry, Histogram, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from .streaming_dispatch import longest_silence_seconds


class _WeightedHistogram:
    """Aggregate equal observations with one bucket lookup and one lock.

    Own the counters and expose them through Prometheus' public collector API.
    A scrape snapshots bucket counts and the sum under the same lock.
    """

    def __init__(self, name, documentation, *, buckets, registry):
        self._name = name
        self._documentation = documentation
        self._bounds = (*buckets, float("inf"))
        self._counts = [0] * len(self._bounds)
        self._sum = 0.0
        self._created = time.time()
        self._lock = threading.Lock()
        registry.register(self)

    def observe_weighted(self, total: float, weight: int) -> None:
        """Record ``weight`` equal samples whose sum is ``total``."""
        if weight <= 0:
            return
        index = bisect_left(self._bounds, total / weight)
        with self._lock:
            self._counts[index] += weight
            self._sum += total

    def collect(self):
        with self._lock:
            counts, total = self._counts.copy(), self._sum
        cumulative = 0
        buckets = []
        for bound, count in zip(self._bounds, counts):
            cumulative += count
            buckets.append(
                ("+Inf" if bound == float("inf") else str(bound), cumulative)
            )
        yield HistogramMetricFamily(
            self._name, self._documentation, buckets=buckets, sum_value=total
        )
        yield GaugeMetricFamily(
            self._name + "_created", self._documentation, value=self._created
        )


class _AtomMetricsCollector:
    def __init__(self, exporter: AtomMetricsExporter):
        self._exporter = exporter

    def collect(self) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
        snapshot, refresh_errors, last_refresh = self._exporter.read()
        available = bool(snapshot.get("enabled", False))

        metric = GaugeMetricFamily(
            "atom:metrics_snapshot_available",
            "Whether a runtime metrics snapshot has been collected successfully.",
        )
        metric.add_metric([], float(available))
        yield metric

        metric = CounterMetricFamily(
            "atom:metrics_refresh_errors",
            "Number of failed runtime metrics refreshes.",
        )
        metric.add_metric([], float(refresh_errors))
        yield metric

        metric = GaugeMetricFamily(
            "atom:metrics_last_refresh_timestamp_seconds",
            "Unix timestamp of the last successful runtime metrics refresh.",
        )
        metric.add_metric([], last_refresh)
        yield metric

        # Read live rather than from the snapshot: the snapshot is refreshed by
        # the engine, and a stream starved by the engine is exactly the case
        # where that refresh may also be late. This one is answered by the
        # event loop that is serving the stalled request.
        metric = GaugeMetricFamily(
            "atom:stream_longest_silence_seconds",
            "Seconds the most starved in-flight SSE stream has gone without a "
            "chunk. Zero when none is waiting. Non-zero and growing is a "
            "response that has stopped delivering while the client waits.",
        )
        metric.add_metric([], longest_silence_seconds())
        yield metric

        gauges = (
            (
                "atom:requests_running",
                "Number of requests currently running across data-parallel ranks.",
                snapshot.get("requests_running", 0),
            ),
            (
                "atom:requests_waiting",
                "Number of requests waiting across data-parallel ranks.",
                snapshot.get("requests_waiting", 0),
            ),
            (
                "atom:requests_parked_kv_load",
                "Number of requests parked for an external KV load.",
                snapshot.get("requests_parked_kv_load", 0),
            ),
            (
                "atom:requests_partial_prefill",
                "Number of requests currently in chunked prefill.",
                snapshot.get("requests_partial_prefill", 0),
            ),
            (
                "atom:kv_cache_blocks_used",
                "Number of allocated KV-cache blocks.",
                snapshot.get("kv_blocks_used", 0),
            ),
            (
                "atom:kv_cache_blocks_free",
                "Number of free KV-cache blocks.",
                snapshot.get("kv_blocks_free", 0),
            ),
            (
                "atom:kv_cache_blocks_total",
                "Total number of KV-cache blocks.",
                snapshot.get("kv_blocks_total", 0),
            ),
            (
                "atom:kv_cache_blocks_indexed",
                "Number of KV-cache blocks reachable by prefix hash.",
                snapshot.get("kv_blocks_indexed", 0),
            ),
            (
                "atom:kv_cache_usage_ratio",
                "Fraction of KV-cache blocks currently allocated.",
                snapshot.get("kv_cache_usage_ratio", 0),
            ),
        )
        for name, documentation, value in gauges:
            metric = GaugeMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        for name, documentation, value in (
            (
                "atom:requests_finished",
                "Number of requests completed by the scheduler.",
                snapshot.get("requests_finished", 0),
            ),
            (
                "atom:prompt_tokens",
                "Number of prompt tokens in completed requests.",
                snapshot.get("prompt_tokens", 0),
            ),
            (
                "atom:generation_tokens",
                "Number of generated tokens in completed requests.",
                snapshot.get("generation_tokens", 0),
            ),
            (
                "atom:preemptions",
                "Number of scheduler preemptions.",
                snapshot.get("preemptions", 0),
            ),
        ):
            metric = CounterMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        dp_router = snapshot.get("dp_router", {})
        for name, documentation, value in (
            (
                "atom:dp_affinity_new",
                (
                    "Number of new sticky DP sessions assigned to a load-aware "
                    "cache owner."
                ),
                dp_router.get("affinity_new_total", 0),
            ),
            (
                "atom:dp_affinity_owner_hit",
                "Number of requests routed to an existing session cache owner.",
                dp_router.get("affinity_owner_hit_total", 0),
            ),
            (
                "atom:dp_affinity_spill",
                (
                    "Number of existing sessions moved off their cache owner; "
                    "strict affinity keeps this zero."
                ),
                dp_router.get("affinity_spill_total", 0),
            ),
            (
                "atom:dp_affinity_parent_ignored",
                (
                    "Number of new child sessions independently placed instead of "
                    "inheriting a parent owner."
                ),
                dp_router.get("affinity_parent_ignored_total", 0),
            ),
            (
                "atom:dp_route_explicit",
                "Number of requests routed by an explicit data-parallel rank.",
                dp_router.get("explicit_total", 0),
            ),
            (
                "atom:dp_route_load_balanced",
                (
                    "Number of sessionless requests routed by the configured load "
                    "balancer."
                ),
                dp_router.get("load_balanced_total", 0),
            ),
        ):
            metric = CounterMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        metric = CounterMetricFamily(
            "atom:dp_requests_routed",
            "Cumulative requests routed to each data-parallel rank.",
            labels=["rank"],
        )
        for rank, value in enumerate(dp_router.get("requests_per_rank", [])):
            metric.add_metric([str(rank)], float(value))
        yield metric

        for name, documentation, values in (
            (
                "atom:dp_inflight_requests",
                "Current in-flight requests charged to each data-parallel rank.",
                dp_router.get("inflight_requests_per_rank", []),
            ),
            (
                "atom:dp_queued_prefill_tokens",
                (
                    "Current estimated uncached prefill-token debt per "
                    "data-parallel rank; later sticky turns charge only positive "
                    "prompt growth."
                ),
                dp_router.get("queued_prefill_tokens_per_rank", []),
            ),
            (
                "atom:dp_sessions",
                "Sticky sessions currently owned by each data-parallel rank.",
                dp_router.get("session_count_per_rank", []),
            ),
        ):
            metric = GaugeMetricFamily(name, documentation, labels=["rank"])
            for rank, value in enumerate(values):
                metric.add_metric([str(rank)], float(value))
            yield metric

        cache = snapshot.get("cache", {})
        cache_counters = (
            (
                "atom:prefix_cache_requests",
                "Number of prefill requests observed by prefix-cache accounting.",
                cache.get("requests", 0),
            ),
            (
                "atom:prefix_cache_cached_tokens",
                "Number of prompt tokens served from the admitted prefix cache.",
                cache.get("cached_tokens", 0),
            ),
            (
                "atom:prefix_cache_compressed_tokens",
                "Number of tokens matched by the compressed-prefix index.",
                cache.get("compressed_tokens", 0),
            ),
            (
                "atom:prefix_cache_full_tokens",
                "Number of prompt tokens considered by prefix-cache accounting.",
                cache.get("full_tokens", 0),
            ),
            (
                "atom:prefix_cache_wanted_tokens",
                "Number of reusable tokens wanted after checkpoint gates.",
                cache.get("wanted_tokens", 0),
            ),
            (
                "atom:prefix_cache_checkpoints_kept",
                "Number of prefix-cache checkpoints kept.",
                cache.get("checkpoints_kept", 0),
            ),
            (
                "atom:prefix_cache_checkpoints_dropped",
                "Number of prefix-cache checkpoints dropped.",
                cache.get("checkpoints_dropped", 0),
            ),
            (
                "atom:prefix_cache_checkpoints_evicted",
                "Number of prefix-cache checkpoints evicted.",
                cache.get("checkpoints_evicted", 0),
            ),
            (
                "atom:prefix_cache_checkpoints_orphaned",
                "Number of prefix-cache checkpoints orphaned.",
                cache.get("checkpoints_orphaned", 0),
            ),
        )
        for name, documentation, value in cache_counters:
            metric = CounterMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        for name, documentation, value in (
            (
                "atom:prefix_cache_hit_ratio",
                "Admitted prefix-cache token hit ratio.",
                cache.get("hit", 0),
            ),
            (
                "atom:prefix_cache_compressed_hit_ratio",
                "Compressed-prefix token hit ratio before state gates.",
                cache.get("compressed_hit", 0),
            ),
            (
                "atom:prefix_cache_lost_to_checkpoint_ratio",
                "Reusable-token ratio lost because a checkpoint was unavailable.",
                cache.get("lost_to_checkpoint", 0),
            ),
            (
                "atom:prefix_cache_lost_unrecoverable_ratio",
                "Reusable-token ratio not recoverable by checkpointing.",
                cache.get("lost_unrecoverable", 0),
            ),
        ):
            metric = GaugeMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        offload = snapshot.get("offload", {})
        for name, documentation, value in (
            (
                "atom:lmcache_load_requests",
                "Number of completed LMCache load operations.",
                offload.get("load_requests", 0),
            ),
            (
                "atom:lmcache_loaded_tokens",
                "Number of tokens loaded from LMCache.",
                offload.get("loaded_tokens", 0),
            ),
            (
                "atom:lmcache_load_failures",
                "Number of failed LMCache load operations.",
                offload.get("load_failures", 0),
            ),
            (
                "atom:lmcache_save_requests",
                "Number of completed LMCache save operations.",
                offload.get("save_requests", 0),
            ),
            (
                "atom:lmcache_saved_tokens",
                "Number of tokens saved to LMCache.",
                offload.get("saved_tokens", 0),
            ),
        ):
            metric = CounterMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        for name, documentation, value in (
            (
                "atom:lmcache_loads_pending",
                "Number of LMCache loads currently in flight.",
                offload.get("loads_pending", 0),
            ),
            (
                "atom:lmcache_saves_pending",
                "Number of LMCache saves currently in flight.",
                offload.get("saves_pending", 0),
            ),
        ):
            metric = GaugeMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        mtp = snapshot.get("mtp", {})
        for name, documentation, value in (
            (
                "atom:mtp_draft_tokens",
                "Number of speculative draft tokens considered.",
                mtp.get("total_draft_tokens", 0),
            ),
            (
                "atom:mtp_accepted_tokens",
                "Number of accepted speculative bonus tokens.",
                mtp.get("total_accepted_tokens", 0),
            ),
        ):
            metric = CounterMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        for name, documentation, value in (
            (
                "atom:mtp_acceptance_rate",
                "Fraction of speculative draft tokens accepted.",
                mtp.get("acceptance_rate", 0),
            ),
            (
                "atom:mtp_average_tokens_per_forward",
                "Average emitted tokens per speculative decode forward.",
                mtp.get("average_tokens_per_forward", 0),
            ),
        ):
            metric = GaugeMetricFamily(name, documentation)
            metric.add_metric([], float(value))
            yield metric

        distribution = CounterMetricFamily(
            "atom:mtp_decode_steps",
            "Number of speculative decode steps by accepted bonus-token count.",
            labels=["accepted_tokens"],
        )
        for accepted, steps in sorted(mtp.get("distribution", {}).items()):
            distribution.add_metric([str(accepted)], float(steps))
        yield distribution

        yield from _gc_metrics()


def _gc_metrics() -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
    """This process's own collector -- the frontend's, not the engine's, since
    each interpreter keeps its own counters.

    `atom:gc_collected_total` is the one to watch, and why the rest are here:
    it is what decides whether spacing this process's collections out with
    `ATOM_GC_THRESHOLD` would cost nothing or would defer real work. Flat after
    startup means the collector is finding nothing; a rising line means the
    process builds reference cycles and raising thresholds has a price.

    O(1) per source is a bound, not a preference: rendering runs inline on the
    loop that delivers every stream, so a scrape pauses all of them. It cost a
    metric. `atom:gc_frozen_objects` came from `gc.get_freeze_count()`, which
    walks the permanent generation -- 11.9 ms at 430k frozen against 0.5 us for
    `gc.get_stats()`, i.e. the cost freezing exists to remove -- for a number
    that changes twice in a process's life. It and the tracked-set size both
    live in `/debug/gc_census` now, which is asked for rather than scraped.
    """
    stats = gc.get_stats()
    for name, key, doc in (
        (
            "atom:gc_collections",
            "collections",
            "Collections run by this process's collector, per generation.",
        ),
        (
            "atom:gc_collected",
            "collected",
            (
                "Objects reclaimed by this process's collector, per generation. "
                "Expected flat after startup; growth means raising this "
                "process's ATOM_GC_THRESHOLD would defer real work."
            ),
        ),
        (
            "atom:gc_uncollectable",
            "uncollectable",
            "Objects found unreclaimable by this process's collector.",
        ),
    ):
        metric = CounterMetricFamily(name, doc, labels=["generation"])
        for generation, per_gen in enumerate(stats):
            metric.add_metric([str(generation)], float(per_gen.get(key, 0)))
        yield metric

    threshold = GaugeMetricFamily(
        "atom:gc_threshold",
        "Collection threshold in effect in this process, per generation.",
        labels=["generation"],
    )
    for generation, value in enumerate(gc.get_threshold()):
        threshold.add_metric([str(generation)], float(value))
    yield threshold


class AtomMetricsExporter:
    """Own a cached runtime snapshot and render it without engine RPCs."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: dict[str, Any] = {}
        self._refresh_errors = 0
        self._last_refresh = 0.0
        self._registry = CollectorRegistry(auto_describe=False)
        self._registry.register(_AtomMetricsCollector(self))
        self._inter_token_latency = _WeightedHistogram(
            "atom:inter_token_latency_seconds",
            "Frontend-observed streaming output interval divided by new token "
            "count, weighted by that count. Excludes the first output batch.",
            buckets=(
                0.002,
                0.004,
                0.006,
                0.008,
                0.010,
                0.015,
                0.020,
                0.025,
                0.030,
                0.035,
                0.040,
                0.060,
                0.080,
                0.100,
                0.200,
                0.400,
                0.600,
                0.800,
                1.000,
                2.000,
                4.000,
                6.000,
                8.000,
            ),
            registry=self._registry,
        )
        self._time_to_first_token = Histogram(
            "atom:time_to_first_token_seconds",
            "Local API request arrival to first output. Streaming observes the "
            "first generated SSE payload; non-streaming observes the first "
            "internal token delivery. One sample per request.",
            labelnames=("streaming",),
            buckets=(
                0.001,
                0.005,
                0.010,
                0.025,
                0.050,
                0.100,
                0.250,
                0.500,
                1.0,
                2.5,
                5.0,
                10.0,
                15.0,
                30.0,
                45.0,
                60.0,
                90.0,
                120.0,
                180.0,
                240.0,
            ),
            registry=self._registry,
        )

        # Expose zero-valued children before traffic so Prometheus can establish
        # a baseline for rate(). Registering labels does not record a sample.
        for streaming in ("true", "false"):
            self._time_to_first_token.labels(streaming=streaming)

    def observe_time_to_first_token(self, interval: float, streaming: bool) -> None:
        self._time_to_first_token.labels(streaming=str(streaming).lower()).observe(
            interval
        )

    def observe_inter_token_latency(self, interval: float, num_new_tokens: int) -> None:
        """Record token-weighted intervals in one aggregation update.

        These cumulative observations are independent of the engine snapshot;
        neither refreshing that snapshot nor scraping resets the histogram.
        """
        self._inter_token_latency.observe_weighted(interval, num_new_tokens)

    def update(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = copy.deepcopy(snapshot)
            self._last_refresh = time.time()

    def record_refresh_error(self) -> None:
        with self._lock:
            self._refresh_errors += 1

    def read(self) -> tuple[dict[str, Any], int, float]:
        with self._lock:
            return (
                copy.deepcopy(self._snapshot),
                self._refresh_errors,
                self._last_refresh,
            )

    def render(self) -> bytes:
        return generate_latest(self._registry)
