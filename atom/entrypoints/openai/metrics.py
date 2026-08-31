"""Prometheus exposition for the standalone ATOM OpenAI server."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterable
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from .streaming_dispatch import longest_silence_seconds


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
