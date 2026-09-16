"""Snapshot cache, metric collection and Prometheus rendering.

Legacy state metrics use engine snapshots. Event observations are exported by
the native Prometheus client, using multiprocess storage in server launches.
"""

from __future__ import annotations

import asyncio
import copy
import gc
import threading
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.exposition import CONTENT_TYPE_LATEST

Snapshot = dict[str, Any]
SnapshotState = tuple[Snapshot, int, float]


class _AtomMetricsCollector:
    def __init__(self, exporter: AtomMetricsExporter):
        self._exporter = exporter

    def describe(self):
        # Reserve all original families, including data-dependent label sets,
        # without reading the runtime snapshot or live process state.
        return self._collect(None)

    def collect(self) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
        return self._collect(self._exporter.read())

    def _collect(
        self, state: SnapshotState | None
    ) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
        describe = state is None
        snapshot, refresh_errors, last_refresh = ({}, 0, 0.0) if describe else state
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
        metric.add_metric([], 0.0 if describe else self._exporter.stream_silence())
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

        ranks = snapshot.get("scheduler_metrics", [])
        for name, help_text, key, states in (
            (
                "atom:scheduler_requests",
                "Requests by scheduler state; waiting excludes KV waits.",
                None,
                ("running", "waiting", "waiting_kv"),
            ),
            (
                "atom:scheduler_kv_cache_blocks",
                "KV block pool by state; used + evictable + vacant = total.",
                "kv_blocks",
                ("used", "evictable", "vacant", "total"),
            ),
        ):
            metric = GaugeMetricFamily(
                name, help_text, labels=["dp_rank", "engine_role", "state"]
            )
            for rank in ranks:
                values = rank if key is None else rank.get(key, {})
                for queue_state in states:
                    if queue_state in values:
                        metric.add_metric(
                            [str(rank["dp_rank"]), rank["engine_role"], queue_state],
                            values[queue_state],
                        )
            yield metric

        cache = snapshot.get("cache", {})
        if describe or "offload_tokens" in cache:
            metric = CounterMetricFamily(
                "atom:prefix_cache_offload_tokens",
                "Prompt tokens reused from LMCache beyond the admitted GPU prefix; shares prefix-cache input accounting, not transfer volume.",
            )
            metric.add_metric([], float(cache.get("offload_tokens", 0)))
            yield metric

        yield from _gc_metrics(describe=describe)


def _gc_metrics(
    *, describe: bool = False
) -> Iterable[GaugeMetricFamily | CounterMetricFamily]:
    """This process's own collector -- the frontend's, not the engine's, since
    each interpreter keeps its own counters.

    `atom:gc_collected_total` is the one to watch, and why the rest are here:
    it is what decides whether spacing this process's collections out with
    `ATOM_GC_THRESHOLD` would cost nothing or would defer real work. Flat after
    startup means the collector is finding nothing; a rising line means the
    process builds reference cycles and raising thresholds has a price.

    Keep each source O(1): rendering runs in a thread but still competes with
    SSE delivery for CPU and the GIL. `atom:gc_frozen_objects` came from
    `gc.get_freeze_count()`, which walks the permanent generation -- 11.9 ms
    at 430k frozen against 0.5 us for
    `gc.get_stats()`, i.e. the cost freezing exists to remove -- for a number
    that changes twice in a process's life. It and the tracked-set size both
    live in `/debug/gc_census` now, which is asked for rather than scraped.
    """
    stats = [] if describe else gc.get_stats()
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
    for generation, value in enumerate(() if describe else gc.get_threshold()):
        threshold.add_metric([str(generation)], float(value))
    yield threshold


class AtomMetricsExporter:
    """Own a cached runtime snapshot and render it without engine RPCs."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self, *, stream_silence: Callable[[], float] | None = None):
        self._stream_silence = stream_silence
        self._lock = threading.Lock()
        self._snapshot: Snapshot = {}
        self._refresh_errors = 0
        self._last_refresh = 0.0
        self._render_silence: ContextVar[float | None] = ContextVar(
            "metrics_render_silence", default=None
        )
        # Owned by the API event loop. Share only in-flight work, never a
        # completed response: the next scrape must observe fresh API metrics.
        self._render_task: asyncio.Task[bytes] | None = None
        self.registry = CollectorRegistry(auto_describe=False)
        self.registry.register(_AtomMetricsCollector(self))

    def update(self, snapshot: dict[str, Any]) -> None:
        snapshot = copy.deepcopy(snapshot)
        with self._lock:
            # Published snapshots are replaced, never mutated or exposed.
            self._snapshot = snapshot
            self._last_refresh = time.time()

    def record_refresh_error(self) -> None:
        with self._lock:
            self._refresh_errors += 1

    def read(self) -> tuple[dict[str, Any], int, float]:
        with self._lock:
            snapshot = self._snapshot
            refresh_errors = self._refresh_errors
            last_refresh = self._last_refresh
        # A renderer must not hold up the API loop's next update while copying
        # the engine state. The captured snapshot is immutable here.
        return copy.deepcopy(snapshot), refresh_errors, last_refresh

    def stream_silence(self) -> float:
        captured = self._render_silence.get()
        if captured is not None:
            return captured
        return self._stream_silence() if self._stream_silence is not None else 0.0

    def render(self, *, stream_silence: float | None = None) -> bytes:
        silence_token = self._render_silence.set(stream_silence)
        try:
            return generate_latest(self.registry)
        finally:
            self._render_silence.reset(silence_token)

    async def render_async(self) -> bytes:
        """Render on demand off the API loop, coalescing concurrent scrapes."""
        task = self._render_task
        if task is None or task.done():
            # Read the provider on its owning event loop before handing
            # serialization to a thread; registry instruments have their own
            # synchronization and are collected in that thread.
            silence = self.stream_silence()
            task = asyncio.create_task(
                asyncio.to_thread(self.render, stream_silence=silence),
                name="metrics_render",
            )
            self._render_task = task
            task.add_done_callback(self._render_finished)
        # Cancelling an HTTP waiter cannot stop the thread. Keep tracking its
        # task so another scrape shares it instead of starting more work.
        return await asyncio.shield(task)

    def _render_finished(self, task: asyncio.Task[bytes]) -> None:
        if self._render_task is task:
            self._render_task = None
        if not task.cancelled():
            task.exception()  # Retrieve failures even if every client left.

    async def wait_for_render(self) -> None:
        """Drain in-flight rendering during API shutdown without blocking it."""
        if self._render_task is not None:
            await asyncio.gather(
                asyncio.shield(self._render_task), return_exceptions=True
            )
