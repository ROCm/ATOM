"""Snapshot cache and explicit Prometheus registration for the OpenAI server.

Business metric definitions live with their owners. The API composition module
registers local instruments and snapshot collectors into this exporter.
"""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Callable, Iterable
from contextvars import ContextVar
from typing import Any

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.exposition import CONTENT_TYPE_LATEST

Snapshot = dict[str, Any]
SnapshotState = tuple[Snapshot, int, float]


class _SnapshotCollector:
    def __init__(
        self, exporter, collect: Callable[[Snapshot | None], Iterable[Metric]]
    ):
        self._exporter = exporter
        self._collect = collect

    def describe(self):
        # None requests all family names, including optional snapshot fields.
        # Registration must never depend on available runtime data or do RPCs.
        return self._collect(None)

    def collect(self):
        return self._collect(self._exporter.read()[0])


class AtomMetricsExporter:
    """Own a cached runtime snapshot and render it without engine RPCs."""

    content_type = CONTENT_TYPE_LATEST

    def __init__(self):
        self._lock = threading.Lock()
        self._snapshot: Snapshot = {}
        self._refresh_errors = 0
        self._last_refresh = 0.0
        self._render_snapshot: ContextVar[SnapshotState | None] = ContextVar(
            "metrics_render_snapshot", default=None
        )
        self.registry = CollectorRegistry(auto_describe=False)
        self.registry.register(self)

    def register_snapshot_collector(
        self, collect: Callable[[Snapshot | None], Iterable[Metric]]
    ) -> None:
        """Register a component's pure snapshot-to-families function once.

        collect(None) must describe every possible family, without runtime I/O.
        collect(snapshot) exports existing cumulative values; it must not mutate
        the snapshot or re-observe values. The registry rejects name collisions.
        """
        self.registry.register(_SnapshotCollector(self, collect))

    def describe(self):
        return self.collect()

    def collect(self):
        snapshot, refresh_errors, last_refresh = self.read()
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

    def update(self, snapshot: dict[str, Any]) -> None:
        with self._lock:
            self._snapshot = copy.deepcopy(snapshot)
            self._last_refresh = time.time()

    def record_refresh_error(self) -> None:
        with self._lock:
            self._refresh_errors += 1

    def read(self) -> tuple[dict[str, Any], int, float]:
        pinned = self._render_snapshot.get()
        if pinned is not None:
            return pinned
        with self._lock:
            return (
                copy.deepcopy(self._snapshot),
                self._refresh_errors,
                self._last_refresh,
            )

    def render(self) -> bytes:
        # All component collectors see the same revision, even if refresh runs
        # concurrently. Context-local state isolates concurrent/nested scrapes.
        token = self._render_snapshot.set(self.read())
        try:
            return generate_latest(self.registry)
        finally:
            self._render_snapshot.reset(token)
