"""Compose component-owned metrics for one API process."""

import os

from prometheus_client import multiprocess, values

from atom.metrics.exporter import AtomMetricsExporter
from atom.metrics.request import RequestMetrics, StreamMetrics

from .streaming_dispatch import longest_silence_seconds


def create_metrics_exporter() -> (
    tuple[AtomMetricsExporter, RequestMetrics, StreamMetrics]
):
    exporter = AtomMetricsExporter(stream_silence=longest_silence_seconds)
    registry = exporter.registry
    if os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        if not values.ValueClass._multiprocess:
            raise RuntimeError(
                "Set PROMETHEUS_MULTIPROC_DIR before importing prometheus_client"
            )
        multiprocess.MultiProcessCollector(registry)
        # Register only the multiprocess collector, not its instruments too.
        registry = None
    request_metrics = RequestMetrics(registry)
    stream_metrics = StreamMetrics(registry)
    return exporter, request_metrics, stream_metrics
