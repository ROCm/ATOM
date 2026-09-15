"""Compose component-owned metrics for one API process."""

import os

from prometheus_client import multiprocess, values

from .metrics import AtomMetricsExporter
from .request_timing import RequestMetrics
from .streaming_dispatch import StreamMetrics


def create_metrics_exporter() -> (
    tuple[AtomMetricsExporter, RequestMetrics, StreamMetrics]
):
    exporter = AtomMetricsExporter()
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
