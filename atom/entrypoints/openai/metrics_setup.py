"""Compose component-owned metrics for one API process."""

from atom.kv_transfer.offload.metrics import collect_offload_metrics
from atom.model_engine.dp_metrics import collect_dp_metrics
from atom.model_engine.engine_stats import collect_engine_metrics
from atom.model_engine.gpu_metrics import collect_gpu_metrics
from atom.model_engine.scheduler_metrics import collect_scheduler_metrics
from atom.utils.gc_utils import GCMetricsCollector

from .metrics import AtomMetricsExporter
from .request_timing import RequestMetrics
from .streaming_dispatch import StreamMetrics


def create_metrics_exporter() -> (
    tuple[AtomMetricsExporter, RequestMetrics, StreamMetrics]
):
    exporter = AtomMetricsExporter()
    for collect in (
        collect_scheduler_metrics,
        collect_gpu_metrics,
        collect_engine_metrics,
        collect_dp_metrics,
        collect_offload_metrics,
    ):
        exporter.register_snapshot_collector(collect)
    exporter.registry.register(GCMetricsCollector())
    request_metrics = RequestMetrics(exporter.registry)
    stream_metrics = StreamMetrics(exporter.registry)
    return exporter, request_metrics, stream_metrics
