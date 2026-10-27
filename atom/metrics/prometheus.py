"""Node-local Prometheus storage, initialized before importing its client."""

import atexit
import os
import tempfile


def initialize_metrics():
    # Spawned children inherit the directory; only its creator removes it.
    # An explicitly supplied directory is owned and cleaned by the launcher.
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        directory = tempfile.TemporaryDirectory(prefix="atom-metrics-")
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = directory.name
        atexit.register(directory.cleanup)


def start_metrics_server(host, port):
    """Expose this node's native instruments, without coordinator snapshots."""
    from prometheus_client import CollectorRegistry, multiprocess, start_http_server

    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
    return start_http_server(port, addr=host, registry=registry)
