"""Rendering metrics must no longer perform process-tree accounting."""

import psutil

from atom.entrypoints.openai.metrics_setup import create_metrics_exporter


def test_metrics_render_without_reading_process_state(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("metrics must not inspect processes")

    monkeypatch.setattr(psutil, "Process", forbidden)
    exporter, *_ = create_metrics_exporter()
    text = exporter.render().decode()
    assert "atom:time_to_first_token_seconds" in text
    assert "atom:process_cpu_seconds" not in text
    assert "atom:process_threads" not in text
    assert "atom:process_cpus" not in text
