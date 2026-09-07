from types import SimpleNamespace

import pytest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter
from atom.model_engine.scheduling_metrics import SchedulingMetrics


def test_batch_metrics_use_actual_cpu_lengths_and_do_not_multiply_dcp(monkeypatch):
    stats = SchedulingMetrics()
    seqs = {1: SimpleNamespace(), 2: SimpleNamespace(), 3: SimpleNamespace()}
    monkeypatch.setattr(
        "atom.model_engine.scheduling_metrics.time.monotonic", lambda: 10
    )
    for seq in seqs.values():
        stats.enqueue(seq)
    monkeypatch.setattr(
        "atom.model_engine.scheduling_metrics.time.monotonic", lambda: 12
    )
    # One chunked prefill (cached 1000 + query 128) and two speculative decodes.
    batch = SimpleNamespace(
        req_ids=[1, 2, 3],
        total_seqs_num_prefill=1,
        num_scheduled_tokens=[128, 4, 4],
        context_lens=[1128, 4096, 8192],
    )
    stats.execute(batch, seqs)
    snapshot = stats.snapshot()
    assert snapshot["last"]["prefill"] == {
        "batch_size": 1,
        "query_tokens": 128,
        "context_tokens_max": 1128,
        "context_tokens_min": 1128,
        "context_tokens_mean": 1128,
        "query_tokens_min": 128,
        "query_tokens_max": 128,
        "query_tokens_mean": 128,
    }
    assert snapshot["last"]["decode"]["batch_size"] == 2
    assert snapshot["stages"]["decode"]["query_tokens"]["sum"] == 8
    assert snapshot["queue"]["sum"] == 6
    stats.execute(batch, seqs)
    assert stats.snapshot()["queue"]["sum"] == 6  # first execution only
    batch.is_dummy_run = True
    stats.execute(batch, seqs)
    assert stats.snapshot()["stages"]["decode"]["batch_size"]["sum"] == 4


def test_lightweight_endpoint_uses_latest_snapshot_without_runtime_refresh():
    exporter = AtomMetricsExporter()
    stats = SchedulingMetrics()
    stats.duration.observe(0.003)
    snapshot = stats.snapshot()
    snapshot["snapshot_timestamp_seconds"] = 123
    exporter.scheduling_provider = lambda: {"0": snapshot}
    body = exporter.render_scheduling().decode()
    assert "atom:scheduler_duration_seconds" in body
    assert "atom:kv_cache_usage_ratio" not in body
    assert "atom:ttft_seconds" not in body
    samples = [s for f in text_string_to_metric_families(body) for s in f.samples]
    assert next(
        s.value for s in samples if s.name == "atom:scheduler_duration_seconds_sum"
    ) == pytest.approx(0.003)
    assert exporter.render_scheduling().decode() == body
    assert "atom:scheduler_duration_seconds" in exporter.render().decode()
