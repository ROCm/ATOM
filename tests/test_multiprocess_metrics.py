"""Native metrics across spawned workers and independently scraped nodes."""

import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

import pytest


def _observe(rank):
    from test_gpu_metrics import Event, complete_event, prefill_batch

    from atom.metrics.gpu import GPUForwardMetrics
    from atom.metrics.scheduler import SchedulerMetrics

    scheduler = SchedulerMetrics(rank, "prefill")
    for _ in range(5):
        scheduler.queue_time.observe(0.05 * (rank + 1))
    with patch("atom.metrics.scheduler.time.perf_counter", return_value=0.0):
        for phase in ("prefill", "decode"):
            context_metrics = (
                scheduler if phase == "prefill" else SchedulerMetrics(rank, phase)
            )
            seq = SimpleNamespace(
                external_request_id=f"request-{rank}", num_prompt_tokens=1000 + rank
            )
            context_metrics.enqueue(seq)
            batch = SimpleNamespace(
                req_ids=[1],
                is_dummy_run=False,
                total_seqs_num_decode=int(phase == "decode"),
                total_seqs_num_prefill=int(phase == "prefill"),
                total_tokens_num_prefill=100,
                num_cached_tokens=[100],
                context_lens=[200 if phase == "prefill" else 1000 + rank],
            )
            context_metrics.record_forward(batch, {1: seq})
            batch.context_lens = [400 if phase == "prefill" else 2000 + rank]
            context_metrics.record_forward(batch, {1: seq})
    for pp_rank in (0, 1):
        gpu = GPUForwardMetrics(
            Event, dp_rank=rank, tp_rank=rank, pp_rank=pp_rank, engine_role="prefill"
        )
        with gpu.measure(prefill_batch((rank, 1, True))):
            pass
        complete_event(gpu, milliseconds=8 * (rank + 1))
        gpu.poll()  # The last forward must be published without another forward.


def _run_node():
    # This runs in a fresh interpreter, as a server launch does.
    from atom.metrics.prometheus import initialize_metrics, start_metrics_server

    initialize_metrics()
    from prometheus_client import values
    from prometheus_client.parser import text_string_to_metric_families

    from atom.entrypoints.openai import api_server

    assert values.ValueClass._multiprocess
    exporter = api_server._metrics_exporter
    api_server._request_metrics.observe_time_to_first_token(0.5, True)
    api_server._stream_metrics.observe_inter_token_latency(0.020, 4)
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_observe, args=(rank,)) for rank in (0, 1)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
        assert process.exitcode == 0

    def samples(text):
        result = {}
        for family in text_string_to_metric_families(text):
            if family.name.endswith("_request_context_tokens"):
                assert family.type == "gauge"
            for sample in family.samples:
                if sample.name.startswith("atom:gc_"):
                    continue
                key = (sample.name, tuple(sorted(sample.labels.items())))
                assert key not in result, key  # No direct + multiprocess duplicates.
                result[key] = sample.value
        return result

    first = samples(exporter.render().decode())
    assert samples(exporter.render().decode()) == first
    assert not any(name.endswith("_created") for name, _ in first)
    assert (
        first[("atom:time_to_first_token_seconds_count", (("streaming", "true"),))] == 1
    )
    assert first[("atom:inter_token_latency_seconds_count", ())] == 4
    for rank in (0, 1):
        labels = (("dp_rank", str(rank)), ("engine_role", "prefill"))
        assert first[("atom:request_queue_time_seconds_count", labels)] == 6
        assert first[("atom:request_queue_time_seconds_sum", labels)] == 0.25 * (
            rank + 1
        )
        assert (
            first[("atom:request_queue_time_seconds_bucket", labels + (("le", "0.1"),))]
            == 6
        )
        for pp_rank in (0, 1):
            gpu_labels = (*labels, ("pp_rank", str(pp_rank)), ("tp_rank", str(rank)))
            for name in (
                "atom:gpu_forward_seconds",
                "atom:prefill_request_gpu_forward_seconds",
            ):
                assert first[(name + "_count", gpu_labels)] == 1
                assert first[(name + "_sum", gpu_labels)] == pytest.approx(
                    0.008 * (rank + 1)
                )

    server, thread = start_metrics_server("127.0.0.1", 0)
    try:
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/metrics", timeout=5
        ) as response:
            remote = samples(response.read().decode())
        assert "atom:metrics_snapshot_available" not in {name for name, _ in remote}
        assert remote == {key: first[key] for key in remote}
        # Requests and workers are gone; repeated node scrapes retain their
        # first-dispatch context, including distinct prefill/decode semantics.
        for phase in ("prefill", "decode"):
            name = f"atom:{phase}_request_context_tokens"
            contexts = [
                (dict(labels), value)
                for (metric, labels), value in remote.items()
                if metric == name
            ]
            assert len(contexts) == 2
            assert {
                (labels["dp_rank"], labels["request_id"], value)
                for labels, value in contexts
            } == {("0", "request-0", 1000), ("1", "request-1", 1001)}
            assert all(
                labels["engine_role"] == phase
                and labels["sequence_id"] == "1"
                and float(labels["started_at"]) > 0
                for labels, _ in contexts
            )
        with urlopen(
            f"http://127.0.0.1:{server.server_port}/metrics", timeout=5
        ) as response:
            assert samples(response.read().decode()) == remote
        assert (
            "atom:request_queue_time_seconds_count",
            (("dp_rank", "1"), ("engine_role", "prefill")),
        ) in remote
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
    print(
        json.dumps(
            {"directory": os.environ["PROMETHEUS_MULTIPROC_DIR"], "series": len(first)}
        )
    )


def test_spawned_metrics_are_exported_without_snapshots_and_restart_cleanly():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    results = []
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, __file__],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        node = json.loads(result.stdout.splitlines()[-1])
        assert not Path(node["directory"]).exists()
        results.append(node)
    assert results[0]["directory"] != results[1]["directory"]
    assert results[0]["series"] == results[1]["series"]


@pytest.mark.parametrize(
    "module", ["atom.entrypoints.openai_server", "atom.entrypoints.openai.api_server"]
)
def test_server_entrypoints_initialize_native_storage_before_import(module):
    env = dict(os.environ)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    script = """
import argparse, json, os, runpy, sys
# Stop at argument parsing: startup imports must already select native storage.
argparse.ArgumentParser.parse_args = lambda self: sys.exit(0)
sys.argv = [sys.argv[1]]
try:
    runpy.run_module(sys.argv[0], run_name="__main__")
except SystemExit as exc:
    assert exc.code == 0
from prometheus_client import values
assert values.ValueClass._multiprocess
print(json.dumps(os.environ["PROMETHEUS_MULTIPROC_DIR"]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, module],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not Path(json.loads(result.stdout.splitlines()[-1])).exists()


if __name__ == "__main__":
    _run_node()
