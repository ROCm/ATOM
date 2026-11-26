"""Completed requests must not grow default scheduler metrics or mmap storage."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _probe():
    # Import only after the parent configured native multiprocess storage.
    from prometheus_client import CollectorRegistry, generate_latest, multiprocess
    from prometheus_client.parser import text_string_to_metric_families

    from atom.metrics.scheduler import SchedulerMetrics

    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    registry = CollectorRegistry()
    if directory:
        multiprocess.MultiProcessCollector(registry)
    # Ordinary mode has one registry per scheduler-owning process. Native
    # storage merges all rank/phase instruments through its collector.
    registries = [CollectorRegistry() for _ in range(4)]
    schedulers = [
        SchedulerMetrics(
            rank,
            phase,
            registry=None if directory else registries[i],
            **(
                {"request_context_enabled": False}
                if "--explicit-disabled" in sys.argv
                else {}
            ),
        )
        for i, (rank, phase) in enumerate(
            (rank, phase) for rank in range(2) for phase in ("prefill", "decode")
        )
    ]

    def observe(start, stop):
        for request_id in range(start, stop):
            for scheduler in schedulers:
                is_decode = scheduler._labels["engine_role"] == "decode"
                seq = SimpleNamespace(
                    id=request_id,
                    external_request_id=f"request-{request_id}",
                    num_prompt_tokens=1000,
                )
                scheduler.enqueue(seq)
                batch = SimpleNamespace(
                    req_ids=[request_id],
                    is_dummy_run=False,
                    total_seqs_num_decode=int(is_decode),
                    total_seqs_num_prefill=int(not is_decode),
                    total_tokens_num_prefill=100,
                    num_cached_tokens=[900],
                    context_lens=[1000],
                )
                # Repeated forwards must keep the existing once-per-request
                # observations while batch histograms continue to advance.
                scheduler.record_forward(batch, {request_id: seq})
                scheduler.record_forward(batch, {request_id: seq})

    def snapshot():
        payload = (
            generate_latest(registry)
            if directory
            else b"".join(generate_latest(r) for r in registries)
        )
        samples = {
            (sample.name, tuple(sorted(sample.labels.items()))): sample.value
            for family in text_string_to_metric_families(payload.decode())
            for sample in family.samples
        }
        assert not any("request_context_tokens" in name for name, _ in samples)
        assert not any(
            key in {"request_id", "sequence_id", "started_at"}
            for _, labels in samples
            for key, _ in labels
        )
        storage = (
            {p.name: p.stat().st_size for p in Path(directory).glob("*.db")}
            if directory
            else {}
        )
        return samples, len(payload), storage

    observe(0, 100)
    first, first_bytes, first_storage = snapshot()
    observe(100, 10000)
    last, last_bytes, last_storage = snapshot()
    assert first.keys() == last.keys()
    assert first_storage == last_storage
    if directory:
        assert first_storage  # Exercise real storage, not an empty directory.
    for scheduler in schedulers:
        labels = tuple(sorted(scheduler._labels.items()))
        phase = scheduler._labels["engine_role"]
        assert last[("atom:request_queue_time_seconds_count", labels)] == 10000
        assert last[(f"atom:{phase}_context_tokens_count", labels)] == 20000
        assert last[(f"atom:{phase}_context_tokens_sum", labels)] == 20000000
        if phase == "prefill":
            assert last[("atom:prefill_request_tokens_count", labels)] == 10000
            assert last[("atom:prefill_request_tokens_sum", labels)] == 1000000
        else:
            assert last[("atom:decode_batch_size_count", labels)] == 20000
    print(
        json.dumps(
            {
                "series_before": len(first),
                "series_after": len(last),
                "bytes_before": first_bytes,
                "bytes_after": last_bytes,
                "storage_before": first_storage,
                "storage_after": last_storage,
            }
        )
    )


@pytest.mark.parametrize("native_storage", [False, True])
@pytest.mark.parametrize("explicit_disabled", [False, True])
def test_completed_request_cardinality_is_bounded(
    tmp_path, native_storage, explicit_disabled
):
    env = dict(os.environ)
    env.pop("PROMETHEUS_MULTIPROC_DIR", None)
    env.pop("prometheus_multiproc_dir", None)
    if native_storage:
        env["PROMETHEUS_MULTIPROC_DIR"] = str(tmp_path)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join([str(root), env.get("PYTHONPATH", "")])
    result = subprocess.run(
        [sys.executable, __file__]
        + (["--explicit-disabled"] if explicit_disabled else []),
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["series_after"] == report["series_before"]


if __name__ == "__main__":
    _probe()
