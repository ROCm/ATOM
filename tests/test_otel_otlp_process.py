# SPDX-License-Identifier: MIT
"""A spawned engine process exports real OTLP to a loopback HTTP receiver.

This validates the transport and context contract on CPU, not GPU inference.
"""

# Pytest discovers the imported spans fixture by its original name.
# ruff: noqa: F811

import json
import multiprocessing
import os
import pickle
import subprocess
import sys
import threading
import time
from pathlib import Path

from test_otel_tracing import (
    PARENT_ID,
    TRACE_ID,
    execute,
    sequence,
    spans,  # noqa: F401
)

from atom.observability import tracing

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "scripts" / "observability")
)
from otlp_sink import receiver


def engine_process(payload):
    tracing.initialize()
    seq = pickle.loads(payload)
    tracing.enqueue(seq)
    time.sleep(0.002)
    tracing.forward(seq, is_decode=False)
    time.sleep(0.002)
    tracing.prefill_complete(seq)
    tracing.first_token(seq)
    time.sleep(0.002)
    tracing.finish(seq, reason="max_tokens", num_tokens=104)
    tracing.shutdown()


def test_spawned_worker_otlp_preserves_parent_and_dye(spans, monkeypatch, tmp_path):
    output = tmp_path / "cpu-otlp-spans.jsonl"
    server = receiver(output)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("ATOM_ENABLE_OTEL_TRACE", "1")
    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        f"http://127.0.0.1:{server.server_port}/v1/traces",
    )
    monkeypatch.setenv("OTEL_SERVICE_NAME", "atom")
    child_pid = []

    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        child = multiprocessing.get_context("spawn").Process(
            target=engine_process, args=(pickle.dumps(seq),)
        )
        child.start()
        child_pid.append(child.pid)
        child.join(20)
        if child.is_alive():
            child.terminate()
            child.join(5)
            raise AssertionError("CPU worker did not finish")
        assert child.exitcode == 0

    try:
        execute(app, flags="00")
    finally:
        server.shutdown()
        thread.join(5)
        server.server_close()
    result = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(result) == 4
    assert {row["name"] for row in result} == {
        "engine.request",
        "prefill_waiting",
        "prefill_forward",
        "decode_forward",
    }
    http = spans.get_finished_spans()[0]
    engine = next(row for row in result if row["name"] == "engine.request")
    assert engine["parent_span_id"] == f"{http.context.span_id:016x}"
    assert http.parent.span_id == int(PARENT_ID, 16)
    for row in result:
        assert row["trace_id"] == TRACE_ID
        assert "nsdye=1" in row["trace_state"]
        assert row["resource"]["service.instance.id"].endswith(":" + str(child_pid[0]))
        assert row["resource"]["service.name"] == "atom"
        assert row["start_ns"] <= row["end_ns"]
        assert row["status_code"] != 2
    assert child_pid[0] != os.getpid()


def test_disabled_startup_and_request_do_not_import_sdk():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from atom.observability import tracing
tracing.initialize()
assert tracing._tracer() is None
assert not any(name.startswith('opentelemetry') for name in sys.modules)
tracing.shutdown()
""",
        ],
        env={**os.environ, "ATOM_ENABLE_OTEL_TRACE": "0"},
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
