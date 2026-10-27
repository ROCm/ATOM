# SPDX-License-Identifier: MIT
"""Exercise real client I/O and shell exit traps, without a GPU or model."""

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from atom.benchmarks.results.io import iter_jsonl, read_json
from atom.benchmarks.results.records import RequestRecorder

ROOT = Path(__file__).resolve().parents[3]


def test_real_stream_timing_and_usage(tmp_path):
    pytest.importorskip("aiohttp")
    pytest.importorskip("transformers")
    from atom.benchmarks.backend_request_func import (
        RequestFuncInput,
        async_request_openai_completions,
    )

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for data, delay in [
                ({"choices": [{"text": "one"}]}, 0.01),
                ({"choices": [{"text": " two"}]}, 0.01),
                ({"choices": [{"text": ""}], "usage": {"completion_tokens": 5}}, 0.04),
            ]:
                time.sleep(delay)
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()
            # Usage may precede EOF; full-response duration must include the tail.
            time.sleep(0.03)
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def run():
        recorder = RequestRecorder(tmp_path / "requests.jsonl")
        request = RequestFuncInput(
            prompt="fixture",
            prompt_len=2,
            output_len=5,
            model="fixture",
            api_url=f"http://127.0.0.1:{server.server_port}/v1/completions",
        )
        result = await recorder.call(async_request_openai_completions, request)
        recorder.close()
        assert result.success
        assert result.output_tokens == 5
        assert result.full_response_duration_s > result.latency

    try:
        asyncio.run(run())
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    row = next(iter_jsonl(tmp_path / "requests.jsonl"))
    assert row["output_tokens"] == 5
    assert row["e2el_s"] > row["harness_e2el_s"]
    assert int(row["end_ns"]) > int(row["start_ns"])


def lifecycle_env(tmp_path):
    env = os.environ.copy()
    env.update(
        MODEL_PATH="fixture",
        CONC="1",
        ATOM_SERVER_PORT="1",
        RESULT_FILENAME=str(tmp_path / "result"),
        ATOM_CLIENT_LOG=str(tmp_path / "client.log"),
        ATOM_BUNDLE_WORK=str(tmp_path / "evidence"),
        ATOM_BUNDLE_OUTPUT=str(tmp_path / "bundle"),
        BENCH_KIND="random",
    )
    return env


def test_aiperf_failure_after_export_is_not_reported_as_success(tmp_path):
    executable = tmp_path / "bin/aiperf"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\nprintf '{}' > \"$TEST_OUT/profile_export_aiperf.json\"\n"
        "exit 17\n"
    )
    executable.chmod(0o755)
    output = tmp_path / "output"
    env = os.environ.copy()
    env.pop("ATOM_BUNDLE_WORK", None)
    env.update(
        AIPERF_VENV=str(tmp_path),
        AIPERF_BENCHMARK_DURATION="900",
        MODEL_PATH="fixture",
        TEST_OUT=str(output),
    )
    result = subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            (
                "source .github/scripts/aiperf_agentic.sh; "
                'run_aiperf_agentic http://localhost 1 "$TEST_OUT"'
            ),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert (output / "profile_export_aiperf.json").is_file()
    assert result.returncode == 17, result.stderr


@pytest.mark.parametrize("code", [17, 143])
def test_exit_trap_preserves_failure(tmp_path, code):
    command = f"""set -euo pipefail
source .github/scripts/benchmark_bundle.sh
BENCH_CMD=(false)
start_benchmark_bundle
exit {code}
"""
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        env=lifecycle_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == code, result.stderr
    validation = read_json(tmp_path / "bundle/validation.json")
    assert validation["exit_code"] == code
    assert not validation["measurement_valid"]


def test_signal_stops_client_then_packages(tmp_path):
    command = """set -euo pipefail
source .github/scripts/benchmark_bundle.sh
BENCH_CMD=(false)
start_benchmark_bundle
set -m
sleep 60 &
CLIENT_PID=$!
set +m
printf '%s' "$CLIENT_PID" > "$ATOM_BUNDLE_WORK/client.pid"
wait "$CLIENT_PID"
"""
    process = subprocess.Popen(
        ["bash", "-c", command],
        cwd=ROOT,
        env=lifecycle_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    marker = tmp_path / "evidence/client.pid"
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert marker.exists()
        pid = int(marker.read_text())
        process.send_signal(signal.SIGTERM)
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, stderr
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert read_json(tmp_path / "bundle/validation.json")["exit_code"] == 143
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("code", [0, 17])
def test_client_exit_stops_scraping_before_drain(tmp_path, code):
    # Run the actual client subprocess block without launching a model. An
    # idle collector represents /metrics traffic that prevents a quiet log.
    script = (ROOT / ".github/scripts/atom_test.sh").read_text()
    client = script.rsplit("  set -m\n  (\n", 1)[1].split("  ) &", 1)[0]
    command = (
        """sleep 60 &
ATOM_BUNDLE_COLLECTOR_PID=$!
trap 'kill "$ATOM_BUNDLE_COLLECTOR_PID" 2>/dev/null || true' EXIT
BENCH_KIND=random
BENCH_CMD=(bash -c 'exit "$1"' bash "$1")
ATOM_CLIENT_LOG="$2"
(
"""
        + client
        + """
)
client_rc=$?
wait "$ATOM_BUNDLE_COLLECTOR_PID"
exit "$client_rc"
"""
    )
    result = subprocess.run(
        [
            "bash",
            "-o",
            "pipefail",
            "-c",
            command,
            "bash",
            str(code),
            str(tmp_path / "client.log"),
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == code, result.stderr
