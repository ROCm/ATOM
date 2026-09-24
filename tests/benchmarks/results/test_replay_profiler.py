# SPDX-License-Identifier: MIT
"""Exercise real subprocess/HTTP cleanup without loading an engine or GPU."""

import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "client_failure",
        "start_failure",
        "stop_failure",
        "cancel",
        "no_phase",
        "shell_cancel",
    ],
)
def test_replay_profiler_window_and_cleanup(tmp_path, mode):
    calls = []
    started = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append((self.path, time.monotonic()))
            if mode == "shell_cancel" and self.path == "/stop_profile":
                time.sleep(0.6)
            code = (
                500
                if (mode == "start_failure" and self.path == "/start_profile")
                or (mode == "stop_failure" and self.path == "/stop_profile")
                else 200
            )
            self.send_response(code)
            self.end_headers()
            self.wfile.write(b"{}")
            if self.path == "/start_profile":
                started.set()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    replay = tmp_path / "replay.py"
    replay.write_text(
        "import time, sys\n"
        "print('AIPerf System is PROFILING', flush=True)\n"
        "print('Phase warmup (warmup) started', flush=True)\n"
        "time.sleep(0.25)\n"
        + (
            ""
            if mode == "no_phase"
            else "print('Phase profiling (profiling) started | target: 900s', flush=True)\n"
        )
        + "time.sleep(0.8)\n"
        + f"sys.exit({17 if mode == 'client_failure' else 0})\n"
    )
    report = tmp_path / "profile.json"
    start_time = time.monotonic()
    command = [
        sys.executable,
        str(ROOT / ".github/scripts/profile_agentic_replay.py"),
        "--url",
        f"http://127.0.0.1:{server.server_port}",
        "--output",
        str(report),
        "--seconds",
        "0.1" if mode == "success" else "30",
        "--",
        sys.executable,
        str(replay),
    ]
    if mode == "shell_cancel":
        command = ["bash", "-c", shlex.join(command) + " & wait"]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=mode == "shell_cancel",
    )
    try:
        if mode == "cancel":
            assert started.wait(5)
            process.send_signal(signal.SIGTERM)
        elif mode == "shell_cancel":
            assert started.wait(5)
            cleanup = subprocess.run(
                [
                    "bash",
                    "-c",
                    "source .github/scripts/benchmark_bundle.sh; stop_benchmark_client",
                ],
                cwd=ROOT,
                env={
                    **os.environ,
                    "CLIENT_PID": str(process.pid),
                    "ENABLE_TORCH_PROFILER": "1",
                },
                capture_output=True,
                timeout=8,
                check=False,
            )
            assert cleanup.returncode == 0, cleanup.stderr
        stdout, stderr = process.communicate(timeout=10)
        data = json.loads(report.read_text())
        paths = [path for path, _ in calls]
        if mode == "no_phase":
            assert paths == []
            assert process.returncode != 0
            assert not data["complete"]
        else:
            assert paths == ["/start_profile", "/stop_profile"]
            assert calls[0][1] - start_time >= 0.25
        if mode == "success":
            assert process.returncode == 0, stderr
            assert data["complete"]
            assert 0.05 <= calls[1][1] - calls[0][1] < 0.7
            assert b"Phase warmup" in stdout
        elif mode == "client_failure":
            assert process.returncode == 17
            assert not data["complete"]
        elif mode == "cancel":
            assert process.returncode == 143
            assert not data["complete"]
        elif mode == "shell_cancel":
            assert data["cancelled_signal"] == signal.SIGTERM
            assert data["stopped_at"]
            assert not data["errors"]
            assert not data["complete"]
        elif mode in ("start_failure", "stop_failure"):
            assert process.returncode != 0
            assert data["errors"]
            assert not data["complete"]
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        server.shutdown()
        server.server_close()
        thread.join()
