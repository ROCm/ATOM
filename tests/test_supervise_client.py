# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run .github/scripts/supervise_client.sh (and scripts/wait_infer_drain.sh
under it) against fake server and client processes: a healthy run returns the
client's status, a hang or fault stops the client early with the drain code."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SUPERVISE = REPO / ".github" / "scripts" / "supervise_client.sh"

TICKING = "while :; do echo tick; sleep 0.3; done"
SILENT = "exec sleep 600"
VLLM_IDLE_STATS = (
    "while :; do echo 'Engine 000: Avg prompt throughput: 0.0 tokens/s, "
    "Avg generation throughput: 0.0 tokens/s, Running: 4 reqs'; sleep 0.3; done"
)


@pytest.fixture
def server(tmp_path):
    procs = []

    def start(script):
        with open(tmp_path / "server.log", "w") as log:
            proc = subprocess.Popen(
                ["bash", "-c", script],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        procs.append(proc)
        return proc

    yield start
    for proc in procs:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def _supervise(tmp_path, server_pid, client, engine="sglang"):
    env = {
        **os.environ,
        "WAIT_SERVER_PID": str(server_pid),
        "WAIT_POLL_SEC": "1",
        "WAIT_STUCK_POLLS": "3",
    }
    started = time.monotonic()
    result = subprocess.run(
        [
            "bash",
            str(SUPERVISE),
            engine,
            str(tmp_path / "client.log"),
            "1",
            "--",
            "bash",
            "-c",
            f"echo $$ > {tmp_path / 'client.pid'}; {client}",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return result.returncode, result.stdout, time.monotonic() - started


def _client_gone(tmp_path):
    pid = int((tmp_path / "client.pid").read_text())
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return True
    return state == "Z"


def test_healthy_run_returns_the_clients_status(tmp_path, server):
    srv = server(TICKING)
    code, _, _ = _supervise(tmp_path, srv.pid, "sleep 2; echo done")
    assert code == 0
    assert "done" in (tmp_path / "client.log").read_text()


def test_client_failure_status_is_kept(tmp_path, server):
    srv = server(TICKING)
    code, _, _ = _supervise(tmp_path, srv.pid, "sleep 1; exit 3")
    assert code == 3


def test_a_silent_server_with_a_waiting_client_is_a_hang(tmp_path, server):
    srv = server(SILENT)
    code, out, elapsed = _supervise(tmp_path, srv.pid, "exec sleep 600")
    assert code == 1
    assert "HANG detected" in out
    assert elapsed < 30
    assert _client_gone(tmp_path)


def test_vllm_zero_throughput_stats_are_not_progress(tmp_path, server):
    srv = server(VLLM_IDLE_STATS)
    code, out, _ = _supervise(tmp_path, srv.pid, "exec sleep 600", engine="vllm")
    assert code == 1
    assert "HANG detected" in out


def test_gpu_fault_in_the_server_log_stops_the_client(tmp_path, server):
    srv = server("sleep 1; echo 'Memory access fault by GPU node-2'; exec sleep 600")
    code, out, _ = _supervise(tmp_path, srv.pid, "exec sleep 600")
    assert code == 2
    assert "GPU fault detected" in out
    assert _client_gone(tmp_path)


def test_vllm_engine_death_is_a_fault(tmp_path, server):
    srv = server(
        "echo tick; sleep 1; echo 'ERROR EngineCore_DP0 died unexpectedly'; exec sleep 600"
    )
    code, _, _ = _supervise(tmp_path, srv.pid, "exec sleep 600", engine="vllm")
    assert code == 2


def test_server_exiting_under_a_running_client_is_a_fault(tmp_path, server):
    srv = server("echo tick; sleep 1")
    code, out, _ = _supervise(tmp_path, srv.pid, "exec sleep 600")
    assert code == 2
    assert "server exited while the client is still running" in out
    assert _client_gone(tmp_path)


def test_native_atom_defaults_are_unchanged():
    # Without the WAIT_* overrides the drain script still finds the native
    # ATOM server by pattern and treats its exit as a drain.
    text = (REPO / "scripts" / "wait_infer_drain.sh").read_text()
    assert (
        "SERVER_PATTERN='atom\\.entrypoints|atom\\.examples\\.simple_inference'" in text
    )
    assert 'WAIT_SERVER_EXIT_IS_FAULT="${WAIT_SERVER_EXIT_IS_FAULT:-0}"' in text
