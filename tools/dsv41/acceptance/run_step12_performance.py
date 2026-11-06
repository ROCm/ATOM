"""One unprofiled, source-bound arm; all requests and cold operations count."""

import json
import os
import signal
import subprocess
import threading
import urllib.request

from rebase_graph_audit_r6 import validate_interval, write_json
from run_owner_packet_service import commit_counts
from run_prefill_route_capture import bounded_log
from run_rebase_graph_coverage_r6 import (
    OUTPUT,
    SERVER_COMMAND,
    URL,
    WARM_CASES,
    set_phase,
    snapshots,
    wait_for,
)
from serving_benchmark_storage import benchmark


def main():
    candidate = os.environ["DSV41_OWNER_SERVING"] == "1"
    mode = "packed" if candidate else "native"
    argv = (
        SERVER_COMMAND[:]
        if candidate
        else SERVER_COMMAND[: SERVER_COMMAND.index("--method")]
    )
    path = OUTPUT / "step12_performance.json"
    if path.exists():
        raise RuntimeError("Preserve each independent performance arm")
    environment = dict(os.environ, DSV41_REBASE_SERVICE_SERVER="1")
    client = os.environ["DSV41_REBASE_ATOM_SOURCE"]
    subject = environment["DSV41_REBASE_ATOM_SOURCE"]
    environment["PYTHONPATH"] = ":".join(
        subject if entry == client else entry
        for entry in environment["PYTHONPATH"].split(":")
        if candidate or not entry.startswith(client + "/")
    )
    state = {
        "status": "loading",
        "subject": subject,
        "candidate": candidate,
        "server_command": argv,
        "formal_requests": 2560,
        "new_GSM": False,
        "profiled": False,
        "warm_cases": WARM_CASES,
    }
    set_phase("warmup")
    (OUTPUT / "capacity.txt").write_text(mode + "\n")
    write_json(path, state)
    server = subprocess.Popen(
        argv,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    drain = threading.Thread(
        target=bounded_log,
        args=(server.stdout, OUTPUT / "step12_server.log", 2_000_000),
    )
    drain.start()
    try:

        def ready():
            try:
                with urllib.request.urlopen(URL + "/v1/models", timeout=2) as response:
                    return response.status == 200
            except OSError:
                return False

        wait_for(server, ready, 5400)
        state["preload"] = [
            json.loads((OUTPUT / f"kernel_preload_rank{rank}.json").read_text())
            for rank in range(4)
        ]
        if any(row["status"] != "PASS" for row in state["preload"]):
            raise RuntimeError("Each formal worker must finish its native preload plan")
        state["warm"] = []
        for label, count, concurrency, output in WARM_CASES:
            state.update(status="warming", current_case=label)
            write_json(path, state)
            state["warm"].append(
                benchmark(
                    "warm_" + label,
                    mode,
                    server,
                    count=count,
                    concurrency=concurrency,
                    output_length=output,
                    seed=2026102517,
                )
            )
        set_phase("sealed")
        before = snapshots(server, "audit_before")
        committed = commit_counts()
        state["status"] = "formal"
        write_json(path, state)
        state["formal"] = benchmark("formal", mode, server, count=2560, seed=2026102518)
        after = snapshots(server, "audit_after")
        state["scheduler_commit_delta"] = delta = {
            key: value - committed.get(key, 0) for key, value in commit_counts().items()
        }
        state["actual_C256_q"] = (
            delta["C256_decode_committed_tokens"]
            / delta["C256_decode_source_request_cycles"]
        )
        state["audit_delta"] = validate_interval(before, after)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(path, state)
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=15)
        drain.join(timeout=10)


if __name__ == "__main__":
    main()
