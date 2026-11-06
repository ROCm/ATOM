"""Discover actual keys, seal once, then validate a bounded complete workload.

The finite warm cases target the previously observed fill/replenish/drain and
tail paths. They do NOT prove all possible scheduler keys have been visited.
Sealed guards and cold-operation deltas decide coverage for the new measured
workload. Any miss fails this named attempt; no automatic more-warm retry.
"""

import json
import os
import time
from pathlib import Path

from rebase_graph_audit_r6 import write_json
from run_owner_serving import OUTPUT, URL

__all__ = (
    "OUTPUT",
    "SERVER_COMMAND",
    "URL",
    "WARM_CASES",
    "set_phase",
    "snapshots",
    "wait_for",
)

HERE = Path(__file__).resolve().parent
PHASE = Path(os.environ["DSV41_OWNER_GRAPH_PHASE_FILE"])
SERVER_COMMAND = [
    "/opt/venv/bin/python",
    "-m",
    "atom.entrypoints.openai_server",
    "--model",
    "/models/DeepSeek-V4.1-Flash",
    "--served-model-name",
    "/models/DeepSeek-V4.1-Flash",
    "--host",
    "127.0.0.1",
    "--server-port",
    "19125",
    "--kv_cache_dtype",
    "bf16",
    "-tp",
    "4",
    "--index_cache_dtype",
    "fp4",
    "--level",
    "3",
    "--cudagraph-mode",
    "FULL",
    "--max-num-seqs",
    "256",
    "--gpu-memory-utilization",
    "0.85",
    "--disable-uvicorn-access-log",
    "--method",
    "dspark",
    "--num-speculative-tokens",
    "5",
]
WARM_CASES = (
    ("fill_replenish_drain", 512, 256, 1024),
    ("tail64_full_proposal", 64, 64, 128),
    ("tail16_full_proposal", 16, 16, 128),
    ("last_request", 1, 1, 128),
)


def set_phase(value):
    temporary = PHASE.with_suffix(".tmp")
    temporary.write_text(value + "\n")
    temporary.replace(PHASE)


def wait_for(server, predicate, timeout):
    deadline = time.monotonic() + timeout
    while not predicate():
        if server.poll() is not None:
            raise RuntimeError("Rebased server exited; preserve this attempt")
        if list(OUTPUT.glob("graph_audit_failure_pid*.json")):
            raise RuntimeError("Runtime audit failed; measurement is invalid")
        if time.monotonic() >= deadline:
            raise TimeoutError("New validation exceeded its bounded wait")
        time.sleep(0.2)


def snapshots(server, identity):
    paths = [OUTPUT / f"graph_audit_rank{r}.json" for r in range(4)]
    write_json(OUTPUT / "graph_audit_request.json", {"id": identity})

    def complete():
        return all(
            p.exists() and json.loads(p.read_text())["request_id"] == identity
            for p in paths
        )

    wait_for(server, complete, 45)
    result = [json.loads(p.read_text()) for p in paths]
    for rank, row in enumerate(result):
        write_json(OUTPUT / f"{identity}_rank{rank}.json", row)
    return result
