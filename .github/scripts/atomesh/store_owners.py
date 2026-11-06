#!/usr/bin/env python3
"""Start a standalone-store case's Mooncake Store owners and wait until the master counts their segments.

ATOM's Store layout: mooncake_client owners mount the segments before vLLM loads weights, right after the page
cache is dropped and the nodes compacted, and the ranks (mode standalone-store) mount nothing. A rank's embedded
segment mounts only after the checkpoint read has refilled the page cache, and its tail comes as 4 KiB pages that
ionic refuses (mc56, mc57). One owner per NIC on the NIC's NUMA node keeps the segments where the ranks' were.

Usage: store_owners.py <case json> <client binary> <host ip> <master port> <metrics port> <first owner port> <log dir>
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILED = ("Failed to setup client", "Failed to mount", "Check failed", "FATAL")
TIMEOUT = 1200


def owner_plan(case: dict) -> list[tuple[str, int, int, dict[str, str]]]:
    """(device, NUMA node, segment bytes, env) per owner: the prefill role's Mooncake env plus the owners' own."""
    owners = case["mooncake"]["store_owners"]
    env = {k: v for k, v in case["roles"]["prefill"]["env"].items() if k.startswith("MC_") or k == "GLIBC_TUNABLES"}
    env.update(owners.get("env", {}))
    # No GPU: a process that sees one makes Mooncake probe every buffer through HIP.
    env.update(HIP_VISIBLE_DEVICES="-1", CUDA_VISIBLE_DEVICES="-1")
    plan = []
    for dev in owners["devices"]:
        numa = int(Path(f"/sys/class/infiniband/{dev}/device/numa_node").read_text())
        assert numa >= 0, f"{dev} reports no NUMA node"
        plan.append((dev, numa, int(owners["segment_size"]), env))
    return plan


def capacity(metrics_port: int) -> int:
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{metrics_port}/metrics", timeout=5).read().decode()
    except OSError:
        return 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "master_total_capacity_bytes":
            return int(float(fields[1]))
    return 0


def stop(owners: list) -> None:
    for _, proc, _ in owners:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def main(argv: list[str]) -> None:
    case, client, host = json.loads(Path(argv[0]).read_text()), argv[1], argv[2]
    master_port, metrics_port, owner_port = map(int, argv[3:6])
    plan, owners = owner_plan(case), []
    for i, (dev, numa, size, env) in enumerate(plan):
        log = Path(argv[6]) / f"mooncake-owner-{dev}.log"
        cmd = [sys.executable, str(HERE / "numa_exec.py"), str(numa), client, f"--host={host}",
               f"--port={owner_port + i}", f"--master_server_address=127.0.0.1:{master_port}",
               "--metadata_server=P2PHANDSHAKE", "--protocol=rdma", f"--device_names={dev}",
               f"--global_segment_size={size}", "--local_buffer_size=0", "--threads=4"]
        print(f"[mooncake] owner {dev} numa={numa} env={env} cmd={' '.join(cmd)}", flush=True)
        with log.open("w") as out:
            proc = subprocess.Popen(cmd, env={**os.environ, **env}, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        owners.append((dev, proc, log))
    want, start = sum(size for _, _, size, _ in plan), time.monotonic()
    # shortcut: nothing watches the owners once they mount (a dead one only shrinks the Store); read their logs
    # and the master's capacity after the run, or track their pids in pd_c48_server.sh if a run depends on it.
    while (have := capacity(metrics_port)) < want:
        for dev, proc, log in owners:
            text = log.read_text(errors="replace")
            if (rc := proc.poll()) is not None or any(s in text for s in FAILED):
                stop(owners)
                sys.exit(f"[mooncake][FAIL] owner {dev} (rc={rc}):\n" + "\n".join(text.splitlines()[-30:]))
        if time.monotonic() - start > TIMEOUT:
            stop(owners)
            sys.exit(f"[mooncake][FAIL] owners mounted {have} of {want} bytes in {TIMEOUT} s")
        time.sleep(2)
    print(f"[mooncake] owners mounted {have} bytes in {time.monotonic() - start:.0f} s", flush=True)
    for dev, proc, _ in owners:
        rollup = Path(f"/proc/{proc.pid}/smaps_rollup").read_text().splitlines()
        print(f"[mooncake] owner {dev} pid={proc.pid} {next(l for l in rollup if l.startswith('AnonHugePages'))}",
              flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
