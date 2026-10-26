#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Read an existing TW MiniMax C32 run from a CI submitter; never submit work."""

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

CASE = "minimax-m3-mxfp4-minimax-m3-fp4-1p1d-tp4-eagle3-agentic-lmcache-1m-c32-nightly"
ROOTS = (
    Path("/share_nfs/ATOMESH_RUNNER/ATOMESH_LOG"),
    Path("/it-share/ATOMESH_LOG"),
)
PATTERNS = (
    "slurm-*.out",
    "slurm-*.err",
    "slurm_job-*/rank-*/*.log",
    "slurm_job-*/logs/*.log",
    "slurm_job-*/logs/*/*.log",
    "slurm_job-*/benchmark_results/*/aiperf.log",
    "slurm_job-*/benchmark_results/*/*/aiperf.log",
    "slurm_job-*/benchmark_results/lmcache-host-memory/*.log",
    "slurm_job-*/benchmark_results/lmcache-host-memory/summary.json",
    "slurm_job-*/benchmark_results/lmcache-host-memory/host/hugetlb.json",
    "slurm_job-*/benchmark_results/runtime-diagnostics/*",
    "slurm_job-*/benchmark_results/runtime-diagnostics/*/*",
    "slurm_job-*/benchmark_results/*/benchmark-guard.json",
    "slurm_job-*/job-result.json",
    "slurm_job-*/rank-rc-*",
)
MAX_FILE_BYTES = 8 << 20
MAX_TOTAL_BYTES = 64 << 20
MAX_DIAGNOSTIC_BYTES = 2 << 20


def run_id(value):
    if not re.fullmatch(r"[0-9]{5,20}", value):
        raise argparse.ArgumentTypeError("expected a numeric GitHub run ID")
    return value


def scheduler_state(target):
    command = ["squeue"]
    if os.environ.get("USES_SPUR_CONTROLLER") == "1" and os.environ.get(
        "SPUR_CONTROLLER_ADDR"
    ):
        command += ["--controller", os.environ["SPUR_CONTROLLER_ADDR"]]
    base_command = command.copy()
    command += ["--noheader", "--format=%i|%j|%T|%M|%R|%N"]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=False
        )
        rows = []
        for line in result.stdout.splitlines():
            fields = line.split("|")
            if len(fields) >= 6 and fields[1].strip().startswith(f"{CASE}-{target}-"):
                rows.append(line)
        state = {
            "returncode": result.returncode,
            "rows": rows,
            "stderr": result.stderr[-4096:],
        }
        pending = [
            line.split("|")[0].strip()
            for line in rows
            if line.split("|")[2].strip() == "PENDING"
        ]
        pending = [job for job in pending if re.fullmatch(r"[0-9]+", job)]
        if pending:
            try:
                estimate = subprocess.run(
                    base_command
                    + [
                        "--start",
                        "--noheader",
                        "--jobs=" + ",".join(pending),
                        "--format=%i|%S|%R|%N",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                state["start_estimate"] = {
                    "returncode": estimate.returncode,
                    "rows": [
                        line
                        for line in estimate.stdout.splitlines()
                        if line.split("|")[0].strip() in pending
                    ],
                    "stderr": estimate.stderr[-4096:],
                }
            except (OSError, subprocess.TimeoutExpired) as exc:
                state["start_estimate"] = {"error": str(exc)}
        return state
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def collect(target, output, *, roots=ROOTS):
    run_id(target)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": target,
        "host": socket.gethostname(),
        "unix_time": time.time(),
        "scheduler": scheduler_state(target),
        "roots": [],
        "files": [],
        "errors": [],
    }
    used = 0
    seen = set()
    for root in roots:
        row = {"root": str(root), "exists": root.is_dir(), "runs": []}
        manifest["roots"].append(row)
        for source in sorted(root.glob(f"{CASE}-{target}-*")):
            source = source.resolve()
            if not source.is_relative_to(root.resolve()) or not source.is_dir():
                continue
            row["runs"].append(str(source))
            for pattern in PATTERNS:
                # Rotated history must not crowd out another rank's live evidence.
                for path in sorted(
                    source.glob(pattern),
                    key=lambda path: (".previous." in path.name, path),
                ):
                    resolved = path.resolve()
                    if (
                        resolved in seen
                        or not resolved.is_relative_to(source)
                        or not resolved.is_file()
                    ):
                        continue
                    seen.add(resolved)
                    if used >= MAX_TOTAL_BYTES or len(manifest["files"]) >= 200:
                        manifest["limit_reached"] = True
                        break
                    relative = path.relative_to(source)
                    destination = output / source.name / relative
                    try:
                        with resolved.open("rb") as stream:
                            size = os.fstat(stream.fileno()).st_size
                            file_limit = MAX_FILE_BYTES
                            if "runtime-diagnostics" in relative.parts:
                                file_limit = min(file_limit, MAX_DIAGNOSTIC_BYTES)
                            count = min(size, file_limit, MAX_TOTAL_BYTES - used)
                            stream.seek(max(0, size - count))
                            data = stream.read(count)
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        destination.write_bytes(data)
                        used += len(data)
                        manifest["files"].append(
                            {
                                "source": str(path),
                                "artifact": str(destination.relative_to(output)),
                                "original_bytes": size,
                                "copied_bytes": len(data),
                                "tail_only": size > len(data),
                            }
                        )
                    except OSError as exc:
                        manifest["errors"].append(
                            {"path": str(path), "error": str(exc)}
                        )
    (output / "snapshot.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", type=run_id)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.run_id, args.output)


if __name__ == "__main__":
    main()
