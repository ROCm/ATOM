#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only host memory preparation, before either model service starts."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numa_memory_budget as budget

HERE = Path(__file__).resolve().parent
GIB = 1 << 30


def read_optional(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"unavailable: {exc}"


def cgroup_memory(root=Path("/sys/fs/cgroup"), membership=Path("/proc/self/cgroup")):
    """Read visible v2 ancestors, or the usual v1 memory controller mount."""
    candidates = {root}
    for line in read_optional(membership).splitlines():
        if line.startswith("0::"):
            relative = Path(line[3:].lstrip("/"))
            if ".." not in relative.parts:
                path = root / relative
                candidates.update(
                    [path, *[p for p in path.parents if p == root or root in p.parents]]
                )
    rows = []
    for path in sorted(candidates):
        if (path / "memory.max").is_file():
            limit = (path / "memory.max").read_text().strip()
            current = int((path / "memory.current").read_text())
            rows.append(
                {
                    "path": str(path),
                    "limit": None if limit == "max" else int(limit),
                    "current": current,
                }
            )
    v1 = root / "memory"
    if not rows and (v1 / "memory.limit_in_bytes").is_file():
        limit = int((v1 / "memory.limit_in_bytes").read_text())
        rows.append(
            {
                "path": str(v1),
                "limit": None if limit >= (1 << 60) else limit,
                "current": int((v1 / "memory.usage_in_bytes").read_text()),
            }
        )
    return rows


def snapshot(planned):
    return {
        "unix_time": time.time(),
        "uid": os.getuid(),
        "numa": {node: budget.node_meminfo(Path("/sys"), node) for node in planned},
        "cgroup": cgroup_memory(),
        "meminfo": read_optional("/proc/meminfo"),
        "buddyinfo": read_optional("/proc/buddyinfo"),
        "vmstat": read_optional("/proc/vmstat"),
        "thp_enabled": read_optional("/sys/kernel/mm/transparent_hugepage/enabled"),
        "thp_defrag": read_optional("/sys/kernel/mm/transparent_hugepage/defrag"),
    }


def capacity_error(state, planned, reserve_gib):
    reserve = reserve_gib * GIB
    for node, gib in planned.items():
        info = state["numa"][node]
        if gib * GIB + reserve > info.get("MemTotal", 0):
            return "numa_capacity_exceeded"
        # The pool itself must fit in free memory before first-touch. The
        # additional reserve may include clean file cache that can be reclaimed;
        # requiring the reserve to be free too rejected a node with 1056 GiB
        # free plus 414 GiB file cache for a 1024 GiB pool.
        free = info.get("MemFree", 0)
        clean_file_cache = max(
            0,
            info.get("Active(file)", 0)
            + info.get("Inactive(file)", 0)
            - info.get("Dirty", 0)
            - info.get("Writeback", 0),
        )
        if gib * GIB > free or gib * GIB + reserve > free + clean_file_cache:
            return "insufficient_numa_free_memory"
    if not state["cgroup"]:
        return "cgroup_memory_unknown"
    needed = sum(planned.values()) * GIB + reserve
    for row in state["cgroup"]:
        if row["limit"] is not None and needed > row["limit"] - row["current"]:
            return "insufficient_cgroup_headroom"
    return None


def run_step(command, seconds, output):
    """Bound each child; never launch a model while a timed-out child remains."""
    print(f"[host-memory] {output.name}: {command}", flush=True)
    started = time.monotonic()
    with output.open("w") as log:
        child = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "LC_ALL": "C"},
        )
        timed_out = False
        try:
            child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(child.pid, signal.SIGKILL)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                # A process in kernel uninterruptible sleep may not exit yet.
                raise RuntimeError(
                    f"preparation child {child.pid} did not exit after SIGKILL; refusing model startup"
                ) from exc
    return {
        "command": command,
        "returncode": child.returncode,
        "timed_out": timed_out,
        "seconds": time.monotonic() - started,
        "log": str(output),
        "text": output.read_text(),
    }


def prepare(args, *, run=run_step, take_snapshot=snapshot, clock=time.monotonic):
    args.output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "preparing",
        "attempts": [],
        "planned_gib": {},
        "settings": {
            "attempts": args.attempts,
            "timeout": args.timeout,
            "step_timeout": args.step_timeout,
            "reserve_gib": args.reserve_gib,
        },
    }
    began = clock()
    deadline = began + args.timeout

    def save():
        result["seconds"] = clock() - began
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for directory in args.directories:
            if not Path(directory).is_dir():
                raise ValueError(f"model directory is missing: {directory}")
        planned = budget.plan_pins([], args.gpus, args.per_gpu_gib, Path("/sys"))
        if not planned or min(planned.values()) <= 0:
            raise ValueError("a non-empty positive LMCache memory plan is required")
        result["planned_gib"] = planned
        direct_available = True
        for attempt in range(1, args.attempts + 1):
            if clock() >= deadline:
                result["status"] = "preparation_deadline"
                break
            row = {"attempt": attempt, "before": take_snapshot(planned), "steps": []}
            result["attempts"].append(row)
            save()

            def step(label, command, *, attempt=attempt, row=row):
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("host-memory preparation deadline exceeded")
                completed = run(
                    command,
                    min(args.step_timeout, remaining),
                    args.output / f"attempt-{attempt}-{label}.log",
                )
                row["steps"].append(completed)
                save()
                return completed

            direct_ok = False
            if direct_available:
                # No sudo, privilege escalation, or container capability changes.
                direct = step(
                    "direct",
                    [
                        "sh",
                        "-c",
                        'set -e; sync; printf "3\\n" > /proc/sys/vm/drop_caches; printf "1\\n" > /proc/sys/vm/compact_memory',
                    ],
                )
                if direct["timed_out"]:
                    result["status"] = "direct_compaction_timeout"
                    break
                direct_ok = direct["returncode"] == 0
                if not direct_ok:
                    direct_available = False
                    row["direct_unavailable"] = True
                    print(
                        "[host-memory] Direct VM preparation unavailable; using file-cache and NUMA helpers",
                        flush=True,
                    )
            dropped = step(
                "file-cache",
                [sys.executable, str(HERE / "drop_page_cache.py"), *args.directories],
            )
            row["after_drop"] = take_snapshot(planned)
            problem = capacity_error(row["after_drop"], planned, args.reserve_gib)
            if dropped["timed_out"] or dropped["returncode"]:
                problem = "file_cache_step_failed"
            if problem:
                row["status"] = problem
                result["status"] = problem
                if problem in {
                    "numa_capacity_exceeded",
                    "cgroup_memory_unknown",
                    "insufficient_cgroup_headroom",
                }:
                    break
                continue
            complete = direct_ok
            if not direct_ok:
                complete = True
                for node, gib in sorted(planned.items()):
                    compact = step(
                        f"numa-{node}",
                        [
                            sys.executable,
                            str(HERE / "numa_exec.py"),
                            str(node),
                            sys.executable,
                            str(HERE / "compact_huge_pages.py"),
                            str(gib),
                        ],
                    )
                    # The existing script exits 0 even for partial THP coverage.
                    full_thp = any(
                        line.startswith("[thp-compact] ")
                        for line in compact["text"].splitlines()
                    )
                    if compact["timed_out"] or compact["returncode"] or not full_thp:
                        complete = False
                        row["status"] = "numa_preparation_incomplete"
                        # A permission denial will not improve by retrying it.
                        if any(
                            word in compact["text"]
                            for word in (
                                "Operation not permitted",
                                "Permission denied",
                                "Read-only file system",
                            )
                        ):
                            result["status"] = "numa_permission_denied"
                            return 2
                        break
            row["after"] = take_snapshot(planned)
            row["status"] = "prepared" if complete else "numa_preparation_incomplete"
            result["status"] = row["status"]
            if complete:
                result["method"] = "direct_vm" if direct_ok else "file_cache_and_numa"
                # This establishes preparation, not the latency of LMCache pinning.
                return 0
        return 2
    except TimeoutError as exc:
        result.update(status="preparation_deadline", error=repr(exc))
        return 2
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        result.update(status="preparation_error", error=repr(exc))
        return 2
    finally:
        save()
        print(f"[host-memory] {result['status']}; evidence={args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--per-gpu-gib", required=True, type=budget.non_negative_gib)
    parser.add_argument("--reserve-gib", type=budget.non_negative_gib, default=128)
    parser.add_argument("--attempts", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--timeout", type=budget.positive_seconds, default=120)
    parser.add_argument("--step-timeout", type=budget.positive_seconds, default=35)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("directories", nargs="+")
    return prepare(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
