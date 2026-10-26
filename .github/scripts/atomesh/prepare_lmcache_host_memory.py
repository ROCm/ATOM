#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only memory preparation for LMCache or Store, before services start."""

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


def capacity_error(state, planned, reserve_gib, *, require_free=False):
    reserve = reserve_gib * GIB
    for node, gib in planned.items():
        info = state["numa"][node]
        if gib * GIB + reserve > info.get("MemTotal", 0):
            return "numa_capacity_exceeded"
        # Host preparation deliberately reclaims clean file cache before model
        # startup. Requiring the whole pool to be free beforehand prevents that
        # preparation on otherwise idle nodes (769 GiB free + 701 GiB cache).
        free = info.get("MemFree", 0)
        clean_file_cache = max(
            0,
            info.get("Active(file)", 0)
            + info.get("Inactive(file)", 0)
            - info.get("Dirty", 0)
            - info.get("Writeback", 0),
        )
        if gib * GIB + reserve > free + clean_file_cache:
            return "insufficient_numa_reclaimable_memory"
        if require_free and gib * GIB > free:
            return "insufficient_numa_free_memory"
    if not state["cgroup"]:
        return "cgroup_memory_unknown"
    needed = sum(planned.values()) * GIB + reserve
    for row in state["cgroup"]:
        if row["limit"] is not None and needed > row["limit"] - row["current"]:
            return "insufficient_cgroup_headroom"
    return None


def base_page_reclaim_gib(state, node, planned_gib, reserve_gib):
    """Scratch capacity, retaining the reserve in both NUMA and cgroup budgets."""
    info = state["numa"][node]
    free = info.get("MemFree", 0)
    clean = max(
        0,
        info.get("Active(file)", 0)
        + info.get("Inactive(file)", 0)
        - info.get("Dirty", 0)
        - info.get("Writeback", 0),
    )
    reserve = reserve_gib * GIB
    if clean <= reserve:
        return 0
    # Ordinary free memory can still be scattered among cached file pages.
    # Count free blocks of order >= 9 (2 MiB on these Linux hosts).
    large_free = 0
    found = False
    for line in state.get("buddyinfo", "").splitlines():
        fields = line.split()
        if len(fields) > 4 and fields[:2] == ["Node", f"{node},"]:
            found = True
            large_free += sum(
                int(count) * (os.sysconf("SC_PAGE_SIZE") << order)
                for order, count in enumerate(fields[4:])
                if order >= 9
            )
    if free >= planned_gib * GIB and (not found or large_free >= planned_gib * GIB):
        return 0
    available = min(free + clean, info.get("MemTotal", 0)) - reserve
    if not state["cgroup"]:
        return 0  # The caller rejects unknown cgroup capacity before allocation.
    for row in state["cgroup"]:
        if row["limit"] is not None:
            available = min(available, row["limit"] - row["current"] - reserve)
    return max(0, int(available // GIB))


def child_state(pid, elapsed):
    row = {"pid": pid, "elapsed_seconds": round(elapsed, 3)}
    for line in read_optional(f"/proc/{pid}/status").splitlines():
        key, _, value = line.partition(":")
        if key in {"State", "VmRSS", "RssAnon"}:
            row[key] = value.strip()
    row["wchan"] = read_optional(f"/proc/{pid}/wchan").strip()
    return row


def file_cache_delta(before, after):
    fields = ("MemFree", "Active(file)", "Inactive(file)", "Dirty", "Writeback")
    return {
        node: {
            field: {
                "before_gib": round(info.get(field, 0) / GIB, 3),
                "after_gib": round(after["numa"][node].get(field, 0) / GIB, 3),
                "delta_gib": round(
                    (after["numa"][node].get(field, 0) - info.get(field, 0)) / GIB, 3
                ),
            }
            for field in fields
        }
        for node, info in before["numa"].items()
    }


def run_step(command, seconds, output):
    """Bound each child; never launch a model while a timed-out child remains."""
    print(f"[host-memory] {output.name}: {command}", flush=True)
    started = time.monotonic()
    heartbeats = []
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
            while True:
                remaining = seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, seconds)
                try:
                    child.wait(timeout=min(10, remaining))
                    break
                except subprocess.TimeoutExpired:
                    row = child_state(child.pid, time.monotonic() - started)
                    heartbeats.append(row)
                    print("[host-memory][child] " + json.dumps(row), flush=True)
                    if time.monotonic() - started >= seconds:
                        raise
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
        "heartbeats": heartbeats,
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
            "base_page_step_timeout": args.step_timeout + 30,
        },
    }
    cache_roots = getattr(args, "cache_roots", [])
    result["settings"]["cache_roots"] = cache_roots
    began = clock()
    deadline = began + args.timeout

    def save():
        result["seconds"] = clock() - began
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")

    try:
        for directory in [*args.directories, *cache_roots]:
            if not Path(directory).is_dir():
                raise ValueError(f"model directory is missing: {directory}")
        planned = budget.plan_pins(
            getattr(args, "pins", []), args.gpus, args.per_gpu_gib, Path("/sys")
        )
        if not planned or min(planned.values()) <= 0:
            raise ValueError("a non-empty positive memory plan is required")
        result["planned_gib"] = planned
        direct_available = not getattr(args, "skip_direct", False)
        reclaimed_nodes = set()
        for attempt in range(1, args.attempts + 1):
            if clock() >= deadline:
                result["status"] = "preparation_deadline"
                break
            row = {"attempt": attempt, "before": take_snapshot(planned), "steps": []}
            result["attempts"].append(row)
            save()

            def step(label, command, *, release_allowance=0, attempt=attempt, row=row):
                remaining = deadline - clock()
                if remaining <= 0:
                    raise TimeoutError("host-memory preparation deadline exceeded")
                completed = run(
                    command,
                    min(args.step_timeout + release_allowance, remaining),
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
            row["before_drop"] = take_snapshot(planned)
            dropped = step(
                "file-cache",
                [
                    sys.executable,
                    str(HERE / "drop_page_cache.py"),
                    "--stats",
                    *args.directories,
                ],
            )
            row["after_drop"] = take_snapshot(planned)
            row["file_cache_delta"] = file_cache_delta(
                row["before_drop"], row["after_drop"]
            )
            print(
                "[host-memory][file-cache-delta] "
                + json.dumps(row["file_cache_delta"]),
                flush=True,
            )
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
            # Other models can occupy nearly all of an idle node's RAM. On
            # g08 (38788119513), advising only MiniMax left 1.4 TiB of cache;
            # repeatedly faulting THP then reclaimed just 84 GiB in 270 s.
            if cache_roots and capacity_error(
                row["after_drop"], planned, args.reserve_gib, require_free=True
            ):
                row["after_target_drop"] = row["after_drop"]
                broader = step(
                    "model-cache",
                    [
                        sys.executable,
                        str(HERE / "drop_page_cache.py"),
                        "--stats",
                        *cache_roots,
                    ],
                )
                row["after_drop"] = take_snapshot(planned)
                row["model_cache_delta"] = file_cache_delta(
                    row["after_target_drop"], row["after_drop"]
                )
                print(
                    "[host-memory][model-cache-delta] "
                    + json.dumps(row["model_cache_delta"]),
                    flush=True,
                )
                if broader["timed_out"] or broader["returncode"]:
                    row["status"] = result["status"] = "model_cache_step_failed"
                    continue
                problem = capacity_error(row["after_drop"], planned, args.reserve_gib)
                if problem:
                    row["status"] = result["status"] = problem
                    break
            complete = direct_ok and not capacity_error(
                row["after_drop"], planned, args.reserve_gib, require_free=True
            )
            used_numa_helper = not complete
            if used_numa_helper:
                complete = True
                for node, gib in sorted(planned.items()):
                    current = take_snapshot(planned)
                    problem = capacity_error(current, planned, args.reserve_gib)
                    if problem:
                        result["status"] = problem
                        return 2
                    scratch_gib = base_page_reclaim_gib(
                        current, node, gib, args.reserve_gib
                    )
                    if scratch_gib and node not in reclaimed_nodes:
                        row.setdefault("before_base_reclaim", {})[node] = current
                        reclaimed = step(
                            f"base-pages-{node}",
                            [
                                sys.executable,
                                str(HERE / "numa_exec.py"),
                                str(node),
                                sys.executable,
                                str(HERE / "reclaim_file_cache.py"),
                                str(scratch_gib),
                            ],
                            # On g35, 1344 GiB populated in 67s, but freeing
                            # its base-page PTEs was still running at 90s.
                            release_allowance=30,
                        )
                        row.setdefault("after_base_reclaim", {})[node] = take_snapshot(
                            planned
                        )
                        if reclaimed["timed_out"] or reclaimed["returncode"]:
                            complete = False
                            row["status"] = "base_page_reclaim_failed"
                            break  # Retry only after run_step has reaped the child.
                        reclaimed_nodes.add(node)
                    compact = step(
                        f"numa-{node}",
                        [
                            sys.executable,
                            str(HERE / "numa_exec.py"),
                            str(node),
                            sys.executable,
                            str(HERE / "compact_huge_pages.py"),
                            "--progress",
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
            row["status"] = (
                "prepared"
                if complete
                else row.get("status", "numa_preparation_incomplete")
            )
            result["status"] = row["status"]
            if complete:
                problem = capacity_error(
                    row["after"], planned, args.reserve_gib, require_free=True
                )
                if problem:
                    row["status"] = result["status"] = problem
                    continue
                result["method"] = (
                    "file_cache_and_numa" if used_numa_helper else "direct_vm"
                )
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
    parser.add_argument("--gpus", default="")
    parser.add_argument("--per-gpu-gib", default=0, type=budget.non_negative_gib)
    parser.add_argument(
        "--pin",
        dest="pins",
        action="append",
        default=[],
        metavar="NODE:GIB",
        help="explicit NUMA owner memory, added to any per-GPU memory",
    )
    parser.add_argument(
        "--skip-direct",
        action="store_true",
        help="use bounded file-cache reclaim and full-THP checks without VM sysctl writes",
    )
    parser.add_argument("--reserve-gib", type=budget.non_negative_gib, default=128)
    parser.add_argument("--attempts", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--timeout", type=budget.positive_seconds, default=120)
    parser.add_argument("--step-timeout", type=budget.positive_seconds, default=35)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache-root",
        dest="cache_roots",
        action="append",
        default=[],
        help="additional model-cache directory to advise only when free RAM is insufficient",
    )
    parser.add_argument("directories", nargs="+")
    return prepare(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
