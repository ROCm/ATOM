#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Temporarily shrink unused 2 MiB HugeTLB pools on a CI allocation's NUMA nodes."""

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path

import numa_memory_budget as budget
from prepare_lmcache_host_memory import GIB, run_step

PAGE_BYTES = 2 << 20


def save(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def pool_path(sysfs, node):
    return sysfs / f"devices/system/node/node{node}/hugepages/hugepages-2048kB"


def read_pool(path):
    return {
        key: int((path / key).read_text())
        for key in ("nr_hugepages", "free_hugepages", "surplus_hugepages")
    }


def change_pool(path, count, log):
    # A separate process bounds kernel work; no sudo or container privilege change.
    result = run_step(
        [
            sys.executable,
            "-c",
            'import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2]+"\\n")',
            str(path / "nr_hugepages"),
            str(count),
        ],
        60,
        log,
    )
    if result["timed_out"] or result["returncode"]:
        raise RuntimeError(
            f"HugeTLB write failed: {result['text'].strip()} (timeout={result['timed_out']})"
        )


def prepare(
    output, job_id, planned, reserve_gib=128, *, sysfs=Path("/sys"), change=change_pool
):
    output.mkdir(parents=True, exist_ok=True)
    journal = output / "hugetlb.json"
    # Never overwrite recovery data from an earlier attempt.
    if journal.exists():
        raise RuntimeError(f"HugeTLB journal already exists: {journal}")
    state = {
        "job_id": job_id,
        "host": socket.gethostname(),
        "uid": os.getuid(),
        "status": "preparing",
        "pools": [],
    }
    save(journal, state)
    try:
        for node, gib in sorted(planned.items()):
            memory = budget.node_meminfo(sysfs, node)
            needed = math.ceil((gib + reserve_gib) * GIB) - memory["MemFree"]
            if needed <= 0:
                continue
            path = pool_path(sysfs, node)
            if not path.exists():
                continue
            before = read_pool(path)
            if not before["nr_hugepages"]:
                continue
            row = {"node": node, "before": before, "memory_before": memory}
            state["pools"].append(row)
            save(journal, state)
            reserved = int(
                (
                    sysfs / "kernel/mm/hugepages/hugepages-2048kB/resv_hugepages"
                ).read_text()
            )
            if (
                reserved
                or before["nr_hugepages"] != before["free_hugepages"]
                or before["surplus_hugepages"]
            ):
                raise RuntimeError(
                    f"NUMA{node} HugeTLB pool is in use or reserved; refusing to shrink it"
                )
            count = max(0, before["nr_hugepages"] - math.ceil(needed / PAGE_BYTES))
            # Journal before mutation, including partial writes or interruption.
            row.update(target=count, writer="pending")
            save(journal, state)
            try:
                change(path, count, output / f"release-node{node}.log")
            except RuntimeError as exc:
                if "did not exit after SIGKILL" in str(exc):
                    row["writer"] = "unreaped"
                    raise
                row["writer"] = "finished"
                raise
            row["writer"] = "finished"
            row["after"] = read_pool(path)
            row["memory_after"] = budget.node_meminfo(sysfs, node)
            save(journal, state)
            if row["after"]["nr_hugepages"] != count:
                raise RuntimeError(
                    f"NUMA{node} HugeTLB shrink did not reach {count} pages"
                )
            released = (before["nr_hugepages"] - count) * PAGE_BYTES
            if row["memory_after"]["MemFree"] < memory["MemFree"] + released - GIB:
                raise RuntimeError(
                    f"NUMA{node} ordinary free memory did not recover after HugeTLB shrink"
                )
            print(
                f"[host-memory] NUMA{node}: released {released / GIB:.2f} GiB idle HugeTLB; restore journal={journal}",
                flush=True,
            )
        state["status"] = "prepared"
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        state.update(status="failed", error=str(exc))
        raise
    finally:
        save(journal, state)


def restore(output, job_id, *, sysfs=Path("/sys"), change=change_pool):
    journal = output / "hugetlb.json"
    if not journal.exists():
        return
    state = json.loads(journal.read_text())
    if (
        state["job_id"] != job_id
        or state["host"] != socket.gethostname()
        or state["uid"] != os.getuid()
    ):
        raise RuntimeError(
            "HugeTLB restore journal does not belong to this job/host/user"
        )
    errors = []
    for row in reversed(state["pools"]):
        if "target" not in row:
            continue
        try:
            if row.get("writer") != "finished":
                raise RuntimeError(
                    "previous HugeTLB writer may still be running; manual recovery required"
                )
            path = pool_path(sysfs, row["node"])
            current = read_pool(path)
            original = row["before"]["nr_hugepages"]
            if current["nr_hugepages"] != original:
                if not row["target"] <= current["nr_hugepages"] < original:
                    raise RuntimeError(
                        "HugeTLB count changed outside the job; refusing to overwrite it"
                    )
                # Restoration only increases the configured pool after services exit.
                row["writer"] = "pending"
                save(journal, state)
                try:
                    change(path, original, output / f"restore-node{row['node']}.log")
                except RuntimeError as exc:
                    row["writer"] = (
                        "unreaped"
                        if "did not exit after SIGKILL" in str(exc)
                        else "finished"
                    )
                    raise
                row["writer"] = "finished"
                if read_pool(path)["nr_hugepages"] != original:
                    raise RuntimeError(
                        "HugeTLB restore incomplete; node administrator recovery required"
                    )
            row["restored"] = True
        except (OSError, ValueError, RuntimeError) as exc:
            errors.append(f"NUMA{row['node']}: {exc}")
        finally:
            save(journal, state)
    state.update(
        restore_status="failed" if errors else "restored", restore_errors=errors
    )
    save(journal, state)
    if errors:
        raise RuntimeError("; ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "restore"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpus")
    parser.add_argument("--per-gpu-gib", type=budget.non_negative_gib)
    args = parser.parse_args()
    try:
        if not args.job_id.isdigit():
            raise ValueError("requires a Slurm allocation")
        if args.action == "restore":
            restore(args.output, args.job_id)
        else:
            if not args.gpus or not args.per_gpu_gib:
                raise ValueError("positive GPU memory plan required")
            planned = budget.plan_pins([], args.gpus, args.per_gpu_gib, Path("/sys"))
            prepare(args.output, args.job_id, planned)
    except (OSError, ValueError, RuntimeError, SystemExit) as exc:
        print(f"[host-memory][FAIL] {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
