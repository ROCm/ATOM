#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CI-only native Python stack dumps and read-only host memory sampling."""

import argparse
import faulthandler
import json
import os
import shutil
import time
from pathlib import Path

_STACK_FILE = None
_STARTED_PID = None


def process_start(pid):
    # comm can contain spaces and parentheses; fields after it start at state.
    return (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[19]


def start():
    global _STACK_FILE, _STARTED_PID
    root = os.environ.get("ATOM_CI_RUNTIME_DIAGNOSTICS")
    if not root or _STARTED_PID == os.getpid():
        return
    role = os.environ.get("ATOM_CI_RUNTIME_ROLE", "unknown")
    folder = Path(root) / role
    folder.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    _STACK_FILE = (folder / f"pid-{pid}.stacks.log").open("a")
    # Native watchdog works even while another thread holds the GIL in a C call.
    faulthandler.enable(file=_STACK_FILE)
    faulthandler.dump_traceback_later(120, repeat=True, file=_STACK_FILE)
    (folder / f"pid-{pid}.json").write_text(
        json.dumps(
            {
                "pid": pid,
                "start_ticks": process_start(pid),
                "role": role,
                "unix_time": time.time(),
            }
        )
        + "\n"
    )
    _STARTED_PID = pid


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=start)


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"unavailable: {exc}"


def sample(root):
    vmstat = read("/proc/vmstat")
    prefixes = (
        "compact_",
        "thp_",
        "pgscan_",
        "pgsteal_",
        "allocstall",
        "pgmajfault",
        "pswp",
        "unevictable_",
        "nr_mlock",
        "nr_foll_pin",
    )
    result = {
        "unix_time": time.time(),
        "vmstat": [line for line in vmstat.splitlines() if line.startswith(prefixes)],
        "meminfo": read("/proc/meminfo"),
        "memory_pressure": read("/proc/pressure/memory"),
        "buddyinfo": read("/proc/buddyinfo"),
        "numa": {
            p.parent.name: read(p)
            for p in Path("/sys/devices/system/node").glob("node*/meminfo")
        },
        "processes": [],
    }
    for path in sorted(root.glob("*/pid-*.json")):
        try:
            row = json.loads(path.read_text())
            pid = row["pid"]
            if process_start(pid) != row["start_ticks"]:
                continue  # Never inspect a reused PID belonging to another run.
            proc = Path("/proc") / str(pid)
            row["status"] = read(proc / "status")
            row["threads"] = {
                p.name: read(p / "wchan") for p in (proc / "task").iterdir()
            }
            result["processes"].append(row)
        except (OSError, ValueError, KeyError):
            continue
    return result


def prepare(source, runtime, output):
    shutil.copytree(
        source / "atom",
        runtime / "atom",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    helper = Path(__file__).read_text()
    (runtime / "atom/_ci_runtime_watch.py").write_text(helper)
    init = runtime / "atom/__init__.py"
    init.write_text(
        "from atom._ci_runtime_watch import start as _ci_watch_start\n"
        "_ci_watch_start()\n" + init.read_text()
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "runtime-watch-source.py").write_text(helper)
    (output / "runtime-overlay.json").write_text(
        json.dumps(
            {
                "source": str(source),
                "runtime": str(runtime),
                "change": "start native stack watchdog at atom import; no model hooks",
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--runtime", type=Path)
    args = parser.parse_args()
    if args.source:
        prepare(args.source, args.runtime, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    log = args.output / "host-memory.jsonl"
    while True:
        if log.exists() and log.stat().st_size > 8 << 20:
            log.replace(log.with_suffix(".previous.jsonl"))
        with log.open("a") as stream:
            stream.write(json.dumps(sample(args.output)) + "\n")
        time.sleep(30)


if __name__ == "__main__":
    main()
