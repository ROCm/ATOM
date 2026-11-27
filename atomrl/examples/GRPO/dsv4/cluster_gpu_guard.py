#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Keep reserved RL nodes free from external GPU workloads."""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

ALLOWED_CONTAINERS = frozenset({"dsv4-rl", "node-exporter.service"})
TARGET_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"/home/[^/]+/(?:[^/]+/)*\.?actions-runner/",
        r"/home/[^/]+/frameworks-qa-ci-runner/",
        r"\brocm_test_executor\.py\b",
        r"/ROCmTest/",
        r"\brun_mad\.py\b",
        r"\bmadengine\b",
        r"(?:^|/)\bgpu_stress\b",
        r"\bLSTM_pytorch\.py\b",
        r"\bcontainer_ci-pyt_lstm\b",
    )
)


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    ppid: int
    cmdline: str


def foreign_containers(names: list[str]) -> list[str]:
    return [name for name in names if name not in ALLOWED_CONTAINERS]


def find_target_tree(processes: dict[int, ProcessInfo]) -> set[int]:
    selected = {
        process.pid
        for process in processes.values()
        if any(pattern.search(process.cmdline) for pattern in TARGET_PATTERNS)
    }
    while True:
        descendants = {
            process.pid for process in processes.values() if process.ppid in selected
        }
        expanded = selected | descendants
        if expanded == selected:
            return selected
        selected = expanded


def read_processes(proc_root: Path = Path("/proc")) -> dict[int, ProcessInfo]:
    processes: dict[int, ProcessInfo] = {}
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = dict(
                line.split(":", 1)
                for line in (entry / "status").read_text(errors="replace").splitlines()
                if ":" in line
            )
            cmdline = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
            if not cmdline:
                cmdline = fields.get("Name", "").strip()
            pid = int(entry.name)
            processes[pid] = ProcessInfo(
                pid=pid,
                ppid=int(fields["PPid"].strip()),
                cmdline=cmdline,
            )
        except (
            FileNotFoundError,
            KeyError,
            PermissionError,
            ProcessLookupError,
            ValueError,
        ):
            continue
    return processes


def running_containers() -> list[str]:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def enforce(grace_seconds: float, dry_run: bool) -> None:
    for name in foreign_containers(running_containers()):
        logger.warning(
            "%s foreign container=%s", "would stop" if dry_run else "stopping", name
        )
        if not dry_run:
            subprocess.run(
                ["docker", "stop", "-t", str(max(0, int(grace_seconds))), name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    processes = read_processes()
    targets = find_target_tree(processes)
    targets.discard(os.getpid())
    for pid in sorted(targets):
        process = processes.get(pid)
        logger.warning(
            "%s external process pid=%d command=%s",
            "would terminate" if dry_run else "terminating",
            pid,
            process.cmdline if process else "<exited>",
        )
        if not dry_run:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    if dry_run or not targets:
        return
    time.sleep(grace_seconds)
    for pid in targets:
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--grace-seconds", type=float, default=1.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s cluster-gpu-guard: %(message)s",
    )
    logger.info("allowing containers=%s", sorted(ALLOWED_CONTAINERS))
    while True:
        try:
            enforce(args.grace_seconds, args.dry_run)
        except Exception:
            logger.exception("watchdog iteration failed")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
