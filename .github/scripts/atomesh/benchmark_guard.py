#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Stop a CI benchmark on fatal worker errors or sustained inference inactivity."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

FATAL = re.compile(
    r"Watchdog caught collective operation timeout|"
    r"c10::DistBackendError|HIP error:|CUDA error:|"
    r"Engine core process .* (?:died|exited)|"
    r"Torch\.OutOfMemoryError|torch\.OutOfMemoryError"
)
PROGRESS = re.compile(
    r"Scheduled prefill batch:|Engine Core: output send|"
    r"\[OFFLOAD-SAVE-PROF\]"
)
COUNTERS = re.compile(r"returned=([\d,]+)/[\d,]+.*sent=([\d,]+).*in_flight=([\d,]+)")
PHASE = re.compile(r"Phase \S+ \([^)]+\) (started|sending complete|complete)\b")
DRAIN_COUNTERS = re.compile(
    r"sent=([\d,]+), completed=([\d,]+), (?:cancelled=[\d,]+, )?in_flight=([\d,]+)"
)


class Guard:
    def __init__(self, logs, client, timeout):
        self.logs = logs
        self.client = client
        self.timeout = timeout
        self.positions = {}
        self.partial = {}
        self.last_progress = time.monotonic()
        self.returned = None
        self.in_flight = 0
        self.phase_active = True

    def lines(self, path):
        try:
            with path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                position = self.positions.get(path, 0)
                if size < position:
                    position = 0
                    self.partial.pop(path, None)
                stream.seek(position)
                data = stream.read(4 << 20)
                self.positions[path] = stream.tell()
            parts = (self.partial.get(path, b"") + data).split(b"\n")
            self.partial[path] = parts.pop()[-65536:]
            return [p.decode("utf-8", "replace") for p in parts]
        except OSError:
            return []

    def check(self, now=None):
        now = time.monotonic() if now is None else now
        for path in self.logs:
            for line in self.lines(path):
                if FATAL.search(line):
                    return {
                        "reason": "fatal_worker_error",
                        "file": str(path),
                        "line": line[:2048],
                    }
                if PROGRESS.search(line):
                    self.last_progress = now
        for line in self.lines(self.client):
            phase = PHASE.search(line)
            if phase:
                event = phase.group(1)
                if event == "started":
                    self.phase_active = True
                    self.returned, self.in_flight = None, 0
                    self.last_progress = now
                elif event == "complete":
                    # Exporting records/metrics needs no inference progress.
                    # A subsequent phase start re-arms the inactivity check.
                    self.phase_active = False
                    self.in_flight = 0
            if not self.phase_active:
                continue
            match = COUNTERS.search(line)
            drain = DRAIN_COUNTERS.search(line)
            if match or drain:
                values = [
                    int(value.replace(",", "")) for value in (match or drain).groups()
                ]
                returned, _sent, in_flight = (
                    values if match else (values[1], values[0], values[2])
                )
                if self.returned != returned:
                    self.last_progress = now
                self.returned, self.in_flight = returned, in_flight
        if (
            self.phase_active
            and self.in_flight
            and now - self.last_progress >= self.timeout
        ):
            return {
                "reason": "inference_stalled",
                "seconds": now - self.last_progress,
                "returned": self.returned,
                "in_flight": self.in_flight,
            }
        return None


def run(command, guard, output):
    child = subprocess.Popen(command, start_new_session=True)
    received = None
    failure = None
    stopped = None

    def stop(signum, _frame):
        nonlocal received, stopped
        received = signum
        stopped = time.monotonic()
        if child.poll() is None:
            os.killpg(child.pid, signum)

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        while child.poll() is None:
            if stopped is None:
                failure = guard.check()
                if failure:
                    failure["unix_time"] = time.time()
                    output.write_text(json.dumps(failure, indent=2) + "\n")
                    print("[benchmark-guard][FAIL] " + json.dumps(failure), flush=True)
                    stop(signal.SIGTERM, None)
            elif time.monotonic() - stopped > 120:
                os.killpg(child.pid, signal.SIGKILL)
            time.sleep(1)
        return 86 if failure else (128 + received if received else child.returncode)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-log", action="append", type=Path, default=[])
    parser.add_argument("--client-log", type=Path, required=True)
    parser.add_argument("--stall-timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.stall_timeout <= 0:
        parser.error("command and positive stall timeout are required")
    return run(
        command,
        Guard(args.server_log, args.client_log, args.stall_timeout),
        args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
