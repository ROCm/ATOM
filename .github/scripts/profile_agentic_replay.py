#!/usr/bin/env python3
"""Run AIPerf with one bounded torch-profiler window in its profiling phase."""

import argparse
import json
import os
import re
import select
import signal
import subprocess
import sys
import time
from http.client import HTTPException
from pathlib import Path
from urllib.request import Request, urlopen


def profile_replay(command, url, output, seconds=30):
    """Stream replay output unchanged; always stop an attempted profiler session."""
    if not 0 < seconds <= 120:
        raise ValueError(
            "Profiler window must be greater than 0 and at most 120 seconds"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    status = {"requested_seconds": seconds, "phase": "profiling", "errors": []}
    cancelled = 0
    attempted = False
    stopped = False
    stop_attempted = False
    deadline = None
    started_at = None

    def cancel(signum, _frame):
        nonlocal cancelled
        cancelled = signum

    old_handlers = {
        sig: signal.signal(sig, cancel) for sig in (signal.SIGTERM, signal.SIGINT)
    }

    def request(endpoint, timeout):
        with urlopen(
            Request(url.rstrip("/") + endpoint, data=b"", method="POST"),
            timeout=timeout,
        ) as response:
            response.read()

    def stop():
        nonlocal stopped, stop_attempted
        if attempted and not stop_attempted:
            stop_attempted = True
            try:
                status.setdefault("stop_requested_at", time.time())
                status.setdefault(
                    "sample_seconds", time.monotonic() - started_at if started_at else 0
                )
                request("/stop_profile", 120)
                stopped = True
                status["stopped_at"] = time.time()
            except (OSError, HTTPException, ValueError) as exc:
                status["errors"].append(f"stop_profile: {exc}")

    process = None
    try:
        # Keep the caller's process group: benchmark supervision can terminate
        # AIPerf's workers together with this wrapper on a drain failure.
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0
        )
        pending = b""
        eof = False
        while not eof or process.poll() is None:
            if cancelled:
                break
            if deadline is not None and time.monotonic() >= deadline:
                stop()
                deadline = None
                if not stopped:
                    break
            readable, _, _ = select.select(
                [process.stdout] if not eof else [], [], [], 0.2
            )
            if not readable:
                continue
            chunk = os.read(process.stdout.fileno(), 65536)
            if not chunk:
                eof = True
                continue
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            pending += chunk
            lines = pending.split(b"\n")
            pending = lines.pop()
            for line in lines:
                clean = re.sub(r"\x1b\[[0-9;]*m", "", line.decode(errors="replace"))
                # "System is PROFILING" occurs before warmup; only this phase
                # event (AIPerf runner.py) identifies the measured replay.
                if not attempted and re.search(
                    r"\bPhase \S+ \(profiling\) started\b", clean
                ):
                    attempted = True
                    status["start_attempted_at"] = time.time()
                    try:
                        request("/start_profile", 10)
                        started_at = time.monotonic()
                        status["started_at"] = time.time()
                        deadline = started_at + seconds
                    except (OSError, HTTPException, ValueError) as exc:
                        status["errors"].append(f"start_profile: {exc}")
                        break
            if status["errors"]:
                break
    finally:
        stop()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            status["client_exit_code"] = process.wait()
            process.stdout.close()
        if not attempted:
            status["errors"].append("AIPerf ended before a profiling-phase start event")
        status["cancelled_signal"] = cancelled or None
        status["complete"] = bool(
            started_at
            and stopped
            and not status["errors"]
            and not cancelled
            and status.get("client_exit_code") == 0
        )
        output.write_text(json.dumps(status, indent=2) + "\n")
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
    if cancelled:
        return 128 + cancelled
    code = status.get("client_exit_code", 1)
    if code:
        return code if code > 0 else 128 - code
    return 0 if status["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("an AIPerf command is required")
    return profile_replay(command, args.url, args.output, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
