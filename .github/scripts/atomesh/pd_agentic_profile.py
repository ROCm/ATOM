"""Capture bounded P/D torch traces while the real AIPerf replay continues."""

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PHASE_START = re.compile(r"Phase profiling \(profiling\) started", re.IGNORECASE)


def inspect_trace(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        data = json.load(stream)
    events = data["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    cpu_count = sum(event.get("cat") == "cpu_op" for event in events)
    if not kernels or not cpu_count:
        raise ValueError(f"Missing GPU kernels or CPU operations: {path}")
    rank = re.search(r"rank(\d+)\.", path.name)
    if rank is None:
        raise ValueError(f"Missing worker rank: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "file": str(path),
        "rank": int(rank[1]),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "events": len(events),
        "cpu_ops": cpu_count,
        "gpu_kernels": len(kernels),
        "first_kernel_ts_us": min(event["ts"] for event in kernels),
        "last_kernel_end_us": max(
            event["ts"] + event.get("dur", 0) for event in kernels
        ),
        "baseTimeNanoseconds": data.get("baseTimeNanoseconds"),
    }


class Capture:
    def __init__(self, args):
        self.args = args
        self.endpoints = {"prefill": args.prefill, "decode": args.decode}
        self.lock = threading.Lock()
        self.args.output.mkdir(parents=True, exist_ok=True)
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def event(self, kind, **fields):
        record = {
            "event": kind,
            "unix_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        with self.lock, (self.args.output / "control.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print("[pd-profile] " + json.dumps(record), flush=True)

    def call(self, role, path, window, timeout=900):
        self.event("http_begin", role=role, path=path, window=window)
        try:
            req = urllib.request.Request(
                self.endpoints[role] + path, data=b"", method="POST"
            )
            with self.opener.open(req, timeout=timeout) as response:
                response.read()  # vLLM returns an empty body on success.
                status = response.status
            self.event("http_end", role=role, path=path, window=window, status=status)
        except Exception as exc:
            self.event(
                "http_error", role=role, path=path, window=window, error=repr(exc)
            )
            raise

    def both(self, path, window):
        errors = []
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(self.call, role, path, window) for role in self.endpoints
            ]
            for future in futures:
                try:
                    future.result()
                except (OSError, RuntimeError) as exc:
                    errors.append(repr(exc))
        if errors:
            raise RuntimeError(f"{path}: {errors}")

    def metrics(self, tag):
        for role, url in self.endpoints.items():
            start = time.time_ns()
            with self.opener.open(url + "/metrics", timeout=30) as response:
                body = response.read()
                server_date = response.headers.get("Date")
            (self.args.output / f"{tag}-{role}.metrics.txt").write_bytes(body)
            self.event(
                "metrics",
                tag=tag,
                role=role,
                begin_unix_ns=start,
                server_http_date=server_date,
            )

    def window(self, number, child):
        self.metrics(f"window-{number}-before")
        self.event("window_begin", window=number)
        before = set(self.args.traces.glob("*/*.pt.trace.json*"))
        try:
            self.both("/start_profile", number)
            deadline = time.monotonic() + self.args.seconds
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    raise RuntimeError("AIPerf ended during the capture window")
                time.sleep(min(0.25, max(0, deadline - time.monotonic())))
        finally:
            # Stop both even after a failed/timed-out start: its remote state is unknown.
            self.both("/stop_profile", number)
        self.event("window_exported", window=number)
        self.metrics(f"window-{number}-after")
        inventory = []
        for role in self.endpoints:
            paths = sorted(
                set((self.args.traces / role).glob("*.pt.trace.json*")) - before
            )
            records = [inspect_trace(path) for path in paths]
            if len(records) != self.args.ranks or {r["rank"] for r in records} != set(
                range(self.args.ranks)
            ):
                raise ValueError(
                    f"{role}: expected {self.args.ranks} unique GPU ranks, got {records}"
                )
            dest = self.args.traces / f"window-{number}" / role
            dest.mkdir(parents=True, exist_ok=False)
            for path, record in zip(paths, records):
                moved = dest / path.name
                path.rename(moved)
                record.update(file=str(moved), role=role, window=number)
                inventory.append(record)
        (self.args.output / f"window-{number}-inventory.json").write_text(
            json.dumps(inventory, indent=2) + "\n"
        )
        self.event(
            "window_validated",
            window=number,
            files=len(inventory),
            gpu_kernels=sum(record["gpu_kernels"] for record in inventory),
        )
        return inventory

    def run(self, command):
        self.event("workload_start", argv=command)
        with (self.args.output / "workload-console.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            errors, windows = [], []
            try:
                deadline = time.monotonic() + self.args.phase_timeout
                phase_start = None
                while child.poll() is None and time.monotonic() < deadline:
                    # Check the phase-specific marker, not the earlier system PROFILING state.
                    for path in (
                        self.args.aiperf_log,
                        self.args.output / "workload-console.log",
                    ):
                        if path.exists() and PHASE_START.search(
                            path.read_text(errors="replace")
                        ):
                            phase_start = time.monotonic()
                            self.event("agentic_phase_observed", source=str(path))
                            break
                    if phase_start is not None:
                        break
                    time.sleep(1)
                if phase_start is None:
                    raise RuntimeError(
                        "AIPerf never reached the post-warmup profiling phase"
                    )
                for number, offset in enumerate(self.args.offsets, 1):
                    while time.monotonic() < phase_start + offset:
                        if child.poll() is not None:
                            raise RuntimeError(
                                "AIPerf ended before all windows were captured"
                            )
                        time.sleep(
                            min(1, max(0, phase_start + offset - time.monotonic()))
                        )
                    windows.extend(self.window(number, child))
            except (OSError, ValueError, RuntimeError, EOFError, KeyError) as exc:
                errors.append(repr(exc))
                self.event("capture_failed", error=repr(exc))
            finally:
                # Keep AIPerf's normal result export/draining; bound failed workloads.
                try:
                    rc = child.wait(timeout=self.args.finish_timeout)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    try:
                        rc = child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        rc = child.wait()
                    errors.append("AIPerf finish timeout")
        result = {
            "workload_rc": rc,
            "errors": errors,
            "traces": windows,
            "complete": not errors
            and rc == 0
            and len(windows) == 2 * self.args.ranks * len(self.args.offsets),
            "note": "Profiling overhead affects these results; this is not an unprofiled throughput acceptance run.",
        }
        (self.args.output / "result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        self.event(
            "capture_complete",
            complete=result["complete"],
            workload_rc=rc,
            errors=errors,
        )
        return 0 if result["complete"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--aiperf-log", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--offsets", type=float, nargs="+", default=[60, 240])
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--phase-timeout", type=float, default=7200)
    parser.add_argument("--finish-timeout", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.seconds <= 0 or args.ranks < 1:
        parser.error(
            "a workload command, positive capture seconds, and rank count are required"
        )
    raise SystemExit(Capture(args).run(command))


if __name__ == "__main__":
    main()
