"""Three bounded single-role (P or D) captures inside the unchanged full C48 replay."""

import argparse
import json
import subprocess
import time
from pathlib import Path

from pd_agentic_profile import PHASE_START, Capture


class RoleCapture(Capture):
    def __init__(self, args):
        super().__init__(args)
        # Base methods operate on the overridden endpoint map.
        self.endpoints = {role: url for role, url in self.endpoints.items() if url}

    def run(self, command):
        self.event("workload_start", argv=command)
        windows, errors = [], []
        with (self.args.output / "workload-console.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + self.args.phase_timeout
                phase_start = None
                while child.poll() is None and time.monotonic() < deadline:
                    for path in (
                        self.args.aiperf_log,
                        self.args.output / "workload-console.log",
                    ):
                        if path.exists() and PHASE_START.search(
                            path.read_text(errors="replace")
                        ):
                            phase_start = time.monotonic()
                            plan = {
                                "phase_start_unix_ns": time.time_ns(),
                                "phase_seconds": 3930,
                                "detail_windows": [
                                    [offset - 60, offset + 120]
                                    for offset in self.args.offsets
                                ],
                            }
                            self.args.timeline.mkdir(parents=True, exist_ok=True)
                            temporary = self.args.timeline / "windows.tmp"
                            temporary.write_text(json.dumps(plan) + "\n")
                            temporary.replace(self.args.timeline / "windows.json")
                            self.event(
                                "agentic_phase_observed",
                                source=str(path),
                                timeline=plan,
                            )
                            break
                    if phase_start is not None:
                        break
                    time.sleep(1)
                if phase_start is None:
                    raise RuntimeError(
                        "AIPerf did not reach the formal profiling phase"
                    )
                for number, offset in enumerate(self.args.offsets, 1):
                    while time.monotonic() < phase_start + offset:
                        if child.poll() is not None:
                            raise RuntimeError("AIPerf ended before all trace windows")
                        time.sleep(
                            min(1, max(0, phase_start + offset - time.monotonic()))
                        )
                    windows.extend(self.window(number, child))
            except (OSError, ValueError, RuntimeError, EOFError, KeyError) as exc:
                errors.append(repr(exc))
                self.event("capture_failed", error=repr(exc))
            finally:
                # Preserve the full workload even when capture fails.
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
            and len(windows) == self.args.ranks * len(self.args.offsets),
            "note": "Single-role profiler windows and export perturb latency; use the unprofiled control for performance.",
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
    role = parser.add_mutually_exclusive_group(required=True)
    role.add_argument("--prefill")
    role.add_argument("--decode")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--aiperf-log", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--offsets", type=float, nargs="+", default=[600, 1500, 3000])
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--phase-timeout", type=float, default=7200)
    parser.add_argument("--finish-timeout", type=float, default=5400)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0 < args.seconds <= 20 or args.ranks != 8:
        parser.error("Require the full workload, <=20s captures and eight ranks")
    if args.offsets != sorted(set(args.offsets)) or not all(
        60 <= x <= 3300 for x in args.offsets
    ):
        parser.error("Offsets must be unique, ordered and within the full send window")
    raise SystemExit(RoleCapture(args).run(command))


if __name__ == "__main__":
    main()
