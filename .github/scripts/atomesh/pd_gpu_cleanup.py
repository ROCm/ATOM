#!/usr/bin/env python3
"""Clear GPU users on an explicitly allocated exclusive AMD CI node."""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

KFD = Path("/sys/class/kfd/kfd/proc")
DRM = Path("/sys/class/drm")
PROC = Path("/proc")


def identity(pid):
    try:
        text = (PROC / str(pid) / "stat").read_text()
    except FileNotFoundError:
        return None
    tail = text.rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "starttime": int(tail[19]),
        "comm": text.split("(", 1)[1].rsplit(")", 1)[0],
    }


def gpu_processes():
    # KFD exposes host process IDs. This program is invoked by the node worker,
    # before Docker, so pidfd and /proc use that same host PID namespace.
    result = []
    for entry in KFD.iterdir():
        if entry.name.isdecimal():
            row = identity(int(entry.name))
            if row is not None:
                result.append(row)
    return sorted(result, key=lambda row: row["pid"])


def memory():
    devices = {}
    for card in DRM.glob("card[0-9]*"):
        if not re.fullmatch(r"card[0-9]+", card.name):
            continue
        device = (card / "device").resolve()
        if not (device / "mem_info_vram_total").exists():
            continue
        if (device / "vendor").read_text().strip().lower() != "0x1002":
            continue
        total = int((device / "mem_info_vram_total").read_text())
        used = int((device / "mem_info_vram_used").read_text())
        if total <= 0 or not 0 <= used <= total:
            raise RuntimeError(f"Invalid VRAM accounting for {device.name}")
        devices[str(device)] = {
            "pci": device.name,
            "total": total,
            "used": used,
            "free": total - used,
        }
    if len(devices) != 8:
        raise RuntimeError(
            f"Exclusive K3 cleanup requires exactly 8 AMD GPUs, found {len(devices)}"
        )
    return sorted(devices.values(), key=lambda row: row["pci"])


def command(args, timeout=10):
    env = os.environ.copy()
    if args[0] == "docker":
        # KFD and /proc describe this host, never a remote Docker context.
        for key in (
            "DOCKER_HOST",
            "DOCKER_CONTEXT",
            "DOCKER_TLS_VERIFY",
            "DOCKER_CERT_PATH",
        ):
            env.pop(key, None)
        args = ["docker", "--host", "unix:///var/run/docker.sock", *args[1:]]
    proc = subprocess.run(
        args, env=env, text=True, capture_output=True, timeout=timeout, check=False
    )
    if proc.returncode:
        raise RuntimeError(
            f"{args[0]} failed rc={proc.returncode}: {proc.stderr[-1500:]}"
        )
    return proc.stdout


def protected_pids():
    result = {1}
    pid = os.getpid()
    while pid > 1 and pid not in result:
        result.add(pid)
        text = (PROC / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        pid = int(text[1])
    return result


def signal_gpu_process(pid, starttime, sig):
    if pid <= 1 or pid in protected_pids():
        raise RuntimeError("Refusing to signal cleanup process or its ancestors")
    before = identity(pid)
    if (
        before is None
        or before["starttime"] != starttime
        or not (KFD / str(pid)).exists()
    ):
        return "gone_or_replaced"
    try:
        fd = os.pidfd_open(pid)
    except ProcessLookupError:
        return "gone"
    try:
        after = identity(pid)
        if (
            after is None
            or after["starttime"] != starttime
            or not (KFD / str(pid)).exists()
        ):
            return "gone_or_replaced"
        try:
            signal.pidfd_send_signal(fd, sig)
        except ProcessLookupError:
            return "gone"
    finally:
        os.close(fd)
    return "signalled"


class Cleanup:
    def __init__(self, args):
        self.args = args
        self.report = {
            "job_id": args.job_id,
            "run_token": args.run_token,
            "node": args.node,
            "rank": args.rank,
            "required_free_ratio": 0.88,
            "docker_host": "unix:///var/run/docker.sock",
            "events": [],
            "passed": False,
        }
        self.started = time.monotonic()

    def save(self):
        self.args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.args.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.report, indent=2) + "\n")
        tmp.replace(self.args.out)

    def event(self, kind, **data):
        row = {"kind": kind, "elapsed": time.monotonic() - self.started, **data}
        self.report["events"].append(row)
        self.save()
        print("GPU_CLEANUP " + json.dumps(row), flush=True)

    def snapshot(self, stage):
        data = {"processes": gpu_processes(), "memory": memory()}
        self.event(stage, **data)
        if any(row["pid"] in protected_pids() for row in data["processes"]):
            raise RuntimeError("A protected supervisor process uses the GPU")
        return data

    def stop_containers(self, processes):
        targets = {p["pid"] for p in processes}
        if not targets:
            return
        for cid in command(
            ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"]
        ).split():
            if not re.fullmatch(r"[0-9a-f]{64}", cid):
                raise RuntimeError("Invalid Docker container ID")
            try:
                text = command(["docker", "top", cid, "-eo", "pid"])
            except RuntimeError:
                # A concurrent natural exit is fine only if Docker confirms it.
                active = command(
                    ["docker", "ps", "--no-trunc", "--format", "{{.ID}}"]
                ).split()
                if cid not in active:
                    continue
                raise
            values = text.split()
            if (
                not values
                or values[0] != "PID"
                or any(not p.isdecimal() for p in values[1:])
            ):
                raise RuntimeError("Invalid Docker host PID listing")
            pids = {int(p) for p in values[1:]}
            live_targets = {
                row["pid"]
                for row in processes
                if row["pid"] in pids
                and identity(row["pid"]) == row
                and (KFD / str(row["pid"])).exists()
            }
            if not live_targets:
                continue
            if pids & protected_pids():
                raise RuntimeError("GPU container includes a protected supervisor")
            info = json.loads(
                command(["docker", "inspect", "--format", "{{json .Name}}", cid])
            )
            self.event(
                "stop_container",
                container=cid,
                name=info,
                gpu_pids=sorted(live_targets),
            )
            command(["docker", "stop", "--time", "5", cid], timeout=15)

    def signal_remaining(self, sig):
        for row in gpu_processes():
            try:
                outcome = signal_gpu_process(row["pid"], row["starttime"], sig)
            except PermissionError:
                # Uses only existing host sudo policy; no privileged container.
                command(
                    [
                        "sudo",
                        "-n",
                        "--",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--signal-pid",
                        str(row["pid"]),
                        "--starttime",
                        str(row["starttime"]),
                        "--signal-name",
                        signal.Signals(sig).name,
                    ],
                    timeout=10,
                )
                outcome = "host_sudo_signal"
            self.event("signal", **row, signal=signal.Signals(sig).name, result=outcome)

    def run(self):
        args = self.args
        if socket.gethostname().split(".")[0] != args.node:
            raise RuntimeError("Allocated node does not match this host")
        job_ids = [
            os.environ[k] for k in ("SLURM_JOB_ID", "SPUR_JOB_ID") if os.environ.get(k)
        ]
        if not job_ids or any(value != args.job_id for value in job_ids):
            raise RuntimeError("Cleanup requires the current Slurm/Spur allocation")
        if os.environ.get("ATOMESH_EXCLUSIVE_GPU_CLEANUP") != "1":
            raise RuntimeError("Cleanup must be invoked by the exclusive CI submitter")
        if not re.fullmatch(r"[0-9a-f]{32}", args.run_token):
            raise RuntimeError("Missing cleanup run identity")
        initial = self.snapshot("before")
        self.stop_containers(initial["processes"])
        self.signal_remaining(signal.SIGTERM)
        deadline = time.monotonic() + 60
        kill_at = time.monotonic() + 5
        while True:
            current = self.snapshot("check")
            if not current["processes"] and all(
                p["free"] >= 0.88 * p["total"] for p in current["memory"]
            ):
                self.report["passed"] = True
                self.event("complete")
                return
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "GPU processes or insufficient free VRAM remain after cleanup"
                )
            if time.monotonic() >= kill_at:
                self.stop_containers(current["processes"])
                self.signal_remaining(signal.SIGKILL)
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--run-token")
    parser.add_argument("--node")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--signal-pid", type=int)
    parser.add_argument("--starttime", type=int)
    parser.add_argument("--signal-name", choices=("SIGTERM", "SIGKILL"))
    args = parser.parse_args()
    if args.signal_pid is not None:
        if args.starttime is None or args.signal_name is None:
            parser.error("signal helper requires an immutable process identity")
        print(
            signal_gpu_process(
                args.signal_pid, args.starttime, getattr(signal, args.signal_name)
            )
        )
        return
    if None in (args.job_id, args.run_token, args.node, args.rank, args.out):
        parser.error("cleanup requires allocation, node, rank, token and output path")
    cleanup = Cleanup(args)
    try:
        cleanup.run()
    except Exception as error:
        cleanup.event("failure", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
