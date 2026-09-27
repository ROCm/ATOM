#!/usr/bin/env python3
"""Remove stale containers for this case on an exclusively allocated CI node."""

import argparse
import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path

KFD = Path("/sys/class/kfd/kfd/proc")
DRM = Path("/sys/class/drm")


def gpu_processes():
    # Spur isolates job PIDs; these are diagnostic IDs, never signal targets.
    return sorted(int(entry.name) for entry in KFD.iterdir() if entry.name.isdecimal())


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
        # Use the allocated host daemon, independent of the job PID namespace.
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


class Cleanup:
    def __init__(self, args):
        self.args = args
        self.report = {
            "job_id": args.job_id,
            "run_token": args.run_token,
            "node": args.node,
            "rank": args.rank,
            "cell_id": args.cell_id,
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
        return data

    def active_jobs(self):
        args = ["squeue"]
        if controller := os.environ.get("SPUR_CONTROLLER_ADDR"):
            args += ["--controller", controller]
        rows = command(args + ["--noheader", "--format=%i"]).split()
        if any(not row.isdecimal() for row in rows) or self.args.job_id not in rows:
            raise RuntimeError(
                "Cannot verify the current allocation in the active queue"
            )
        return set(rows)

    def stop_containers(self):
        pattern = re.compile(
            rf"atomesh-{re.escape(self.args.cell_id)}-([0-9]+)-[0-9]+"
            r"(?:-benchmark|-eval|-router|-benchmark-router|-eval-router)?"
        )
        user = f"{os.getuid()}:{os.getgid()}"
        for cid in command(
            ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}"]
        ).split():
            if not re.fullmatch(r"[0-9a-f]{64}", cid):
                raise RuntimeError("Invalid Docker container ID")
            try:
                name, owner = json.loads(
                    command(
                        [
                            "docker",
                            "inspect",
                            "--format",
                            "[{{json .Name}},{{json .Config.User}}]",
                            cid,
                        ]
                    )
                )
            except RuntimeError:
                # A concurrent natural exit is fine only if Docker confirms it.
                active = command(
                    ["docker", "ps", "-a", "--no-trunc", "--format", "{{.ID}}"]
                ).split()
                if cid not in active:
                    continue
                raise
            match = pattern.fullmatch(name.removeprefix("/"))
            if match is None or owner != user:
                continue
            old_job = match[1]
            if old_job in self.active_jobs():
                self.event(
                    "skip_active_container", container=cid, name=name, job_id=old_job
                )
                continue
            self.event(
                "stop_container",
                container=cid,
                name=name,
                job_id=old_job,
                user=owner,
            )
            command(["docker", "stop", "--time", "5", cid], timeout=15)
            command(["docker", "rm", cid])
            self.event("removed_container", container=cid, name=name)

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
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.cell_id):
            raise RuntimeError("Invalid case identity")
        self.snapshot("before")
        self.stop_containers()
        deadline = time.monotonic() + 60
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
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id")
    parser.add_argument("--run-token")
    parser.add_argument("--node")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--cell-id")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if None in (
        args.job_id,
        args.run_token,
        args.node,
        args.rank,
        args.cell_id,
        args.out,
    ):
        parser.error("cleanup requires allocation, node, rank, token and output path")
    cleanup = Cleanup(args)
    try:
        cleanup.run()
    except Exception as error:
        cleanup.event("failure", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
