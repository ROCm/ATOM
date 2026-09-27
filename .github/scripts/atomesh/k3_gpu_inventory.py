#!/usr/bin/env python3
"""Read GPU ownership and memory on the configured CI pool, without allocation."""

import concurrent.futures
import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


def read(path):
    try:
        return path.read_text().strip()
    except OSError as exc:
        return {"error": str(exc)}


def command(args):
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=False
        )
        return {
            "rc": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def worker():
    result = {"host": socket.gethostname(), "uid": os.getuid(), "memory": []}
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if not re.fullmatch(r"card[0-9]+", card.name):
            continue
        device = card / "device"
        if read(device / "vendor") != "0x1002":
            continue
        result["memory"].append(
            {
                "card": card.name,
                "pci": device.resolve().name,
                "total": read(device / "mem_info_vram_total"),
                "used": read(device / "mem_info_vram_used"),
            }
        )
    kfd = Path("/sys/class/kfd/kfd/proc")
    try:
        result["kfd_pids"] = sorted(p.name for p in kfd.iterdir())
    except OSError as exc:
        result["kfd_error"] = str(exc)
    result["namespaces"] = {}
    for pid in ("1", "self"):
        for namespace in ("pid", "mnt", "user"):
            try:
                value = os.readlink(f"/proc/{pid}/ns/{namespace}")
            except OSError as exc:
                value = str(exc)
            result["namespaces"][f"{pid}/{namespace}"] = value
    result["gpu_fds"] = []
    denied = 0
    for proc in Path("/proc").iterdir():
        if not proc.name.isdecimal():
            continue
        try:
            fds = list((proc / "fd").iterdir())
        except PermissionError:
            denied += 1
            continue
        except FileNotFoundError:
            continue
        devices = set()
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target == "/dev/kfd" or target.startswith("/dev/dri/"):
                devices.add(target)
        if devices:
            result["gpu_fds"].append(
                {
                    "pid": int(proc.name),
                    "comm": read(proc / "comm"),
                    "devices": sorted(devices),
                }
            )
    result["fd_permission_denied_processes"] = denied
    docker = ["docker", "--host", "unix:///var/run/docker.sock"]
    result["containers"] = command(
        docker + ["ps", "--no-trunc", "--format", "{{.ID}} {{.Names}} {{.Status}}"]
    )
    result["old_job_containers"] = []
    for line in result["containers"].get("stdout", "").splitlines():
        fields = line.split()
        if len(fields) < 2 or not re.fullmatch(r"[0-9a-f]{64}", fields[0]):
            continue
        if not re.fullmatch(
            r"atomesh-.*-4640-[01](-benchmark|-eval|-router|-benchmark-router|-eval-router)?",
            fields[1],
        ):
            continue
        result["old_job_containers"].append(
            {
                "name": fields[1],
                "state": command(
                    docker + ["inspect", "--format", "{{json .State}}", fields[0]]
                ),
                "pids": command(
                    docker + ["top", fields[0], "-eo", "pid,ppid,stat,comm"]
                ),
            }
        )
    result["kernel_gpu_events"] = command(
        [
            "journalctl",
            "-k",
            "--since",
            "2026-09-27 15:25:00 UTC",
            "--no-pager",
            "-n",
            "300",
        ]
    )
    events = result["kernel_gpu_events"]
    events["stdout"] = "\n".join(
        line
        for line in events.get("stdout", "").splitlines()
        if re.search(
            r"amdgpu|kfd|gpu|oom|out of memory|killed process|permission|no journal",
            line,
            re.IGNORECASE,
        )
    )
    print(json.dumps(result, indent=2))


def main():
    nodes = os.environ["ATOMESH_NODE_POOL"].split(",")
    if not nodes or any(
        not re.fullmatch(r"pit2-p03-g[0-9]{2}", node) for node in nodes
    ):
        raise ValueError("Invalid configured node pool")
    out = Path("inspection/gpu-inventory")
    out.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).read_text()

    def inspect(node):
        try:
            proc = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "StrictHostKeyChecking=accept-new",
                    "-o",
                    "ConnectTimeout=4",
                    node,
                    "timeout -k 2s 25s python3 - --worker",
                ],
                input=script,
                capture_output=True,
                text=True,
                timeout=35,
                check=False,
            )
            row = {
                "node": node,
                "rc": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except (OSError, subprocess.TimeoutExpired) as exc:
            row = {"node": node, "error": str(exc)}
        (out / f"{node}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(node, row.get("rc", row.get("error")), flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(inspect, nodes))


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        worker()
    else:
        main()
