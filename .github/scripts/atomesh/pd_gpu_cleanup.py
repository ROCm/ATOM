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


def remaining_process_owners(pids, *, seconds=10, max_containers=64):
    """Inspect host Docker membership; never interpret job-local /proc PIDs."""
    result = {
        "gpu_pids": sorted(set(pids)),
        "matches": [],
        "errors": [],
        "containers_checked": 0,
        "limited": False,
        "basis": "Host Docker top PID membership with stable container init identity",
        "scope": "Read-only diagnostic snapshot, not cleanup authorization",
    }
    deadline = time.monotonic() + seconds

    def read(args):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Ownership diagnostic time budget reached")
        return command(args, timeout=min(3, remaining))

    def inspect(cid):
        return json.loads(
            read(
                [
                    "docker",
                    "inspect",
                    "--format",
                    (
                        "[{{json .Name}},{{json .Config.User}},{{json .State.Pid}},"
                        "{{json .State.StartedAt}}]"
                    ),
                    cid,
                ]
            )
        )

    unmatched = set(pids)
    try:
        containers = read(["docker", "ps", "--no-trunc", "--format", "{{.ID}}"]).split()
        result["limited"] = len(containers) > max_containers
        for cid in containers[:max_containers]:
            if not unmatched:
                break
            if not re.fullmatch(r"[0-9a-f]{64}", cid):
                raise ValueError("Invalid Docker container ID")
            try:
                identity = inspect(cid)
                name, owner, init_pid, started_at = identity
                output = read(["docker", "top", cid, "-eo", "pid,ppid,uid,gid"])
                rows = output.splitlines()
                if not rows or rows[0].split() != ["PID", "PPID", "UID", "GID"]:
                    raise ValueError("Unexpected Docker top numeric columns")
                if len(rows) > 4097:
                    result["limited"] = True
                processes = [list(map(int, row.split())) for row in rows[1:4097]]
                if any(len(row) != 4 for row in processes):
                    raise ValueError("Malformed Docker top row")
                result["containers_checked"] += 1
                # A host init PID anchors the namespace; restarts invalidate it.
                if not init_pid or init_pid not in {row[0] for row in processes}:
                    raise ValueError("Docker top does not contain the host init PID")
                if inspect(cid) != identity:
                    raise ValueError("Container identity changed during observation")
                matches = [row for row in processes if row[0] in unmatched]
                if matches:
                    result["matches"].append(
                        {
                            "container": cid,
                            "name": name,
                            "configured_user": owner,
                            "host_init_pid": init_pid,
                            "started_at": started_at,
                            "processes": [
                                dict(zip(("pid", "ppid", "uid", "gid"), row))
                                for row in matches
                            ],
                        }
                    )
                    unmatched.difference_update(row[0] for row in matches)
            except (
                OSError,
                RuntimeError,
                ValueError,
                subprocess.TimeoutExpired,
            ) as error:
                result["errors"].append({"container": cid, "error": str(error)[:500]})
                if time.monotonic() >= deadline:
                    result["limited"] = True
                    break
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as error:
        result["errors"].append({"error": str(error)[:500]})
        result["limited"] = True
    result["unmatched_pids"] = sorted(unmatched)
    return result


class Cleanup:
    def __init__(self, args):
        self.args = args
        self.targets = None
        if args.targets:
            manifest = json.loads(args.targets.read_text())
            if manifest.get("schema_version") != 1:
                raise ValueError("Unsupported cleanup target manifest")
            self.targets = {}
            terminal = {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "NODE_FAIL",
                "OUT_OF_MEMORY",
            }
            for target in manifest["targets"]:
                cell, job = target["cell_id"], target["slurm_job_id"]
                if (
                    not re.fullmatch(r"[a-zA-Z0-9_.-]+", cell)
                    or not re.fullmatch(r"[0-9]+", job)
                    or target["terminal_state"] not in terminal
                    or job == args.job_id
                ):
                    raise ValueError(
                        "Cleanup targets must identify archived terminal jobs"
                    )
                self.targets[(cell, job)] = target
            if not self.targets:
                raise ValueError("Explicit cleanup target manifest is empty")
        self.report = {
            "job_id": args.job_id,
            "run_token": args.run_token,
            "node": args.node,
            "rank": args.rank,
            "cell_id": args.cell_id,
            "required_free_ratio": args.required_free_ratio,
            "targets": (
                list(self.targets.values()) if self.targets is not None else None
            ),
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
        suffix = r"(?:-benchmark|-eval|-router|-benchmark-router|-eval-router)?"
        if self.targets is None:
            patterns = [
                re.compile(
                    rf"atomesh-{re.escape(self.args.cell_id)}-([0-9]+)-[0-9]+" + suffix
                )
            ]
        else:
            patterns = [
                re.compile(rf"atomesh-{re.escape(cell)}-({job})-[0-9]+" + suffix)
                for cell, job in self.targets
            ]
        user = f"{os.getuid()}:{os.getgid()}"
        self.active_jobs()  # Validate the current allocation before any cleanup.
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
            matches = [
                pattern.fullmatch(name.removeprefix("/")) for pattern in patterns
            ]
            match = next((match for match in matches if match is not None), None)
            if match is None:
                continue
            if owner != user:
                self.event("skip_different_owner", container=cid, name=name, user=owner)
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
                p["free"] >= args.required_free_ratio * p["total"]
                for p in current["memory"]
            ):
                self.report["passed"] = True
                self.event("complete")
                return
            if time.monotonic() >= deadline:
                if current["processes"]:
                    try:
                        ownership = remaining_process_owners(current["processes"])
                    except (
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                        subprocess.TimeoutExpired,
                    ) as error:
                        ownership = {"diagnostic_error": type(error).__name__}
                    self.event("remaining_gpu_process_owners", **ownership)
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
    parser.add_argument(
        "--targets",
        type=Path,
        help="Optional exact cell/job allowlist from terminal archives",
    )
    parser.add_argument("--required-free-ratio", type=float, default=0.88)
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
    if not 0 < args.required_free_ratio < 1:
        parser.error("required-free-ratio must be between zero and one")
    cleanup = Cleanup(args)
    try:
        cleanup.run()
    except Exception as error:
        cleanup.event("failure", error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
