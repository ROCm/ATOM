# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Machine-wide GPU and port leases.

Choosing "free" GPUs by looking at sysfs and then launching is check-then-act:
two agents (two users, or two swarm workers) can both see GPU 4 free and both
start on it, and a server only shows up in sysfs after it has loaded weights.
A lease closes that window. Every launch first takes a lease on its GPUs and
port, atomically, under one lock shared by every user on the machine.

Leases live in one JSON file in a shared directory (default
``/tmp/atom-agent-leases``, override with ``ATOM_AGENT_LEASE_DIR``). The file
and its lock are world-writable on purpose: this is coordination between
cooperating agents, not a security boundary -- sysfs (who really holds VRAM)
is still checked on every acquire.

A lease is live while the process or container it names is alive. Before the
server exists (between acquire and launch) it is live for PENDING_TTL_S.
Stale leases are dropped by whoever next takes the lock.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pwd
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PENDING_TTL_S = 900


def lease_dir() -> Path:
    return Path(os.environ.get("ATOM_AGENT_LEASE_DIR", "/tmp/atom-agent-leases"))


def current_user() -> str:
    try:
        return pwd.getpwuid(os.getuid()).pw_name
    except KeyError:
        return str(os.getuid())


@dataclass
class Lease:
    id: str
    user: str
    host: str
    gpus: list[int]
    port: int | None
    runtime: str
    created: float
    agent_pid: int
    pid: int | None = None
    container: str | None = None
    label: str = ""


class LeaseConflict(RuntimeError):
    def __init__(self, message: str, holders: list[dict]):
        super().__init__(message)
        self.holders = holders


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    return state != "Z"


def _container_alive(name: str) -> bool:
    try:
        res = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return True  # cannot tell: keep the lease (conservative)
    if res.returncode != 0:
        # "No such object" means gone; any other failure means we cannot tell.
        return "No such" not in res.stderr and "no such" not in res.stderr
    return res.stdout.strip() == "true"


def lease_alive(lease: Lease, now: float | None = None) -> bool:
    if lease.host != socket.gethostname():
        return True  # another host sharing the directory: not ours to judge
    if lease.container:
        return _container_alive(lease.container)
    if lease.pid:
        return _pid_alive(lease.pid)
    # Pending: the agent took the lease and is starting the server.
    now = now or time.time()
    return now - lease.created < PENDING_TTL_S and _pid_alive(lease.agent_pid)


def _port_bindable(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
        except OSError:
            return False
    return True


class LeaseManager:
    def __init__(self, directory: Path | None = None):
        self.dir = Path(directory) if directory else lease_dir()
        self.file = self.dir / "leases.json"

    def _ensure_dir(self) -> None:
        if not self.dir.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(PermissionError):
                os.chmod(self.dir, 0o1777)  # shared, like /tmp

    def _open_shared(self, path: Path, flags: int):
        fd = os.open(path, flags | os.O_CREAT, 0o666)
        with contextlib.suppress(PermissionError):
            if os.fstat(fd).st_uid == os.getuid():
                os.fchmod(fd, 0o666)  # umask must not lock other users out
        return fd

    @contextlib.contextmanager
    def _locked(self):
        self._ensure_dir()
        lock_fd = self._open_shared(self.dir / ".lock", os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            data_fd = self._open_shared(self.file, os.O_RDWR)
            with os.fdopen(data_fd, "r+") as f:
                raw = f.read()
                try:
                    leases = (
                        [Lease(**x) for x in json.loads(raw)] if raw.strip() else []
                    )
                except (ValueError, TypeError):
                    # A torn write; sysfs still guards real conflicts.
                    leases = []
                box = {"leases": leases, "dirty": False}
                yield box
                if box["dirty"]:
                    # In place, not rename: in a sticky shared dir another
                    # user may not replace a file they do not own.
                    f.seek(0)
                    f.write(json.dumps([asdict(x) for x in box["leases"]], indent=1))
                    f.truncate()
                    f.flush()
                    os.fsync(f.fileno())
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    @staticmethod
    def _prune(box: dict) -> None:
        now = time.time()
        live = [x for x in box["leases"] if lease_alive(x, now)]
        if len(live) != len(box["leases"]):
            box["leases"] = live
            box["dirty"] = True

    def held(self) -> list[Lease]:
        with self._locked() as box:
            self._prune(box)
            return list(box["leases"])

    def acquire(
        self,
        lease_id: str,
        gpus: list[int],
        port: int | None,
        runtime: str,
        *,
        label: str = "",
        gpu_busy=None,
        check_port: bool = True,
    ) -> Lease:
        """Take GPUs + port or raise LeaseConflict naming the holders.

        ``gpu_busy(index) -> str | None`` reports a GPU held by something
        that has no lease (another engine, another user's job).
        """
        with self._locked() as box:
            self._prune(box)
            holders = []
            for x in box["leases"]:
                if x.id == lease_id:
                    continue
                clash = sorted(set(x.gpus) & set(gpus))
                if clash or (port is not None and x.port == port):
                    holders.append(
                        {
                            "lease": x.id,
                            "user": x.user,
                            "label": x.label,
                            "gpus": clash,
                            "port": x.port if x.port == port else None,
                        }
                    )
            for g in gpus if gpu_busy else []:
                why = gpu_busy(g)
                if why:
                    holders.append({"gpus": [g], "held_by": why})
            if (
                port is not None
                and check_port
                and not _port_bindable(port)
                and not any(h.get("port") == port for h in holders)
            ):
                holders.append({"port": port, "held_by": "a process outside any lease"})
            if holders:
                raise LeaseConflict(f"GPUs {gpus} / port {port} are not free", holders)
            lease = Lease(
                id=lease_id,
                user=current_user(),
                host=socket.gethostname(),
                gpus=list(gpus),
                port=port,
                runtime=runtime,
                created=time.time(),
                agent_pid=os.getpid(),
                label=label,
            )
            box["leases"] = [x for x in box["leases"] if x.id != lease_id] + [lease]
            box["dirty"] = True
            return lease

    def bind(
        self, lease_id: str, *, pid: int | None = None, container: str | None = None
    ):
        """Attach the running server, so liveness follows it from now on."""
        with self._locked() as box:
            for x in box["leases"]:
                if x.id == lease_id:
                    x.pid = pid
                    x.container = container
                    box["dirty"] = True

    def release(self, lease_id: str) -> bool:
        with self._locked() as box:
            before = len(box["leases"])
            box["leases"] = [x for x in box["leases"] if x.id != lease_id]
            box["dirty"] = len(box["leases"]) != before
            return box["dirty"]

    def reserved(self) -> tuple[set[int], set[int]]:
        """(GPUs, ports) under live leases of anyone on this machine."""
        held = self.held()
        return (
            {g for x in held for g in x.gpus},
            {x.port for x in held if x.port is not None},
        )
