# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run commands where ATOM runs: inside a deployment's container (or this
Python's ATOM install), in the foreground or as background jobs.

This is how skills execute. ATOM's and AITER's skills are procedures written
for an engineer's shell inside the ATOM container ("cd /app/ATOM; bash
scripts/run_benchmark.sh ...", "zcat $TRACE_DIR/rank_0/*.gz | ..."), so a
job runs there: ``docker exec -w <repo root> <container> bash -lc CMD``.

A background job is detached from the agent (new session) and writes its
output and exit code to ``<state_dir>/jobs/``, so it survives the agent
restarting. Inside the container the command runs under ``setsid`` with its
process-group id recorded, so cancelling kills the whole tree there -- not
just the ``docker exec`` client on the host.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from atom.agent.deploy import Deployment

CONTAINER_LOG_DIR = "/app/logs_claude"  # where ATOM's scripts write


@dataclass
class Job:
    id: str
    deployment: str | None
    runtime: str
    container: str | None
    command: str
    cwd: str
    log: str
    exit_file: str
    pgid_file: str
    created: float
    host_pid: int | None = None
    label: str = ""
    cancelled: bool = False


class JobStore:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir) / "jobs"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "jobs.json"

    @contextlib.contextmanager
    def _locked(self):
        with open(self.dir / ".lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = json.loads(self.path.read_text()) if self.path.exists() else []
            box = {"data": data}
            yield box
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(box["data"], indent=1))
            tmp.replace(self.path)

    def all(self) -> list[Job]:
        if not self.path.exists():
            return []
        try:
            return [Job(**j) for j in json.loads(self.path.read_text())]
        except (ValueError, TypeError):
            return []

    def get(self, job_id: str) -> Job | None:
        return next((j for j in self.all() if j.id == job_id), None)

    def create(self, **fields) -> Job:
        with self._locked() as box:
            n = 1 + max(
                (
                    int(j["id"][1:])
                    for j in box["data"]
                    if re.fullmatch(r"j\d+", j["id"])
                ),
                default=0,
            )
            job_id = f"j{n}"
            job = Job(
                id=job_id,
                log=str(self.dir / f"{job_id}.log"),
                exit_file=str(self.dir / f"{job_id}.exit"),
                pgid_file=f"/tmp/atom-agent-{job_id}-{os.getpid()}.pgid",
                created=time.time(),
                **fields,
            )
            box["data"].append(asdict(job))
        return job

    def save(self, job: Job) -> None:
        with self._locked() as box:
            box["data"] = [j for j in box["data"] if j["id"] != job.id] + [asdict(job)]


def _inner_script(job: Job) -> str:
    """What runs inside the target: its own process group, pgid recorded."""
    body = f"echo $$ > {shlex.quote(job.pgid_file)}; {job.command}"
    return (
        f"mkdir -p {CONTAINER_LOG_DIR} 2>/dev/null; "
        f"exec setsid -w bash -c {shlex.quote(body)}"
    )


def _target_argv(job: Job, env: dict[str, str]) -> list[str]:
    if job.runtime == "docker":
        argv = ["docker", "exec", "-w", job.cwd]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        return argv + [job.container, "bash", "-lc", _inner_script(job)]
    return ["bash", "-lc", _inner_script(job)]


def start(
    store: JobStore,
    dep: Deployment | None,
    command: str,
    *,
    cwd: str,
    env: dict[str, str] | None = None,
    label: str = "",
) -> Job:
    """Start a detached job; its output goes to job.log, exit code to job.exit_file."""
    runtime = dep.runtime if dep else "local"
    job = store.create(
        deployment=dep.id if dep else None,
        runtime=runtime,
        container=dep.container if dep else None,
        command=command,
        cwd=cwd,
        label=label,
    )
    full_env = dict(env or {})
    if dep is not None and runtime == "local":
        full_env.setdefault("HIP_VISIBLE_DEVICES", ",".join(map(str, dep.gpus)))
    argv = _target_argv(job, full_env if runtime == "docker" else {})
    # The host wrapper outlives the agent and records the exit code.
    wrapper = (
        f"{shlex.join(argv)} > {shlex.quote(job.log)} 2>&1; "
        f"echo $? > {shlex.quote(job.exit_file)}"
    )
    proc = subprocess.Popen(
        ["sh", "-c", wrapper],
        start_new_session=True,
        cwd=cwd if runtime == "local" else None,
        env={**os.environ, **full_env} if runtime == "local" else None,
    )
    job.host_pid = proc.pid
    store.save(job)
    return job


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    if state == "Z":
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, os.WNOHANG)
        return False
    return True


def exit_code(job: Job) -> int | None:
    try:
        return int(Path(job.exit_file).read_text().strip())
    except (OSError, ValueError):
        return None


def tail(job: Job, lines: int = 40, grep: str | None = None) -> list[str]:
    from atom.agent.diagnose import clean

    try:
        with open(job.log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 400_000))
            text = clean(f.read().decode(errors="replace"))
    except OSError:
        return []
    rows = [r for r in text.splitlines() if r.strip()]
    if grep:
        rx = re.compile(grep, re.IGNORECASE)
        rows = [r for r in rows if rx.search(r)]
    return [r[:400] for r in rows[-lines:]]


def status(job: Job, lines: int = 20) -> dict:
    code = exit_code(job)
    running = code is None and _alive(job.host_pid)
    state = (
        "cancelled"
        if job.cancelled
        else (
            "running"
            if running
            else "succeeded" if code == 0 else "failed" if code is not None else "lost"
        )  # wrapper gone without an exit code (host reboot, kill -9)
    )
    return {
        "job": job.id,
        "state": state,
        "exit_code": code,
        "deployment": job.deployment,
        "command": job.command,
        "runtime_s": int(time.time() - job.created),
        "log": job.log,
        "tail": tail(job, lines),
    }


def wait(
    job: Job, timeout_s: float = 1800, poll_s: float = 5, on_progress=None
) -> dict:
    deadline = time.monotonic() + timeout_s
    last_line, last_report = None, 0.0
    while True:
        st = status(job, lines=1)
        if st["state"] != "running":
            return status(job, lines=40)
        now = time.monotonic()
        line = st["tail"][-1] if st["tail"] else ""
        if on_progress and line != last_line and now - last_report >= 15:
            on_progress(f"{job.id} [{st['runtime_s']}s] {line[:160]}")
            last_line, last_report = line, now
        if now > deadline:
            st = status(job, lines=40)
            st["timed_out"] = True
            st["note"] = "still running; wait again or cancel_job"
            return st
        time.sleep(poll_s)


def cancel(store: JobStore, job: Job) -> dict:
    # `kill -SIG -PGID` without "--": dash (sh on Debian/Ubuntu, and in most
    # images) rejects "--", which would silently signal nothing.
    kill = (
        f'pg=$(cat {shlex.quote(job.pgid_file)} 2>/dev/null); [ -n "$pg" ] || exit 0; '
        'kill -TERM -"$pg" 2>/dev/null; sleep 3; kill -KILL -"$pg" 2>/dev/null; true'
    )
    if job.runtime == "docker" and job.container:
        subprocess.run(
            ["docker", "exec", job.container, "sh", "-c", kill],
            capture_output=True,
            timeout=30,
            check=False,
        )
    else:
        subprocess.run(["sh", "-c", kill], capture_output=True, timeout=30, check=False)
    if job.host_pid and _alive(job.host_pid):
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(job.host_pid), signal.SIGTERM)
    job.cancelled = True
    store.save(job)
    return status(job)


def run_foreground(
    store: JobStore,
    dep: Deployment | None,
    command: str,
    *,
    cwd: str,
    env: dict[str, str] | None = None,
    timeout_s: float = 120,
) -> dict:
    """Start a job and wait for it; a slow one is left running as a job."""
    job = start(store, dep, command, cwd=cwd, env=env)
    st = wait(job, timeout_s=timeout_s, poll_s=0.5)
    if st.get("timed_out"):
        st["note"] = (
            f"still running after {timeout_s:g}s as background job {job.id}; "
            "use wait_job / job_log / cancel_job"
        )
    return st


def copy_out(dep: Deployment | None, src: str, dest_dir: Path, cwd: str) -> dict:
    """Copy a file or directory out of the deployment's container (or the
    local install) into dest_dir on this host."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(src.rstrip("/")).name or "out"
    dest = dest_dir / name
    if dep is not None and dep.runtime == "docker":
        src_path = src if src.startswith("/") else f"{cwd}/{src}"
        res = subprocess.run(
            ["docker", "cp", f"{dep.container}:{src_path}", str(dest)],
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if res.returncode != 0:
            return {"error": res.stderr.strip()[-500:] or "docker cp failed"}
    else:
        src_path = Path(src if src.startswith("/") else Path(cwd) / src)
        if not src_path.exists():
            return {"error": f"{src_path} does not exist"}
        if src_path.is_dir():
            shutil.copytree(src_path, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(src_path, dest)
    files = [p for p in dest.rglob("*") if p.is_file()] if dest.is_dir() else [dest]
    return {
        "copied_to": str(dest),
        "files": [
            {"path": str(p), "mib": round(p.stat().st_size / 2**20, 2)}
            for p in sorted(files)[:50]
        ],
        "file_count": len(files),
    }
