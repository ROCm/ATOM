# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Start, watch and stop ATOM servers -- only the ones this agent started.

Two runtimes:

* ``local``  -- ``python -m atom.entrypoints.openai_server`` in this Python
  (the usual case inside an ATOM container). Its own session/process group,
  so stopping it takes every engine worker with it.
* ``docker`` -- ``docker run -d`` of an ATOM image from a bare host, GPUs
  selected with HIP_VISIBLE_DEVICES, the model mounted read-only at the same
  path, and a per-image volume for the compile cache.

Deployments are recorded in ``<state_dir>/deployments.json`` so a later chat
session can still see, diagnose and stop them. Nothing outside that record is
ever killed: shared GPU boxes run other people's servers.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from atom.agent import hardware
from atom.agent.config import DEFAULT_IMAGE, AgentConfig
from atom.agent.diagnose import analyze_log, last_exception
from atom.agent.leases import LeaseConflict, LeaseManager, current_user

SERVER_MODULE = "atom.entrypoints.openai_server"
DOCKER_LABEL = "atom-agent"


@dataclass
class Deployment:
    id: str
    model: str
    runtime: str
    gpus: list[int]
    port: int
    env: dict[str, str]
    args: list[str]
    log: str
    created: float
    role: str = "service"
    pid: int | None = None
    container: str | None = None
    image: str | None = None
    stopped: bool = False
    source: dict = field(default_factory=dict)
    lease: str | None = None
    image_id: str | None = None
    # Versions of what actually runs (image probe): ATOM / AITER / ROCm.
    provenance: dict = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


class DeploymentStore:
    def __init__(self, state_dir: Path):
        self.path = Path(state_dir) / "deployments.json"

    @contextlib.contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = []
            if self.path.exists():
                try:
                    data = json.loads(self.path.read_text())
                except ValueError:
                    data = []
            box = {"data": data}
            yield box
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(box["data"], indent=1))
            tmp.replace(self.path)

    def all(self) -> list[Deployment]:
        if not self.path.exists():
            return []
        try:
            return [Deployment(**d) for d in json.loads(self.path.read_text())]
        except (ValueError, TypeError):
            return []

    def get(self, key: str | int) -> Deployment | None:
        key = str(key)
        deps = self.all()
        for d in reversed(deps):
            if d.id == key or str(d.port) == key:
                return d
        return None

    def save(self, dep: Deployment) -> None:
        with self._locked() as box:
            box["data"] = [d for d in box["data"] if d["id"] != dep.id]
            box["data"].append(asdict(dep))

    def new_id(self) -> str:
        return self._next_id([Deployment(**d) for d in self._read()])

    @staticmethod
    def _next_id(deps: list[Deployment]) -> str:
        n = 1 + max(
            (int(d.id[1:]) for d in deps if re.fullmatch(r"d\d+", d.id)),
            default=0,
        )
        return f"d{n}"

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text())
        except ValueError:
            return []

    def create(self, logs_dir: Path, **fields) -> Deployment:
        """Allocate the next id and record the deployment in one locked step,
        so two agents of the same user never get the same id."""
        with self._locked() as box:
            dep_id = self._next_id([Deployment(**d) for d in box["data"]])
            now = time.time()
            dep = Deployment(
                id=dep_id,
                log=str(Path(logs_dir) / f"{dep_id}-{int(now)}.log"),
                created=now,
                **fields,
            )
            box["data"].append(asdict(dep))
        return dep

    def delete(self, dep_id: str) -> None:
        with self._locked() as box:
            box["data"] = [d for d in box["data"] if d["id"] != dep_id]

    def active(self) -> list[Deployment]:
        return [d for d in self.all() if not d.stopped]


# --------------------------------------------------------------------------
# Command construction
# --------------------------------------------------------------------------


def choose_runtime(cfg: AgentConfig, requested: str | None = None) -> str:
    runtime = requested or cfg.runtime
    if runtime in ("local", "docker"):
        return runtime
    return "local" if hardware.atom_runnable_here() else "docker"


def choose_image(cfg: AgentConfig, requested: str | None = None) -> str:
    if requested or cfg.image:
        return requested or cfg.image
    try:
        res = subprocess.run(
            [
                "docker",
                "images",
                "--format",
                "{{.Repository}}:{{.Tag}}\t{{.CreatedAt}}",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        local = [
            line.split("\t")
            for line in res.stdout.splitlines()
            if re.match(r"(docker\.io/)?rocm/atom", line)
        ]
        if local:
            return max(local, key=lambda x: x[1])[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return DEFAULT_IMAGE


def model_mount(model_path: str) -> str:
    """The directory to mount so the model (and its HF blob links) resolve."""
    p = Path(model_path)
    if "snapshots" in p.parts:  # HF cache: snapshot files link into ../../blobs
        return str(Path(*p.parts[: p.parts.index("snapshots")]))
    return str(p)


def docker_volume_name(image: str) -> str:
    return "atom-agent-cache-" + re.sub(r"[^a-zA-Z0-9_.-]", "_", image)[-80:]


def build_local_command(dep: Deployment) -> tuple[list[str], dict[str, str]]:
    env = dict(os.environ)
    env.update(dep.env)
    env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, dep.gpus))
    env["PYTHONUNBUFFERED"] = "1"
    return [sys.executable, "-u", "-m", SERVER_MODULE, *dep.args], env


def build_docker_command(dep: Deployment) -> list[str]:
    model_path = _arg(dep.args, "--model") or dep.model
    mount = model_mount(model_path)
    cmd = [
        "docker", "run", "-d",
        # tini as PID 1: `docker stop` reaches the server and its workers,
        # and their zombies are reaped.
        "--init",
        "--name", dep.container,
        "--label", f"{DOCKER_LABEL}={dep.id}",
        "--label", f"{DOCKER_LABEL}.user={current_user()}",
        "--network=host", "--ipc=host",
        "--device=/dev/kfd", "--device=/dev/dri",
        "--group-add", "video",
        "--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined",
        "--shm-size=16G", "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-v", f"{mount}:{mount}:ro",
        "-v", f"{docker_volume_name(dep.image)}:/root/.cache",
        "-e", f"HIP_VISIBLE_DEVICES={','.join(map(str, dep.gpus))}",
        "-e", "PYTHONUNBUFFERED=1",
    ]  # fmt: skip
    for k, v in dep.env.items():
        cmd += ["-e", f"{k}={v}"]
    return cmd + [dep.image, "python3", "-u", "-m", SERVER_MODULE, *dep.args]


def _pretty(argv: list[str], indent: str = "  ") -> str:
    """argv as a shell command, one flag (with its value) per line."""
    lines: list[str] = []
    for tok in argv:
        q = shlex.quote(tok)
        if (
            lines
            and not tok.startswith("-")
            and lines[-1].lstrip().startswith("-")
            and (" " not in lines[-1].strip())
        ):
            lines[-1] += " " + q
        elif (
            lines and not lines[-1].lstrip().startswith("-") and not tok.startswith("-")
        ):
            lines[-1] += " " + q  # words before the first flag stay together
        else:
            lines.append(indent + q if lines else q)
    return " \\\n".join(lines)


def shell_command(dep: Deployment) -> str:
    """The deployment as a command a person can paste."""
    if dep.runtime == "docker":
        argv = build_docker_command(dep)
        cut = argv.index(dep.image)
        return _pretty(argv[:cut]) + " \\\n  " + _pretty(argv[cut:], "    ")
    env = {**dep.env, "HIP_VISIBLE_DEVICES": ",".join(map(str, dep.gpus))}
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    return f"{prefix} \\\n" + _pretty(["python3", "-m", SERVER_MODULE, *dep.args])


def _arg(args: list[str], name: str) -> str | None:
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def _gpu_busy_checker():
    """index -> who holds it (sysfs), for GPUs that are not free."""
    gpus = {g.index: g for g in hardware.list_gpus(with_names=False)}

    def busy(index: int) -> str | None:
        if not gpus:
            return None  # no sysfs (CI, unusual container): leases only
        g = gpus.get(index)
        if g is None:
            return "no such GPU on this machine"
        if g.free:
            return None
        top = sorted(g.processes, key=lambda p: -p.vram_mib)[:1]
        who = f"{top[0].user} pid {top[0].pid}" if top else "unknown process"
        return f"{who}, {g.vram_used_mib // 1024} GiB used"

    return busy


def _image_id(image: str) -> str | None:
    try:
        res = subprocess.run(
            ["docker", "image", "inspect", "-f", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() or None if res.returncode == 0 else None


def launch(
    cfg: AgentConfig,
    store: DeploymentStore,
    *,
    model: str,
    gpus: list[int],
    port: int,
    args: list[str],
    env: dict[str, str] | None = None,
    runtime: str | None = None,
    image: str | None = None,
    role: str = "service",
    source: dict | None = None,
    leases: LeaseManager | None = None,
    force: bool = False,
    provenance: dict | None = None,
) -> Deployment:
    """Lease the GPUs and port, then start the server.

    Raises LeaseConflict when another agent's lease or a process outside any
    lease holds them. ``force`` skips only the sysfs / port-bind checks (for a
    GPU that shows stale memory); another agent's live lease always wins.
    """
    runtime = choose_runtime(cfg, runtime)
    leases = leases or LeaseManager(cfg.lease_dir)
    dep = store.create(
        cfg.logs_dir,
        model=model,
        runtime=runtime,
        gpus=list(gpus),
        port=int(port),
        env=dict(env or {}),
        args=list(args),
        role=role,
        source=source or {},
        provenance=provenance or {},
    )
    # Unique across users, state dirs and time: two agents of the same user
    # each launching their first deployment ("d1") in the same second must
    # not share a lease id, or the second would "renew" the first's lease.
    dep.lease = f"{current_user()}:{dep.id}:{uuid.uuid4().hex[:12]}"
    try:
        leases.acquire(
            dep.lease,
            dep.gpus,
            dep.port,
            runtime,
            label=f"{Path(model).name} ({dep.id})",
            gpu_busy=None if force else _gpu_busy_checker(),
            check_port=not force,
        )
    except LeaseConflict:
        store.delete(dep.id)  # never started; do not leave a ghost record
        raise
    try:
        if runtime == "docker":
            dep.image = choose_image(cfg, image)
            dep.image_id = _image_id(dep.image)
            user = re.sub(r"[^a-zA-Z0-9_.-]", "_", current_user())
            dep.container = f"atom-agent-{user}-{dep.id}-{int(time.time()) % 100000}"
            res = subprocess.run(
                build_docker_command(dep),
                capture_output=True,
                text=True,
                timeout=900,  # includes an implicit pull of a missing image
                check=False,
            )
            if res.returncode != 0:
                raise RuntimeError(f"docker run failed: {res.stderr.strip()[-800:]}")
            dep.image_id = dep.image_id or _image_id(dep.image)
            leases.bind(dep.lease, container=dep.container)
        else:
            argv, full_env = build_local_command(dep)
            with open(dep.log, "ab") as log:
                log.write(f"$ {shell_command(dep)}\n".encode())
                log.flush()
                proc = subprocess.Popen(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=full_env,
                    start_new_session=True,
                    cwd=str(cfg.state_dir),
                )
            dep.pid = proc.pid
            leases.bind(dep.lease, pid=proc.pid)
    except BaseException:
        leases.release(dep.lease)
        dep.stopped = True
        store.save(dep)
        raise
    store.save(dep)
    return dep


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return False
    if state == "Z":
        with contextlib.suppress(ChildProcessError, OSError):
            os.waitpid(pid, os.WNOHANG)  # reap our own zombie
        return False
    return True


def _container_state(name: str | None) -> dict:
    if not name:
        return {"running": False}
    try:
        res = subprocess.run(
            [
                "docker", "inspect", "-f",
                "{{.State.Status}} {{.State.ExitCode}} {{.State.OOMKilled}}", name,
            ],
            capture_output=True, text=True, timeout=15, check=False,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return {"running": False, "error": "docker unavailable"}
    if res.returncode != 0:
        return {"running": False, "status": "missing"}
    status, code, oom = (res.stdout.split() + ["", "", ""])[:3]
    return {
        "running": status == "running",
        "status": status,
        "exit_code": int(code) if code.lstrip("-").isdigit() else None,
        "oom_killed": oom == "true",
    }


def _docker_logs(container: str, tail: int | None = None) -> str | None:
    cmd = ["docker", "logs"] + (["--tail", str(tail)] if tail else []) + [container]
    try:
        # One pipe for both streams keeps stdout/stderr lines in order.
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.decode(errors="replace")


def persist_logs(dep: Deployment) -> None:
    """Copy a container's whole log to dep.log (before it is removed)."""
    if dep.runtime != "docker" or not dep.container:
        return
    text = _docker_logs(dep.container)
    if text:
        with contextlib.suppress(OSError):
            Path(dep.log).write_text(text)


def read_log_tail(dep: Deployment, max_bytes: int = 400_000) -> str:
    # A live container is the source of truth; the file is the post-mortem
    # copy (written at stop / failure) and the only log of local servers.
    if dep.runtime == "docker" and dep.container and not dep.stopped:
        text = _docker_logs(dep.container, tail=6000)
        if text is not None:
            return text[-max_bytes:]
    try:
        with open(dep.log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def http_get_json(url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def status(dep: Deployment) -> dict:
    if dep.runtime == "docker":
        cstate = _container_state(dep.container)
        alive = cstate.get("running", False)
    else:
        cstate = {}
        alive = _pid_alive(dep.pid)
    models = http_get_json(f"{dep.base_url}/v1/models") if alive else None
    served = [m.get("id") for m in (models or {}).get("data", [])]
    text = read_log_tail(dep)
    log = analyze_log(text)
    gpu_mem = {
        g.index: round(g.vram_used_mib / 1024, 1)
        for g in hardware.list_gpus(with_names=False)
        if g.index in dep.gpus
    }
    fatal = [f for f in log["findings"] if f["fatal"]]
    if dep.stopped:
        state = "stopped"
    elif served:
        # /v1/models can answer before the engine holds the GPU; trust both.
        state = "ready" if all(v > 1 for v in gpu_mem.values()) else "http_up_gpu_idle"
    elif not alive:
        state = "failed"
    elif fatal:
        state = "failing"
    else:
        state = "starting"
    out = {
        "id": dep.id,
        "role": dep.role,
        "state": state,
        "model": dep.model,
        "served_model": served[0] if served else None,
        "base_url": f"{dep.base_url}/v1",
        "runtime": dep.runtime,
        "gpus": dep.gpus,
        "gpu_mem_used_gib": gpu_mem,
        "uptime_s": int(time.time() - dep.created),
        "versions": dep.provenance or None,
        "phase": log["phase"],
        "progress": log["progress"],
        "findings": log["findings"],
        "log": dep.log,
    }
    if dep.runtime == "docker":
        out["container"] = dep.container
        out["image"] = dep.image
        out["container_state"] = cstate
        if cstate.get("oom_killed"):
            out["findings"].insert(
                0,
                {
                    "key": "host_oom",
                    "title": "Container was OOM-killed (host RAM)",
                    "fatal": True,
                },
            )
    if state in ("failed", "failing"):
        out["last_exception"] = last_exception(text)
        out["last_lines"] = log["last_lines"]
    return out


def wait_ready(
    dep: Deployment,
    timeout_s: float = 900,
    poll_s: float = 5,
    on_progress=None,
) -> dict:
    """Block until ready / failed / timeout, reporting phase changes."""
    deadline = time.monotonic() + timeout_s
    last = None
    st = status(dep)
    while True:
        marker = (st["state"], st["phase"], st["progress"])
        if on_progress and marker != last:
            on_progress(st)
            last = marker
        if st["state"] in ("ready", "failed", "stopped"):
            break
        if st["state"] == "failing":
            # Give the process a moment to exit; a fatal signature with a
            # live process is usually a dying engine.
            time.sleep(poll_s)
            st = status(dep)
            if st["state"] == "failing":
                break
            continue
        if time.monotonic() > deadline:
            st["timed_out"] = True
            break
        time.sleep(poll_s)
        st = status(dep)
    st["waited_s"] = int(timeout_s - max(deadline - time.monotonic(), 0))
    return st


def _kill_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(os.getpgid(pid), sig)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, sig)


def stop(
    store: DeploymentStore,
    dep: Deployment,
    grace_s: float = 30,
    leases: LeaseManager | None = None,
) -> dict:
    if dep.runtime == "docker":
        persist_logs(dep)
        subprocess.run(
            ["docker", "stop", "-t", str(int(grace_s)), dep.container],
            capture_output=True,
            timeout=grace_s + 60,
            check=False,
        )
        subprocess.run(
            ["docker", "rm", "-f", dep.container],
            capture_output=True,
            timeout=60,
            check=False,
        )
    elif dep.pid and _pid_alive(dep.pid):
        # The server runs in its own session, so the group holds every engine
        # worker -- and nothing that is not ours.
        _kill_group(dep.pid, signal.SIGTERM)
        deadline = time.monotonic() + grace_s
        while _pid_alive(dep.pid) and time.monotonic() < deadline:
            time.sleep(1)
        if _pid_alive(dep.pid):
            _kill_group(dep.pid, signal.SIGKILL)
    dep.stopped = True
    store.save(dep)
    if dep.lease:
        (leases or LeaseManager()).release(dep.lease)
    # Wait for its GPUs to drain so the next launch does not collide.
    deadline = time.monotonic() + 60
    busy: list[int] = []
    while time.monotonic() < deadline:
        busy = [
            g.index
            for g in hardware.list_gpus(with_names=False)
            if g.index in dep.gpus and not g.free
        ]
        if not busy:
            break
        time.sleep(2)
    return {"id": dep.id, "stopped": True, "gpus_still_busy": busy}


def is_alive(dep: Deployment) -> bool:
    """Process or container still running (a failed server holds no GPU)."""
    if dep.stopped:
        return False
    if dep.runtime == "docker":
        return _container_state(dep.container).get("running", False)
    return _pid_alive(dep.pid)


def live(store: DeploymentStore) -> list[Deployment]:
    return [d for d in store.active() if is_alive(d)]


def reconcile(store: DeploymentStore, leases: LeaseManager | None = None) -> None:
    """Bring records in line with reality after a restart or a crash.

    A server that died keeps its record (for diagnosis) but loses its lease
    and, for docker, gets its log saved; records older than an hour whose
    server is gone are marked stopped.
    """
    leases = leases or LeaseManager()
    for dep in store.active():
        if dep.runtime == "docker":
            cstate = _container_state(dep.container)
            exists = cstate.get("status") not in (None, "missing")
            alive = cstate.get("running", False)
            log = Path(dep.log)
            if exists and not alive and (not log.exists() or log.stat().st_size == 0):
                persist_logs(dep)
        else:
            exists = alive = _pid_alive(dep.pid)
        if not alive and dep.lease:
            leases.release(dep.lease)
        if not exists and time.time() - dep.created > 3600:
            dep.stopped = True
            store.save(dep)
