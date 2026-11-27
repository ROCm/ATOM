# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_PRELOAD: start a daemon with a server's own launch, stop one."""

from __future__ import annotations

import atexit
import base64
import contextlib
import copy
import glob
import hashlib
import json
import os
import pickle
import signal
import subprocess
import sys
import time

import psutil

from atom.model_loader.weight_cache.key import _ENV_VARS
from atom.model_loader.weight_cache.transport import (
    PROTOCOL,
    _connect,
    _recv,
    _send,
    daemon_alive,
    daemon_state,
    socket_dir,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.utils import envs

# Set by a launch's main process for its others (record_launch): its name.
_LAUNCH_ENV = "ATOM_PRELOAD_LAUNCH"
# Starts the daemon in a new session and exits at once, so the daemon is no
# server's descendant: a server's shutdown kills its whole process tree, and a
# terminal's Ctrl-C signals its whole process group.
# It runs on the CPUs the launch had: a runner that starts it is bound to
# its own GPU's NUMA node.
_DETACH = (
    "import os, subprocess, sys\n"
    "os.sched_setaffinity(0, [int(c) for c in sys.argv[2].split(',')])\n"
    "with open(sys.argv[1], 'ab') as log:\n"
    "    p = subprocess.Popen(sys.argv[3:], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)\n"
    "print(p.pid)\n"
)


def _initial_environ() -> dict[str, str]:
    """This process's environment as it was at exec, before imports changed it."""
    try:
        with open("/proc/self/environ", "rb") as f:
            raw = f.read()
    except OSError:
        return dict(os.environ)
    env = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep and k:
            env[os.fsdecode(k)] = os.fsdecode(v)
    return env


def _pid_running(pid, created=None) -> bool:
    """Alive and not a zombie (the container's PID 1 may never reap);
    ``created``: and the process started then, not a later one with its pid."""
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        # Within a tick or two: the boot time psutil adds can shift a second.
        if created is not None and abs(proc.create_time() - created) > 2:
            return False
        status = proc.status()
    except (psutil.Error, ValueError, TypeError):
        return False
    return status not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD)


def _ancestors(pid: int) -> list[int]:
    try:
        return [p.pid for p in psutil.Process(pid).parents()]
    except psutil.Error:
        return []


def _remove(*paths: str) -> None:
    for p in paths:
        with contextlib.suppress(OSError):
            os.unlink(p)


def _prune_logs(d: str, keep: int = 9) -> None:
    """Keep the newest daemon logs only (one more is about to start)."""
    logs = []
    for p in glob.glob(os.path.join(d, "daemon_*.log")):
        with contextlib.suppress(OSError):
            logs.append((os.path.getmtime(p), p))
    _remove(*[p for _, p in sorted(logs)[:-keep]])


def prune_sockets() -> None:
    """Remove the sockets of daemons that are gone: one that was killed
    leaves its own."""
    for p in glob.glob(os.path.join(socket_dir(), "daemon_*.sock")):
        with contextlib.suppress(ValueError):
            if not _pid_running(int(os.path.basename(p)[7:-5])):
                _remove(p)


def _launch_file() -> str | None:
    """Where this launch's main process recorded it, if it did."""
    launch = envs.ATOM_PRELOAD_LAUNCH
    return os.path.join(socket_dir(), f"launch_{launch}.json") if launch else None


def launch_ranks(config) -> list[tuple]:
    """(rank, its engine's Config) of every rank process a launch runs on
    this node, in the order CoreManager lays them out. Rewrites ``config``
    as that does (DP attention): give it a copy to keep."""
    from atom.model_engine.engine_core_mgr import engine_config, layout_engines

    engines = layout_engines(config)
    span = config.tp_world_size * config.pcp_group_size
    stages = range(config.pipeline_parallel_size)
    return [
        (rank, engine_config(config, dp, local, pp))
        for dp, local in engines
        for pp in stages
        for rank in range(span)
    ]


# Asked of a process of its own: a GPU runtime in a launch's main process
# would hold device memory on its first GPU for good.
_PROBE = (
    "import json, torch\n"
    "from atom.model_loader.weight_cache.transport import gpu_identity\n"
    "n = torch.cuda.device_count()\n"
    "print(json.dumps([gpu_identity(torch.device('cuda', i)) for i in range(n)]))\n"
)


def gpu_identities() -> list[str]:
    """gpu_identity of the GPUs this process's children see, by index."""
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(f"no GPU could be asked: {out.stderr.strip()[-500:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def vram_used(identity: str) -> int | None:
    """Device memory in use on a GPU, as the kernel counts it (no GPU
    runtime needed); None: it does not say."""
    pci = identity.rsplit("|", 1)[-1]
    try:
        with open(f"/sys/bus/pci/devices/{pci}.0/mem_info_vram_used") as f:
            return int(f.read())
    except (OSError, ValueError):
        return None


def write_session(session: dict) -> None:
    """What the launch's main process decided, for its rank processes."""
    path = _launch_file() + ".session"
    with open(path + ".tmp", "w") as f:
        json.dump(session, f)
    os.replace(path + ".tmp", path)


def read_session() -> dict | None:
    """The daemon this launch's main process found or started for it and the
    GPUs it serves; None: the launch loads from disk."""
    launch = _launch_file()
    if launch is None:
        return None
    try:
        with open(launch + ".session") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def unsupported(config) -> list[str]:
    """What in a launch the cache does not serve. Servers map the daemon's
    weights: nothing may write them in place (EPLB, RLHF updates, RapidServe's
    own sharing), and a daemon serves the GPUs of its own node. A server
    builds without its weights (--fake-eplb makes a tensor from one).
    Every rank of a launch gets the same answer, on every node."""
    default_runner = "atom.model_engine.model_runner.ModelRunner"
    pc = config.parallel_config
    checks = (
        (config.runner_qualname, config.runner_qualname != default_runner),
        ("EPLB", config.eplb_enable),
        ("--fake-eplb", config.fake_eplb),
        # Not is_multinode_dp, which is also true of a worker's own rank > 0.
        (
            "data parallelism across nodes",
            pc.data_parallel_size_local < pc.data_parallel_size,
        ),
        ("tensor parallelism across nodes", config.is_multinode_tp),
    )
    return [name for name, used in checks if used]


class _InUse(RuntimeError):
    """The daemon on a launch's GPUs can neither be used nor be stopped."""


def _detail(reasons: list[str]) -> str:
    return "\n  ".join(reasons[:25]) or "another rank cannot (see its log)"


def _fail(label: str, reasons: list[str], pid=None) -> _InUse:
    return _InUse(
        f"{_TAG} {label}: ATOM_PRELOAD cannot use or replace the weight cache daemon"
        f"{f' (pid {pid})' if pid else ''}: {_detail(reasons)}\nFix the cause above, or launch "
        "without ATOM_PRELOAD once the daemon is gone (python -m "
        "atom.model_loader.weight_cache stop; --force stops one that servers use)."
    )


def _off(why: str) -> None:
    """This launch goes without the cache: the processes it starts from here
    on find no launch."""
    os.environ.pop(_LAUNCH_ENV, None)
    logger.warning(
        f"{_TAG} ATOM_PRELOAD is off for this launch, which loads from disk: {why}"
    )


class _Engine:
    """A launch's engine arguments, as `run_daemon` takes an API server's."""

    def __init__(self, model: str, kwargs: dict):
        self.model, self.kwargs, self._recorded = model, kwargs, model

    def _get_engine_kwargs(self) -> dict:
        """A copy: a Config keeps and fills what it is given. A drafter that
        is the checkpoint's own follows ``model`` (a daemon's `load`)."""
        kwargs = copy.deepcopy(self.kwargs)
        spec = kwargs.get("speculative_config")
        if spec is not None and spec.model == self._recorded:
            spec.model = self.model
        return kwargs


def launch_engine(path: str) -> _Engine:
    """The engine arguments of the launch recorded at ``path``."""
    with open(path) as f:
        return _Engine(*pickle.loads(base64.b64decode(json.load(f)["engine"])))


def record_launch(model: str, kwargs: dict) -> None:
    """Called by a launch's main process (ATOM_PRELOAD=1) before it starts
    any process: name the launch for them, and save its engine's arguments
    (LLMEngine's) and its environment, to start a daemon with."""
    if os.getpid() == 1:
        # Its exit stops the container, and the daemon in it.
        _off(
            "this server is its container's PID 1, so no daemon would outlive "
            "it. Give the container a long-lived command and start the server "
            "inside it"
        )
        return
    try:
        _record_launch(model, kwargs)
    except Exception as e:  # noqa: BLE001 - the launch runs all the same
        _off(f"its launch cannot be recorded ({e!r})")


def _record_launch(model: str, kwargs: dict) -> None:
    os.environ[_LAUNCH_ENV] = f"{os.getpid()}_{time.time()!r}"
    d = socket_dir()
    for old in glob.glob(os.path.join(d, "launch_*.json")):
        with contextlib.suppress(ValueError, IndexError):
            if not _pid_running(int(os.path.basename(old).split("_")[1])):
                _remove(old, old + ".session")
    path = _launch_file()
    # As it was at exec, with the settings loading goes by as they are now: a
    # launch that is a script may have set them itself.
    env = _initial_environ()
    env.update({k: os.environ[k] for k in _ENV_VARS if k in os.environ})
    launch = {
        # The daemon reads the engine's arguments back from this file.
        "argv": [sys.executable, "-m", __package__, "daemon", path],
        "engine": base64.b64encode(pickle.dumps((model, kwargs))).decode(),
        "cwd": os.getcwd(),
        "env": env,
        "cpus": sorted(os.sched_getaffinity(0)),
    }
    # 0600: it holds the environment.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(launch, f)
    atexit.register(_remove, path, path + ".session")


def launch_recorded() -> bool:
    """Whether this process may start daemons: its main process saved the
    launch to start them with (record_launch)."""
    path = _launch_file()
    return path is not None and os.path.exists(path)


def _spawn_daemon() -> dict:
    """Start a daemon with this launch's engine arguments and environment (its
    main process does, once): the daemon's main pid, when it started, and
    its log. Raises _Miss."""
    try:
        with open(_launch_file()) as f:  # callers check launch_recorded()
            launch = json.load(f)
        # The same launch, as a daemon (it names itself anew).
        env = {k: v for k, v in launch["env"].items() if k != _LAUNCH_ENV}
        env["ATOM_PRELOAD"] = "daemon"
        _prune_logs(socket_dir())
        log = os.path.join(
            socket_dir(), f"daemon_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.log"
        )
        cpus = ",".join(map(str, launch["cpus"]))
        out = subprocess.run(
            [sys.executable, "-c", _DETACH, log, cpus, *launch["argv"]],
            cwd=launch["cwd"],
            env=env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            start_new_session=True,
            timeout=120,
            check=False,
        )
        if out.returncode != 0:
            raise RuntimeError(
                f"launcher exited with {out.returncode}: {out.stderr.strip()[-2000:]}"
            )
        pid = int(out.stdout.split()[-1])  # last: a startup hook may print too
    except Exception as e:  # noqa: BLE001
        raise _Miss(f"could not start a weight cache daemon: {e!r}") from None
    # With when it started: its pid alone may come to name another process.
    created = None  # gone already: nobody finds it running
    with contextlib.suppress(psutil.Error):
        created = psutil.Process(pid).create_time()
    return {"pid": pid, "log": log, "created": created}


def stop_daemon(path: str, timeout_s: float = 180) -> int | None:
    """Stop the daemon holding this GPU (SIGTERM to its main process) and wait
    until its process on this GPU has exited, so its device memory is free.
    Returns the main pid; None when no daemon holds the GPU."""
    deadline = time.monotonic() + 5
    while True:
        if not daemon_alive(path):
            return None
        state = daemon_state(path) or {}
        if _pid_running(state.get("pid")) or time.monotonic() > deadline:
            break
        # The lock is taken just before the state is written (over that of a
        # daemon that died here).
        time.sleep(0.2)
    proc, main = state.get("pid"), state.get("daemon_pid")
    why = None
    if not _pid_running(proc):
        why = (
            f"the daemon holding {path}.lock (pid {proc}) does not run in this PID "
            "namespace (another container?); stop it where it runs"
        )
    elif not main or main not in _ancestors(proc):
        why = f"cannot find the main process of the daemon holding {path}.lock (process {proc})"
    if why:
        if not daemon_alive(path):
            return None  # it exited by itself meanwhile
        raise RuntimeError(why)
    if main == 1:
        raise RuntimeError(
            "the daemon's main process is this container's PID 1: stopping it would "
            "stop the container. Stop the container, or run the daemon under an init "
            "process"
        )
    with contextlib.suppress(ProcessLookupError):
        os.kill(main, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    # The GPU's process lets go of the lock early in its shutdown; its device
    # memory is free only once the process has exited.
    while daemon_alive(path) or _pid_running(proc):
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"the weight cache daemon (pid {main}) has not exited on this GPU "
                f"{timeout_s:.0f}s after SIGTERM (process {proc})"
            )
        time.sleep(0.5)
    return main


def _uncacheable_file(key: dict) -> str:
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode())
    return os.path.join(socket_dir(), f"uncacheable_{digest.hexdigest()[:16]}")


def mark_uncacheable(key: dict, why: list[str]) -> None:
    """A daemon's load with this key left state the cache cannot share, and
    would again: launches with it load from disk, until `stop` (after a
    change to loading code, as for everything the key does not see)."""
    with contextlib.suppress(OSError), open(_uncacheable_file(key), "w") as f:
        json.dump(why[:10], f)


def uncacheable(key: dict) -> list[str] | None:
    """Why (mark_uncacheable); None: not marked."""
    try:
        with open(_uncacheable_file(key)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def forget_uncacheable() -> int:
    marks = glob.glob(os.path.join(socket_dir(), "uncacheable_*"))
    _remove(*marks)
    return len(marks)


def _hold_unused(path: str, wait_s: float = 5):
    """Before stopping the daemon that holds this GPU. (connection, None):
    no server uses that daemon, and none gets its weights while the
    connection is open; the ranks of one launch each get one. (None, None):
    no daemon holds this GPU. (None, why): it must not be stopped."""
    # Whoever asks for the same launch stops the same daemon together.
    by = envs.ATOM_PRELOAD_LAUNCH or f"pid {os.getpid()}"
    deadline = time.monotonic() + wait_s
    while True:  # this server has just closed its own connection
        others, sock = None, None
        try:
            sock = _connect(path, timeout=10)
            _send(sock, {"cmd": "hold", "v": PROTOCOL, "by": by})
            reply = _recv(sock)
            if reply.get("held"):
                sock.settimeout(None)
                return sock, None
            if "error" in reply:
                sock.close()
                return None, reply["error"]
            others = reply.get("clients") or None  # none: it is being stopped
        except (FileNotFoundError, ConnectionRefusedError):
            # Its socket is not there (any more): gone, or of another version.
            if not daemon_alive(path):
                return None, None
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            if sock is not None:
                sock.close()
            if not daemon_alive(path):
                return None, None  # it exited by itself while it was asked
            return None, f"could not ask the daemon whether servers use it: {e!r}"
        if sock is not None:
            sock.close()
        if time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if others is None:
        return None, (
            "the daemon on this GPU is being stopped, or does not answer (one "
            "started by another version of ATOM does not: stop it with python -m "
            "atom.model_loader.weight_cache stop --force)"
        )
    return None, (
        f"{others} other running server(s) use the daemon's weights; they keep them "
        "in memory, so a new daemon would not fit. Stop them first"
    )
