# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_PRELOAD: start a daemon with a server's own launch, stop one."""

from __future__ import annotations

import atexit
import contextlib
import glob
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time

import psutil

from atom.model_loader.weight_cache.transport import (
    _recv,
    _send,
    daemon_alive,
    daemon_state,
    socket_dir,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.utils import envs

# Set by the API server for its processes (record_launch): the launch's name.
_LAUNCH_ENV = "ATOM_PRELOAD_LAUNCH"
# Starts the daemon in a new session and exits at once, so the daemon is no
# server's descendant: a server's shutdown kills its whole process tree, and a
# terminal's Ctrl-C signals its whole process group.
# It runs on the CPUs the API server had: a runner that starts it is bound to
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


def _launch_file() -> str | None:
    """Where this launch's API server saved its command line, if it did."""
    launch = envs.ATOM_PRELOAD_LAUNCH
    return os.path.join(socket_dir(), f"launch_{launch}.json") if launch else None


def unsupported(config) -> list[str]:
    """What in a launch the cache does not serve. Servers map the daemon's
    weights: nothing may write them in place (EPLB, RLHF updates, RapidServe's
    own sharing), and a daemon serves the GPUs of its own node. A daemon has no
    device communicators (sequence parallelism builds on them), a server
    builds without its weights (--fake-eplb makes a tensor from one), and
    streaming online quantization swaps parameters the daemon's diff holds.
    Every rank of a launch gets the same answer, on every node."""
    default_runner = "atom.model_engine.model_runner.ModelRunner"
    streams = envs.ATOM_ONLINE_QUANT_STREAMING and config.online_quant_config
    pc = config.parallel_config
    checks = (
        (config.runner_qualname, config.runner_qualname != default_runner),
        ("EPLB", config.eplb_enable),
        ("--fake-eplb", config.fake_eplb),
        ("streaming online quantization", bool(streams)),
        ("pipeline parallelism", config.pipeline_parallel_size > 1),
        ("sequence parallelism", config.sequence_parallel_size > 1),
        # Not is_multinode_dp, which is also true of a worker's own rank > 0.
        (
            "data parallelism across nodes",
            pc.data_parallel_size_local < pc.data_parallel_size,
        ),
        ("tensor parallelism across nodes", config.is_multinode_tp),
    )
    return [name for name, used in checks if used]


def _off(why: str) -> None:
    """This launch goes without the cache: the processes it starts from here
    on find no launch."""
    os.environ.pop(_LAUNCH_ENV, None)
    logger.warning(
        f"{_TAG} ATOM_PRELOAD is off for this launch, which loads from disk: {why}"
    )


def record_launch() -> None:
    """Called by the API server (ATOM_PRELOAD=1) before it starts any
    process: name the launch for them, and save its command line and
    environment, to start a daemon with."""
    if os.getpid() == 1:
        # Its exit stops the container, and the daemon in it.
        _off(
            "this server is its container's PID 1, so no daemon would outlive "
            "it. Give the container a long-lived command and start the server "
            "inside it"
        )
        return
    try:
        _record_launch()
    except OSError as e:
        _off(str(e))


def _record_launch() -> None:
    os.environ[_LAUNCH_ENV] = f"{os.getpid()}_{time.time()!r}"
    d = socket_dir()
    for old in glob.glob(os.path.join(d, "launch_*.json")):
        with contextlib.suppress(ValueError, IndexError):
            if not _pid_running(int(os.path.basename(old).split("_")[1])):
                _remove(old, old + ".daemon")
    path = _launch_file()
    launch = {
        "argv": [sys.executable, *sys.orig_argv[1:]],
        "cwd": os.getcwd(),
        "env": _initial_environ(),
        "cpus": sorted(os.sched_getaffinity(0)),
    }
    # 0600: it holds the environment.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(launch, f)
    atexit.register(_remove, path, path + ".daemon")


def launch_recorded() -> bool:
    """Whether this process may start daemons: its API server saved the
    launch to start them with (record_launch)."""
    path = _launch_file()
    return path is not None and os.path.exists(path)


def _started_by_launch() -> dict | None:
    """The daemon this launch has started already, if any."""
    launch = _launch_file()
    if launch is None:
        return None
    try:
        with open(launch + ".daemon") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    return None if "error" in info else info


def _spawn_daemon() -> dict:
    """Start a daemon with this server's command line and environment. Every
    rank calls this; the first one starts it, the others get its pid and log."""
    launch_path = _launch_file()  # callers check launch_recorded()
    marker = launch_path + ".daemon"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            with contextlib.suppress(OSError, ValueError):
                with open(marker) as f:
                    info = json.load(f)
                if "error" in info:
                    raise _Miss(info["error"])
                return info
            time.sleep(0.2)
        raise _Miss("the rank that starts the daemon did not report it")
    pid = None
    try:
        with open(launch_path) as f:
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
        # With when it started: its pid alone may come to name another process.
        info = {"pid": pid, "log": log, "created": psutil.Process(pid).create_time()}
    except Exception as e:  # noqa: BLE001 - the other ranks wait for the marker
        info = {"error": f"could not start a weight cache daemon: {e!r}"}
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(info, f)
    except OSError as e:  # the other ranks do not learn of the daemon
        info = {"error": f"could not record the weight cache daemon: {e!r}"}
    if "error" in info:
        if pid is not None:  # started all the same: not next to the disk load
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGTERM)
        raise _Miss(info["error"])
    return info


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
    """Before stopping this GPU's daemon. (connection, None): no server uses
    it, and none gets its weights while the connection is open. (None, None):
    no daemon serves here. (None, why): it must not be stopped."""
    deadline = time.monotonic() + wait_s
    while True:  # this server has just closed its own connection
        others = None
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(10)
            sock.connect(path)
            _send(sock, {"cmd": "hold"})
            reply = _recv(sock)
            if reply.get("held"):
                sock.settimeout(None)
                return sock, None
            others = reply.get("clients") or None  # none: it is being stopped
        except FileNotFoundError:
            sock.close()
            return None, None  # not serving (yet): nobody can be connected
        except ConnectionRefusedError:
            if not daemon_alive(path):
                sock.close()
                return None, None  # it has stopped serving and exited
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            sock.close()
            if not daemon_alive(path):
                return None, None  # it exited by itself while it was asked
            return None, f"could not ask the daemon whether servers use it: {e!r}"
        sock.close()
        if time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if others is None:
        return None, "the daemon on this GPU is being stopped, or does not answer"
    return None, (
        f"{others} other running server(s) use the daemon's weights on this GPU; they keep "
        "them in memory, so a new daemon would not fit. Stop them first"
    )
