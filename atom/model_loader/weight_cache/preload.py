# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_PRELOAD: start a daemon with a server's own launch, stop one."""

from __future__ import annotations

import atexit
import contextlib
import glob
import json
import os
import signal
import subprocess
import sys
import time

import psutil

from atom.model_loader.weight_cache.key import _LAUNCH_T0_ENV
from atom.model_loader.weight_cache.transport import (
    _DAEMON_LOG_ENV,
    _DAEMON_PID_ENV,
    _ping,
    daemon_alive,
    daemon_state,
    socket_dir,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.utils import envs

_LAUNCH_ENV = "ATOM_PRELOAD_LAUNCH"  # set by the API server: its saved launch
# Starts the daemon in a new session and exits at once, so the daemon is no
# server's descendant: a server's shutdown kills its whole process tree, and a
# terminal's Ctrl-C signals its whole process group.
_DETACH = (
    "import subprocess, sys\n"
    "with open(sys.argv[1], 'ab') as log:\n"
    "    p = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)\n"
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


def _pid_running(pid) -> bool:
    """Alive and not a zombie (the container's PID 1 may never reap)."""
    if not pid:
        return False
    try:
        status = psutil.Process(int(pid)).status()
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


def record_launch() -> None:
    """Called first thing by the API server (weight cache on). The time it
    runs tells the processes that load which sources may have changed under
    them. A daemon tells its processes its main process (stopping that stops
    the daemon). An ATOM_PRELOAD server saves its command line and
    environment, to start a daemon with them."""
    # The processes that load start, and import their code, after this moment.
    os.environ[_LAUNCH_T0_ENV] = repr(time.time())
    if envs.ATOM_WEIGHT_CACHE == "daemon":
        os.environ[_DAEMON_PID_ENV] = str(os.getpid())
        return
    if not envs.ATOM_PRELOAD:
        return
    if os.getpid() == 1:
        logger.warning(
            f"{_TAG} ATOM_PRELOAD: this server is its container's PID 1, so the "
            "container (and the daemon in it) stops when the server does; run "
            "the server under another init process to keep the daemon"
        )
    d = socket_dir()
    for old in glob.glob(os.path.join(d, "launch_*.json")):
        with contextlib.suppress(ValueError, IndexError):
            if not _pid_running(int(os.path.basename(old).split("_")[1])):
                _remove(old, old + ".daemon")
    path = os.path.join(d, f"launch_{os.getpid()}_{time.time_ns()}.json")
    launch = {
        "argv": [sys.executable, *sys.orig_argv[1:]],
        "cwd": os.getcwd(),
        "env": _initial_environ(),
    }
    # 0600: it holds the environment.
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(launch, f)
    os.environ[_LAUNCH_ENV] = path
    atexit.register(_remove, path, path + ".daemon")


def start_early() -> None:
    """ATOM_PRELOAD, from the API server before it starts its engine: with no
    daemon on any GPU here, start one now, so it loads while this server
    starts up. The runners find it through the launch's marker; with a
    daemon present they decide themselves (reuse or restart)."""
    if not os.environ.get(_LAUNCH_ENV):
        return
    locks = glob.glob(os.path.join(socket_dir(), "gpu_*.sock.lock"))
    if any(daemon_alive(lock[: -len(".lock")]) for lock in locks):
        return
    try:
        info = _spawn_daemon()
    except _Miss as e:
        logger.warning(f"{_TAG} could not start a weight cache daemon early: {e}")
        return
    logger.info(
        f"{_TAG} started weight cache daemon pid {info['pid']} (log {info['log']})"
    )


def _started_by_launch() -> dict | None:
    """The daemon this launch has started already (start_early), if any."""
    launch = os.environ.get(_LAUNCH_ENV)
    if not launch:
        return None
    try:
        with open(launch + ".daemon") as f:
            info = json.load(f)
    except (OSError, ValueError):
        return None
    return None if "error" in info else info


def _spawn_daemon() -> dict:
    """Start a daemon with this server's command line and environment. Every
    rank calls this; the first one starts it, the others get its pid and log.
    (Per API server launch, so each node of a multi-node launch starts its own.)"""
    launch_path = os.environ.get(_LAUNCH_ENV)
    if not launch_path:
        raise _Miss(
            "ATOM_PRELOAD starts the daemon with the API server's command line, "
            "but this engine was not started by the API server"
        )
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
    try:
        with open(launch_path) as f:
            launch = json.load(f)
        env = {
            k: v
            for k, v in launch["env"].items()
            if k not in ("ATOM_PRELOAD", _LAUNCH_ENV)
        }
        log = os.path.join(
            socket_dir(), f"daemon_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}.log"
        )
        env.update({"ATOM_WEIGHT_CACHE": "daemon", _DAEMON_LOG_ENV: log})
        out = subprocess.run(
            [sys.executable, "-c", _DETACH, log, *launch["argv"]],
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
        info = {"pid": int(out.stdout.strip()), "log": log}
    except Exception as e:  # noqa: BLE001 - the other ranks wait for the marker
        info = {"error": f"could not start a weight cache daemon: {e!r}"}
    with os.fdopen(fd, "w") as f:
        json.dump(info, f)
    if "error" in info:
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
        if state.get("pid") or time.monotonic() > deadline:
            break
        time.sleep(0.2)  # the lock is taken just before the state is written
    proc, main = state.get("pid"), state.get("daemon_pid")
    if not _pid_running(proc):
        raise RuntimeError(
            f"the daemon holding {path}.lock (pid {proc}) does not run in this PID "
            "namespace (another container?); stop it where it runs"
        )
    if not main or main not in _ancestors(proc):
        raise RuntimeError(
            f"cannot find the main process of the daemon holding {path}.lock (process {proc})"
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


def _in_use_by_others(path: str, wait_s: float = 5) -> str | None:
    """Why this GPU's daemon must not be replaced (None: it may be)."""
    deadline = time.monotonic() + wait_s
    while True:  # this server has just closed its own connection
        try:
            others = _ping(path).get("clients", 0)
        except FileNotFoundError:
            return None  # not serving (yet): nobody can be connected
        except ConnectionRefusedError:
            if not daemon_alive(path):
                return None  # it has stopped serving and exited
            others = None  # stopping: let it finish
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            return f"could not ask the daemon whether servers use it: {e!r}"
        if others == 0:
            return None
        if time.monotonic() > deadline:
            break
        time.sleep(0.1)
    if others is None:
        return "the daemon holds this GPU but does not answer"
    return (
        f"{others} other running server(s) use the daemon's weights on this GPU; they keep "
        "them in memory, so a new daemon would not fit. Stop them first"
    )
