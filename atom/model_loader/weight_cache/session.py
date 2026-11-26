# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A server launch's side of the weight cache, in its main process: find the
daemon that serves the launch, or start or replace one. Once, and before any
rank process runs: the ranks only map what this found (client)."""

from __future__ import annotations

import contextlib
import copy
import os
import signal
import socket
import threading
import time
from typing import NamedTuple

from atom.model_loader.weight_cache.key import compute_key
from atom.model_loader.weight_cache.preload import (
    _detail,
    _fail,
    _hold_unused,
    _InUse,
    _off,
    _spawn_daemon,
    gpu_identities,
    launch_ranks,
    launch_recorded,
    read_session,
    record_launch,
    stop_daemon,
    uncacheable,
    unsupported,
    vram_used,
    write_session,
)
from atom.model_loader.weight_cache.transport import (
    PROTOCOL,
    _connect,
    _ping,
    _recv,
    _send,
    daemon_alive,
    daemon_socket,
    daemon_state,
    gpu_path,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.utils import envs, worker_process_name

# The launch's connection to its daemon's socket: open, the launch uses the
# daemon, from before its ranks take their weights.
_control: socket.socket | None = None
# The launch's ranks (_Rank), kept for serve_notices.
_mine: list = []


class _Rank(NamedTuple):
    """One rank process of the launch, before it runs."""

    label: str
    path: str  # its GPU's name (gpu_path)
    gpu: str  # gpu_identity
    key: dict  # its cache key, as a daemon's main process has its ranks'


def _ranks(config) -> list[_Rank]:
    from atom.config import local_device_index

    gpus = gpu_identities()
    ranks = []
    # A copy: laying the ranks out is the engine's to do to the Config.
    for rank, cfg in launch_ranks(copy.deepcopy(config)):
        gpu = gpus[local_device_index(cfg, rank)]
        key = compute_key(cfg, rank)
        ranks.append(_Rank(worker_process_name(cfg, rank), gpu_path(gpu), gpu, key))
    return ranks


def open_session(config, model: str, kwargs: dict) -> None:
    """Called once by a launch's main process (ATOM_PRELOAD=1) with the
    launch's Config and the engine arguments it comes from (LLMEngine's),
    before its engines start. Raises when the daemon on the launch's GPUs
    would have to be replaced and other servers use it."""
    record_launch(model, kwargs)
    if not launch_recorded():
        return
    why = unsupported(config)
    if why:
        _off(f"the weight cache does not support {', '.join(why)}")
        return
    try:
        session = _find_or_start(_ranks(config))
    except _Miss as e:
        _off(str(e))
        return
    except _InUse:
        raise
    except Exception as e:  # noqa: BLE001 - the launch runs all the same
        _off(f"no weight cache daemon could be looked for or started ({e!r})")
        return
    try:
        write_session(session)
    except OSError as e:
        if session.get("started"):  # not beside the ranks' disk load
            with contextlib.suppress(ProcessLookupError):
                os.kill(session["pid"], signal.SIGTERM)
        _off(f"its ranks could not be told of the weight cache daemon ({e!r})")


def _find_or_start(ranks: list[_Rank]) -> dict:
    """The daemon the launch's ranks take their weights from (read_session).
    One on the launch's GPUs that does not serve it is stopped first. Raises
    _Miss: the launch loads from disk."""
    paths = [r.path for r in ranks]
    _mine[:] = ranks
    held = {p: daemon_state(p) or {} for p in paths if daemon_alive(p)}
    if held:
        daemons = {(s.get("daemon_pid"), s.get("socket")) for s in held.values()}
        pid, sock = next(iter(daemons))
        if len(daemons) > 1:
            pid, reasons = None, ["daemons of several launches hold its GPUs"]
        elif len(held) < len(paths):
            reasons = ["it holds only some of this launch's GPUs"]
        elif sock is None:
            reasons = ["another version of ATOM started it"]
        else:
            reasons = _attach(sock, ranks)
            if not reasons:
                log = next(iter(held.values())).get("log")
                return {"socket": sock, "pid": pid, "log": log, "paths": paths}
        logger.warning(
            f"{_TAG} stopping the weight cache daemon{f' (pid {pid})' if pid else ''}, "
            f"it cannot be used here:\n  {_detail(reasons)}"
        )
        _stop([r for r in ranks if r.path in held], held, reasons, pid)
    marked = next((m for m in map(uncacheable, (r.key for r in ranks)) if m), None)
    if marked is not None:
        raise _Miss(
            "a daemon found that this load leaves state the cache cannot share "
            "(python -m atom.model_loader.weight_cache stop forgets this):\n  "
            + _detail(marked)
        )
    started = _spawn_daemon()
    logger.info(
        f"{_TAG} started weight cache daemon pid {started['pid']} (log "
        f"{started['log']}); the ranks wait for it to load"
    )
    return {
        **started,
        "socket": daemon_socket(started["pid"]),
        "paths": paths,
        "started": True,
    }


def serve_notices(stop, use) -> None:
    """Called by the launch's main process once its engines serve: from here
    on it does what the daemon's `stop` and `load` ask of the servers that
    have its weights. ``stop(rewrite)``: return once nothing of them runs on
    the weights, and nothing does until ``use(changed)`` (``changed``: they
    hold other values now); raise if they cannot, or, with ``rewrite``,
    could not go on with other values."""
    session = read_session()
    if session is None:
        return  # the launch loaded from disk
    # The launch started the daemon: its ranks have its weights by now.
    if _control is None and _attach(session["socket"], _mine):
        return  # gone: its ranks loaded from disk
    try:
        stop(True)
        why = None
    except Exception as e:  # noqa: BLE001 - told to whoever asks for a load
        why = str(e)
    use(False)
    with contextlib.suppress(OSError):  # the daemon may have gone
        _send(_control, {"serves": True, "rewrite": why})
    threading.Thread(
        target=_listen, args=(_control, stop, use), name="weight-cache", daemon=True
    ).start()


def _held() -> list[str]:
    """The GPUs whose weights this launch has from its daemon ([]: none)."""
    session = read_session()
    return session["paths"] if session and _control is not None else []


def change_weights(cmd: str, model: str | None = None) -> list[str]:
    """Have this launch's daemon `stop` its weights, or `load` the checkpoint
    at ``model`` into them (the API server's /model_weight routes): for every
    server that has them. Returns why it did not ([]: it did)."""
    paths = _held()
    if not paths:
        return ["this server's weights are not a weight cache daemon's (ATOM_PRELOAD)"]
    request = {"cmd": cmd, "v": PROTOCOL, **({"model": model} if model else {})}
    try:
        with _connect(paths[0]) as sock:
            sock.settimeout(None)  # a load takes as long as loading does
            _send(sock, request)
            reply = _recv(sock)
    except Exception as e:  # noqa: BLE001
        return [f"the daemon does not answer: {e!r}"]
    return reply.get("why") or ([reply["error"]] if "error" in reply else [])


def weights_status() -> dict | None:
    """What this launch's daemon holds; None: the launch has no daemon's."""
    paths = _held()
    if not paths:
        return None
    state, said = daemon_state(paths[0]) or {}, _ping(paths[0])
    return {
        "model_path": state.get("model"),
        "stopped": bool(said.get("stopped")),
        "servers": said.get("clients"),
        "daemon_pid": state.get("daemon_pid"),
    }


def _listen(conn: socket.socket, stop, use) -> None:
    """Do what the daemon tells this launch, and answer why it could not."""
    while True:
        try:
            told = _recv(conn, limit=1 << 20)
        except Exception:  # noqa: BLE001 - the daemon has gone
            return
        try:
            if told["event"] == "stop":
                stop(False)
            else:
                use(told["changed"])
            why = None
        except Exception as e:  # noqa: BLE001 - told to the daemon
            why = str(e)
        logger.info(
            f"{_TAG} the daemon's weights are "
            f"{'stopped' if told['event'] == 'stop' else 'to be used'}"
            f"{f': {why}' if why else ''}"
        )
        with contextlib.suppress(OSError):
            _send(conn, {"why": why})


def _attach(sock: str, ranks: list[_Rank]) -> list[str]:
    """Ask the daemon at ``sock`` whether it serves this launch, and stay
    connected if it does. Returns why it does not ([]: it does)."""
    global _control
    request = {
        "cmd": "attach",
        "v": PROTOCOL,
        "launch": envs.ATOM_PRELOAD_LAUNCH,
        "who": f"pid {os.getpid()}",
        "keys": {os.path.basename(r.path): r.key for r in ranks},
    }
    deadline = time.monotonic() + 60
    while True:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            conn.settimeout(30)
            conn.connect(sock)
            _send(conn, request)
            reply = _recv(conn)
            conn.settimeout(None)
        except Exception as e:  # noqa: BLE001
            conn.close()
            return [f"it does not answer: {e!r}"]
        if reply.get("fits"):
            _control = conn
            return []
        conn.close()
        # Its ranks have not all taken their GPUs yet: they do in seconds.
        if not reply.get("starting") or time.monotonic() > deadline:
            return reply.get("why") or [reply.get("error") or "it does not start"]
        time.sleep(0.5)


def _stop(ranks: list[_Rank], held: dict, reasons: list[str], pid) -> None:
    """Stop what holds these GPUs and wait until their memory is free. Raises
    (_fail) when another launch's servers use it: they keep its weights in
    memory, so nothing else would fit."""
    holds = []
    try:
        for r in ranks:
            # Held: no server takes the weights between this and the stop.
            hold, why = _hold_unused(r.path)
            holds.append(hold)
            if why:
                why = [why, "it would need a restart, since it cannot be used here:"]
                raise _fail("the launch", why + reasons, pid)
        try:
            for r in ranks:
                _stop_and_reclaim(r, held[r.path].get("bytes") or 0)
        except Exception as e:  # noqa: BLE001 - it still holds the GPUs
            raise _fail(
                "the launch", [f"could not stop the daemon: {e}"], pid
            ) from None
    finally:
        for hold in holds:
            if hold is not None:
                hold.close()


def _stop_and_reclaim(r: _Rank, size: int) -> None:
    """Stop the daemon that holds this GPU, then wait until its memory is
    free: the driver releases an exited process's device memory
    asynchronously, so a load that starts at once can still find the GPU
    full. ``size``: what it held (0: unknown, it was loading)."""
    used0 = vram_used(r.gpu)
    if stop_daemon(r.path) is None or used0 is None:
        return
    deadline = time.monotonic() + 120
    last, steady_since, fell = used0, time.monotonic(), False
    while time.monotonic() < deadline:
        used = vram_used(r.gpu)
        if used is None or (size and used <= used0 - 0.9 * size):
            return
        if used != last:
            fell = fell or used < used0
            last, steady_since = used, time.monotonic()
        elif not size and time.monotonic() - steady_since > (2 if fell else 10):
            return  # freed and settled, or nothing came free
        time.sleep(0.2)
    logger.warning(
        f"{_TAG} the stopped daemon's memory is still not free on {r.gpu} after 120s"
    )
