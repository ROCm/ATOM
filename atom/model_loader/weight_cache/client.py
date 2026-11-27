# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A server rank's side: its model and drafter with their weights, mapped
from the daemon the launch's main process found or started (session) under
ATOM_PRELOAD, read from the checkpoint otherwise."""

from __future__ import annotations

import contextlib
import gc
import os
import re
import socket
import sys
import threading
import time
from typing import Any, NamedTuple

import torch
from torch import nn

from atom.distributed.group_agreement import group_agree
from atom.model_loader.weight_cache.export import (
    _leftover_meta,
    _run_prepare_finalize,
    apply_export,
    apply_process,
)
from atom.model_loader.weight_cache.key import (
    _diff_key,
    compute_key,
    exported_structure,
    structure,
)
from atom.model_loader.weight_cache.preload import (
    _detail,
    _fail,
    _hold_unused,
    _pid_running,
    read_session,
    stop_daemon,
)
from atom.model_loader.weight_cache.transport import (
    PROTOCOL,
    _connect,
    _recv,
    _send,
    daemon_alive,
    daemon_state,
    socket_path,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.model_loader.weight_utils import params_on_meta
from atom.utils import envs, worker_process_name

# [the drafter part] of the payload this process took its model from, for
# load_drafter_from_cache, which runs after the model's.
_draft: list | None = None
# What that daemon's process holds on the GPU beside the weights it maps here.
_load_extra = 0


def daemon_load_extra() -> int:
    """Device memory, in bytes, the daemon this process took its model from
    holds beyond those weights: the KV budget leaves it out, as a disk load
    would not hold it (0: the model is not from a daemon)."""
    return _load_extra


class _Rank(NamedTuple):
    """One runner of the launch."""

    label: str
    path: str  # its GPU's name (socket_path)
    device: torch.device
    group: Any  # every runner of the launch: they decide together


def build_model(config, model_class, device):
    """model_setup's, imported when called: the GPU stack is behind it, and
    what this module decides is tested without."""
    from atom.model_engine import model_setup

    return model_setup.build_model(config, model_class, device)


def load_weights(model, config):
    from atom.model_engine import model_setup

    return model_setup.load_weights(model, config)


def construct_drafter(config, device, runner):
    from atom.model_engine import model_setup

    return model_setup.construct_drafter(config, device, runner)


def _rank(config, device, rank: int) -> _Rank:
    # init_model_env made the groups before anything loads.
    from aiter.dist.parallel_state import get_world_group

    group = get_world_group().cpu_group
    return _Rank(worker_process_name(config, rank), socket_path(device), device, group)


class _Unfit(Exception):
    """Some rank cannot go on with the daemon's weights; ``reasons``: this
    rank's (none: another rank's)."""

    def __init__(self, reasons: list[str]):
        super().__init__(reasons)
        self.reasons = reasons


def _agree(reasons: list[str], group) -> None:
    """Every rank goes on, or none does (raises _Unfit)."""
    if not group_agree(not reasons, group):
        raise _Unfit(reasons)


class _MetaBuild:
    """A model or drafter built with its parameters on meta, and how to
    forget it: construction registers its layers in the config's
    static_forward_context, where a second build would collide with them."""

    def __init__(self, config, build):
        self.layers = config.compilation_config.static_forward_context
        self.before = set(self.layers)
        self.obj, self.error = None, []  # why it could not be built on meta
        t0 = time.perf_counter()
        try:
            with params_on_meta():
                self.obj = build()
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            self.error = [f"building on meta failed: {e!r}"]
        self.seconds = time.perf_counter() - t0

    def drop(self) -> None:
        for name in set(self.layers) - self.before:
            del self.layers[name]
        self.obj = None
        gc.collect()  # unmaps what was mapped


# The loader's progress bar, as tqdm draws it.
_BAR = re.compile(
    rb"Loading safetensors shards\[[^\]\r\n]*\]: *\d+%\|[^|\r\n]*\| \d+/\d+ \[[^\]\r\n]*\]"
)


def _load_bar(log: str | None) -> str:
    """The loader's progress bar as the daemon last drew it in its log
    ("": none there)."""
    try:
        with open(log, "rb") as f:
            f.seek(max(0, os.path.getsize(log) - (1 << 14)))
            bars = _BAR.findall(f.read())
    except (OSError, TypeError):  # no log: a daemon started by hand
        return ""
    return bars[-1].decode(errors="replace") if bars else ""


def _wait_serving(me: _Rank, session: dict) -> bool:
    """Wait until the session's daemon holds the weights of every GPU of the
    launch, or will not (its main process is gone): each rank by itself, on
    what every rank sees, so they go on together. A daemon that hangs is the
    user's to stop. Rank 0 shows the daemon's progress bar as a disk load
    shows its own. Returns whether a daemon held this GPU meanwhile."""
    t0 = last = time.monotonic()
    held, drawn = False, ""
    shows = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    while True:
        held = held or daemon_alive(me.path)
        serving = all(
            daemon_alive(p) and (daemon_state(p) or {}).get("state") == "serving"
            for p in session["paths"]
        )
        if serving or not _pid_running(session["pid"], session.get("created")):
            return held
        bar = _load_bar(session.get("log")) if shows else ""
        if bar != drawn:
            done = " 100%|" in bar
            sys.stderr.write(f"\r{_TAG} daemon: {bar}" + "\n" * done)
            sys.stderr.flush()
            drawn, last = bar, time.monotonic()
        elif time.monotonic() - last > 30:  # no bar: before it reads, or after
            logger.info(
                f"{_TAG} {me.label}: loading weights in daemon "
                f"({time.monotonic() - t0:.0f}s)"
            )
            last = time.monotonic()
        time.sleep(0.5)


def _fetch(me: _Rank) -> tuple[socket.socket, dict]:
    """Get this GPU's payload from the daemon; the connection stays open
    (raises _Miss)."""
    sock = None
    try:
        sock = _connect(me.path)
        _send(
            sock,
            {
                "cmd": "get",
                "v": PROTOCOL,
                "gpu": os.path.basename(me.path),
                "launch": envs.ATOM_PRELOAD_LAUNCH,
                "who": f"pid {os.getpid()} {me.label}",
            },
        )
        sock.settimeout(600)  # the daemon checksums its storages first
        payload = _recv(sock)
        sock.settimeout(None)
    except Exception as e:  # noqa: BLE001
        _close(sock)
        raise _Miss(f"the daemon holding {me.path} is unreachable: {e}") from None
    if "error" in payload:
        sock.close()
        raise _Miss(payload["error"])
    return sock, payload


def _check_payload(payload: dict, key: dict) -> list[str]:
    """Why this payload cannot be used here (empty: it can), as far as it
    shows before this server builds its model. ``key``: this server's."""
    diffs = _diff_key(payload["key"], key)
    if diffs:
        return [
            "the checkpoint, the shard or a setting that decides it differ:"
        ] + diffs
    return []


def _probe(me: _Rank, session: dict, key: dict):
    """This rank's weights from the session's daemon, once _wait_serving
    returned: (connection, payload, why they cannot be used)."""
    sock, payload = None, None
    try:
        if not daemon_alive(me.path):
            raise _Miss(
                f"the weight cache daemon (pid {session['pid']}) does not hold this "
                f"GPU: it exited before serving, see {session.get('log') or 'its log'}"
            )
        if (daemon_state(me.path) or {}).get("socket") != session["socket"]:
            raise _Miss(
                "another daemon holds this GPU than the one the launch found "
                f"(pid {session['pid']})"
            )
        sock, payload = _fetch(me)
        reasons = _check_payload(payload, key)
    except _Miss as e:
        reasons = [str(e)]
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        reasons = [f"weight cache error: {e!r}"]
    if reasons:
        _close(sock)
        sock = None
    return sock, payload, reasons


def _close(sock) -> None:
    if sock is not None:
        sock.close()


def _stop_and_reclaim(path: str, device: torch.device, held: bool) -> None:
    """Stop this GPU's daemon, then wait until its memory is free: the driver
    releases an exited process's device memory asynchronously, so a load
    that starts at once can still find the GPU full. ``held``: a daemon
    held this GPU while this launch waited (it may have exited since)."""
    size = (daemon_state(path) or {}).get("bytes") or 0 if daemon_alive(path) else 0
    on_gpu = device.type == "cuda"
    free0 = torch.cuda.mem_get_info(device)[0] if on_gpu else 0
    if (stop_daemon(path) is None and not held) or not on_gpu:
        return  # no daemon had this GPU: nothing to wait for
    deadline = time.monotonic() + 120
    last, steady_since, rose = free0, time.monotonic(), False
    while time.monotonic() < deadline:
        free = torch.cuda.mem_get_info(device)[0]
        if size and free >= free0 + 0.9 * size:
            return
        if free != last:
            rose = rose or free > free0
            last, steady_since = free, time.monotonic()
        elif not size and time.monotonic() - steady_since > (2 if rose else 10):
            # Size unknown (it was loading): freed and settled, or nothing came free.
            return
        time.sleep(0.2)
    logger.warning(
        f"{_TAG} the stopped daemon's memory is still not free on {device} after 120s"
    )


def _stop_together(me: _Rank, why, held: bool, pid=None) -> None:
    """Stop the daemon on every rank's GPU, or on none: fail if another
    launch's servers use any rank's. Returns once the memory is free."""
    # Held from here: no server takes the weights between this check, the
    # ranks' agreement and the stop. Let go, the daemon serves again.
    hold, err = _hold_unused(me.path)
    try:
        if not group_agree(err is None, me.group):
            raise _fail(me.label, ([err] if err else []) + why, pid)
        err = None
        try:
            _stop_and_reclaim(me.path, me.device, held)
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            err = f"could not stop the daemon: {e}"
        if not group_agree(err is None, me.group):
            raise _fail(me.label, [err] if err else [], pid)
    finally:
        _close(hold)


def _differs(part: dict, model: nn.Module) -> list[str]:
    """Where this server's meta-built ``model`` differs in structure from
    the one the daemon built."""
    diffs = _diff_key(
        part["structure"], exported_structure(structure(model), part["export"])
    )
    return ["the model this server builds differs:"] + diffs if diffs else []


def _map(model, export: dict, device) -> tuple[int, list[str]]:
    """Point the meta-built ``model`` at the daemon's tensors: (bytes
    mapped, why it failed)."""
    try:
        return apply_export(model, export, device), []
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        return 0, [f"mapping the daemon's tensors failed: {e!r}"]


def _on_meta(modules) -> list[str]:
    left = [name for m in modules for name in _leftover_meta(m)]
    if not left:
        return []
    return [f"{len(left)} tensors still on meta after mapping: {left[:8]}"]


def _apply_process(me: _Rank, export: dict) -> int:
    """Make the process changes the daemon's load made, on every rank."""
    error, mapped = None, 0
    try:
        mapped = apply_process(export, me.device)
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        error = f"applying what the daemon's load changed failed: {e!r}"
    if not group_agree(error is None, me.group):
        raise _fail(me.label, [error] if error else [])
    return mapped


def load_from_cache(config, model_class, device, *, rank: int) -> nn.Module | None:
    """This rank's model with the weights of the daemon the launch's main
    process found or started; None: load from disk (no such daemon, or its
    weights cannot be used here: it is stopped then, and the next launch
    starts its own). Every rank of the launch takes the same decisions."""
    global _draft, _load_extra
    session = read_session()
    if session is None:
        return None
    t0 = time.perf_counter()
    me = _rank(config, device, rank)
    # As the daemon's rank took its own: before this server builds anything.
    key = compute_key(config, rank)
    held = _wait_serving(me, session)
    sock, payload, reasons = _probe(me, session, key)
    built = None
    try:
        _agree(reasons, me.group)
        built = _MetaBuild(config, lambda: build_model(config, model_class, device))
        _agree(built.error or _differs(payload, built.obj), me.group)
        t_map = time.perf_counter()
        mapped, error = _map(built.obj, payload["export"], device)
        _agree(error + _on_meta([built.obj]), me.group)
    except _Unfit as e:
        # Unmapped before the daemon stops, or its memory stays in use.
        if built is not None:
            built.drop()
        _close(sock)
        _stop_together(me, e.reasons, held, session["pid"])
        logger.warning(
            f"{_TAG} {me.label}: the weight cache daemon (pid {session['pid']}) cannot "
            f"be used here, so it was stopped and this server loads from disk. If "
            f"every launch of this configuration ends here, launch it without "
            f"ATOM_PRELOAD:\n  {_detail(e.reasons)}"
        )
        return None

    model = built.obj
    try:
        mapped += _apply_process(me, payload["process"])
    except Exception:
        sock.close()
        raise
    torch.cuda.reset_peak_memory_stats()
    # Collective on EP setups; a failure here must not be swallowed.
    n_moe = _run_prepare_finalize(model)
    _draft = [payload["draft"]]
    _load_extra = payload["load_extra"]
    # Holds the connection open for this process's lifetime: the daemon
    # counts it as a server using its weights.
    threading.Thread(target=_watch_daemon, args=(sock, me), daemon=True).start()
    waited = (
        "waited for the daemon the launch started, " if session.get("started") else ""
    )
    logger.info(
        f"{_TAG} {me.label}: cache hit, mapped {mapped / (1 << 30):.2f} GiB from daemon "
        f"pid {session['pid']} in {time.perf_counter() - t0:.2f}s ({waited}"
        f"fetch+checks {t_map - t0 - built.seconds:.2f}s, meta build {built.seconds:.2f}s, "
        f"map {time.perf_counter() - t_map:.2f}s, {n_moe} MoE prepare/finalize)"
    )
    return model


def load_drafter_from_cache(config, device, runner, target, *, rank: int):
    """The drafter with its own weights from the daemon the model came from;
    None: load it from disk (a drafter is small, so one the cache cannot give
    does not fail the launch). All ranks take the same one."""
    global _draft
    taken, _draft = _draft, None
    if taken is None:
        return None  # the model is not from a daemon (on any rank)
    (draft,) = taken
    t0 = time.perf_counter()
    me = _rank(config, device, rank)
    built = None
    try:
        _agree([] if draft else ["the daemon holds no drafter"], me.group)
        built = _MetaBuild(config, lambda: construct_drafter(config, device, runner))
        drafter = built.obj
        _agree(built.error or _differs(draft, drafter.model), me.group)
        mapped, error = _map(drafter.model, draft["export"], device)
        _agree(error, me.group)
        try:
            mapped += apply_process(draft["process"], device)
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            error = [f"applying what the drafter's load changed failed: {e!r}"]
        _agree(error, me.group)
        try:
            _run_prepare_finalize(drafter.model)  # collective on EP setups
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            error = [f"the drafter's MoE setup failed: {e!r}"]
        _agree(error, me.group)
        try:
            # Replaces what the daemon left out with the target's modules.
            drafter.attach_to_target(target, set(draft["loaded"]))
            error = []
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            error = [f"attaching it to the target failed: {e!r}"]
        own = [m for m in vars(drafter).values() if isinstance(m, nn.Module)]
        _agree(error or _on_meta(own), me.group)
    except _Unfit as e:
        if built is not None:
            built.drop()
        logger.warning(
            f"{_TAG} {me.label}: this server loads its drafter from disk, the "
            f"daemon's cannot be used here:\n  {_detail(e.reasons)}"
        )
        return None
    logger.info(
        f"{_TAG} {me.label}: drafter from the cache, mapped "
        f"{mapped / (1 << 30):.2f} GiB in {time.perf_counter() - t0:.2f}s"
    )
    return drafter


def build_and_load_model(config, model_class, device, *, rank: int):
    """This rank's model with its weights, a daemon's under ATOM_PRELOAD or
    read from the checkpoint, and the names of the parameters the checkpoint
    wrote (None: none was read here)."""
    if envs.ATOM_PRELOAD == "server":
        model = load_from_cache(config, model_class, device, rank=rank)
        if model is not None:
            return model, None
    model = build_model(config, model_class, device)
    loaded = load_weights(model, config)
    return model, None if config.load_dummy else frozenset(loaded)


def build_and_load_drafter(config, device, runner, target, *, rank: int):
    """The drafter with its weights, from where ``target`` (the loaded model)
    has its own; it shares with it what the draft does not hold itself."""
    drafter = load_drafter_from_cache(config, device, runner, target, rank=rank)
    if drafter is None:
        drafter = construct_drafter(config, device, runner)
        drafter.load_model(target)
    return drafter


def _watch_daemon(sock: socket.socket, me: _Rank) -> None:
    with contextlib.suppress(OSError):
        while sock.recv(1 << 12):
            pass
    # The mapping stays valid (the memory is freed when its last user exits),
    # so this server keeps working; only the next restart loses the cache.
    logger.warning(
        f"{_TAG} {me.label}: the weight cache daemon ({me.path}) exited; this server "
        "keeps its mapped weights, and its next restart starts a new daemon"
    )
