# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A server's side (ATOM_PRELOAD): take the weights from the daemon on its
GPUs, starting or restarting the daemon as needed."""

from __future__ import annotations

import contextlib
import gc
import os
import re
import signal
import socket
import sys
import threading
import time
from typing import Any, NamedTuple

import torch
from torch import nn

from atom.distributed.group_agreement import group_agree, group_max, group_min
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
    _hold_unused,
    _pid_running,
    _spawn_daemon,
    _started_by_launch,
    launch_recorded,
    stop_daemon,
    uncacheable,
    unsupported,
)
from atom.model_loader.weight_cache.transport import (
    _recv,
    _send,
    daemon_alive,
    daemon_state,
    socket_path,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.model_loader.weight_utils import params_on_meta
from atom.utils import worker_process_name

# What a rank finds (the launch acts on the most severe): no daemon, one it
# can use, one it cannot.
_NONE, _USABLE, _UNUSABLE = range(3)
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
    path: str  # its GPU's daemon socket
    device: torch.device
    group: Any  # every runner of the launch: they decide together


def build_model(config, model_class, device):
    """model_setup's, imported when called: the GPU stack is behind it, and
    what this module decides is tested without."""
    from atom.model_engine import model_setup

    return model_setup.build_model(config, model_class, device)


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


def _detail(reasons: list[str]) -> str:
    return "\n  ".join(reasons[:25]) or "another rank cannot (see its log)"


def _is_running(started: dict) -> bool:
    """The daemon this launch started (its main process) is still there."""
    return _pid_running(started["pid"], started.get("created"))


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


def _wait_together(me: _Rank, started: dict | None) -> bool:
    """Wait until every GPU's daemon serves or will not: all ranks together,
    so none sits in a collective while another waits out a load. A daemon
    that hangs is the user's to stop. Rank 0 shows the daemon's progress bar
    as a disk load shows its own. Returns whether a daemon held this GPU
    meanwhile."""
    t0 = last = time.monotonic()
    held, drawn = False, ""
    shows = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    while True:
        alive = daemon_alive(me.path)
        held = held or alive
        waiting = (alive and not os.path.exists(me.path)) or (
            # The daemon this launch started has not taken the GPU yet.
            started is not None
            and not held
            and _is_running(started)
        )
        if not group_max(int(waiting), me.group):
            return held
        log = (started or daemon_state(me.path) or {}).get("log") if shows else None
        bar = _load_bar(log)
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
    """Get the daemon's payload; the connection stays open (raises _Miss)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30)
        sock.connect(me.path)
        _send(sock, {"cmd": "get", "who": f"pid {os.getpid()} {me.label}"})
        sock.settimeout(600)  # the daemon checksums its storages first
        payload = _recv(sock)
        sock.settimeout(None)
    except Exception as e:  # noqa: BLE001
        sock.close()
        raise _Miss(f"daemon at {me.path} unreachable: {e}") from None
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


def _probe(me: _Rank, started: dict | None, held: bool, key: dict):
    """This rank's daemon, once _wait_together returned (``held``: one held
    the GPU meanwhile): (connection, payload, why it cannot be used, status)."""
    sock, payload, status = None, None, _USABLE
    try:
        if not daemon_alive(me.path):
            if held:
                # Its memory may still be in use: not a GPU without a daemon.
                raise _Miss(
                    "the daemon on this GPU exited before serving; see its log in "
                    + os.path.dirname(me.path)
                )
            if started is None:
                return None, None, [], _NONE
            raise _Miss(
                f"the weight cache daemon (pid {started['pid']}) did not take this GPU; "
                f"see {started['log']}"
            )
        sock, payload = _fetch(me)
        reasons = _check_payload(payload, key)
    except _Miss as e:
        reasons = [str(e)]
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        reasons = [f"weight cache error: {e!r}"]
    if reasons:
        status = max(status, _UNUSABLE)
        _close(sock)
        sock = None
    return sock, payload, reasons, status


def _close(sock) -> None:
    if sock is not None:
        sock.close()


def _fail(label: str, reasons: list[str], pid=None) -> RuntimeError:
    return RuntimeError(
        f"{_TAG} {label}: ATOM_PRELOAD cannot use or replace the weight cache daemon"
        f"{f' (pid {pid})' if pid else ''}: {_detail(reasons)}\nFix the cause above, or launch "
        "without ATOM_PRELOAD once the daemon is gone (python -m "
        "atom.model_loader.weight_cache stop; --force stops one that servers use)."
    )


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


def _stop_together(me: _Rank, why, held: bool, pid=None, started=None) -> None:
    """Stop the daemon on every rank's GPU, or on none: fail if servers use
    any rank's. With ``started``, this launch's own, that one goes too,
    whichever holds the GPUs (an aborted launch may have left one starting,
    which then took them first). Returns once the memory is free."""
    # Held from here: no server takes the weights between this check, the
    # ranks' agreement and the stop. Let go, the daemon serves again.
    hold, err = _hold_unused(me.path)
    try:
        if not group_agree(err is None, me.group):
            raise _fail(me.label, ([err] if err else []) + why, pid)
        err = None
        try:
            _stop_and_reclaim(me.path, me.device, held)
            if started is not None and _is_running(started):
                with contextlib.suppress(ProcessLookupError):  # another rank's did
                    os.kill(started["pid"], signal.SIGTERM)  # it never took a GPU
        except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
            err = f"could not stop the daemon: {e}"
        if not group_agree(err is None, me.group):
            raise _fail(me.label, [err] if err else [], pid)
    finally:
        _close(hold)


def _replace_daemon(
    me: _Rank, reasons, held, pid, started, present, key
) -> dict | None:
    """The launch cannot use what is on its GPUs (``reasons``; ``present``:
    a daemon is on some): stop it and start a daemon of this launch's own.
    Returns the daemon started; None: load from disk, as the one this launch
    started cannot be used either, none can be started, or a daemon found
    that loading ``key`` leaves state it cannot share."""
    marked = uncacheable(key)  # written by that daemon
    skip = group_max(int(marked is not None), me.group)  # on any rank
    if started is not None:
        _stop_together(me, reasons, held, started=started)
        logger.warning(
            f"{_TAG} {me.label}: the weight cache daemon this launch started cannot "
            f"be used, so it was stopped and this server loads from disk (daemon "
            f"log {started['log']}). If every launch of this configuration ends "
            f"here, launch it without ATOM_PRELOAD:\n  {_detail(marked or reasons)}"
        )
        return None
    if present:
        logger.warning(
            f"{_TAG} {me.label}: {'stopping' if skip else 'restarting'} the weight "
            f"cache daemon{f' (pid {pid})' if pid else ''}, it cannot be used here:"
            f"\n  {_detail(reasons)}"
        )
        why = ["it would need a restart, since it cannot be used here:"] + reasons
        _stop_together(me, why, held, pid)
    if skip:
        logger.warning(
            f"{_TAG} {me.label}: a daemon found that this load leaves state the "
            "cache cannot share, so no daemon is started and this server loads from "
            "disk (python -m atom.model_loader.weight_cache stop forgets this):"
            f"\n  {_detail(marked or [])}"
        )
        return None
    if not present:
        logger.info(f"{_TAG} {me.label}: no weight cache daemon here; starting one")
    err = None
    try:
        started = _spawn_daemon()
    except _Miss as e:
        err = str(e)
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        err = f"could not start a weight cache daemon: {e!r}"
    if not group_agree(err is None, me.group):
        # No daemon holds these GPUs now: none was there, or it was stopped.
        logger.warning(
            f"{_TAG} {me.label}: {err or 'another rank could not start the daemon'}; "
            "loading from disk"
        )
        return None
    logger.info(
        f"{_TAG} {me.label}: started weight cache daemon pid {started['pid']} "
        f"(log {started['log']}); waiting for it to load"
    )
    return started


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
    """This rank's model with a daemon's weights; None: load from disk.

    Starts a daemon when there is none; when the one there cannot be used,
    stops it and starts a new one (once). If the daemon this launch started
    cannot be used either, or none can be started, the server loads from
    disk. Every rank of the launch takes the same decisions."""
    global _draft, _load_extra
    if not launch_recorded():
        logger.warning(
            f"{_TAG} {worker_process_name(config, rank)}: ATOM_PRELOAD needs the "
            "launch its API server records (see that server's log if there is one); "
            "loading from disk"
        )
        return None
    # Before the first collective: every rank, on every node, leaves here.
    why = unsupported(config)
    if why:
        logger.warning(
            f"{_TAG} {worker_process_name(config, rank)}: the weight cache does "
            f"not support {', '.join(why)}; loading from disk"
        )
        return None
    t0 = time.perf_counter()
    me = _rank(config, device, rank)
    # The daemon ranks of this launch started before this call, if any.
    started = _started_by_launch()
    # Taken once, before this server builds anything (as the daemon's was).
    key = compute_key(config, rank)
    while True:
        held = _wait_together(me, started)
        sock, payload, reasons, status = _probe(me, started, held, key)
        lo, hi = group_min(status, me.group), group_max(status, me.group)
        state = daemon_state(me.path) or {}
        pid = state.get("daemon_pid") or state.get("pid")  # as `status` shows it
        if lo == hi == _USABLE:
            built = _MetaBuild(config, lambda: build_model(config, model_class, device))
            try:
                _agree(built.error or _differs(payload, built.obj), me.group)
                t_map = time.perf_counter()
                mapped, error = _map(built.obj, payload["export"], device)
                _agree(error + _on_meta([built.obj]), me.group)
                break
            except _Unfit as e:
                reasons = e.reasons
                # Unmapped before the daemon stops, or its memory stays in use.
                built.drop()
        elif lo == _NONE < hi and not reasons:
            reasons = [
                (
                    "a daemon serves only some of this launch's GPUs"
                    if status != _NONE
                    else "no daemon serves this GPU, but other GPUs of this launch have one"
                )
            ]
        _close(sock)
        started = _replace_daemon(me, reasons, held, pid, started, hi > _NONE, key)
        if started is None:
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
    logger.info(
        f"{_TAG} {me.label}: cache hit, mapped {mapped / (1 << 30):.2f} GiB from daemon pid {pid} "
        f"in {time.perf_counter() - t0:.2f}s ({'started the daemon and waited for it, ' if started else ''}"
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
