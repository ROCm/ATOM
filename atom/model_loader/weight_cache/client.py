# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A server's side: take the weights from the daemon on its GPUs, starting or
restarting the daemon under ATOM_PRELOAD."""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import time

import torch

from atom.distributed.group_agreement import group_agree, group_max, group_min
from atom.model_loader.weight_cache.export import (
    _leftover_meta,
    _run_prepare_finalize,
    apply_export,
    params_on_meta,
)
from atom.model_loader.weight_cache.key import (
    _diff_key,
    _SourceChanged,
    compute_key,
    exported_structure,
    structure,
    verify_code,
)
from atom.model_loader.weight_cache.preload import (
    _in_use_by_others,
    _pid_running,
    _spawn_daemon,
    _started_by_launch,
    stop_daemon,
)
from atom.model_loader.weight_cache.recorder import STATIC_READS, diff_reads
from atom.model_loader.weight_cache.transport import (
    PROTOCOL_VERSION,
    _check_peer,
    _recv,
    _send,
    daemon_alive,
    daemon_state,
    socket_path,
)
from atom.model_loader.weight_cache.values import _TAG, _Miss, logger
from atom.utils import envs

_STOP_HINT = "python -m atom.model_loader.weight_cache stop"


def _agree_group():
    """The CPU group of every runner in this launch (they all load together)."""
    try:
        from aiter.dist.parallel_state import get_world_group

        return get_world_group().cpu_group
    except Exception:  # noqa: BLE001 - single process
        return None


def _wait_serving(path: str, label: str, started: dict | None) -> None:
    """Wait until this GPU's daemon serves: first, for a daemon this launch
    started, until it holds the GPU, then until it has loaded (raises _Miss)."""
    t0 = last = time.monotonic()
    seen = started is None  # a daemon that was there holds the GPU already
    while True:
        alive = daemon_alive(path)
        if alive and os.path.exists(path):
            return
        waited = time.monotonic() - t0
        if alive:
            seen = True
        elif seen:
            raise _Miss("the daemon exited while loading")
        elif not _pid_running(started["pid"]):
            raise _Miss(
                f"the weight cache daemon (pid {started['pid']}) exited while starting; see {started['log']}"
            )
        if waited > envs.ATOM_WEIGHT_CACHE_WAIT_S:
            raise _Miss(f"the daemon is still not serving after {waited:.0f}s")
        if time.monotonic() - last > 30:
            logger.info(
                f"{_TAG} {label}: waiting for the daemon to finish loading ({waited:.0f}s)"
            )
            last = time.monotonic()
        time.sleep(0.5)


def _fetch(path: str, label: str) -> tuple[socket.socket, dict]:
    """Get the daemon's payload; the connection stays open (raises _Miss)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(30)
        sock.connect(path)
        _check_peer(sock)
        _send(sock, {"cmd": "get", "who": f"pid {os.getpid()} {label}"})
        sock.settimeout(600)  # the daemon digests its storages first
        payload = _recv(sock)
        sock.settimeout(None)
    except Exception as e:  # noqa: BLE001
        sock.close()
        raise _Miss(f"daemon at {path} unreachable: {e}") from None
    if "error" in payload:
        sock.close()
        raise _Miss(payload["error"])
    return sock, payload


def _check_payload(payload: dict, runner) -> list[str]:
    """Why this payload cannot be used here (empty: it can)."""
    if payload.get("protocol") != PROTOCOL_VERSION:
        return [f"protocol {payload.get('protocol')} != {PROTOCOL_VERSION}"]
    diffs = _diff_key(
        payload["key"], compute_key(runner.config, runner.rank, runner.device)
    )
    if diffs:
        return ["checkpoint, model, rank, GPU or library versions differ:"] + diffs
    diffs = diff_reads(payload["reads"], STATIC_READS, config=runner.config)
    if diffs:
        return ["settings the daemon's load read differ:"] + diffs
    try:
        changed = verify_code(payload["code"])
    except _SourceChanged as e:
        return [f"{e} changed after this server started; restart it"]
    if changed:
        return [
            f"{len(changed)} function(s) or kernel(s) that ran while the daemon loaded have changed:"
        ] + changed
    if payload["export"]["unshippable"]:
        return ["the daemon's load created state that cannot be shared:"] + payload[
            "export"
        ]["unshippable"]
    return []


def _fail(
    label: str, reasons: list[str], pid=None, preload: bool = False
) -> RuntimeError:
    detail = (
        "\n  ".join(reasons[:25])
        if reasons
        else "another rank could not use it; see that rank's log"
    )
    if preload:
        hint = (
            "ATOM_PRELOAD could not set up a weight cache daemon for this launch. Fix the "
            f"cause above, or stop the daemons ({_STOP_HINT}) and launch without ATOM_PRELOAD "
            "to load from disk."
        )
    else:
        hint = (
            "The daemon keeps its copy on these GPUs, so this server will not load a second "
            "one from disk. Restart the daemon with this code and configuration (ATOM_PRELOAD=1 "
            f"has servers do that), or stop it ({_STOP_HINT}) to load from disk."
        )
    return RuntimeError(
        f"{_TAG} {label}: cannot use the weight cache daemon{f' (pid {pid})' if pid else ''}: {detail}\n{hint}"
    )


def _probe(path: str, label: str, runner, started: dict | None):
    """This rank's daemon: (connection, payload, why it cannot be used, whether
    there is one). Waits for a daemon that is starting or loading."""
    sock, payload = None, None
    try:
        if started is None and not daemon_alive(path):
            return None, None, [], False
        _wait_serving(path, label, started)
        sock, payload = _fetch(path, label)
        reasons = _check_payload(payload, runner)
    except _Miss as e:
        reasons = [str(e)]
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        reasons = [f"weight cache error: {e!r}"]
    if reasons and sock is not None:
        sock.close()
        sock = None
    return sock, payload, reasons, True


def _stop_and_reclaim(path: str, device: torch.device) -> None:
    """Stop this GPU's daemon, then wait until its memory is free: the driver
    releases an exited process's device memory asynchronously, so a load
    that starts at once can still find the GPU full."""
    held = (daemon_state(path) or {}).get("bytes") or 0
    on_gpu = device.type == "cuda"
    free0 = torch.cuda.mem_get_info(device)[0] if on_gpu else 0
    stop_daemon(path)
    if not on_gpu:
        return
    deadline = time.monotonic() + 120
    last, steady_since = free0, time.monotonic()
    while time.monotonic() < deadline:
        free = torch.cuda.mem_get_info(device)[0]
        if held and free >= free0 + 0.9 * held:
            return
        if free != last:
            last, steady_since = free, time.monotonic()
        elif not held and time.monotonic() - steady_since > 2:
            return  # size unknown (it was loading): settled
        time.sleep(0.2)
    logger.warning(
        f"{_TAG} the stopped daemon's memory is still not free on {device} after 120s"
    )


def _forward_context(runner) -> dict | None:
    cc = getattr(runner.config, "compilation_config", None)
    return getattr(cc, "static_forward_context", None)


def _give_up(label, path, reasons, started, group, runner, layers) -> bool:
    """The daemon this launch started cannot be used either: stop it and
    load from disk (its memory is free again), or fail if it won't stop."""
    err = None
    try:
        _stop_and_reclaim(path, runner.device)
    except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
        err = f"could not stop the daemon this launch started: {e}"
    if not group_agree(err is None, group):
        raise _fail(label, [err] if err else [], preload=True)
    _drop_layers(runner, layers)
    detail = "\n  ".join(reasons[:25]) if reasons else "another rank cannot use it"
    logger.warning(
        f"{_TAG} {label}: the weight cache daemon this launch started cannot be used, "
        f"so it was stopped and this server loads from disk (daemon log {started['log']}):"
        f"\n  {detail}"
    )
    return False


def _drop_layers(runner, layers) -> None:
    """Unregister the layers a meta-built model registered (a disk load
    builds the model again)."""
    ctx = _forward_context(runner)
    if ctx is not None and layers is not None:
        for name in set(ctx) - layers:
            del ctx[name]


def try_load_from_cache(runner, model_class) -> bool:
    """Build ``runner.model`` from a daemon's weights; False: load from disk.

    Client mode uses the daemon that is there: False when none of this
    launch's GPUs has one, an error when one is there but cannot be used.
    ATOM_PRELOAD manages the daemon: it starts one when there is none, and
    when the one there cannot be used, stops it and starts a new one (once).
    If the daemon it started cannot be used either, or none can be started,
    it loads from disk. Every rank of the launch takes the same decisions."""
    t0 = time.perf_counter()
    label = f"rank {runner.rank}"
    group = _agree_group()
    auto = envs.ATOM_PRELOAD
    path = socket_path(runner.device)
    # The daemon this launch started (the API server may have already).
    started = _started_by_launch() if auto else None
    settled = False
    model, t_meta, layers = None, 0.0, None
    while True:
        sock, payload, reasons, present = _probe(path, label, runner, started)
        status = 0 if not present else (2 if reasons else 1)
        lo, hi = group_min(status, group), group_max(status, group)
        summary = (payload or {}).get("summary", {})
        pid = summary.get("daemon_pid") or summary.get("pid")  # as `status` shows it
        if hi == 0 and not auto:
            logger.info(
                f"{_TAG} {label}: no weight cache daemon on this launch's GPUs; loading from disk"
            )
            return False
        if lo == 0 and hi > 0 and started is None and not settled:
            # A daemon takes its GPUs one rank at a time: give the others a moment.
            settled = True
            if sock is not None:
                sock.close()
            deadline = time.monotonic() + 60
            while not daemon_alive(path) and time.monotonic() < deadline:
                time.sleep(0.5)
            continue
        if lo == hi == 1:
            if model is None:  # built once: a model registers its layers globally
                ctx = _forward_context(runner)
                layers = set(ctx) if ctx is not None else None
                t_build = time.perf_counter()
                with params_on_meta():
                    model = model_class(runner.config)
                torch.set_default_device(None)
                t_meta = time.perf_counter() - t_build
            diffs = _diff_key(
                payload["structure"],
                exported_structure(structure(model), payload["export"]),
            )
            diffs += diff_reads(payload["reads"], ("attrs",), model=model)
            if group_agree(not diffs, group):
                break
            reasons = (
                (["model construction differs where it matters to loading:"] + diffs)
                if diffs
                else []
            )
        elif lo == 0 and hi > 0 and not reasons:
            reasons = (
                ["a daemon serves only some of this launch's GPUs"]
                if present
                else [
                    "no daemon serves this GPU, but other GPUs of this launch have one"
                ]
            )
        if sock is not None:
            sock.close()
            sock = None
        if not auto:
            raise _fail(label, reasons, pid)
        if started is not None:
            return _give_up(label, path, reasons, started, group, runner, layers)
        if hi > 0:
            err = _in_use_by_others(path)
            if not group_agree(err is None, group):
                why = (
                    ["it would need a restart, since it cannot be used here:"] + reasons
                    if reasons
                    else []
                )
                raise _fail(label, ([err] if err else []) + why, pid, preload=True)
            if reasons:
                logger.warning(
                    f"{_TAG} {label}: restarting the weight cache daemon{f' (pid {pid})' if pid else ''}, "
                    "it cannot be used here:\n  " + "\n  ".join(reasons[:25])
                )
            else:
                logger.info(
                    f"{_TAG} {label}: restarting the weight cache daemon (another rank cannot use its daemon)"
                )
            err = None
            try:
                _stop_and_reclaim(path, runner.device)
            except Exception as e:  # noqa: BLE001 - every rank must reach the agreement
                err = f"could not stop the daemon: {e}"
            if not group_agree(err is None, group):
                raise _fail(label, [err] if err else [], pid, preload=True)
        else:
            logger.info(
                f"{_TAG} {label}: no weight cache daemon on this launch's GPUs; starting one (ATOM_PRELOAD)"
            )
        err = None
        try:
            started = _spawn_daemon()
        except _Miss as e:
            err = str(e)
        if not group_agree(err is None, group):
            # No daemon holds these GPUs now: none was there, or it was stopped.
            _drop_layers(runner, layers)
            logger.warning(
                f"{_TAG} {label}: {err or 'another rank could not start the daemon'}; "
                "loading from disk"
            )
            return False
        logger.info(
            f"{_TAG} {label}: started weight cache daemon pid {started['pid']} (log {started['log']}); "
            "waiting for it to load"
        )

    t_map = time.perf_counter()
    error = None
    try:
        mapped = apply_export(model, payload["export"], runner.device)
        left = _leftover_meta(model)
        if left:
            error = f"{len(left)} tensors still on meta after mapping: {left[:8]}"
    except Exception as e:  # noqa: BLE001
        error = f"mapping the daemon's tensors failed: {e!r}"
    if not group_agree(error is None, group):
        sock.close()
        raise _fail(label, [error] if error else [], pid)
    torch.cuda.reset_peak_memory_stats()
    # Collective on EP setups; a failure here must not be swallowed.
    n_moe = _run_prepare_finalize(model)
    runner.model = model
    runner._weight_cache_conn = sock
    threading.Thread(
        target=_watch_daemon, args=(sock, label, path), daemon=True
    ).start()
    logger.info(
        f"{_TAG} {label}: cache hit, mapped {mapped / (1 << 30):.2f} GiB from daemon pid {pid} "
        f"in {time.perf_counter() - t0:.2f}s ({'started the daemon and waited for it, ' if started else ''}"
        f"fetch+checks {t_map - t0 - t_meta:.2f}s, meta build {t_meta:.2f}s, "
        f"map {time.perf_counter() - t_map:.2f}s, {n_moe} MoE prepare/finalize)"
    )
    return True


def _watch_daemon(sock: socket.socket, label: str, path: str) -> None:
    with contextlib.suppress(OSError):
        while sock.recv(1 << 12):
            pass
    # The mapping stays valid (the memory is freed when its last user exits),
    # so this server keeps working; only the next restart loses the cache.
    logger.warning(
        f"{_TAG} {label}: the weight cache daemon ({path}) exited; this server keeps "
        "its mapped weights, but its next restart needs a new daemon (ATOM_PRELOAD starts one)"
    )
