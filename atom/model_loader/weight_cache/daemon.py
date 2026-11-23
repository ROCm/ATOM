# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon (``ATOM_PRELOAD=daemon``): one process per GPU that builds
and loads its shard as a server does (``init_model_env``, ``get_model``, and
the drafter's own weights), then serves it.

No engine, runner, scheduler or HTTP server: the daemon shares no port, store
or communicator with the servers that use it, and serving features need no
daemon mode."""

from __future__ import annotations

import contextlib
import functools
import gc
import glob
import multiprocessing
import os
import queue
import signal
import time
from types import SimpleNamespace

import torch

from atom.model_engine.model_setup import (
    build_model,
    construct_drafter,
    get_model,
    init_model_env,
    load_weights,
)
from atom.model_loader import loader
from atom.model_loader.weight_cache.export import (
    build_export,
    leave_out,
    process_export,
    process_snapshot,
    share_storages,
    snapshot,
)
from atom.model_loader.weight_cache.key import (
    compute_key,
    exported_structure,
    structure,
)
from atom.model_loader.weight_cache.preload import mark_uncacheable, unsupported
from atom.model_loader.weight_cache.transport import (
    _DaemonLock,
    _Server,
    gpu_identity,
    socket_path,
)
from atom.model_loader.weight_cache.values import (
    _TAG,
    logger,
    release_checksum_buffers,
)
from atom.utils import (
    get_hf_text_config,
    shutdown_all_processes,
    worker_process_name,
)

# Without a connected server: an idle daemon still holds every GPU's weights.
_IDLE_TIMEOUT_S = 3600


def run_daemon(engine_args) -> None:
    """Called by the API server in place of starting an engine: start one
    process per GPU of the launch, report when all serve, and stop them on
    SIGINT/SIGTERM (``python -m atom.model_loader.weight_cache stop``, the
    idle timeout) or when one of them fails."""
    from torch.distributed import TCPStore

    from atom.model_engine.engine_core_mgr import engine_config, layout_engines
    from atom.model_engine.llm_engine import prepare_config

    # The Config a server's runners get.
    config, _ = prepare_config(engine_args.model, **engine_args._get_engine_kwargs())
    if unsupported(config):
        raise ValueError(
            "the weight cache (ATOM_PRELOAD) does not support "
            + ", ".join(unsupported(config))
        )
    engines = layout_engines(config)
    # The processes' rendezvous, as the engine's CoreManager hosts it.
    pc = config.parallel_config
    store = TCPStore(
        pc.data_parallel_master_ip, 0, is_master=True, wait_for_workers=False
    )
    pc.data_parallel_base_port = store.port
    pc._managed_distributed_store = True

    # Every rank of every DP rank's engine, as the engines would run them.
    span = config.tp_world_size * config.pcp_group_size
    ranks = [
        (rank, cfg)
        for cfg in (engine_config(config, dp, local) for dp, local in engines)
        for rank in range(span)
    ]
    n = len(ranks)
    ctx = multiprocessing.get_context("spawn")
    ready, all_serving = ctx.Queue(), ctx.Event()
    procs = [
        ctx.Process(
            target=_serve_rank,
            args=(rank, cfg, ready, i, all_serving),
            name=f"WeightCache{i}",
        )
        for i, (rank, cfg) in enumerate(ranks)
    ]
    # A flag, and no logging: a signal can land inside a lock or a log write.
    signaled = []
    signal.signal(signal.SIGINT, lambda *_: signaled.append(1))
    signal.signal(signal.SIGTERM, lambda *_: signaled.append(1))
    t0 = time.monotonic()
    for p in procs:
        p.start()
    serving: set[int] = set()
    exited = []
    while not signaled:
        with contextlib.suppress(queue.Empty):
            serving.add(ready.get(timeout=1.0))
            if len(serving) == n:
                all_serving.set()
                logger.info(
                    f"{_TAG} daemon ready: {n} rank(s) serve their weights "
                    f"({time.monotonic() - t0:.0f}s)"
                )
        exited = [p for p in procs if p.exitcode is not None]
        if exited:
            break
    # A Ctrl-C reaches the ranks too: their exit is then no failure. Nor is
    # an exit with 0: that rank's load cannot be cached (_load_and_serve).
    gone = [] if signaled else exited
    failed = [p.name for p in gone if p.exitcode != 0]
    if gone:
        say = logger.error if failed else logger.info
        names = [p.name for p in gone]
        say(f"{_TAG} daemon process(es) {names} exited; stopping the daemon")
    shutdown_all_processes(procs, 60)  # each stops serving and exits
    del store
    logger.info(f"{_TAG} daemon stopped")
    if failed:
        raise SystemExit(1)


def _serve_rank(rank: int, config, ready, index: int, all_serving) -> None:
    """One GPU's daemon process: load this rank's shard, serve it, and wait
    to be stopped. ``index``: what it reports on ``ready`` once serving;
    ``all_serving``: set once every rank does."""
    from atom.config import local_device_index
    from atom.utils import enable_orphan_reaping, set_process_title
    from atom.utils.gc_utils import tune_gc
    from atom.utils.numa_utils import numa_bind_to_node

    enable_orphan_reaping()  # dies with the daemon's main process
    set_process_title(f"WeightCache_{worker_process_name(config, rank)}")
    tune_gc()
    gpu = torch.device("cuda", local_device_index(config, rank))
    with contextlib.suppress(Exception):  # locality only, as in the engine's workers
        numa_bind_to_node(gpu.index, f"weight cache rank {rank}")
    t0 = time.perf_counter()
    # Hold the GPU first: until then servers and `stop` cannot see this daemon.
    path = socket_path(gpu)
    lock = _DaemonLock(
        path,
        name=worker_process_name(config, rank),
        model=config.model,
        gpu=gpu_identity(gpu),
        # The launch's main process: SIGTERM there stops the whole daemon.
        daemon_pid=os.getppid(),
        log=_log_file(),
        idle_timeout_s=_IDLE_TIMEOUT_S,
    )
    # Loading does without device communicators (several GiB per GPU), and
    # leaves MoE communication setup to the servers that take the weights.
    loader.MOE_PREPARE_FINALIZE = False
    device = init_model_env(config, rank, device_communicators=False)
    server, pool = _load_and_serve(config, rank, device, path, lock, t0)
    # Flags: a handler that takes a lock could land inside its holder.
    signaled, idle = [], []

    def on_signal(*_):
        signaled.append(1)
        server.stopping = True  # no server gets weights that are about to go

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    server.watch_idle(
        _IDLE_TIMEOUT_S, functools.partial(_stop_launch, lock, idle), all_serving
    )
    ready.put(index)
    while not (signaled or idle):
        time.sleep(1.0)
    # Clients that mapped the weights keep them until they exit.
    server.close()
    lock.release()
    del pool  # held while the weights in it are served
    logger.info(f"{_TAG} {worker_process_name(config, rank)}: stopped serving")


def _log_file() -> str | None:
    """The file this process writes its output to (a daemon ATOM_PRELOAD
    started has one), for `status`."""
    with contextlib.suppress(OSError):
        out = os.readlink("/proc/self/fd/1")
        if os.path.isfile(out):
            return out
    return None


def _stop_launch(lock, idle: list) -> None:
    """Idle timeout: shut the whole daemon down through its main process,
    as SIGTERM to the launch would."""
    main_pid = lock.info.get("daemon_pid")
    if main_pid and os.getppid() == main_pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(main_pid, signal.SIGTERM)
    else:
        idle.append(1)  # main process unknown or gone: stop serving and exit


def _load_and_export(config, build, load) -> SimpleNamespace:
    """Build and load as a server does, and take what loading changed. The
    export is taken at once: nothing that runs later (the drafter) may show
    in it."""
    rec = SimpleNamespace(model=build())
    rec.built, before = structure(rec.model), snapshot(rec.model, config)
    # Around the load only: what constructing does, clients do.
    states = [process_snapshot(config)]
    rec.loaded = load(rec.model)
    states.append(process_snapshot(config))
    rec.export = build_export(rec.model, before, snapshot(rec.model))
    rec.process = process_export(states)
    rec.export["unshippable"] += rec.process["unshippable"]
    return rec


class _DaemonLoad:
    """get_model's loader in the daemon: the disk load and what it changed;
    then the drafter's own weights, apart (each client binds them to its own
    target)."""

    def __init__(self, rank: int):
        self.rank = rank
        self.draft = None

    def __call__(self, config, model_class, device):
        self.key = compute_key(config, self.rank)
        self.rec = _load_and_export(
            config,
            lambda: build_model(config, model_class, device),
            lambda model: load_weights(model, config),
        )
        if config.speculative_config:
            self.draft = _load_draft(config, device, self.rec.model, self.rank)
        return self.rec.model


def _load_draft(config, device, target, rank: int) -> SimpleNamespace | None:
    """The drafter's load; None: servers load it from disk."""
    from atom.kv_transfer.disaggregation import kv_config_has_producer

    # What building and attaching a drafter reads from its runner, as
    # ModelRunner sets it.
    rope = getattr(get_hf_text_config(config.hf_config), "rope_parameters", None)
    runner = SimpleNamespace(
        model=target,
        config=config,
        device=device,
        rank=rank,
        world_size=config.tensor_parallel_size,
        use_mrope="mrope_section" in (rope or {}),
        tokenID_processor=SimpleNamespace(
            is_pd_producer=kv_config_has_producer(config.kv_transfer_config or {})
        ),
    )
    drafters = []

    def build():
        drafters.append(construct_drafter(config, device, runner))
        return drafters[0].model

    try:
        rec = _load_and_export(config, build, lambda _: drafters[0].load_weights())
    except Exception as e:  # noqa: BLE001 - only the drafter goes uncached
        logger.warning(
            f"{_TAG} {worker_process_name(config, rank)}: the drafter is not cached "
            f"({e!r}); servers load it from disk"
        )
        return None
    rec.drafter = drafters[0]
    return rec


def _draft_payload(d, target, label: str) -> dict | None:
    """The drafter's part of the payload; None: servers load it from disk."""
    if d is None:
        return None
    why = d.export["unshippable"] or _leave_out_shared(d, target)
    if why:
        logger.warning(
            f"{_TAG} {label}: the drafter is not cached, servers load it from disk: "
            f"{why[:10]}"
        )
        return None
    return {
        "export": d.export,
        "process": d.process,
        "structure": exported_structure(d.built, d.export),
        "loaded": sorted(d.loaded),
    }


def _leave_out_shared(d, target) -> list[str]:
    """Each client's attach_to_target replaces some drafter modules with the
    target's (embed, lm_head); left out, the daemon holds no copies of them.
    Returns why the drafter cannot be cached."""
    before = dict(d.model.named_modules(remove_duplicate=False))
    try:
        d.drafter.attach_to_target(target, d.loaded)
    except Exception as e:  # noqa: BLE001 - only the drafter goes uncached
        return [f"attach_to_target failed: {e!r}"]
    after = dict(d.model.named_modules(remove_duplicate=False))
    lost = leave_out(d.export, {p for p, m in before.items() if after.get(p) is not m})
    # An object below a replaced module may be one a kept module holds too.
    return [
        f"{p}: changed by the load, below a module attaching replaces" for p in lost
    ]


def _own_vram(pdev: str, fdinfo: str = "/proc/self/fdinfo") -> int | None:
    """The memory this process holds on the GPU at PCI address ``pdev``, as
    the kernel counts it for the process (DRM fdinfo); None: it does not say."""
    units, held = {"KiB": 10, "MiB": 20}, []
    for path in glob.glob(os.path.join(fdinfo, "*")):
        with contextlib.suppress(OSError, ValueError, KeyError), open(path) as f:
            info = dict(line.split(":", 1) for line in f if ":" in line)
            if info["drm-pdev"].strip() == pdev:
                n, unit = info["drm-memory-vram"].split()
                held.append(int(n) << units[unit])
    return max(held, default=None)


def _own_device_memory(device) -> int:
    """What this process holds on ``device``: the device's total would count
    a server that starts beside a daemon started by hand. No less than the
    allocator holds, where the kernel does not say."""
    own = _own_vram(gpu_identity(device).split("|")[1] + ".0")
    return max(own or 0, torch.cuda.memory_reserved(device))


def _load_and_serve(config, rank, device, path, lock, t0):
    label = worker_process_name(config, rank)
    load = _DaemonLoad(rank)
    get_model(config, device, load)
    rec = load.rec
    export, process = rec.export, rec.process
    if export["unshippable"]:
        # The same load leaves the same state: no daemon is started for it again.
        mark_uncacheable(load.key, export["unshippable"])
        logger.warning(
            f"{_TAG} {label}: loading created state that cannot be shared, so this "
            "daemon exits, and launches with these settings load from disk until "
            "python -m atom.model_loader.weight_cache stop: "
            f"{export['unshippable'][:10]}"
        )
        raise SystemExit(0)
    draft = _draft_payload(load.draft, rec.model, label)
    parts = [export, process] + ([draft["export"], draft["process"]] if draft else [])
    keys = [list(p["views"]) for p in parts]
    views = {k: v for p in parts for k, v in p.pop("views").items()}
    pool = _release_other_memory(views, device, label)
    handles = share_storages(views)
    for p, ks in zip(parts, keys):
        p["storages"] = {k: handles[k] for k in ks}
    total = sum(v.numel() for v in views.values())
    # What this process holds beside the weights (its runtime, the kernels it
    # loaded, gaps between the weights); a server's KV budget leaves it out.
    load_extra = max(_own_device_memory(device) - total, 0)
    payload = {
        "key": load.key,
        "structure": exported_structure(rec.built, export),
        "export": export,
        "process": process,
        "draft": draft,
        "load_extra": load_extra,
    }
    server = _Server(path, payload, views, label)
    lock.set_state("serving", bytes=total)
    entries = sum(
        len(i["params"]) + len(i["buffers"]) + len(i["attrs"])
        for i in export["plan"].values()
    )
    logger.info(
        f"{_TAG} {label}: serving {entries} entries "
        f"{'and the drafter ' if draft else ''}({total / (1 << 30):.2f} GiB, and "
        f"{load_extra / (1 << 30):.2f} GiB beside them; loading changed "
        f"{len(process['changes'])} globals and {len(process['env'])} env vars) "
        f"at {path} after {time.perf_counter() - t0:.1f}s"
    )
    return server, pool


def _release_other_memory(views: dict, device, label: str) -> torch.cuda.MemPool:
    """Everything but the weights is a client's KV cache lost: free all
    other device memory this process holds, and the gaps between the
    weights (_compact). Returns the pool now holding the weights."""
    from aiter import destroy_dist_env

    with contextlib.suppress(Exception):
        destroy_dist_env()
    if torch.distributed.is_initialized():
        with contextlib.suppress(Exception):
            torch.distributed.destroy_process_group()
    release_checksum_buffers()
    gc.collect()
    # The daemon runs nothing on the GPU but checksums of what it exports,
    # so every other storage still alive (tensors the clients build for
    # themselves, such as RoPE tables, wherever they are held) can go.
    shared = {v.untyped_storage().data_ptr() for v in views.values()}
    for obj in gc.get_objects():
        with contextlib.suppress(Exception):
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                storage = obj.untyped_storage()
                if storage.nbytes() and storage.data_ptr() not in shared:
                    storage.resize_(0)
    torch.cuda.empty_cache()
    pool = _compact(views, label)
    # Pinned staging buffers from loading stay cached in host memory.
    with contextlib.suppress(Exception):
        torch._C._host_emptyCache()
    free, total = torch.cuda.mem_get_info(device)
    gib, weights = 1 << 30, sum(v.numel() for v in views.values())
    reserved = torch.cuda.memory_reserved()
    logger.info(
        f"{_TAG} {label}: device memory used {(total - free) / gib:.2f} GiB: "
        f"weights {weights / gib:.2f}, other allocated "
        f"{(torch.cuda.memory_allocated() - weights) / gib:.2f}, allocator reserved-but-free "
        f"{(reserved - torch.cuda.memory_allocated()) / gib:.2f}, outside the allocator "
        f"{(total - free - reserved) / gib:.2f}"
    )
    return pool


def _compact(views: dict, label: str) -> torch.cuda.MemPool:
    """Move the exported storages out of segments with free space in them,
    into a pool of their own, so those segments can be returned. Loading
    replaces construction-time tensors, and the replacements land in free
    parts of other segments (about 0.9 GiB per GPU on DeepSeek-V4-Pro)."""
    device = torch.cuda.current_device()
    gappy = set()
    for seg in torch.cuda.memory_snapshot():
        free = sum(b["size"] for b in seg["blocks"] if b["state"] == "inactive")
        # More than a segment's own rounding or packing leaves.
        if seg["device"] == device and free >= 20 << 20:
            gappy.update(
                b["address"] for b in seg["blocks"] if b["state"] == "active_allocated"
            )
    pool = torch.cuda.MemPool()
    for key, old in views.items():
        if old.untyped_storage().data_ptr() not in gappy:
            continue
        try:
            with torch.cuda.use_mem_pool(pool):
                views[key] = torch.empty_like(old).copy_(old)
        except torch.OutOfMemoryError:
            logger.warning(f"{_TAG} {label}: no device memory left to compact the rest")
            break
        old.untyped_storage().resize_(0)
    torch.cuda.empty_cache()  # outside the pool's context, which would only empty the pool
    return pool
