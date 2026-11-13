# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon (``ATOM_WEIGHT_CACHE=daemon``): one process per GPU that builds
and loads its shard as a server does (``init_model_env``, ``get_model``, and
the drafter's own weights), then serves it.

No engine, runner, scheduler or HTTP server: the daemon shares no port, store
or communicator with the servers that use it, and serving features need no
daemon mode."""

from __future__ import annotations

import contextlib
import functools
import gc
import multiprocessing
import os
import queue
import signal
import threading
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
from atom.model_loader.weight_cache.export import (
    _skip_prepare_finalize,
    build_export,
    leave_out,
    process_export,
    process_snapshot,
    share_storages,
    snapshot,
)
from atom.model_loader.weight_cache.key import (
    _CodeTracer,
    _record_jit,
    _SourceChanged,
    code_fingerprint,
    compute_key,
    exported_structure,
    structure,
)
from atom.model_loader.weight_cache.recorder import InputRecorder
from atom.model_loader.weight_cache.transport import (
    PROTOCOL_VERSION,
    _DaemonLock,
    _Server,
    gpu_identity,
    socket_path,
)
from atom.model_loader.weight_cache.values import (
    _TAG,
    logger,
    release_digest_buffers,
)
from atom.model_loader.weight_utils import local_model_dir
from atom.utils import envs, get_hf_text_config, worker_process_name


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
    ready = ctx.Queue()
    procs = [
        ctx.Process(
            target=_serve_rank, args=(rank, cfg, ready, i), name=f"WeightCache{i}"
        )
        for i, (rank, cfg) in enumerate(ranks)
    ]
    stop = threading.Event()

    def on_signal(signum, frame):
        stop.set()  # no logging here: a signal can land inside a log write

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    t0 = time.monotonic()
    for p in procs:
        p.start()
    serving: set[int] = set()
    failed = []
    while not stop.is_set():
        with contextlib.suppress(queue.Empty):
            serving.add(ready.get(timeout=1.0))
            if len(serving) == n:
                logger.info(
                    f"{_TAG} daemon ready: {n} rank(s) serve their weights "
                    f"({time.monotonic() - t0:.0f}s)"
                )
        failed = [p.name for p in procs if p.exitcode is not None]
        if failed:
            logger.error(
                f"{_TAG} daemon process(es) {failed} exited; stopping the daemon"
            )
            break
    for p in procs:
        if p.is_alive():
            p.terminate()  # each stops serving and exits
    deadline = time.monotonic() + 60
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
        if p.is_alive():
            p.kill()
            p.join(5)
    del store
    logger.info(f"{_TAG} daemon stopped")
    if failed:
        raise SystemExit(1)


def _serve_rank(rank: int, config, ready, index: int) -> None:
    """One GPU's daemon process: load this rank's shard, serve it, and wait
    to be stopped. ``index``: what it reports on ``ready`` once serving."""
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
        rank=rank,
        model=config.model,
        gpu=gpu_identity(gpu),
        idle_timeout_s=envs.ATOM_PRELOAD_IDLE_TIMEOUT_S,
    )
    recorder = InputRecorder.start()
    # Loading runs no device collectives.
    device = init_model_env(config, rank, device_communicators=False)
    stop = threading.Event()
    server, pool = _load_and_serve(
        config,
        rank,
        device,
        recorder,
        path,
        lock,
        functools.partial(_stop_launch, lock, stop),
        t0,
    )
    # Not stop.set(): a handler that runs inside stop.wait() would deadlock.
    signaled = []
    signal.signal(signal.SIGTERM, lambda *_: signaled.append(1))
    signal.signal(signal.SIGINT, lambda *_: signaled.append(1))
    ready.put(index)
    while not (signaled or stop.is_set()):
        time.sleep(1.0)
    # Clients that mapped the weights keep them until they exit.
    server.close()
    lock.release()
    del pool  # held while the weights in it are served
    logger.info(f"{_TAG} {worker_process_name(config, rank)}: stopped serving")


def _stop_launch(lock, stop: threading.Event) -> None:
    """Idle timeout: shut the whole daemon down through its main process,
    as SIGTERM to the launch would."""
    main_pid = lock.info.get("daemon_pid")
    if main_pid and os.getppid() == main_pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(main_pid, signal.SIGTERM)
    else:
        stop.set()  # main process unknown or gone: stop serving and exit


class _RecordingLoad:
    """get_model's loader in the daemon: the disk load, recording what it
    reads and runs and the state it starts from; then the drafter's own
    weights, recorded apart (servers take or load them on their own)."""

    def __init__(self, rank: int, recorder: InputRecorder):
        self.rank, self.recorder = rank, recorder
        self.tracer, self.kernels = _CodeTracer(), {}
        self.draft = None

    def __call__(self, config, model_class, device):
        self.key = compute_key(config, self.rank, device)
        self.recorder.enter_construct(config)
        try:
            model = build_model(config, model_class, device)
            self.built, self.before = structure(model), snapshot(model)
            with self.tracer, _record_jit(self.kernels):
                self.recorder.enter_load(model)
                # Around the load only: what constructing does, clients do.
                self.states = [process_snapshot(config)]
                with _skip_prepare_finalize(model):
                    load_weights(model, config)
                self.states.append(process_snapshot(config))
        finally:
            # Before anything else reads the model.
            self.recorder.finish_load(self.tracer.codes)
        self.after = snapshot(model)
        self.reads = self.recorder.key_reads(local_model_dir(config.model))
        if config.speculative_config:
            self.draft = _load_draft(config, device, model, self.rank)
        return model


def _load_draft(config, device, target, rank: int):
    """The drafter's own weights, loaded as a server's runner loads them and
    recorded on their own (each client binds them to its own target). None:
    not cached, and servers load the drafter from disk."""
    recorder, tracer, kernels = InputRecorder.start(), _CodeTracer(), {}
    # What building a drafter reads from its runner, as ModelRunner sets it.
    rope = getattr(get_hf_text_config(config.hf_config), "rope_parameters", None)
    runner = SimpleNamespace(
        model=target,
        config=config,
        device=device,
        rank=rank,
        world_size=config.tensor_parallel_size,
        use_mrope="mrope_section" in (rope or {}),
    )
    try:
        recorder.enter_construct(config)
        drafter = construct_drafter(config, device, runner)
        built, before = structure(drafter.model), snapshot(drafter.model)
        with tracer, _record_jit(kernels):
            recorder.enter_load(drafter.model)
            states = [process_snapshot(config)]
            with _skip_prepare_finalize(drafter.model):
                loaded = drafter.load_weights()
            states.append(process_snapshot(config))
    except Exception as e:  # noqa: BLE001 - only the drafter goes uncached
        logger.warning(
            f"{_TAG} {worker_process_name(config, rank)}: the drafter is not cached "
            f"({e!r}); servers load it from disk"
        )
        return None
    finally:
        recorder.finish_load(tracer.codes)
    return SimpleNamespace(
        drafter=drafter,
        model=drafter.model,
        built=built,
        before=before,
        after=snapshot(drafter.model),
        loaded=loaded,
        reads=recorder.key_reads(),
        codes=tracer.codes,
        kernels=kernels,
        states=states,
    )


def _draft_payload(d, target, label: str) -> dict | None:
    """The drafter's part of the payload; None: servers load it from disk."""
    if d is None:
        return None
    try:
        code = code_fingerprint(d.codes, d.kernels.values())
    except _SourceChanged as e:
        why = [f"{e} changed after this daemon started"]
    else:
        export = build_export(d.model, d.before, d.after)
        process = process_export(d.states)
        why = export["unshippable"] + process["unshippable"]
        if not why:
            why = _leave_out_shared(d, target, export)
    if why:
        logger.warning(
            f"{_TAG} {label}: the drafter is not cached, servers load it from disk: "
            f"{why[:10]}"
        )
        return None
    return {
        "export": export,
        "process": process,
        "structure": exported_structure(d.built, export),
        "reads": d.reads,
        "code": code,
        "loaded": sorted(d.loaded),
    }


def _leave_out_shared(d, target, export: dict) -> list[str]:
    """Each client's attach_to_target replaces some drafter modules with the
    target's (embed, lm_head); left out, the daemon holds no copies of them.
    Returns why the drafter cannot be cached."""
    before = dict(d.model.named_modules(remove_duplicate=False))
    try:
        d.drafter.attach_to_target(target, d.loaded)
    except Exception as e:  # noqa: BLE001 - only the drafter goes uncached
        return [f"attach_to_target failed: {e!r}"]
    after = dict(d.model.named_modules(remove_duplicate=False))
    leave_out(export, {p for p, m in before.items() if after.get(p) is not m})
    return []


def _load_and_serve(config, rank, device, recorder, path, lock, on_idle, t0):
    label = worker_process_name(config, rank)
    load = _RecordingLoad(rank, recorder)
    model = get_model(config, device, load)
    reads = load.reads
    try:
        codes = code_fingerprint(load.tracer.codes, load.kernels.values())
    except _SourceChanged as e:
        raise RuntimeError(
            f"{_TAG} {label}: {e} changed after this daemon started, so the weights it "
            "loaded may not match the file; restart the daemon"
        ) from None
    export = build_export(model, load.before, load.after)
    process = process_export(load.states)
    export["unshippable"] += process["unshippable"]
    draft = _draft_payload(load.draft, model, label)
    parts = [export, process] + ([draft["export"], draft["process"]] if draft else [])
    keys = [list(p["views"]) for p in parts]
    views = {k: v for p in parts for k, v in p.pop("views").items()}
    pool = _release_other_memory(views, device, label)
    handles = share_storages(views)
    for p, ks in zip(parts, keys):
        p["storages"] = {k: handles[k] for k in ks}
    total = sum(v.numel() for v in views.values())
    payload = {
        "protocol": PROTOCOL_VERSION,
        "key": load.key,
        "reads": reads,
        "code": codes,
        "structure": exported_structure(load.built, export),
        "export": export,
        "process": process,
        "draft": draft,
        "summary": {
            "pid": os.getpid(),
            "daemon_pid": lock.info.get("daemon_pid"),
            "bytes": total,
        },
    }
    logger.info(
        f"{_TAG} {label}: load read {len(reads['env'])} env vars, {len(reads['config'])} config "
        f"fields, {len(reads['attrs'])} construction attributes, {len(reads['classattrs'])} class "
        f"attributes, {len(reads['globals'])} globals, {len(reads['files'])} files; it changed "
        f"{len(process['changes'])} globals and {len(process['env'])} env vars"
    )
    server = _Server(
        path,
        payload,
        views,
        label,
        idle_timeout_s=lock.info["idle_timeout_s"],
        on_idle=on_idle,
    )
    lock.set_state("serving", bytes=total)
    entries = sum(
        len(i["params"]) + len(i["buffers"]) + len(i["attrs"])
        for i in export["plan"].values()
    )
    logger.info(
        f"{_TAG} {label}: serving {entries} entries "
        f"{'and the drafter ' if draft else ''}({total / (1 << 30):.2f} GiB, "
        f"{len(codes['entries'])} code entries) at {path} after {time.perf_counter() - t0:.1f}s"
    )
    if export["unshippable"]:
        logger.warning(
            f"{_TAG} {label}: loading created state that cannot be shared, so clients will "
            f"refuse this cache: {export['unshippable'][:10]}"
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
    release_digest_buffers()
    gc.collect()
    # The daemon runs nothing on the GPU but digests of what it exports,
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
