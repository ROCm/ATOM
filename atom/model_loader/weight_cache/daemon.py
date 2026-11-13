# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon (``ATOM_WEIGHT_CACHE=daemon``): one process per GPU that builds
and loads its shard as a server does (``init_model_env``, ``get_model``),
then serves it.

No engine, runner, scheduler or HTTP server: the daemon shares no port, store
or communicator with the servers that use it, and serving features need no
daemon mode."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import gc
import multiprocessing
import os
import queue
import signal
import threading
import time

import torch

from atom.model_engine.model_setup import (
    build_model,
    get_model,
    init_model_env,
    load_weights,
)
from atom.model_loader.weight_cache.export import (
    _skip_prepare_finalize,
    build_export,
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
from atom.utils import envs


def run_daemon(engine_args) -> None:
    """Called by the API server in place of starting an engine: start one
    process per GPU of the launch, report when all serve, and stop them on
    SIGINT/SIGTERM (``python -m atom.model_loader.weight_cache stop``, the
    idle timeout) or when one of them fails."""
    from torch.distributed import TCPStore

    config = _engine_config(engine_args)
    # The processes' rendezvous, as the engine's CoreManager hosts it.
    pc = config.parallel_config
    store = TCPStore(
        pc.data_parallel_master_ip, 0, is_master=True, wait_for_workers=False
    )
    pc.data_parallel_base_port = store.port
    pc._managed_distributed_store = True

    n = config.tp_world_size * config.pcp_group_size
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    procs = [
        ctx.Process(
            target=_serve_rank, args=(rank, config, ready), name=f"WeightCache{rank}"
        )
        for rank in range(n)
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


def _engine_config(engine_args):
    """The Config a server's runners get: LLMEngine's preparation before it
    starts its engine."""
    from atom.config import Config
    from atom.model_engine.llm_engine import load_tokenizer

    fields = {f.name for f in dataclasses.fields(Config)}
    kwargs = engine_args._get_engine_kwargs()
    config = Config(
        engine_args.model, **{k: v for k, v in kwargs.items() if k in fields}
    )
    tokenizer = load_tokenizer(config.model, config.trust_remote_code)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    stop_token_ids = set(config.stop_token_ids)
    stop_token_ids.discard(config.eos_token_id)
    config.stop_token_ids = list(stop_token_ids)
    return config


def _serve_rank(rank: int, config, ready) -> None:
    """One GPU's daemon process: load this rank's shard, serve it, and wait
    to be stopped."""
    from atom.config import local_device_index
    from atom.utils import enable_orphan_reaping, set_process_title, worker_process_name
    from atom.utils.gc_utils import tune_gc
    from atom.utils.numa_utils import numa_bind_to_node

    enable_orphan_reaping()  # dies with the daemon's main process
    set_process_title(f"WeightCache_{worker_process_name(config, rank)}")
    tune_gc()
    with contextlib.suppress(Exception):  # locality only, as in the engine's workers
        numa_bind_to_node(local_device_index(config, rank), f"weight cache rank {rank}")
    t0 = time.perf_counter()
    recorder = InputRecorder.start()
    # Loading runs no device collectives.
    device = init_model_env(config, rank, device_communicators=False)
    path = socket_path(device)
    lock = _DaemonLock(
        path,
        rank=rank,
        model=config.model,
        gpu=gpu_identity(device),
        idle_timeout_s=envs.ATOM_PRELOAD_IDLE_TIMEOUT_S,
    )
    stop = threading.Event()
    server = _load_and_serve(
        config,
        rank,
        device,
        recorder,
        path,
        lock,
        functools.partial(_stop_launch, lock, stop),
        t0,
    )
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    ready.put(rank)
    while not stop.wait(1.0):
        pass
    # Clients that mapped the weights keep them until they exit.
    server.close()
    lock.release()
    logger.info(f"{_TAG} rank {rank}: stopped serving")


def _stop_launch(lock, stop: threading.Event) -> None:
    """Idle timeout: shut the whole daemon down through its main process,
    as SIGTERM to the launch would."""
    main_pid = lock.info.get("daemon_pid")
    if main_pid:
        with contextlib.suppress(ProcessLookupError):
            os.kill(main_pid, signal.SIGTERM)
    else:
        stop.set()  # main process unknown: at least stop serving


class _RecordingLoad:
    """get_model's loader in the daemon: the disk load, recording what it
    reads and runs, and the model's state before loading."""

    def __init__(self, rank: int, recorder: InputRecorder):
        self.rank, self.recorder = rank, recorder
        self.tracer, self.kernels = _CodeTracer(), {}

    def __call__(self, config, model_class, device):
        self.key = compute_key(config, self.rank, device)
        self.recorder.enter_construct(config)
        try:
            model = build_model(config, model_class, device)
            self.built, self.before = structure(model), snapshot(model)
            with self.tracer, _record_jit(self.kernels):
                self.recorder.enter_load(model)
                with _skip_prepare_finalize(model):
                    load_weights(model, config)
        finally:
            # Before anything else reads the model.
            self.recorder.finish_load(self.tracer.codes)
        return model


def _load_and_serve(config, rank, device, recorder, path, lock, on_idle, t0) -> _Server:
    label = f"rank {rank}"
    load = _RecordingLoad(rank, recorder)
    model = get_model(config, device, load)
    reads = recorder.key_reads(config.model)
    after = snapshot(model)
    try:
        codes = code_fingerprint(load.tracer.codes, load.kernels.values())
    except _SourceChanged as e:
        raise RuntimeError(
            f"{_TAG} {label}: {e} changed after this daemon started, so the weights it "
            "loaded may not match the file; restart the daemon"
        ) from None
    export = build_export(model, load.before, after)
    views = export.pop("views")
    _release_other_memory(export, device, label)
    payload = {
        "protocol": PROTOCOL_VERSION,
        "key": load.key,
        "reads": reads,
        "code": codes,
        "structure": exported_structure(load.built, export),
        "export": export,
        "summary": {
            "pid": os.getpid(),
            "daemon_pid": lock.info.get("daemon_pid"),
            "bytes": export["bytes"],
        },
    }
    logger.info(
        f"{_TAG} {label}: load read {len(reads['env'])} env vars, {len(reads['config'])} config "
        f"fields, {len(reads['attrs'])} construction attributes, {len(reads['classattrs'])} class "
        f"attributes, {len(reads['globals'])} globals, {len(reads['files'])} files"
    )
    server = _Server(
        path,
        payload,
        views,
        label,
        idle_timeout_s=lock.info["idle_timeout_s"],
        on_idle=on_idle,
    )
    lock.set_state("serving", bytes=export["bytes"])
    entries = sum(
        len(i["params"]) + len(i["buffers"]) + len(i["attrs"])
        for i in export["plan"].values()
    )
    logger.info(
        f"{_TAG} {label}: serving {entries} entries ({export['bytes'] / (1 << 30):.2f} GiB, "
        f"{len(codes['entries'])} code entries) at {path} after {time.perf_counter() - t0:.1f}s"
    )
    if export["unshippable"]:
        logger.warning(
            f"{_TAG} {label}: loading created state that cannot be shared, so clients will "
            f"refuse this cache: {export['unshippable'][:10]}"
        )
    return server


def _release_other_memory(export: dict, device, label: str) -> None:
    """Everything but the weights is a client's KV cache lost: free all
    other device memory this process holds."""
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
    shared = export["storages"]
    for obj in gc.get_objects():
        with contextlib.suppress(Exception):
            if isinstance(obj, torch.Tensor) and obj.is_cuda:
                storage = obj.untyped_storage()
                if storage.nbytes() and storage.data_ptr() not in shared:
                    storage.resize_(0)
    torch.cuda.empty_cache()
    # Pinned staging buffers from loading stay cached in host memory.
    with contextlib.suppress(Exception):
        torch._C._host_emptyCache()
    free, total = torch.cuda.mem_get_info(device)
    gib, weights = 1 << 30, export["bytes"]
    reserved = torch.cuda.memory_reserved()
    logger.info(
        f"{_TAG} {label}: device memory used {(total - free) / gib:.2f} GiB: "
        f"weights {weights / gib:.2f}, other allocated "
        f"{(torch.cuda.memory_allocated() - weights) / gib:.2f}, allocator reserved-but-free "
        f"{(reserved - torch.cuda.memory_allocated()) / gib:.2f}, outside the allocator "
        f"{(total - free - reserved) / gib:.2f}"
    )
