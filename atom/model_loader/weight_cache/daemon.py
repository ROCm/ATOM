# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The daemon (``ATOM_PRELOAD=daemon``): one process per GPU that builds
and loads its shard as a server does (``init_model_env``, ``get_model``, and
the drafter's own weights) and holds it, and a main process whose socket
serves them all.

No engine, runner, scheduler or HTTP server: the daemon shares no port, store
or communicator with the servers that use it, and serving features need no
daemon mode."""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import gc
import glob
import logging
import multiprocessing
import os
import pickle
import queue
import signal
import threading
import time
from types import SimpleNamespace

import torch
from torch import nn

from atom.model_engine.model_setup import (
    build_model,
    construct_drafter,
    get_model,
    init_model_env,
    load_weights,
)
from atom.model_loader import loader
from atom.model_loader.weight_cache.export import (
    _leftover_meta,
    _maker,
    build_export,
    install,
    leave_out,
    process_export,
    process_snapshot,
    refill,
    share_storages,
    snapshot,
)
from atom.model_loader.weight_cache.key import (
    _diff_key,
    compute_key,
    exported_structure,
    structure,
)
from atom.model_loader.weight_cache.preload import (
    launch_ranks,
    mark_uncacheable,
    prune_sockets,
    unsupported,
)
from atom.model_loader.weight_cache.transport import (
    _DaemonLock,
    _Holder,
    _Server,
    daemon_socket,
    gpu_identity,
    socket_path,
)
from atom.model_loader.weight_cache.values import (
    _TAG,
    logger,
    release_checksum_buffers,
)
from atom.model_loader.weight_utils import local_model_dir, params_on_meta, swap_param
from atom.utils import (
    envs,
    get_hf_text_config,
    shutdown_all_processes,
    worker_process_name,
)

# Without a connected server: an idle daemon still holds every GPU's weights.
_IDLE_TIMEOUT_S = 3600


def _launch(engine_args, model: str):
    """The daemon's launch with the checkpoint at ``model``: (its rank
    processes' rendezvous, every (rank, its engine's Config) as the engines
    would run them, the key a server launch's main process has for each if
    this daemon serves it (session))."""
    from torch.distributed import TCPStore

    from atom.model_engine.llm_engine import prepare_config

    # The Config a server's runners get, launched with this --model.
    args = copy.copy(engine_args)
    args.model = model
    config, _ = prepare_config(model, **args._get_engine_kwargs())
    if unsupported(config):
        raise ValueError(
            "the weight cache (ATOM_PRELOAD) does not support "
            + ", ".join(unsupported(config))
        )
    # As the engine's CoreManager hosts it.
    pc = config.parallel_config
    store = TCPStore(
        pc.data_parallel_master_ip, 0, is_master=True, wait_for_workers=False
    )
    pc.data_parallel_base_port = store.port
    pc._managed_distributed_store = True
    ranks = launch_ranks(config)
    return store, ranks, [compute_key(cfg, rank) for rank, cfg in ranks]


def run_daemon(engine_args) -> None:
    """Called by the API server in place of starting an engine: open the
    daemon's socket, start one process per GPU of the launch, report when all
    hold their weights, and stop them on SIGINT/SIGTERM (``python -m
    atom.model_loader.weight_cache stop``), after the idle timeout or when
    one of them fails."""
    store, ranks, keys = _launch(engine_args, engine_args.model)
    n = len(ranks)
    ctx = multiprocessing.get_context("spawn")
    ready, all_serving = ctx.Queue(), threading.Event()
    # The server asks each rank process for its weights over a pipe.
    pipes = [ctx.Pipe() for _ in ranks]
    procs = [
        ctx.Process(
            target=_serve_rank,
            args=(rank, cfg, ready, i, pipes[i][1], keys[i]),
            name=f"WeightCache{i}",
        )
        for i, (rank, cfg) in enumerate(ranks)
    ]
    # A flag, and no logging: a signal can land inside a lock or a log write.
    signaled, idle = [], []
    signal.signal(signal.SIGINT, lambda *_: signaled.append(1))
    signal.signal(signal.SIGTERM, lambda *_: signaled.append(1))
    t0 = time.monotonic()
    # Before the ranks take their GPUs: their state names it to servers.
    prune_sockets()
    server = _Server(daemon_socket(os.getpid()), "daemon", ranks=n)
    server.watch_idle(_IDLE_TIMEOUT_S, lambda: idle.append(1), all_serving)
    # The GPU each rank holds, and the processes of a `load` under way.
    gpus, loaders = [None] * n, []
    server.load = lambda model: _load(engine_args, model, server, gpus, keys, loaders)
    for p in procs:
        p.start()
    serving: set[int] = set()
    exited = []
    while not (signaled or idle):
        with contextlib.suppress(queue.Empty):
            what, i, gpu = ready.get(timeout=1.0)
            if what == "holds":  # it has taken its GPU and loads
                gpus[i] = gpu
                server.add(gpu, pipes[i][0], keys[i])
                continue
            serving.add(i)
            if len(serving) == n:
                all_serving.set()
                logger.info(
                    f"{_TAG} daemon ready: {n} rank(s) hold their weights "
                    f"({time.monotonic() - t0:.0f}s)"
                )
        exited = [p for p in procs if p.exitcode is not None]
        if exited:
            break
    # A Ctrl-C reaches the ranks too: their exit is then no failure. Nor is
    # an exit with 0: that rank's load cannot be cached (_load_and_serve).
    gone = [] if signaled or idle else exited
    failed = [p.name for p in gone if p.exitcode != 0]
    if gone:
        say = logger.error if failed else logger.info
        names = [p.name for p in gone]
        say(f"{_TAG} daemon process(es) {names} exited; stopping the daemon")
    server.stopping = True  # no server gets weights that are about to go
    # Each lets go of its GPU and exits.
    shutdown_all_processes(procs + loaders, 60)
    server.close()
    del store
    logger.info(f"{_TAG} daemon stopped")
    if failed:
        raise SystemExit(1)


# What tells two checkpoints of one model apart in a launch's key.
_CHECKPOINT_KEYS = ("checkpoint:", "hf_config:", "draft.checkpoint:")


def _load(engine_args, model, server, gpus, keys, loaders) -> list[str]:
    """`load`: write the checkpoint at ``model`` into the storages this
    daemon's ranks hold, for the servers that have them and for launches to
    come: a group of modules at a time (_load_rank), with the servers stopped
    meanwhile. Returns why it was not done ([]: it was)."""
    if None in gpus:
        return ["the daemon's ranks have not all taken their GPUs yet"]
    why = server.cannot_stop(rewrite=True)
    if why:
        return ["a server that has the weights cannot take others:"] + why
    try:
        store, ranks, new = _launch(engine_args, model)
    except Exception as e:  # noqa: BLE001 - told to whoever asked
        return [f"this daemon's launch cannot be laid out for it: {e!r}"]
    why = [
        d
        for old_key, new_key in zip(keys, new)
        for d in _diff_key(old_key, new_key)
        if not d.startswith(_CHECKPOINT_KEYS)
    ]
    reads = [d for d in dict.fromkeys(why) if d.startswith("server_reads")]
    if reads:
        told = (
            "a server reads these from the checkpoint it starts with, by itself, "
            "and a load does not change them (start the servers on it instead):"
        )
        return [told, *reads]
    if why or len(new) != len(keys):
        return ["it is not another checkpoint of the model this daemon holds:"] + why
    if new == keys and not any(server._ask(gpu, "poisoned") for gpu in gpus):
        # What the daemon holds: there is nothing to write, only to use again,
        # also by a server that could not go on when it was loaded.
        waits = server.stopped or server.unfollowed
        return server.resume(changed=False) if waits else []
    t0 = time.monotonic()
    ctx = multiprocessing.get_context("spawn")
    pipes = [ctx.Pipe() for _ in ranks]
    conns = [conn for conn, _ in pipes]
    loaders += [
        ctx.Process(
            target=_load_rank, args=(rank, cfg, pipes[i][1]), name=f"WeightLoad{i}"
        )
        for i, (rank, cfg) in enumerate(ranks)
    ]
    try:
        for p in loaders:
            p.start()
        for gpu, conn in zip(gpus, conns):
            conn.send(server._ask(gpu, "payload"))
        why, _ = _loaded(loaders, conns)  # ready: only the writing is left
        if why:
            return why
        t1 = time.monotonic()
        why = [] if server.stopped else server.stop(load=model)
        if why:
            return ["a server that has the weights cannot stop using them:"] + why
        for gpu, conn in zip(gpus, conns):
            server._tell(gpu, "writing")
            conn.send("go")
        why, written = _loaded(loaders, conns)  # the keys they are under now
        for gpu, key in zip(gpus, written):
            why += server._ask(gpu, ("filled", key, model))
        if why:  # stays stopped: no server may go on with them
            return ["the weights are neither checkpoint's now, load again:"] + why
        keys[:] = new
        why = server.resume(changed=True, keys=dict(zip(gpus, new)), model=model)
        logger.info(
            f"{_TAG} daemon: loaded {model} into the weights it holds (ready in "
            f"{t1 - t0:.1f}s, servers stopped for {time.monotonic() - t1:.1f}s)"
        )
        if why:
            return ["the weights are loaded, but a server cannot go on:"] + why
        return []
    finally:
        shutdown_all_processes([p for p in loaders if p.pid], 10)
        del loaders[:], store


# A module this large loads by itself when a checkpoint is loaded into the
# weights a daemon holds; the smaller ones load together, first.
_GROUP_BYTES = 256 << 20


class _Groups:
    """`load_model`'s ``units`` for a load into the storages a daemon rank
    holds (``plan`` and ``make``: its export's): everything but the children
    of the model's outermost ModuleLists first, then those one at a time. A
    parameter the daemon holds as the model builds it is loaded where it is
    held; another one has memory of its own until its group is written.

    A module is processed seeing what it sees in any load: the modules before
    it processed, the ones after it not. Only that a group after it is not
    loaded yet either: reading its weights fails (they are on meta), where
    any other load would have read them unprocessed."""

    REST = ""

    def __init__(self, model: nn.Module, plan: dict, make, device):
        self.model, self.make, self.device = model, make, device
        self.written = 0
        # Where the time goes: reading the checkpoint into the groups, their
        # processing, writing them where the daemon holds them.
        self.seconds = {"read": 0.0, "process": 0.0, "write": 0.0}
        self._since = time.perf_counter()
        self.groups: dict[str, nn.Module] = {}
        for name, mod in model.named_modules():
            inside = any(name.startswith(g + ".") for g in self.groups)
            if isinstance(mod, nn.ModuleList) and not inside:
                for i, child in enumerate(mod):
                    held = sum(p.numel() * p.element_size() for p in child.parameters())
                    if held >= _GROUP_BYTES:
                        self.groups[f"{name}.{i}" if name else str(i)] = child
        self._of: dict[str, str] = {}
        # Each group's part of the plan, and its parameters as built:
        # (module, name, parameter, where the daemon holds it as built).
        self.plans: dict[str, dict] = {key: {} for key in self.order()}
        for path, item in plan.items():
            self.plans[self.of(path.partition("::")[0] + ".")][path] = item
        self.params: dict[str, list] = {key: [] for key in self.order()}
        for path, mod in model.named_modules():
            for k, p in mod._parameters.items():
                if p is None:
                    continue
                meta = plan.get(path, {}).get("params", {}).get(k) or {}
                there = (
                    isinstance(meta.get("storage"), int)
                    and tuple(meta["shape"]) == tuple(p.shape)
                    and meta["dtype"] == p.dtype
                )
                self.params[self.of(path + ".")].append(
                    (mod, k, p, meta if there else None)
                )

    def _lap(self, phase: str) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)  # what was started is that phase's
        now = time.perf_counter()
        self.seconds[phase] += now - self._since
        self._since = now

    def order(self) -> list[str]:
        return [self.REST, *self.groups]

    def of(self, name: str) -> str:
        key = self._of.get(name)
        if key is None:
            match = (g for g in self.groups if name.startswith(g + "."))
            key = self._of[name] = next(match, self.REST)
        return key

    def modules(self, key: str) -> list:
        self._lap("read")  # asked for once the group's tensors are in
        if key != self.REST:
            named = self.groups[key].named_modules()
            return [(f"{key}.{name}" if name else key, mod) for name, mod in named]
        # The ones before the first group; those after the last one are
        # processed after all, as in any load.
        before = []
        for name, mod in self.model.named_modules():
            if name in self.groups:
                break
            before.append((name, mod))
        return before

    def own_bytes(self, key: str) -> int:
        """What a group's parameters take beside the weights while it loads."""
        return sum(
            p.numel() * p.element_size()
            for _, _, p, there in self.params[key]
            if there is None
        )

    def begin(self, key: str) -> None:
        self._since = time.perf_counter()
        for _, k, p, there in self.params[key]:
            if not p.is_meta:
                continue  # tied to one that has memory
            data = None if there is None else self.make(there)
            if data is None or not data.is_contiguous():
                data = torch.empty(p.shape, dtype=p.dtype, device=self.device)
                data.view(torch.uint8).zero_()
            new = nn.Parameter(data, requires_grad=p.requires_grad)
            # The object the loader has, with the attributes it reads.
            if not swap_param(p, new, dict(vars(p))):
                raise RuntimeError(f"{key}.{k}: its parameter cannot be given memory")

    def end(self, key: str) -> None:
        if key != self.REST:  # that one: once all its modules are processed
            self.write(key)

    def write(self, key: str) -> None:
        """Write a loaded group where the daemon holds it, and have the model
        hold that from here on: the group's own memory is free again."""
        self._lap("process")
        self.written += refill(self.model, self.plans[key], self.make)
        install(self.model, self.plans[key], self.make)
        for mod, k, p, _ in self.params[key]:
            if mod._parameters.get(k) is not p:  # replaced: what was read
                p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        torch.cuda.empty_cache()
        self._lap("write")


class _HeldLoad:
    """get_model's loader for a `load`: build on meta, and write the
    checkpoint into the storages the daemon's rank holds (``held``: their
    payload), a group of modules at a time (_Groups), once the main process
    says the servers are stopped. So the weights are never there twice."""

    def __init__(self, rank: int, held: dict, conn):
        self.rank, self.held, self.conn = rank, held, conn

    def __call__(self, config, model_class, device):
        held, label = self.held, worker_process_name(config, self.rank)
        self.key = compute_key(config, self.rank)
        export, process = held["export"], held["process"]
        with params_on_meta():
            model = build_model(config, model_class, device)
        why = _diff_key(held["structure"], exported_structure(structure(model), export))
        if why:
            raise RuntimeError(f"it builds another model than the one held: {why[:8]}")
        if any(hasattr(m, "get_streaming_deferred_modules") for m in model.modules()):
            raise RuntimeError("its modules combine each other's weights at load")
        if process["storages"] or process["host"]:
            raise RuntimeError("loading it creates tensors outside the model")
        if _has_drafter(config) != bool(held["draft"]):
            raise RuntimeError("the servers have a drafter the daemon does not hold")
        make, _ = _maker(export, device)
        groups = _Groups(model, export["plan"], make, device)
        # A group's own memory twice, for what its processing makes of it.
        own = list(map(groups.own_bytes, groups.order()))
        need = own[0] + 2 * max(own) + (2 << 30)
        free = torch.cuda.mem_get_info(device)[0]
        if free < need:
            raise RuntimeError(
                f"{free / (1 << 30):.0f} GiB free on its GPU, and its largest "
                f"group of modules takes about {need / (1 << 30):.0f} GiB to load"
            )
        _read_ahead(config)
        loader.safetensors_weights_iterator = _indexed_iterator(config)
        states = [process_snapshot(config)]
        self.conn.send((None, "ready"))
        self.conn.recv()  # the go: the servers are stopped
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)
        level = logger.level
        logger.setLevel(logging.WARNING)  # the loader says the same of every group
        try:
            load_weights(model, config, units=groups)
        finally:
            logger.setLevel(level)
        groups.write(groups.REST)
        states.append(process_snapshot(config))
        changed = process_export(states)
        why = _leftover_meta(model) + changed["unshippable"]
        if (changed["changes"], changed["env"]) != (process["changes"], process["env"]):
            why.append("other globals or environment variables")
        if why:
            raise RuntimeError(
                f"loading it leaves other state than the held: {why[:8]}"
            )
        if held["draft"]:
            draft = _load_draft(config, device, model, self.rank)
            if draft is None:
                raise RuntimeError("its drafter could not be loaded")
            plan = held["draft"]["export"]["plan"]
            groups.written += refill(
                draft.model, plan, _maker(held["draft"]["export"], device)[0]
            )
        torch.cuda.synchronize(device)
        took = ", ".join(f"{k} {s:.1f}s" for k, s in groups.seconds.items())
        logger.info(
            f"{_TAG} {label}: wrote {groups.written / (1 << 30):.2f} GiB into the "
            f"weights the daemon holds in {time.perf_counter() - t0:.1f}s ({took}), "
            f"{len(groups.groups)} module(s) by themselves, with at most "
            f"{torch.cuda.max_memory_allocated(device) / (1 << 30):.2f} GiB of its own"
        )
        return model


def _read_ahead(config) -> None:
    """Read this rank's share of the checkpoint's files once: the page cache
    has them when the servers stop for the load, and the loader's own
    prefetch, which reads whole files, is of no use to a group's tensors."""
    from atom.model_loader.weight_iterator import _node_local_rank, _read_whole_file

    if envs.ATOM_LOADER_PREFETCH:
        root = local_model_dir(config.model)
        files = sorted(glob.glob(os.path.join(root, "*.safetensors")))
        rank, world = _node_local_rank()
        block = envs.ATOM_LOADER_PREFETCH_BLOCK_MB << 20
        threads = max(1, envs.ATOM_LOADER_PREFETCH_THREADS)
        with concurrent.futures.ThreadPoolExecutor(threads) as pool:
            list(pool.map(lambda p: _read_whole_file(p, block), files[rank::world]))
    os.environ["ATOM_LOADER_PREFETCH"] = "false"


def _indexed_iterator(config):
    """The loader's weights iterator for a load that reads the checkpoint
    once per group (_Groups): which shard holds which tensors is read here,
    once, not from every shard's header at every pass (~1 s each on
    DeepSeek-V4.1-Flash)."""
    from atom.model_loader import weight_iterator as wi

    root = os.path.realpath(local_model_dir(config.model))
    files = sorted(
        wi.filter_duplicate_safetensors_files(
            glob.glob(os.path.join(root, "*.safetensors")),
            root,
            wi.SAFE_WEIGHTS_INDEX_NAME,
        )
    )
    names = {f: wi._shard_tensor_names(f) for f in files}

    def weights(path, disable_mmap=False, wants=None):
        if os.path.realpath(local_model_dir(path)) != root:  # a drafter's own
            yield from wi.safetensors_weights_iterator(path, disable_mmap, wants)
            return
        kept = [f for f in files if names[f] is None or any(map(wants, names[f]))]
        yield from wi._iter_shards(kept, disable_mmap, False, wants)

    return weights


def _load_rank(rank: int, config, conn) -> None:
    """One GPU's process of a `load`: the daemon rank's payload comes on
    ``conn``; it says when it is ready to write, writes on the go, and sends
    the key the weights are under then."""
    from atom.utils import enable_orphan_reaping, set_process_title

    enable_orphan_reaping()
    label = worker_process_name(config, rank)
    set_process_title(f"WeightLoad_{label}")
    try:
        loader.MOE_PREPARE_FINALIZE = False
        device = _init_env(config, rank)
        load = _HeldLoad(rank, pickle.loads(conn.recv()), conn)
        get_model(config, device, load)
        conn.send((None, load.key))
    except Exception as e:  # told to the main process
        conn.send((f"{label}: {e!r}", None))
        raise
    signal.pause()


def _loaded(procs, conns) -> tuple[list[str], list]:
    """What a `load`'s rank processes send when they have loaded: (why one
    failed, each one's payload as it travels). The first failure ends the
    wait: the others may wait for that rank in a collective."""
    sent = [None] * len(procs)
    while None in sent:
        for i, (p, conn) in enumerate(zip(procs, conns)):
            if sent[i] is None and conn.poll(0.2):
                sent[i] = conn.recv()
            elif sent[i] is None and p.exitcode is not None and not conn.poll():
                sent[i] = (f"{p.name} exited with {p.exitcode}", None)
        failed = [s[0] for s in sent if s and s[0]]
        if failed:
            return failed, []
    return [], [s[1] for s in sent]


def _serve_rank(rank: int, config, ready, index: int, conn, launch_key=None) -> None:
    """One GPU's daemon process: load this rank's shard, then give it to the
    main process when that asks on ``conn``, until stopped. ``index``: this
    rank on ``ready``, where it reports the GPU it holds and that it serves;
    ``launch_key``: what a launch's main process knows this load by."""
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
        # The launch's main process: SIGTERM there stops the whole daemon,
        # and servers connect to its socket.
        daemon_pid=os.getppid(),
        socket=daemon_socket(os.getppid()),
        log=_log_file(),
        idle_timeout_s=_IDLE_TIMEOUT_S,
    )
    ready.put(("holds", index, os.path.basename(path)))
    # MoE communication setup is left to the servers that take the weights.
    loader.MOE_PREPARE_FINALIZE = False
    device = _init_env(config, rank)
    holder, pool = _load_and_serve(config, rank, device, conn, lock, t0, launch_key)
    # A flag: a handler that takes a lock could land inside its holder.
    signaled = []
    signal.signal(signal.SIGTERM, lambda *_: signaled.append(1))
    signal.signal(signal.SIGINT, lambda *_: signaled.append(1))
    ready.put(("serves", index, None))
    holder.serve(lambda: bool(signaled))
    # Clients that mapped the weights keep them until they exit.
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
    # What streaming online quantization kept on the modules for the load
    # only (locks among it).
    for mod in rec.model.modules():
        for k in [k for k in vars(mod) if k.startswith("_stream_")]:
            if k != "_stream_online_quant":
                delattr(mod, k)
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
        if _has_drafter(config):
            self.draft = _load_draft(config, device, self.rec.model, self.rank)
        return self.rec.model


def _has_drafter(config) -> bool:
    """Whether this rank's runner has a drafter: the last PP stage's do."""
    stage = config.parallel_config.pipeline_parallel_rank
    return bool(config.speculative_config) and (
        stage == config.pipeline_parallel_size - 1
    )


def _init_env(config, rank: int):
    """`init_model_env` for a process that only loads: without the device
    communicators (several GiB per GPU), unless building the model takes one
    (sequence parallelism's head exchange)."""
    return init_model_env(
        config, rank, device_communicators=config.sequence_parallel_size > 1
    )


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


def _load_and_serve(config, rank, device, conn, lock, t0, launch_key=None):
    """Load this rank's shard and export it: (what gives it to the main
    process on ``conn``, the pool that holds it)."""
    label = worker_process_name(config, rank)
    load = _DaemonLoad(rank)
    get_model(config, device, load)
    rec = load.rec
    export, process = rec.export, rec.process
    if export["unshippable"]:
        # The same load leaves the same state: no daemon is started for it
        # again (a launch's main process looks it up by its own key).
        mark_uncacheable(launch_key or load.key, export["unshippable"])
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
    holder = _Holder(conn, payload, views, label, lock)
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
        f"after {time.perf_counter() - t0:.1f}s"
    )
    return holder, pool


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
