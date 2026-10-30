# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tensor parallelism across nodes.

The node that runs TP rank 0 runs the engine, the API server and its own ranks'
workers. Every other node runs only its workers (:func:`serve_remote_tp_ranks`).
The engine reaches them the way it reaches local ones, through
:class:`~atom.model_engine.async_proc.AsyncIOProcManager`'s broadcast queue and
reply channels, whose remote ends listen on TCP. The manager publishes those
endpoints, with the runner's constructor arguments, in the model-runner
TCPStore that CoreManager serves on ``data_parallel_master_ip`` and
``data_parallel_base_port``; the workers' process group meets there too.
"""

import logging
import pickle
import signal
import time
from datetime import timedelta

from atom.utils import envs, get_mp_context, shutdown_all_processes

logger = logging.getLogger("atom")

BOOTSTRAP_KEY = "atom/tp_multinode/bootstrap"
_READY = b"READY"
# What an XPUB socket in verbose mode receives when a subscriber joins.
_SUBSCRIBED = b"\x01"
# A short-timeout store client, so that losing the engine's store is noticed
# in seconds rather than after ATOM_MULTINODE_TP_TIMEOUT_S.
_WATCHDOG_TIMEOUT_S = 30
_WATCHDOG_MISSES = 3


def _store(ip: str, port: int, timeout_s: float):
    from torch.distributed import TCPStore

    return TCPStore(ip, port, is_master=False, timeout=timedelta(seconds=timeout_s))


def publish_bootstrap(config, info: dict) -> None:
    """Engine side: hand the other nodes what their workers are built from."""
    pc = config.parallel_config
    store = _store(
        pc.data_parallel_master_ip,
        pc.data_parallel_base_port,
        envs.ATOM_MULTINODE_TP_TIMEOUT_S,
    )
    store.set(BOOTSTRAP_KEY, pickle.dumps(info))


def release_remote_readers(mq, n_remote: int, is_running) -> None:
    """Engine side: wait for every remote reader to subscribe to the broadcast
    queue, then let them read. A PUB socket drops whatever it sends before a
    subscriber is connected."""
    sock = mq.remote_socket
    timeout_s = envs.ATOM_MULTINODE_TP_TIMEOUT_S
    deadline = time.monotonic() + timeout_s
    subscribed, next_log = 0, 0.0
    while subscribed < n_remote:
        now = time.monotonic()
        if now >= next_log:
            logger.info(
                "Waiting for %d of %d TP ranks on other nodes to subscribe",
                n_remote - subscribed,
                n_remote,
            )
            next_log = now + 60
        if not is_running():
            raise RuntimeError("shut down while waiting for TP ranks on other nodes")
        if now > deadline:
            raise TimeoutError(
                f"{n_remote - subscribed} of {n_remote} TP ranks on other nodes "
                f"did not subscribe within {timeout_s:.0f}s"
            )
        if sock.poll(timeout=1000) and sock.recv()[:1] == _SUBSCRIBED:
            subscribed += 1
    sock.send(_READY)
    logger.info("All %d TP ranks on other nodes subscribed", n_remote)


def await_release(mq) -> None:
    """Remote rank side of :func:`release_remote_readers`."""
    timeout_s = envs.ATOM_MULTINODE_TP_TIMEOUT_S
    if not mq.remote_socket.poll(timeout=int(timeout_s * 1000)):
        raise TimeoutError(
            f"the engine did not release this rank's broadcast queue within "
            f"{timeout_s:.0f}s"
        )
    msg = mq.remote_socket.recv()
    if msg != _READY:
        raise RuntimeError(
            f"expected {_READY!r} on the broadcast queue, got {msg[:32]!r}"
        )


def serve_remote_tp_ranks(engine_args) -> int:
    """Run this node's TP ranks for the engine on another node, until they end.

    Returns 0 when every worker exited cleanly (the engine shut them down),
    1 otherwise.
    """
    from atom.model_engine.async_proc import AsyncIOProc

    start = engine_args.tensor_parallel_rank
    local = engine_args.tensor_parallel_size_local
    ip = engine_args.data_parallel_master_ip
    port = engine_args.data_parallel_base_port
    if not local or start % local or not port or ip in ("127.0.0.1", "localhost"):
        raise ValueError(
            "a node that runs only TP workers needs --tensor-parallel-size-local, "
            "a --tensor-parallel-rank that is a multiple of it, and the engine "
            "node's --data-parallel-master-ip and --data-parallel-base-port"
        )
    ranks = range(start, start + local)
    logger.info(
        "TP ranks %d-%d: waiting for the engine at %s:%d", start, ranks[-1], ip, port
    )
    store = _store(ip, port, envs.ATOM_MULTINODE_TP_TIMEOUT_S)
    payload = store.get(BOOTSTRAP_KEY)
    # The engine's config holds classes that a trust_remote_code model defines
    # in transformers' dynamic modules; loading the model's config registers
    # them here, and the workers inherit the path through spawn.
    from atom.config import get_hf_config

    get_hf_config(engine_args.model, trust_remote_code=engine_args.trust_remote_code)
    info = pickle.loads(payload)
    proc_num = info["proc_num"]
    if ranks[-1] >= proc_num:
        raise ValueError(
            f"TP ranks {start}-{ranks[-1]} do not exist: the engine runs {proc_num}"
        )
    config, *runner_args = info["args"]
    # The engine's copy describes its own node.
    config.parallel_config.tensor_parallel_size_local = local
    config.parallel_config.tensor_parallel_rank = start

    ctx = get_mp_context()
    procs = []
    for rank in ranks:
        label = f"ModelRunner{rank}/{proc_num}"
        proc = ctx.Process(
            target=AsyncIOProc,
            name=label,
            args=(
                label,
                [None, None],
                info["handle"],
                info["runner"],
                rank,
                info["kv_output_addrs"][rank],
                None,
                config,
                *runner_args,
            ),
            kwargs={"rpc_output_addr": info["rpc_output_addrs"][rank]},
        )
        proc.start()
        procs.append(proc)
    logger.info("TP ranks %d-%d started for the engine at %s", start, ranks[-1], ip)
    return _supervise(procs, ip, port)


def _supervise(procs, ip: str, port: int) -> int:
    """Wait for the workers. One that fails, a signal, or losing the engine's
    store (it lives in the engine node's server process) stops all of them."""
    stopping = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))

    watchdog, misses = None, 0
    status = 1
    while not stopping:
        exited = [p for p in procs if not p.is_alive()]
        failed = [p for p in exited if p.exitcode != 0]
        if failed:
            logger.error(
                "%s exited with %s; stopping this node's TP ranks",
                failed[0].name,
                failed[0].exitcode,
            )
            break
        if len(exited) == len(procs):
            status = 0
            break
        try:
            if watchdog is None:
                watchdog = _store(ip, port, _WATCHDOG_TIMEOUT_S)
            watchdog.check([BOOTSTRAP_KEY])
            misses = 0
        except Exception as e:  # noqa: BLE001 - any store failure counts as a miss
            watchdog, misses = None, misses + 1
            logger.warning("Engine store at %s:%d unreachable: %s", ip, port, e)
            if misses >= _WATCHDOG_MISSES:
                logger.error("Lost the engine; stopping this node's TP ranks")
                break
        time.sleep(5)
    if stopping:
        logger.info("Signal %d: stopping this node's TP ranks", stopping[0])
    shutdown_all_processes(procs, allowed_seconds=5)
    return status
