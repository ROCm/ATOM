# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Asynchronous I/O process management for model runner workers.

This module provides:

- :class:`AsyncIOProc`: A single worker process that runs a model runner
  and communicates via ZMQ sockets and shared-memory broadcast queues.
- :class:`AsyncIOProcManager`: Manages multiple ``AsyncIOProc`` workers,
  routes function calls via broadcast, and aggregates KV transfer outputs
  from all workers.
"""

import logging
import multiprocessing
import pickle
import queue
import threading
import time
import weakref
from contextlib import ExitStack
from dataclasses import dataclass
from threading import Thread
from typing import ClassVar

import zmq
import zmq.asyncio
from aiter.dist.shm_broadcast import MessageQueue

from atom.kv_transfer.disaggregation import KVConnectorOutput, KVOutputAggregator
from atom.model_engine.collective_rpc import RpcPayload, RpcResult
from atom.utils import (
    get_mp_context,
    get_open_zmq_ipc_path,
    init_exit_handler,
    make_zmq_socket,
    resolve_obj_by_qualname,
    set_process_title,
    shutdown_all_processes,
    worker_process_name,
)
from atom.utils.gc_utils import maybe_attach_gc_debug_callback, tune_gc
from atom.utils.numa_utils import numa_bind_to_node

logger = logging.getLogger("atom")


@dataclass
class _PendingKvAggregation:
    """Worker outputs retained for one outstanding KV aggregation RPC.

    Worker replies drain once and carry no generation identifier, so at most
    one aggregation may be outstanding. ``worker_outputs[rank]`` stays
    ``None`` until that rank's queue is consumed.
    """

    worker_outputs: list[KVConnectorOutput | None]
    started_at: float
    wait_warning_logged: bool = False

    def missing_worker_ranks(self) -> list[int]:
        return [
            rank for rank, output in enumerate(self.worker_outputs) if output is None
        ]


class AsyncIOProc:
    """A single worker process that runs a model runner with ZMQ I/O.

    Each worker receives function calls via shared-memory broadcast,
    executes them on the runner, and sends results back via ZMQ.
    KV aggregation outputs are sent on a dedicated channel to avoid
    mixing with regular forward outputs.

    Args:
        label: Human-readable label for logging.
        io_addrs: ``(input_addr, output_addr)`` ZMQ endpoints.
        input_shm_handle: Shared memory handle for the broadcast queue.
        runner_qualname: Fully qualified class name of the runner to instantiate.
        rank: TP rank of this worker.
        kv_output_addr: Optional ZMQ endpoint for KV aggregation output.
        rpc_output_addr: Optional ZMQ endpoint for generic collective-RPC
            replies. Every rank owns one, unlike the primary channel which
            only rank 0 has, so the caller can prove all ranks answered.
    """

    # Function names whose output goes to the KV channel instead of primary
    _KV_FUNC_NAMES = frozenset(["async_proc_aggregation"])

    def __init__(
        self,
        label: str,
        io_addrs: tuple[str, str],
        input_shm_handle: int,
        runner_qualname: str,
        rank: int,
        kv_output_addr: str | None = None,
        all_ranks_barrier=None,
        *args,
        rpc_output_addr: str | None = None,
        **kwargs,
    ):
        # Bind this worker's lifetime to its parent EngineCore: if the parent
        # exits for any reason, have the kernel reap this process immediately
        # instead of leaving it orphaned. A ModelRunner worker holds a large GPU
        # allocation and the custom all-reduce IPC resources; an orphan blocks
        # forever in busy_loop() on the shm dequeue while keeping those pinned,
        # causing the stale-IPC all-reduce crash on the next restart. Must be
        # armed here, before any GPU / IPC state is created.
        from atom.utils import enable_orphan_reaping

        enable_orphan_reaping()

        # Named once, before anything logs: `ps`, the GC lines here and the
        # freeze line the runner emits over RPC all have to be the same string,
        # or a dp deployment's workers are indistinguishable from each other.
        name = worker_process_name(args[0] if args else None, rank)
        set_process_title(name)

        # Second-order next to the EngineCore's call but not zero: without it
        # a freeze still lands at the wave boundary, where the prefill burst
        # churns enough objects to trigger gen-2 here. Runs before the model
        # is built so the thresholds cover the startup heap. Freezing that heap
        # happens later, via the EngineCore's `freeze_gc_heap` RPC.
        tune_gc()
        maybe_attach_gc_debug_callback(name)

        # NUMA-local CPU/memory pinning (see atom.utils.numa_utils).
        # Auto-detects the GPU's local node by default; gated by
        # ATOM_NUMA_BIND. Must run before any large allocation / native
        # (mooncake) thread spawn so the mask is inherited by child threads and
        # first-touch lands memory locally. The node-local GPU index is
        # dp_local_rank*tp_size+tp_rank (engine_core_mgr GPU assignment).
        # Best-effort: any failure here is logged and skipped rather than
        # taking the worker down, hence the blanket except.
        try:
            cfg = args[0]
            # Node-local: this indexes a GPU on THIS machine. The global rank
            # would run off the end of a later node's device list.
            dp_local_rank = cfg.parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = cfg.parallel_config.data_parallel_rank
            gpu = dp_local_rank * cfg.tp_world_size + rank
            numa_bind_to_node(gpu, label)
        except Exception as e:  # noqa: BLE001 - binding is an optimization
            # NUMA binding only affects locality, never correctness, so any
            # failure (missing libnuma, restricted cpuset, odd topology) must
            # degrade to an unbound worker rather than kill it.
            logger.warning(f"AsyncIOProc({label}): NUMA bind skipped: {e}")
        self.label = f"AsyncIOProc({label})"
        self.rank = rank
        self.io_addrs = io_addrs
        self.io_queues = queue.Queue(), queue.Queue()
        self.io_threads: list[threading.Thread] = []

        # KV aggregation output channel
        self.kv_output_addr = kv_output_addr
        self.kv_queue: queue.Queue | None = None

        # Generic collective-RPC reply channel. Every rank has one, so a caller
        # can tell "all ranks finished" from "rank 0 finished".
        self.rpc_output_addr = rpc_output_addr
        self.rpc_queue: queue.Queue | None = None

        self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, rank)
        import atexit

        atexit.register(self._cleanup_shared_memory)
        init_exit_handler(self)

        # Start I/O threads for primary input/output
        for addr, q, func in zip(
            self.io_addrs,
            self.io_queues,
            [self.recv_input_from_socket, self.send_output_to_socket],
        ):
            if addr is None:
                continue
            t = threading.Thread(target=func, args=(addr, q), daemon=True)
            t.start()
            self.io_threads.append(t)

        # Dedicated KV aggregation output thread
        if self.kv_output_addr is not None:
            self.kv_queue = queue.Queue()
            t = threading.Thread(
                target=self.send_output_to_socket,
                args=(self.kv_output_addr, self.kv_queue),
                daemon=True,
            )
            t.start()
            self.io_threads.append(t)

        # Dedicated collective-RPC reply thread
        if self.rpc_output_addr is not None:
            self.rpc_queue = queue.Queue()
            t = threading.Thread(
                target=self.send_output_to_socket,
                args=(self.rpc_output_addr, self.rpc_queue),
                daemon=True,
            )
            t.start()
            self.io_threads.append(t)

        self.all_ranks_barrier = all_ranks_barrier

        runner_class = resolve_obj_by_qualname(runner_qualname)
        self.runners: list[object] = []
        self.runners = [runner_class(rank, *args, **kwargs)]
        self.busy_loop()

    def exit(self):
        if not getattr(self, "still_running", True):
            return
        self.still_running = False
        logger.debug(f"{self.label}: Shutting down runner...")
        for el in self.runners:
            el.exit()
        # Close shared memory reader handle to prevent resource_tracker leak
        self._cleanup_shared_memory()
        for t in self.io_threads:
            t.join(timeout=0.5)

    def _cleanup_shared_memory(self):
        """Close shared memory handles owned by this process."""
        if hasattr(self, "rpc_broadcast_mq"):
            mq = self.rpc_broadcast_mq
            if hasattr(mq, "buffer") and hasattr(mq.buffer, "shared_memory"):
                try:
                    mq.buffer.shared_memory.close()
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    # Already-closed handles and a racing peer's unlink both
                    # land here; neither is actionable during shutdown, but a
                    # silent pass hides genuine leaks from anyone debugging one.
                    logger.debug(f"{self.label}: shm close failed: {e}")

    def recv_input_from_socket(self, addr: str, input_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            socket.send(b"")
            poller.register(socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")

            while getattr(self, "still_running", True):
                for socket, _ in poller.poll(timeout=1000):
                    serialized_obj = socket.recv(copy=False)
                    input_obj = pickle.loads(serialized_obj)
                    input_queue.put_nowait(input_obj)

    def send_output_to_socket(self, addr: str, output_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                result = output_queue.get()
                serialized_obj = pickle.dumps(result)
                socket.send(serialized_obj)

    # Functions that require all TP ranks to synchronize via barrier before
    # rank 0 returns, so the caller can safely reuse/overwrite shared buffers.
    _BARRIER_FUNCS: ClassVar[set[str]] = {
        "update_weights_from_ipc",
        "update_weights_from_shm",
    }

    def busy_loop(self):
        """Main event loop: dequeue RPCs and dispatch to runners."""
        while True:
            func_name, args = self.get_func()
            payload = (
                args[0] if len(args) == 1 and isinstance(args[0], RpcPayload) else None
            )
            if payload is None:
                call_args, call_kwargs = args, {}
                need_barrier = func_name in self._BARRIER_FUNCS
            else:
                call_args = payload.args
                call_kwargs = payload.call_kwargs()
                need_barrier = payload.barrier

            for runner in self.runners:
                if payload is not None:
                    out = self._run_generic_rpc(
                        runner, func_name, call_args, call_kwargs, payload
                    )
                else:
                    func = getattr(runner, func_name, None)
                    if func is None:
                        continue
                    out = func(*call_args)
                if need_barrier and self.all_ranks_barrier is not None:
                    self.all_ranks_barrier.wait()
                if payload is not None:
                    # Generic replies go to this rank's own channel. Routing
                    # them to the primary would drop every rank but 0, which is
                    # the whole limitation the channel exists to remove.
                    if self.rpc_queue is not None:
                        self.rpc_queue.put_nowait(out)
                    elif self.io_addrs[1] is not None:
                        # No dedicated channel configured (older manager, or a
                        # test): fall back rather than silently discard.
                        self.io_queues[1].put_nowait(out)
                elif out is not None:
                    if (
                        self.io_addrs[1] is not None
                        and func_name not in self._KV_FUNC_NAMES
                    ):
                        self.io_queues[1].put_nowait(out)
                    if self.kv_queue is not None and func_name in self._KV_FUNC_NAMES:
                        self.kv_queue.put_nowait(out)
            if func_name == "exit":
                break
        logger.debug(f"{self.label}: exit busy_loop...")

    def _run_generic_rpc(
        self,
        runner: object,
        func_name: str,
        call_args: tuple,
        call_kwargs: dict,
        payload: RpcPayload,
    ) -> RpcResult:
        """Invoke one generic RPC, converting every outcome into a reply.

        Never raises and never returns ``None``: a missing method, a raising
        target, and an unpicklable return all become an ``RpcResult`` carrying
        ``error``. Anything else would leave the caller blocked in an untimed
        queue get, which is how a typo in a method name currently costs five
        minutes and reports a timeout instead of the typo.
        """
        func = getattr(runner, func_name, None)
        if func is None:
            return RpcResult(
                payload.request_id,
                self.rank,
                error=f"{type(runner).__name__} has no method {func_name!r}",
            )
        try:
            result = RpcResult(
                payload.request_id, self.rank, value=func(*call_args, **call_kwargs)
            )
        except Exception as exc:  # noqa: BLE001 - reported to the caller instead
            return RpcResult(
                payload.request_id,
                self.rank,
                error=f"{type(exc).__name__}: {exc}",
            )
        try:
            # The reply crosses a ZMQ socket, so an unpicklable value would kill
            # the sender thread rather than fail this call. Find out here.
            pickle.dumps(result)
        except Exception as exc:  # noqa: BLE001 - same reason
            return RpcResult(
                payload.request_id,
                self.rank,
                error=f"unpicklable result from {func_name!r}: {type(exc).__name__}: {exc}",
            )
        return result

    def get_func(self):
        method_name, *args = self.rpc_broadcast_mq.dequeue()
        return method_name, args


class AsyncIOProcManager:
    """Manages a pool of :class:`AsyncIOProc` workers.

    Handles process lifecycle, function dispatch via shared-memory broadcast,
    and KV output aggregation across all workers.

    The manager maintains two output channels:
    - **Primary channel** (rank 0 only): Regular forward outputs.
    - **KV channels** (all ranks): Per-worker KV transfer status, aggregated
      by :class:`KVOutputAggregator` before returning to the caller.

    Args:
        finalizer: Callback invoked when the manager shuts down.
        proc_num: Number of worker processes (= TP world size).
        runner: Fully qualified class name of the model runner.
        *args: Additional arguments forwarded to the runner constructor.
    """

    def __init__(self, finalizer, proc_num: int, runner: str, *args):
        self.parent_finalizer = finalizer
        self.proc_num = proc_num

        io_addrs = [get_open_zmq_ipc_path(), get_open_zmq_ipc_path()]
        self.procs: list[multiprocessing.Process] = []
        ctx = get_mp_context()
        self.runner_label = runner.split(".")[-1]
        self.label = f"AsyncIOProcManager({self.runner_label})"

        self.rpc_broadcast_mq = MessageQueue(
            proc_num, proc_num, max_chunk_bytes=16 * 1024 * 1024
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        self.still_running = True
        # Register atexit to clean up shared memory even if exit() doesn't complete
        import atexit

        atexit.register(self._cleanup_shared_memory)
        self.all_ranks_barrier = ctx.Barrier(proc_num)
        init_exit_handler(self)

        # KV output aggregation infrastructure
        self.kv_output_aggregator: KVOutputAggregator | None = None
        self.kv_output_addrs = [get_open_zmq_ipc_path() for _ in range(proc_num)]
        self.kv_outputs_queues: list[queue.Queue] = [
            queue.Queue() for _ in range(proc_num)
        ]
        self._pending_kv_aggregation: _PendingKvAggregation | None = None
        self.kv_output_threads: list[threading.Thread] = []

        # Generic collective-RPC reply channels, one per rank
        self.rpc_output_addrs = [get_open_zmq_ipc_path() for _ in range(proc_num)]
        self.rpc_outputs_queues: list[queue.Queue] = [
            queue.Queue() for _ in range(proc_num)
        ]
        self.rpc_output_threads: list[threading.Thread] = []

        for i in range(proc_num):
            label = f"ModelRunner{i}/{proc_num}"
            # Only rank 0 gets the primary output address
            addrs = [None, io_addrs[1]] if i == 0 else [None, None]

            process = ctx.Process(
                target=AsyncIOProc,
                name=label,
                args=(
                    label,
                    addrs,
                    scheduler_output_handle,
                    runner,
                    i,
                    self.kv_output_addrs[i],
                    self.all_ranks_barrier,
                    *args,
                ),
                # Keyword-only on AsyncIOProc, so it cannot be mistaken for one
                # of the *args forwarded to the runner's constructor.
                kwargs={"rpc_output_addr": self.rpc_output_addrs[i]},
            )
            process.start()
            self.procs.append(process)

        self.zmq_ctx = zmq.Context(io_threads=2)

        # Primary output queue (rank 0 only)
        self.outputs_queue: queue.Queue = queue.Queue()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            name=f"{self.label}_output_thread",
            args=(io_addrs[1],),
            daemon=True,
        )
        self.output_thread.start()

        # Per-worker KV output channels
        for i, output_addr in enumerate(self.kv_output_addrs):
            t = threading.Thread(
                target=self.process_kv_output_sockets,
                name=f"{self.label}_kv_output_thread_{i}",
                args=(output_addr, i),
                daemon=True,
            )
            t.start()
            self.kv_output_threads.append(t)

        # Per-worker collective-RPC reply channels
        for i, output_addr in enumerate(self.rpc_output_addrs):
            t = threading.Thread(
                target=self.process_rpc_output_sockets,
                name=f"{self.label}_rpc_output_thread_{i}",
                args=(output_addr, i),
                daemon=True,
            )
            t.start()
            self.rpc_output_threads.append(t)

        self.monitor_procs()

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self._cleanup_shared_memory()
        logger.info(f"{self.label}: shutdown all runners...")
        for proc in self.procs:
            if proc.is_alive():
                proc.join(timeout=5)
        shutdown_all_processes(self.procs, allowed_seconds=1)
        self.procs = []
        self.output_thread.join(timeout=1)
        for thread in self.kv_output_threads:
            thread.join(timeout=0.5)
        for thread in self.rpc_output_threads:
            thread.join(timeout=0.5)
        logger.info(f"{self.label}: All runners are shutdown.")
        self.outputs_queue.put_nowait(SystemExit())

        self.parent_finalizer()

    def _cleanup_shared_memory(self):
        """Clean up shared memory (creator side: close + unlink)."""
        if hasattr(self, "rpc_broadcast_mq"):
            mq = self.rpc_broadcast_mq
            if hasattr(mq, "buffer") and hasattr(mq.buffer, "shared_memory"):
                try:
                    shm = mq.buffer.shared_memory
                    if mq.buffer.is_creator:
                        shm.unlink()
                        mq.buffer.is_creator = False
                    shm.close()
                except Exception as e:  # noqa: BLE001 - teardown must not raise
                    # Creator side: an unlink of an already-removed segment is
                    # the common case and is harmless. Log rather than pass so
                    # a real leak is visible to whoever goes looking.
                    logger.debug(f"{self.label}: shm unlink/close failed: {e}")

    def process_output_sockets(self, output_address: str):
        """Receive results from rank 0's primary output channel."""
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)
                self.outputs_queue.put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: output thread exit")

    def process_kv_output_sockets(self, output_address: str, worker_id: int):
        """Receive KV output from each worker's dedicated channel."""
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)
                self.kv_outputs_queues[worker_id].put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: kv output thread {worker_id} exit")

    def process_rpc_output_sockets(self, output_address: str, worker_id: int):
        """Receive generic collective-RPC replies from one worker."""
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = pickle.loads(output_socket.recv(copy=False))
                self.rpc_outputs_queues[worker_id].put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: rpc output thread {worker_id} exit")

    def collective_rpc(
        self,
        func_name: str,
        payload: RpcPayload,
        timeout: float = 300.0,
    ) -> list[RpcResult]:
        """Run *func_name* on every TP runner and return one reply per rank.

        Unlike :meth:`call_func`, which surfaces only rank 0's return, this
        collects from every rank's own channel, so "all ranks finished" is
        observable. Unlike :meth:`call_func_with_aggregation`, replies carry a
        ``request_id``, so more than one call may be outstanding.

        Always returns ``proc_num`` results in rank order. A rank that died, or
        that did not answer within *timeout*, yields a failed ``RpcResult``
        rather than an exception or a short list -- the caller needs to know
        *which* rank is missing, and a raise here would lose the ranks that did
        answer.
        """
        if not isinstance(payload, RpcPayload):
            raise TypeError(
                f"collective_rpc needs an RpcPayload, got {type(payload).__name__}"
            )

        logger.debug(
            f"{self.label}: collective_rpc {func_name} id={payload.request_id}"
        )
        self.rpc_broadcast_mq.enqueue((func_name, payload))

        deadline = time.monotonic() + timeout
        results: list[RpcResult] = []
        for rank, output_queue in enumerate(self.rpc_outputs_queues):
            results.append(
                self._await_rank_reply(rank, output_queue, func_name, payload, deadline)
            )
        return results

    def _await_rank_reply(
        self,
        rank: int,
        output_queue: queue.Queue,
        func_name: str,
        payload: RpcPayload,
        deadline: float,
    ) -> RpcResult:
        """Wait for one rank's reply, or synthesise the reason there is none."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return RpcResult(
                    payload.request_id,
                    rank,
                    error=f"timed out waiting for {func_name!r} on TP rank {rank}",
                )
            try:
                # Poll rather than block for the whole budget, so a worker that
                # dies mid-call is reported promptly instead of at the deadline.
                reply = output_queue.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if rank < len(self.procs) and not self.procs[rank].is_alive():
                    return RpcResult(
                        payload.request_id,
                        rank,
                        error=f"TP rank {rank} died before answering {func_name!r}",
                    )
                continue

            if not isinstance(reply, RpcResult):
                return RpcResult(
                    payload.request_id,
                    rank,
                    error=f"unexpected reply type {type(reply).__name__} from rank {rank}",
                )
            if reply.request_id != payload.request_id:
                # A late reply from an earlier call. Dropping it is correct:
                # that caller has already been answered or has given up.
                logger.warning(
                    "%s: dropping stale reply %s from rank %d while awaiting %s",
                    self.label,
                    reply.request_id,
                    rank,
                    payload.request_id,
                )
                continue
            if reply.tp_rank != rank:
                return RpcResult(
                    payload.request_id,
                    rank,
                    error=f"reply rank mismatch: channel {rank} carried {reply.tp_rank}",
                )
            return reply

    def call_func(self, func_name: str, *args, wait_out: bool = False):
        """Standard RPC call for non-KV operations."""
        logger.debug(f"{self.label}: call_func {func_name} {args}")
        msg = (func_name, *args)
        self.rpc_broadcast_mq.enqueue(msg)
        if wait_out:
            ret = self.outputs_queue.get()
            if isinstance(ret, SystemExit):
                raise ret
            return ret

    def _start_kv_aggregation(
        self, func_name: str, args: tuple[object, ...]
    ) -> _PendingKvAggregation:
        """Broadcast one KV aggregation RPC and record its empty result slots."""
        self.rpc_broadcast_mq.enqueue((func_name, *args))
        pending = _PendingKvAggregation(
            worker_outputs=[None] * self.proc_num,
            started_at=time.monotonic(),
        )
        self._pending_kv_aggregation = pending
        return pending

    def call_func_with_aggregation(self, func_name: str, *args, timeout: float = 10.0):
        """RPC call with KV output aggregation across all workers.

        At most one aggregation is outstanding. The first call starts it;
        subsequent calls non-blockingly drain its per-rank replies. Once a batch
        is complete, the next batch is started before returning the completed
        result, keeping one poll in flight without delaying every other engine
        step. A missing rank leaves the current aggregation open and returns
        ``None``. ``timeout`` is only the age after which it is logged.

        Args:
            func_name: Method name to invoke on each worker's runner.
            timeout: Seconds an incomplete aggregation may age before logging.

        Returns:
            Aggregated :class:`KVConnectorOutput`, or ``None`` while the
            current aggregation is still missing ranks.
        """
        if self.kv_output_aggregator is None:
            self.kv_output_aggregator = KVOutputAggregator(world_size=self.proc_num)

        logger.debug(f"{self.label}: call_func_with_aggregation {func_name} {args}")
        pending = self._pending_kv_aggregation
        if pending is None:
            pending = self._start_kv_aggregation(func_name, args)

        # Each rank has its own queue, so later ranks can be consumed while
        # an earlier rank is still missing. Replies are drain-once and carry
        # no generation identifier, so they belong to this outstanding
        # aggregation until it closes.
        for rank, output_queue in enumerate(self.kv_outputs_queues):
            if pending.worker_outputs[rank] is not None:
                continue
            try:
                pending.worker_outputs[rank] = output_queue.get_nowait()
            except queue.Empty:
                continue

        missing = pending.missing_worker_ranks()
        if missing:
            waited = time.monotonic() - pending.started_at
            if timeout > 0 and waited >= timeout and not pending.wait_warning_logged:
                logger.error(
                    "%s: KV aggregation still waiting for workers %s after %.1fs",
                    self.label,
                    missing,
                    waited,
                )
                pending.wait_warning_logged = True
            return None

        kv_output = self.kv_output_aggregator.aggregate(
            worker_outputs=[
                output for output in pending.worker_outputs if output is not None
            ]
        )
        self._pending_kv_aggregation = None
        # Keep the completion query pipelined: workers can produce this next
        # snapshot while EngineCore processes the completed one and executes
        # the following forward. This still leaves exactly one outstanding RPC.
        self._start_kv_aggregation(func_name, args)
        logger.debug(f"Aggregated KV output: {kv_output}")
        return kv_output

    def monitor_procs(self):
        self_ref = weakref.ref(self)
        procs = self.procs
        self.keep_monitoring = True

        def monitor_engine_cores():
            sentinels = [proc.sentinel for proc in procs]
            died = multiprocessing.connection.wait(sentinels)
            _self = self_ref()
            if not _self or not _self.keep_monitoring:
                return
            dead_proc = next(proc for proc in procs if proc.sentinel == died[0])
            dead_proc.join(timeout=5)
            logger.error(
                f"{self.label}: [{dead_proc.name}] proc died unexpectedly "
                f"(exitcode={dead_proc.exitcode}), shutting down.",
            )
            _self.exit()

        Thread(
            target=monitor_engine_cores, daemon=True, name=f"{self.runner_label}Monitor"
        ).start()
