# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import logging
import multiprocessing
import multiprocessing.shared_memory
import os
import pickle
import queue
import weakref
from threading import Lock, Thread

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.model_engine.sequence import Sequence
from atom.utils import (
    envs,
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
    set_device_control_env_var,
)

logger = logging.getLogger("atom")

# Valid values for Config.dp_load_balance / --dp-load-balance, and the default.
# Single source of truth for argparse (choices + default) so the CLI flag and
# the Config field can never diverge.
DP_LB_STRATEGIES = ("round_robin", "least_requests", "least_tokens")
DP_LB_DEFAULT = "least_requests"


class CoreManager:
    def _init_shared_state(
        self, config: Config, *, label: str, local_engine_count: int
    ) -> None:
        """Every field the inherited methods touch, before any engine is spawned.

        Subclasses spawn their engines differently and so cannot run this
        class's ``__init__`` -- but they inherit its output threads, ``close()``
        and DP-load bookkeeping, all of which read the fields set here. This is
        the one place to add another such field.

        It exists because the alternative was tried: ``DisaggCoreManager`` used
        to hand-copy this block, and the copy drifted. ``_flush_stream_batch_fn``
        was added to the copy and to the API server that assigns it, but not to
        this class -- so the offline entrypoint, which is the one path that
        neither initialises nor assigns it, had its output thread die on the
        first streamed token and hung until CI timed out an hour later.
        """
        self.label = label
        self._closed = False  # Track whether already closed
        self.local_engine_count = local_engine_count
        self.ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue[list[Sequence]]()
        self.stream_outputs_queue = queue.Queue()
        self.utility_response_queue = queue.Queue()
        self._seq_id_to_callback = {}
        # Batched stream-flush hook, resolved lazily by the API server (avoids
        # an api_server <-> engine_core_mgr import cycle). Stays None on every
        # path that never streams, which the output thread checks for.
        self._flush_stream_batch_fn = None
        self.engine_core_processes = []
        self.input_sockets = []
        self.output_sockets = []
        self.engine_core_identities = []
        self.shutdown_paths = []
        self.output_threads = []
        # Fair-rotation cursor, advanced once per selection. round_robin picks the
        # rank directly (cursor % n); the load-aware strategies use it only to seed
        # the argmin start offset so fully-tied ranks rotate instead of always
        # resolving to rank 0.
        self._rank_rotation_cursor = 0

        # --- DP request load balancing (see _select_dp_rank_locked) ---
        # A subclass may fan out through its own add_request() and never charge
        # load at all, but the inherited output thread still calls
        # _release_seq_load() on every finished sequence, so these MUST exist.
        # Strategy: "round_robin" | "least_requests" | "least_tokens" (validated
        # at the CLI by argparse choices=DP_LB_STRATEGIES).
        self._dp_lb_strategy = config.dp_load_balance
        # Token-equivalent weight of one in-flight request for "least_tokens".
        # Read once here: this is a construction-time config value (CoreManager
        # is built after env/args are finalized), not a runtime-tunable knob.
        self._dp_lb_req_equiv = envs.ATOM_DP_LB_REQ_EQUIV
        # Authoritative in-flight load per rank, maintained locally: incremented
        # on dispatch, decremented on finish/abort. Guarded by _lb_lock because
        # dispatch runs on the request thread while release runs on the per-rank
        # output threads.
        self._rank_reqs = [0] * local_engine_count
        self._rank_tokens = [0] * local_engine_count
        # seq_id -> (dp_rank, req_cost, tok_cost) so release subtracts exactly
        # what dispatch added, and only for ranks that were actually charged.
        self._seq_load = {}
        self._lb_lock = Lock()

    def __init__(self, config: Config):
        pp_size = config.pipeline_parallel_size
        self.pp_size = pp_size
        if config.enable_dp_attention:
            assert pp_size == 1, "Pipeline parallel + DP-attention is not supported yet"
            local_engine_count = (
                config.tensor_parallel_size * config.parallel_config.data_parallel_size
            )
            logger.info(
                f"Enable dp attention, using {local_engine_count} data parallel ranks"
            )
            config.parallel_config.data_parallel_size = local_engine_count
            config.tensor_parallel_size = 1
        else:
            dp_size = config.parallel_config.data_parallel_size
            assert not (
                pp_size > 1 and dp_size > 1
            ), "Pipeline parallel combined with data parallel is not supported yet."
            # One EngineCore per (dp_rank, pp_rank) stage.
            local_engine_count = dp_size * pp_size
        # Inter-stage ZMQ channels (head<->downstream metadata, last->head
        # tokens), shared across the single dp group. PP+DP would need per-group
        # sets — deferred with the assertion above. Not shared state: only this
        # class's spawn loop reads them.
        self.pp_meta_addrs = []
        self.pp_token_addr = ""
        if pp_size > 1:
            self.pp_meta_addrs = [get_open_zmq_ipc_path() for _ in range(pp_size)]
            self.pp_token_addr = get_open_zmq_ipc_path()

        self._init_shared_state(
            config, label="Engine Core Mgr", local_engine_count=local_engine_count
        )

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        processes_info = []
        local_dp_ranks = []

        try:
            for engine_index in range(self.local_engine_count):
                dp_rank = engine_index // self.pp_size
                pp_rank = engine_index % self.pp_size
                logger.info(
                    f"{self.label}: Creating EngineCore engine {engine_index}"
                    f" (dp={dp_rank}, pp={pp_rank}) of {self.local_engine_count}"
                )

                # Create config for this (dp, pp) stage
                import copy

                rank_config = copy.deepcopy(config)
                rank_config.parallel_config.data_parallel_rank = dp_rank
                rank_config.parallel_config.pipeline_parallel_rank = pp_rank
                if self.pp_size > 1:
                    rank_config.parallel_config.pp_meta_addrs = self.pp_meta_addrs
                    rank_config.parallel_config.pp_token_addr = self.pp_token_addr

                engine_core_process, addresses, local_dp_rank = launch_engine_core(
                    rank_config, dp_rank
                )

                processes_info.append(
                    {
                        "process": engine_core_process,
                        "addresses": addresses,
                        "dp_rank": dp_rank,
                        "config": rank_config,
                    }
                )
                local_dp_ranks.append(local_dp_rank)

            data_parallel = config.parallel_config.data_parallel_size > 1
            try:
                for info, local_dp_rank in zip(processes_info, local_dp_ranks):
                    dp_rank = info["dp_rank"]
                    logger.info(
                        f"{self.label}: Starting EngineCore for DP rank {dp_rank}/{self.local_engine_count}"
                    )

                    if data_parallel:
                        with set_device_control_env_var(info["config"], local_dp_rank):
                            info["process"].start()
                    else:
                        info["process"].start()

                    self.engine_core_processes.append(info["process"])

                    input_address = info["addresses"]["input_address"]
                    input_socket = make_zmq_socket(
                        self.ctx, input_address, zmq.ROUTER, bind=True
                    )
                    identity, _ = input_socket.recv_multipart()
                    self.input_sockets.append(input_socket)
                    self.engine_core_identities.append(identity)

                    output_address = info["addresses"]["output_address"]
                    output_socket = make_zmq_socket(self.ctx, output_address, zmq.PULL)
                    self.output_sockets.append(output_socket)

                    shutdown_path = get_open_zmq_inproc_path()
                    self.shutdown_paths.append(shutdown_path)

                self._wait_for_all_ready_signals()
                logger.info(
                    f"{self.label}: All EngineCores are fully initialized and ready"
                )

                for dp_rank in range(self.local_engine_count):
                    output_thread = self._create_output_thread(
                        dp_rank,
                        self.output_sockets[dp_rank],
                        self.shutdown_paths[dp_rank],
                    )
                    output_thread.start()
                    self.output_threads.append(output_thread)

            finally:
                if self.finished_procs():
                    logger.error(
                        f"{self.label}: Some processes failed to start, shutting down all"
                    )
                    self.close()
                    raise RuntimeError("Failed to start all EngineCore processes")

        except Exception as e:
            logger.error(
                f"{self.label}: Failed to initialize all EngineCores, cleaning up: {e}"
            )
            self.close()
            raise

        logger.info(
            f"{self.label}: All {self.local_engine_count} EngineCores initialized and ready"
        )
        self._finalizer = weakref.finalize(self, self.close)
        self.async_output_queue = asyncio.Queue() if config.asyncio_mode else None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    def _wait_for_all_ready_signals(self):
        """Wait for READY signals from all DP ranks in parallel (no timeout)."""
        poller = zmq.Poller()
        for dp_rank, output_socket in enumerate(self.output_sockets):
            poller.register(output_socket, zmq.POLLIN)

        ready_received = [False] * self.local_engine_count
        remaining = self.local_engine_count

        while remaining > 0:
            socks = poller.poll()  # Wait indefinitely
            if not socks:
                continue

            for socket, _ in socks:
                # Find which DP rank this socket belongs to
                dp_rank = self.output_sockets.index(socket)
                if ready_received[dp_rank]:
                    continue

                obj = socket.recv(copy=False)
                request_type, data = pickle.loads(obj)

                if request_type == EngineCoreRequestType.READY:
                    logger.info(
                        f"{self.label}: DP rank {dp_rank} is fully initialized and ready"
                    )
                    ready_received[dp_rank] = True
                    remaining -= 1
                elif request_type == EngineCoreRequestType.SHUTDOWN:
                    raise RuntimeError(
                        f"{self.label}: Received unexpected SHUTDOWN signal from DP rank {dp_rank} during initialization"
                    )
                else:
                    raise RuntimeError(
                        f"{self.label}: Expected READY signal from DP rank {dp_rank}, but got {request_type}"
                    )

    def _create_output_thread(
        self, dp_rank: int, output_socket: zmq.Socket, shutdown_path: str
    ) -> Thread:
        def process_outputs_socket():
            assert isinstance(output_socket, zmq.Socket)
            shutdown_socket = self.ctx.socket(zmq.PAIR)
            try:
                shutdown_socket.bind(shutdown_path)
                poller = zmq.Poller()
                poller.register(shutdown_socket, zmq.POLLIN)
                poller.register(output_socket, zmq.POLLIN)
                logger.debug(f"{self.label} (DP {dp_rank}): output thread started")
                while True:
                    socks = poller.poll()
                    if not socks:
                        continue
                    if len(socks) == 2 or socks[0][0] == shutdown_socket:
                        # shutdown signal, exit thread.
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive shutdown signal"
                        )
                        break

                    obj = output_socket.recv(copy=False)
                    request_type, data = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(
                            f"{self.label} (DP {dp_rank}): output thread receive SHUTDOWN request"
                        )
                        self._shutdown_engine_core_rank(dp_rank)
                        break
                    elif request_type == EngineCoreRequestType.STREAM:
                        stream_outputs = data  # List of (seq_id, RequestOutput) tuples
                        logger.debug(
                            f"{self.label}: Received STREAM message with {len(stream_outputs)} outputs"
                        )
                        self.stream_outputs_queue.put_nowait(stream_outputs)
                        # Also call callbacks if registered
                        for seq_id, request_output in stream_outputs:
                            callback = self._seq_id_to_callback.get(seq_id)
                            logger.debug(
                                f"{self.label}: seq_id={seq_id}, callback={'found' if callback is not None else 'NOT FOUND'}, tokens={request_output.output_tokens}"
                            )
                            if callback is not None:
                                try:
                                    callback(request_output)
                                    logger.debug(
                                        f"{self.label}: Successfully called callback for seq_id={seq_id}"
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Error calling stream_callback for sequence {seq_id}: {e}",
                                        exc_info=True,
                                    )
                            if request_output.finished:
                                self._seq_id_to_callback.pop(seq_id, None)
                                self._release_seq_load(seq_id)
                                logger.debug(
                                    f"{self.label}: Cleaned up callback for finished sequence {seq_id}"
                                )
                        # Batched stream dispatch: the per-seq callbacks only buffer
                        # their chunks into a thread-local; flush the whole step's
                        # buffer into the per-request asyncio queues now (one
                        # call_soon_threadsafe per loop). Resolved lazily by the API
                        # server to avoid the api_server <-> engine_core_mgr import
                        # cycle. No-op when no streaming request is in flight.
                        if self._flush_stream_batch_fn is not None:
                            try:
                                self._flush_stream_batch_fn()
                            except Exception as e:
                                logger.warning(
                                    f"{self.label}: flush_stream_batch failed: {e}",
                                    exc_info=True,
                                )
                    elif request_type == EngineCoreRequestType.UTILITY_RESPONSE:
                        self.utility_response_queue.put_nowait(data)
                    elif request_type == EngineCoreRequestType.ADD:
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        seqs = data
                        # Offline (non-streaming) completions arrive here as
                        # finished sequences; release their in-flight DP load.
                        for seq in seqs:
                            self._release_seq_load(seq.id)
                        self.outputs_queue.put_nowait(seqs)
            finally:
                # Close sockets.
                shutdown_socket.close(linger=0)
                output_socket.close(linger=0)

        return Thread(
            target=process_outputs_socket,
            name=f"EngineCoreOutputThread-DP{dp_rank}",
            daemon=True,
        )

    def _ensure_output_handler_task(self):
        if self._asyncio_mode and self._output_handler_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._output_handler_task = loop.create_task(
                    self._async_output_handler()
                )
            except RuntimeError:
                # If no running event loop, try to get/create one
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    self._output_handler_task = loop.create_task(
                        self._async_output_handler()
                    )
                else:
                    raise RuntimeError(
                        "CoreManager with asyncio_mode requires a running event loop"
                    )

    async def _async_output_handler(self):
        loop = asyncio.get_event_loop()
        while True:
            # Use run_in_executor to avoid blocking event loop
            seqs = await loop.run_in_executor(None, self.outputs_queue.get)
            if isinstance(seqs, BaseException):
                await self.async_output_queue.put(seqs)
                break
            await self.async_output_queue.put(seqs)

    async def get_output_async(self) -> list[Sequence]:
        if not self.async_output_queue:
            raise RuntimeError("Engine async mode not enabled")

        # Ensure output handler task is started
        self._ensure_output_handler_task()

        seqs = await self.async_output_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    def close(self):
        if self._closed:
            return
        self._closed = True

        logger.info(
            f"{self.label}: Shutting down all {self.local_engine_count} EngineCores"
        )

        for dp_rank in range(self.local_engine_count):
            self._shutdown_engine_core_rank(dp_rank)

        for input_socket in self.input_sockets:
            if not input_socket.closed:
                input_socket.close()

        for shutdown_path in self.shutdown_paths:
            if shutdown_path:
                try:
                    with self.ctx.socket(zmq.PAIR) as shutdown_sender:
                        shutdown_sender.connect(shutdown_path)
                        shutdown_sender.send(b"")
                except Exception as e:
                    logger.debug(f"{self.label}: Error sending shutdown signal: {e}")

        for thread in self.output_threads:
            if thread and thread.is_alive():
                thread.join(timeout=0.5)

        # Wait for EngineCore processes to exit gracefully.
        # Use a single deadline so all processes share the grace period
        # instead of sequential per-process timeouts.  This prevents early
        # process exits from destroying the NCCL TCPStore while later
        # processes' HeartbeatMonitor threads still depend on it.
        import time

        deadline = time.monotonic() + 5
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                remaining = max(deadline - time.monotonic(), 0)
                proc.join(timeout=remaining)

        # Terminate any that are still alive.
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.terminate()
        for proc in self.engine_core_processes:
            if proc is not None and proc.is_alive():
                proc.join(timeout=1)

        # Final join + close to release sentinel semaphores
        for proc in self.engine_core_processes:
            if proc is not None:
                if proc.is_alive():
                    proc.kill()
                proc.join(timeout=1)
                try:
                    proc.close()
                except (ValueError, OSError):
                    pass

        logger.info(f"{self.label}: All EngineCores shut down")

    def add_request(self, seqs: list[Sequence]):
        logger.debug(
            f"{self.label}: Add request, sequence ids: {[seq.id for seq in seqs]}"
        )
        # Register callbacks before sending to engine core
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None
        if self.pp_size > 1:
            # Pipeline parallel (dp=1): requests enter only at stage 0, which
            # drives the pipeline downstream.
            logger.debug(f"{self.label}: Add {len(seqs)} requests to PP head 0")
            self.input_sockets[0].send_multipart(
                [
                    self.engine_core_identities[0],
                    pickle.dumps((EngineCoreRequestType.ADD, seqs)),
                ],
                copy=False,
            )
        elif self.local_engine_count == 1:
            # Single DP rank, send all requests
            logger.debug(f"{self.label}: Add {len(seqs)} requests to DP rank 0")
            self.input_sockets[0].send_multipart(
                [
                    self.engine_core_identities[0],
                    pickle.dumps((EngineCoreRequestType.ADD, seqs)),
                ],
                copy=False,
            )
        else:
            self._dispatch_to_dp_ranks(seqs)

    def _resolve_and_validate_hints(self, seqs: list[Sequence]) -> list[int | None]:
        """Resolve every seq's explicit ``data_parallel_rank`` hint and validate
        the whole batch, once.

        Returns the per-seq resolved hint (an int rank, or None for a
        load-balanced seq) so the dispatch loop can reuse it instead of calling
        getattr/int a second time per seq.

        Validation runs BEFORE any load is charged so a bad hint in the middle
        of a batch cannot leave earlier siblings charged-but-undispatched (a
        permanent in-flight-load leak).
        """
        hints: list[int | None] = []
        for seq in seqs:
            raw = getattr(seq, "data_parallel_rank", None)
            hint = None if raw is None else int(raw)
            if hint is not None and not 0 <= hint < self.local_engine_count:
                raise ValueError(
                    f"Invalid data_parallel_rank={hint}; "
                    f"local_engine_count={self.local_engine_count}"
                )
            hints.append(hint)
        return hints

    def _dispatch_to_dp_ranks(self, seqs: list[Sequence]) -> None:
        """Route a batch across DP ranks and send each rank its sub-batch.

        Honors an explicit ``data_parallel_rank`` hint; otherwise picks a rank
        via ``_select_dp_rank_locked`` (load-aware by default). Selection and the
        in-flight-load charge happen atomically under ``_lb_lock`` so a burst of
        requests spreads across ranks instead of all landing on the current
        minimum.
        """
        # Resolve + validate all hints in one pass first — no charging until the
        # whole batch is known good, so a rejected batch never leaks partial
        # load. The resolved hints are reused in the loop below to avoid a second
        # getattr/int pass per seq.
        hints = self._resolve_and_validate_hints(seqs)

        # round_robin is load-agnostic and skips the charge/release bookkeeping;
        # the load-aware strategies track per-rank load.
        track_load = self._dp_lb_strategy != "round_robin"
        dp_seqs = [[] for _ in range(self.local_engine_count)]
        reqs_snapshot = tokens_snapshot = None
        with self._lb_lock:
            for seq, hint in zip(seqs, hints):
                dp_rank = hint if hint is not None else self._select_dp_rank_locked()
                if track_load:
                    self._charge_seq_load_locked(seq, dp_rank)
                dp_seqs[dp_rank].append(seq)
            # Copy the counters under the lock so the snapshot log below is a
            # consistent instant, not a torn read racing _release_seq_load.
            if track_load:
                reqs_snapshot = list(self._rank_reqs)
                tokens_snapshot = list(self._rank_tokens)

        # Track which ranks were actually handed off, plus a compact per-rank
        # delta ("rankR:Nreq/Ttok") for the single summary log after the loop. If
        # a send fails partway, the seqs on the not-yet-dispatched ranks were
        # charged above but will never produce a finished output to release them,
        # so we roll back their in-flight load before propagating — otherwise
        # routing skews forever.
        dispatched = [False] * self.local_engine_count
        added = []
        try:
            for dp_rank, rank_seqs in enumerate(dp_seqs):
                if not rank_seqs:
                    continue
                self.input_sockets[dp_rank].send_multipart(
                    [
                        self.engine_core_identities[dp_rank],
                        pickle.dumps((EngineCoreRequestType.ADD, rank_seqs)),
                    ],
                    copy=False,
                )
                dispatched[dp_rank] = True
                batch_prefill_tokens = sum(
                    int(getattr(seq, "num_prompt_tokens", 0) or 0) for seq in rank_seqs
                )
                added.append(
                    f"rank{dp_rank}: {len(rank_seqs)} req / {batch_prefill_tokens} tok"
                )
        except Exception:
            # Roll back only ranks we never handed off. _release_seq_load is
            # idempotent (pops from _seq_load), so even if a failing send had
            # already delivered its frames and the engine finished + released
            # those seqs on an output thread, this rollback cannot double-count:
            # whichever release runs first wins, the other is a no-op.
            if track_load:
                for dp_rank, rank_seqs in enumerate(dp_seqs):
                    if rank_seqs and not dispatched[dp_rank]:
                        for seq in rank_seqs:
                            self._release_seq_load(seq.id)
            raise

        # One line per add: the per-rank delta this add placed, plus (for the
        # load-aware strategies) the resulting in-flight distribution across all
        # ranks, so a single grep shows both what changed and how balanced it is.
        if reqs_snapshot is not None:
            logger.info(
                "%s: add %s | in-flight reqs=%s prefill_tokens=%s",
                self.label,
                ", ".join(added),
                reqs_snapshot,
                tokens_snapshot,
            )
        else:
            logger.info("%s: add %s", self.label, ", ".join(added))

    def _select_dp_rank_locked(self, lo: int = 0, hi: int | None = None) -> int:
        """Pick a DP engine rank for a new request. Caller must hold _lb_lock.

        `lo`/`hi` bound the candidate range (default: all engines). Disagg uses
        them to select among the decode ranks only — its engine list starts with
        the prefill process at index 0, which must never be chosen as a target.

        - "round_robin": load-agnostic rotation.
        - "least_requests" (default): fewest in-flight requests, ties broken by
          the lighter in-flight prompt-token load. Request count keeps the
          lockstep DP ranks in phase; the token tie-break packs pending prefill
          work evenly across the equal-request ranks.
        - "least_tokens": lowest combined load ``tokens + req_equiv * reqs``
          (prefill pressure + decode-slot pressure).

        Fully-tied ranks are resolved by a rotating cursor so selection does not
        always fall on rank 0. See docs/distributed_guide.md for the rationale.
        """
        hi = self.local_engine_count if hi is None else hi
        n = hi - lo
        if self._dp_lb_strategy == "round_robin":
            dp_rank = lo + (self._rank_rotation_cursor % n)
            self._rank_rotation_cursor += 1
            return dp_rank

        # argmin over per-rank load, scanned from a rotating start offset so a run
        # of fully-equal ranks spreads evenly. Scores are computed inline (no
        # intermediate list) — the loop reads the counters directly.
        least_requests = self._dp_lb_strategy == "least_requests"
        best_rank = lo
        best_score = None
        offset = self._rank_rotation_cursor % n
        for i in range(n):
            r = lo + ((offset + i) % n)
            if least_requests:
                # Lexicographic (request count, prompt-token load): tuples compare
                # element-wise, so tokens only decide among request-count ties.
                score = (self._rank_reqs[r], self._rank_tokens[r])
            else:  # "least_tokens"
                score = (
                    self._rank_tokens[r] + self._dp_lb_req_equiv * self._rank_reqs[r]
                )
            if best_score is None or score < best_score:
                best_score = score
                best_rank = r
        self._rank_rotation_cursor += 1
        return best_rank

    def _charge_seq_load_locked(self, seq: Sequence, dp_rank: int) -> None:
        """Record a seq's in-flight load on dp_rank. Caller must hold _lb_lock."""
        req_cost = 1
        tok_cost = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        self._rank_reqs[dp_rank] += req_cost
        self._rank_tokens[dp_rank] += tok_cost
        self._seq_load[seq.id] = (dp_rank, req_cost, tok_cost)

    def _release_seq_load(self, seq_id) -> None:
        """Undo a seq's in-flight load when it finishes or is aborted.

        Idempotent: a seq is only charged once and released once, so a repeated
        call (e.g. finish followed by abort) is a no-op.
        """
        with self._lb_lock:
            entry = self._seq_load.pop(seq_id, None)
            if entry is None:
                return
            dp_rank, req_cost, tok_cost = entry
            self._rank_reqs[dp_rank] -= req_cost
            self._rank_tokens[dp_rank] -= tok_cost

    def reset_dp_router(self) -> None:
        """Reset all DP routing state (rotation cursor + in-flight load).

        Called at the start of a fresh offline ``generate()`` batch so counts do
        not leak across independent batches and DP assignment is deterministic.

        Precondition: the previous batch has fully drained. If any request is
        still charged when this runs (e.g. this CoreManager is being shared with
        a concurrent streaming path), that request's later release becomes a
        no-op and the per-rank counters would drift — so we warn loudly instead
        of corrupting accounting silently.
        """
        with self._lb_lock:
            if self._seq_load:
                logger.warning(
                    "%s: reset_dp_router() called with %d request(s) still "
                    "charged in-flight; dropping their load. Expected only "
                    "between fully-drained offline batches — a shared/concurrent "
                    "CoreManager will see counters drift.",
                    self.label,
                    len(self._seq_load),
                )
            self._rank_rotation_cursor = 0
            self._rank_reqs = [0] * self.local_engine_count
            self._rank_tokens = [0] * self.local_engine_count
            self._seq_load.clear()

    def get_stream_outputs(self):
        try:
            return self.stream_outputs_queue.get_nowait()
        except queue.Empty:
            return None

    def send_utility_command(self, cmd: str, dp_rank: int = None):
        if dp_rank is None:
            # Send to all DP ranks
            for rank in range(self.local_engine_count):
                logger.debug(
                    f"{self.label}: Send utility command '{cmd}' to DP rank {rank}"
                )
                self.input_sockets[rank].send_multipart(
                    [
                        self.engine_core_identities[rank],
                        pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd})),
                    ],
                    copy=False,
                )
        else:
            logger.debug(
                f"{self.label}: Send utility command '{cmd}' to DP rank {dp_rank}"
            )
            self.input_sockets[dp_rank].send_multipart(
                [
                    self.engine_core_identities[dp_rank],
                    pickle.dumps((EngineCoreRequestType.UTILITY, {"cmd": cmd})),
                ],
                copy=False,
            )

    def abort_request(self, req_id):
        """Tell the engine core(s) to drop a request (client disconnected).

        Broadcast to every DP rank (only the one holding ``req_id`` acts). The
        scheduler finishes the seq at its next step via the normal stop path,
        freeing its KV blocks. Fire-and-forget; safe if the seq already finished.
        """
        # Release DP load bookkeeping now: an aborted seq may never emit a
        # finished STREAM output, so relying on the finish path alone would leak
        # its in-flight count. _release_seq_load is idempotent.
        self._release_seq_load(req_id)
        try:
            self.broadcast_utility_command("abort_request", req_id=req_id)
        except Exception as e:
            logger.warning(f"{self.label}: abort_request({req_id}) failed: {e}")

    def broadcast_utility_command(self, cmd: str, **kwargs):
        payload = {"cmd": cmd, **kwargs}
        # Serialize once and reuse for all ranks (optimization: avoid repeated pickle.dumps)
        serialized_payload = pickle.dumps((EngineCoreRequestType.UTILITY, payload))
        for rank in range(self.local_engine_count):
            logger.debug(
                f"{self.label}: Broadcast utility command '{cmd}' to DP rank {rank}"
            )
            self.input_sockets[rank].send_multipart(
                [
                    self.engine_core_identities[rank],
                    serialized_payload,
                ],
                copy=True,  # Use copy=True since we're reusing the same buffer
            )

    def broadcast_utility_command_sync(
        self, cmd: str, timeout: float = 300.0, **kwargs
    ):
        # Drain any stale responses that might be left over
        while not self.utility_response_queue.empty():
            try:
                self.utility_response_queue.get_nowait()
            except queue.Empty:
                break

        self.broadcast_utility_command(cmd, **kwargs)

        # Collect one response per DP rank
        responses = []
        for _ in range(self.local_engine_count):
            try:
                resp = self.utility_response_queue.get(timeout=timeout)
                responses.append(resp)
            except queue.Empty:
                raise TimeoutError(
                    f"{self.label}: Timed out waiting for UTILITY_RESPONSE "
                    f"for command '{cmd}' (timeout={timeout}s)"
                )
        return responses

    def _shutdown_engine_core_rank(self, dp_rank: int):
        if dp_rank >= len(self.engine_core_processes):
            return

        process = self.engine_core_processes[dp_rank]
        if process is not None and process.is_alive():
            try:
                input_socket = self.input_sockets[dp_rank]
                if not input_socket.closed:
                    input_socket.send_multipart(
                        [
                            self.engine_core_identities[dp_rank],
                            pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)),
                        ],
                        copy=False,
                    )
                    logger.debug(f"{self.label}: Sent shutdown to DP rank {dp_rank}")
            except Exception as e:
                logger.debug(
                    f"{self.label}: Error sending shutdown to DP rank {dp_rank}: {e}"
                )

    def get_output(self) -> list[Sequence]:
        seqs = self.outputs_queue.get()
        if isinstance(seqs, BaseException):
            raise seqs
        return seqs

    def is_rest(self):
        return not self.outputs_queue.empty()

    def is_alive(self):
        return any(
            proc is not None and proc.is_alive() for proc in self.engine_core_processes
        )

    def finished_procs(self):
        return any(
            proc is not None and not proc.is_alive()
            for proc in self.engine_core_processes
        )


def launch_engine_core(config: Config, dp_rank: int = 0):
    input_address = get_open_zmq_ipc_path()
    output_address = get_open_zmq_ipc_path()
    import torch

    # Imported here, not at module scope: EngineCore pulls the heavy
    # engine_core -> async_proc -> aiter chain. Spawning a worker is inherently a
    # GPU-side operation, so the cost belongs here and keeps CoreManager (routing
    # only) importable on a CPU-only runner.
    from atom.model_engine.engine_core import EngineCore

    if torch.multiprocessing.get_start_method(allow_none=True) is None:
        torch.multiprocessing.set_start_method("spawn", force=False)

    config.parallel_config.data_parallel_rank = dp_rank
    config.parallel_config.data_parallel_rank_local = dp_rank

    logger.info(
        f"Creating EngineCore process: DP rank {dp_rank}, will use GPUs {dp_rank * config.tensor_parallel_size} to {(dp_rank + 1) * config.tensor_parallel_size - 1}"
    )

    process = multiprocessing.Process(
        target=EngineCore.run_engine,
        name=f"EngineCore-DP{dp_rank}",
        kwargs={
            "config": config,
            "input_address": input_address,
            "output_address": output_address,
        },
    )

    return (
        process,
        {"input_address": input_address, "output_address": output_address},
        dp_rank,
    )


class DisaggCoreManager(CoreManager):
    """CoreManager for intra-GPU prefill/decode disaggregation.

    Spawns two separate EngineCore processes on the same GPU(s):
      - PrefillEngineCore: runs prefill forward passes, writes KV cache.
      - DecodeEngineCore: owns BlockManager and KV cache, runs decode.

    add_request() fans out every new sequence to BOTH processes.
    Only DecodeEngineCore produces finished sequences back to LLMEngine.

    The two processes coordinate via direct ZMQ PUSH/PULL sockets whose
    addresses are established here before spawning and passed through config.
    """

    def __init__(self, config: Config):
        import copy

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        # Asymmetric topology: --enable-dp-attention under rapidserve means
        # "prefill wide (TP), decode per-GPU (DP)" rather than the whole-engine
        # flattening CoreManager.__init__ does — that flattening never runs here
        # because this class does not call super().__init__().
        asymmetric = config.enable_dp_attention and config.tensor_parallel_size > 1
        prefill_tp = config.tensor_parallel_size if asymmetric else 0

        # Decode rank count. Symmetric rapidserve runs ONE decode process at the
        # same TP as prefill — tensor parallelism drives all its workers from a
        # single scheduler. Data parallelism cannot: each DP rank owns its own
        # scheduler, BlockManager and KV pool, so it needs its own EngineCore
        # process, exactly as CoreManager.__init__ does for non-disagg DPA.
        n_decode = config.tensor_parallel_size if asymmetric else 1
        if asymmetric:
            logger.info(
                "Asymmetric rapidserve: prefill TP=%d (1 process, %d workers), "
                "decode DP=%d (%d processes, TP=1 each). MLA's KV latent is "
                "TP-replicated, so the prefill rank sharing a GPU with a "
                "sequence's target decode rank writes its KV locally and the "
                "other ranks mask to the dump block.",
                config.tensor_parallel_size,
                config.tensor_parallel_size,
                n_decode,
                n_decode,
            )

        # decode → prefill stays single: many PUSHers to one PULLer is native
        # ZMQ and prefill does not care which rank sent an assignment.
        d2p_addr = get_open_zmq_ipc_path()  # decode → prefill (BlockAssignment)
        # Everything prefill sends is per-rank: a PrefillDone must reach the one
        # owner, and each bootstrap pairs decode rank k with prefill rank k.
        p2d_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]
        # Bootstrap round 1: weight IPC handles (prefill → decode) + ACK (decode → prefill)
        weight_ipc_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]
        weight_ack_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]
        # Bootstrap round 2: kvcache handle + num_blocks (prefill → decode)
        kvcache_ipc_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]

        # Shared memory for dynamic CU partitioning: 4 bytes (float32).
        # DecodeScheduler writes the chosen CU fraction; PrefillScheduler reads it.
        # 0.0 means no mask (None).
        # Only created in constrained mode; unconstrained mode runs prefill
        # and decode on plain separate streams with no CU coordination.
        if config.disagg_constrained:
            cu_shm_name = f"atom_cu_split_{os.getpid()}"
            self._cu_shm = multiprocessing.shared_memory.SharedMemory(
                name=cu_shm_name, create=True, size=4
            )
            self._cu_shm.buf[:4] = b"\x00" * 4
        else:
            cu_shm_name = ""
            self._cu_shm = None

        # Build per-process configs.
        from atom.utils import get_open_port as _get_open_port

        prefill_config = copy.deepcopy(config)
        if config.disagg_prefill_max_num_seqs is not None:
            prefill_config.max_num_seqs = config.disagg_prefill_max_num_seqs
        prefill_config.enforce_eager = True
        prefill_config.disagg_prefill_tp_size = prefill_tp
        # Prefill owns the wide TP topology; DP attention must stay off there or
        # the model takes the DP gather path instead of the TP all_reduce path.
        prefill_config.enable_dp_attention = False
        prefill_config.disagg_d2p_addr = d2p_addr
        prefill_config.disagg_p2d_addrs = p2d_addrs
        prefill_config.disagg_weight_ipc_addrs = weight_ipc_addrs
        prefill_config.disagg_weight_ack_addrs = weight_ack_addrs
        prefill_config.disagg_kvcache_ipc_addrs = kvcache_ipc_addrs
        prefill_config.disagg_num_decode_ranks = n_decode
        prefill_config.disagg_cu_shm_name = cu_shm_name
        # Give prefill a distinct distributed rendezvous port so it doesn't
        # collide with decode's data_parallel_base_port (both deep-copy the
        # same port from config).
        prefill_config.parallel_config.data_parallel_base_port = _get_open_port()

        # One rendezvous port shared by every decode rank — that is how the DP
        # group finds itself. Distinct from prefill's (set above) so the two
        # groups do not collide. Mirrors CoreManager.__init__, which likewise
        # varies only data_parallel_rank across its per-rank configs.
        decode_base_port = _get_open_port()

        decode_configs = []
        for k in range(n_decode):
            dc = copy.deepcopy(config)
            dc.disagg_d2p_addr = d2p_addr
            dc.disagg_p2d_addrs = p2d_addrs
            dc.disagg_weight_ipc_addrs = weight_ipc_addrs
            dc.disagg_weight_ack_addrs = weight_ack_addrs
            dc.disagg_kvcache_ipc_addrs = kvcache_ipc_addrs
            dc.disagg_cu_shm_name = cu_shm_name
            # Decode allocates no GPU memory — kvcache and weights are imported
            # from prefill via CUDA IPC after prefill's READY signal.
            dc.disagg_is_decode = True
            dc.disagg_prefill_tp_size = prefill_tp
            dc.disagg_decode_rank = k
            dc.disagg_num_decode_ranks = n_decode
            if asymmetric:
                # One attention rank per GPU, N of them: the DP-attention shape.
                # tp_size=1 with dp_size=N keeps `dp*tp > 1`, so expert
                # parallelism stays ON (moe.py:151) and decode's ep_size matches
                # prefill's — which is what lets the MoE weights, where nearly
                # all the bytes live, stay IPC-aliased. Only the TP=1 attention
                # weights differ in shape and are loaded locally.
                dc.tensor_parallel_size = 1
                dc.parallel_config.data_parallel_size = n_decode
                dc.parallel_config.data_parallel_rank = k
                # ModelRunner derives its GPU as
                #   (data_parallel_rank_local * pp + pp_rank) * (tp * pcp) + rank
                # (model_runner.py:970-983). At tp=1 the ONLY term that varies is
                # data_parallel_rank_local, so without this every decode process
                # computes device 0 and they all pile onto GPU 0 — and each then
                # tries to open an IPC handle from a prefill rank on a different
                # physical GPU. Setting it puts decode rank k on cuda:k, which is
                # where prefill TP rank k already lives, so the pairing holds.
                # launch_engine_core does the same for the non-disagg DP path.
                dc.parallel_config.data_parallel_rank_local = k
                dc.enable_dp_attention = True
                dc.parallel_config.data_parallel_base_port = decode_base_port
            decode_configs.append(dc)

        if config.torch_profiler_dir:
            prefill_config.torch_profiler_dir = os.path.join(
                config.torch_profiler_dir, "prefill"
            )
            os.makedirs(prefill_config.torch_profiler_dir, exist_ok=True)
            for k, dc in enumerate(decode_configs):
                dc.torch_profiler_dir = os.path.join(
                    config.torch_profiler_dir,
                    f"decode{k}" if n_decode > 1 else "decode",
                )
                os.makedirs(dc.torch_profiler_dir, exist_ok=True)

        # Addresses for the standard CoreManager input/output sockets.
        prefill_input_addr = get_open_zmq_ipc_path()
        prefill_output_addr = get_open_zmq_ipc_path()
        decode_input_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]
        decode_output_addrs = [get_open_zmq_ipc_path() for _ in range(n_decode)]

        from atom.model_engine.engine_core import DecodeEngineCore, PrefillEngineCore

        prefill_proc = multiprocessing.Process(
            target=PrefillEngineCore.run_engine,
            name="PrefillEngineCore",
            kwargs={
                "config": prefill_config,
                "input_address": prefill_input_addr,
                "output_address": prefill_output_addr,
            },
        )
        decode_procs = [
            multiprocessing.Process(
                target=DecodeEngineCore.run_engine,
                name=f"DecodeEngineCore{k}" if n_decode > 1 else "DecodeEngineCore",
                kwargs={
                    "config": decode_configs[k],
                    "input_address": decode_input_addrs[k],
                    "output_address": decode_output_addrs[k],
                },
            )
            for k in range(n_decode)
        ]

        # Process-index layout, used by every socket list below: index 0 is
        # prefill, indices 1..n_decode are decode ranks 0..n_decode-1.
        self._n_decode = n_decode
        self._decode_idx0 = 1

        # Set up the inherited state without running CoreManager.__init__,
        # which would spawn its own engines the base way. This manager fans out
        # through its own add_request(), but the inherited output thread still
        # releases DP load on every finished sequence, and with N decode ranks
        # the inherited _select_dp_rank_locked is what picks a request's target.
        self._init_shared_state(
            config,
            label="DisaggCoreManager",
            local_engine_count=1 + n_decode,
        )

        import weakref

        def _connect_proc(proc, in_addr, out_addr, name):
            proc.start()
            self.engine_core_processes.append(proc)
            in_sock = make_zmq_socket(self.ctx, in_addr, zmq.ROUTER, bind=True)
            identity, _ = in_sock.recv_multipart()
            self.input_sockets.append(in_sock)
            self.engine_core_identities.append(identity)
            out_sock = make_zmq_socket(self.ctx, out_addr, zmq.PULL)
            self.output_sockets.append(out_sock)
            self.shutdown_paths.append(get_open_zmq_inproc_path())
            logger.info(f"{self.label}: {name} process started and connected")

        try:
            # Start every process before waiting on any READY. Prefill binds the
            # bootstrap PUSH sockets and blocks on send() until each decode rank
            # connects and calls recv() — they rendezvous naturally, but only if
            # all of them are alive, so no READY may be awaited until then.
            _connect_proc(
                prefill_proc, prefill_input_addr, prefill_output_addr, "prefill"
            )
            for k, proc in enumerate(decode_procs):
                _connect_proc(
                    proc, decode_input_addrs[k], decode_output_addrs[k], f"decode{k}"
                )
            for idx in range(1 + n_decode):
                self._wait_for_single_ready(idx=idx)
            logger.info(f"{self.label}: all {1 + n_decode} EngineCores ready")

            # Prefill's output thread only monitors READY/errors; the decode
            # ranks are the ones that produce finished sequences.
            for idx in range(1 + n_decode):
                t = self._create_output_thread(
                    idx, self.output_sockets[idx], self.shutdown_paths[idx]
                )
                t.start()
                self.output_threads.append(t)

            if self.finished_procs():
                raise RuntimeError("DisaggCoreManager: a process failed to start")

        except Exception:
            self.close()
            raise

        self._finalizer = weakref.finalize(self, self.close)
        self.async_output_queue = None
        self._output_handler_task = None
        self._asyncio_mode = config.asyncio_mode

    def _wait_for_single_ready(self, idx: int):
        """Block until output_sockets[idx] sends a READY signal."""
        sock = self.output_sockets[idx]
        while True:
            obj = sock.recv(copy=False)
            request_type, _ = pickle.loads(obj)
            if request_type == EngineCoreRequestType.READY:
                return
            if request_type == EngineCoreRequestType.SHUTDOWN:
                raise RuntimeError(
                    f"{self.label}: process {idx} sent SHUTDOWN during initialization"
                )

    def add_request(self, seqs: list[Sequence]):
        """Fan-out: every sequence goes to prefill AND to one decode rank.

        Prefill sees all of them because it computes every prefill forward
        cooperatively across its TP ranks. A sequence's KV, though, lives on
        exactly one decode rank — that rank allocates the blocks and runs the
        decode steps — so it must be sent to that rank alone. Sending it to
        several would have each of them allocate blocks for the same sequence.
        """
        logger.debug(f"{self.label}: fan-out {len(seqs)} seqs to prefill and decode")
        # Register stream callbacks before sending (decode will produce output).
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None

        # Group by target decode rank so each rank gets one message. With a
        # single decode rank this reduces to today's behaviour.
        by_idx: dict[int, list[Sequence]] = {}
        for seq in seqs:
            if self._n_decode == 1:
                idx = self._decode_idx0
            else:
                # Select and charge atomically so a burst spreads instead of all
                # landing on the current minimum. The inherited output thread
                # calls _release_seq_load() when the sequence finishes.
                with self._lb_lock:
                    idx = self._select_dp_rank_locked(
                        lo=self._decode_idx0, hi=self.local_engine_count
                    )
                    self._charge_seq_load_locked(seq, idx)
            by_idx.setdefault(idx, []).append(seq)

        for idx, rank_seqs in by_idx.items():
            payload = pickle.dumps((EngineCoreRequestType.ADD, rank_seqs))
            self.input_sockets[idx].send_multipart(
                [self.engine_core_identities[idx], payload],
                copy=False,
            )

        # For prefill: limit each sequence to 1 output token.  Prefill discards
        # all sampled tokens (postprocess is a no-op), but setting max_tokens=1
        # ensures the forward pass terminates after a single generate step and
        # that num_scheduled_tokens correctly reflects only the prompt tokens.
        import copy as _copy

        prefill_seqs = []
        for seq in seqs:
            ps = _copy.copy(seq)
            ps.max_tokens = 1
            prefill_seqs.append(ps)
        prefill_payload = pickle.dumps((EngineCoreRequestType.ADD, prefill_seqs))
        self.input_sockets[0].send_multipart(
            [self.engine_core_identities[0], prefill_payload],
            copy=False,
        )

    def close(self):
        super().close()
        # Clean up dynamic CU partitioning shared memory (if created).
        if getattr(self, "_cu_shm", None) is not None:
            try:
                self._cu_shm.close()
                self._cu_shm.unlink()
            except Exception:
                pass
