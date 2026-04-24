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
from threading import Thread
from typing import List

import zmq
import zmq.asyncio
from atom.config import Config
from atom.model_engine.engine_core import (
    DecodeEngineCore,
    EngineCore,
    EngineCoreRequestType,
    PrefillEngineCore,
)
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
    set_device_control_env_var,
)

logger = logging.getLogger("atom")


class CoreManager:
    def __init__(self, config: Config):
        self.label = "Engine Core Mgr"
        self._closed = False  # Track whether already closed
        if config.enable_dp_attention:
            self.local_engine_count = (
                config.tensor_parallel_size * config.parallel_config.data_parallel_size
            )
            logger.info(
                f"Enable dp attention, using {self.local_engine_count} data parallel ranks"
            )
            config.parallel_config.data_parallel_size = self.local_engine_count
            config.tensor_parallel_size = 1
        else:
            self.local_engine_count = config.parallel_config.data_parallel_size
        self.ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue[List[Sequence]]()
        self.stream_outputs_queue = queue.Queue()
        self._seq_id_to_callback = {}
        self.engine_core_processes = []
        self.input_sockets = []
        self.output_sockets = []
        self.engine_core_identities = []
        self.shutdown_paths = []
        self.output_threads = []
        self._rr_counter = 0

        import torch

        if torch.multiprocessing.get_start_method(allow_none=True) is None:
            torch.multiprocessing.set_start_method("spawn", force=False)

        processes_info = []
        local_dp_ranks = []

        try:
            for dp_rank in range(self.local_engine_count):
                logger.info(
                    f"{self.label}: Creating EngineCore for DP rank {dp_rank}/{self.local_engine_count}"
                )

                # Create config for this DP rank
                import copy

                rank_config = copy.deepcopy(config)
                rank_config.parallel_config.data_parallel_rank = dp_rank

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
                                logger.debug(
                                    f"{self.label}: Cleaned up callback for finished sequence {seq_id}"
                                )
                    elif request_type == EngineCoreRequestType.ADD:
                        # logger.info(f"Engine core output sequence id: {seq.id}")
                        seqs = data
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

    async def get_output_async(self) -> List[Sequence]:
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

        # Clean up dynamic CU partitioning shared memory (if created).
        if hasattr(self, "_cu_shm"):
            try:
                self._cu_shm.close()
                self._cu_shm.unlink()
            except Exception:
                pass

        logger.info(f"{self.label}: All EngineCores shut down")

    def add_request(self, seqs: List[Sequence]):
        logger.debug(
            f"{self.label}: Add request, sequence ids: {[seq.id for seq in seqs]}"
        )
        # Register callbacks before sending to engine core
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None
        if self.local_engine_count == 1:
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
            # DP ranks, round-robin with counter for load balancing for atom server
            dp_seqs = [[] for _ in range(self.local_engine_count)]
            for seq in seqs:
                dp_rank = self._rr_counter % self.local_engine_count
                dp_seqs[dp_rank].append(seq)
                self._rr_counter += 1

            for dp_rank, rank_seqs in enumerate(dp_seqs):
                if rank_seqs:
                    logger.debug(
                        f"{self.label}: Add {len(rank_seqs)} requests to DP rank {dp_rank}"
                    )
                    self.input_sockets[dp_rank].send_multipart(
                        [
                            self.engine_core_identities[dp_rank],
                            pickle.dumps((EngineCoreRequestType.ADD, rank_seqs)),
                        ],
                        copy=False,
                    )

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

    def get_output(self) -> List[Sequence]:
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

        # Generate the inter-process ZMQ addresses before spawning.
        d2p_addr = get_open_zmq_ipc_path()  # decode → prefill (BlockAssignment)
        p2d_addr = get_open_zmq_ipc_path()  # prefill → decode (PrefillDone)
        # Bootstrap round 1: weight IPC handles (prefill → decode) + ACK (decode → prefill)
        weight_ipc_addr = get_open_zmq_ipc_path()
        weight_ack_addr = get_open_zmq_ipc_path()
        # Bootstrap round 2: kvcache handle + num_blocks (prefill → decode)
        kvcache_ipc_addr = get_open_zmq_ipc_path()

        # Shared memory for dynamic CU partitioning: 4 bytes (float32).
        # DecodeScheduler writes the chosen CU fraction; PrefillScheduler reads it.
        # 0.0 means no mask (None).
        cu_shm_name = f"atom_cu_split_{os.getpid()}"
        self._cu_shm = multiprocessing.shared_memory.SharedMemory(
            name=cu_shm_name, create=True, size=4
        )
        self._cu_shm.buf[:4] = b"\x00" * 4

        # Build per-process configs.
        from atom.utils import get_open_port as _get_open_port

        prefill_config = copy.deepcopy(config)
        if config.disagg_prefill_max_num_seqs is not None:
            prefill_config.max_num_seqs = config.disagg_prefill_max_num_seqs
        prefill_config.enforce_eager = True
        prefill_config.disagg_d2p_addr = d2p_addr
        prefill_config.disagg_p2d_addr = p2d_addr
        prefill_config.disagg_weight_ipc_addr = weight_ipc_addr
        prefill_config.disagg_weight_ack_addr = weight_ack_addr
        prefill_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        prefill_config.disagg_cu_shm_name = cu_shm_name
        # Give prefill a distinct distributed rendezvous port so it doesn't
        # collide with decode's data_parallel_base_port (both deep-copy the
        # same port from config).
        prefill_config.parallel_config.data_parallel_base_port = _get_open_port()

        decode_config = copy.deepcopy(config)
        decode_config.disagg_d2p_addr = d2p_addr
        decode_config.disagg_p2d_addr = p2d_addr
        decode_config.disagg_weight_ipc_addr = weight_ipc_addr
        decode_config.disagg_weight_ack_addr = weight_ack_addr
        decode_config.disagg_kvcache_ipc_addr = kvcache_ipc_addr
        decode_config.disagg_cu_shm_name = cu_shm_name
        # Decode allocates no GPU memory — kvcache and weights are imported from
        # prefill via CUDA IPC after prefill's READY signal.
        decode_config.disagg_is_decode = True

        # Addresses for the standard CoreManager input/output sockets.
        prefill_input_addr = get_open_zmq_ipc_path()
        prefill_output_addr = get_open_zmq_ipc_path()
        decode_input_addr = get_open_zmq_ipc_path()
        decode_output_addr = get_open_zmq_ipc_path()

        prefill_proc = multiprocessing.Process(
            target=PrefillEngineCore.run_engine,
            name="PrefillEngineCore",
            kwargs={
                "config": prefill_config,
                "input_address": prefill_input_addr,
                "output_address": prefill_output_addr,
            },
        )
        decode_proc = multiprocessing.Process(
            target=DecodeEngineCore.run_engine,
            name="DecodeEngineCore",
            kwargs={
                "config": decode_config,
                "input_address": decode_input_addr,
                "output_address": decode_output_addr,
            },
        )

        # Initialise the base class fields that close() and other methods use,
        # without calling super().__init__() (which would spawn its own processes).
        self.label = "DisaggCoreManager"
        self._closed = False
        self.local_engine_count = 2  # prefill + decode
        self.ctx = zmq.Context(io_threads=2)
        self.outputs_queue = queue.Queue()
        self.stream_outputs_queue = queue.Queue()
        self._seq_id_to_callback = {}
        self.engine_core_processes = []
        self.input_sockets = []
        self.output_sockets = []
        self.engine_core_identities = []
        self.shutdown_paths = []
        self.output_threads = []
        self._rr_counter = 0

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
            # Start both processes simultaneously.  Prefill binds the bootstrap
            # PUSH socket and blocks on send() until decode connects and calls
            # recv() — they rendezvous naturally without any sequential ordering.
            _connect_proc(
                prefill_proc, prefill_input_addr, prefill_output_addr, "prefill"
            )
            _connect_proc(decode_proc, decode_input_addr, decode_output_addr, "decode")
            self._wait_for_single_ready(idx=0)
            self._wait_for_single_ready(idx=1)
            logger.info(f"{self.label}: both EngineCores ready")

            # Start output thread for decode only (index 1).
            # Prefill has a separate output thread just for READY/error monitoring.
            for idx, name in [(0, "prefill"), (1, "decode")]:
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

    def add_request(self, seqs: List[Sequence]):
        """Fan-out: send every new sequence to BOTH prefill and decode."""
        logger.debug(f"{self.label}: fan-out {len(seqs)} seqs to prefill and decode")
        # Register stream callbacks before sending (decode will produce output).
        for seq in seqs:
            if seq.stream_callback is not None:
                self._seq_id_to_callback[seq.id] = seq.stream_callback
                seq.stream_callback = None

        # Send decode payload as-is.
        decode_payload = pickle.dumps((EngineCoreRequestType.ADD, seqs))
        self.input_sockets[1].send_multipart(
            [self.engine_core_identities[1], decode_payload],
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
