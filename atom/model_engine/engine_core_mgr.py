# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
import logging
import multiprocessing
import pickle
import queue
import weakref
from threading import Thread
from typing import List

import zmq
import zmq.asyncio
from atom.config import Config
from atom.model_engine.engine_core import EngineCore, EngineCoreRequestType
from atom.model_engine.sequence import Sequence
from atom.utils import (
    get_open_zmq_inproc_path,
    get_open_zmq_ipc_path,
    make_zmq_socket,
    set_device_control_env_var,
)

logger = logging.getLogger("atom")


def _cleanup_core_manager_resources(
    label: str,
    engine_core_processes: list,
    input_sockets: list,
    output_sockets: list,
    shutdown_paths: list,
    output_threads: list,
    ctx,
    closed_flag: list,
):
    """
    Static cleanup function for CoreManager resources.
    Used by weakref.finalize to ensure cleanup happens even on abnormal exit.
    
    Args:
        label: Label for logging
        engine_core_processes: List of engine core processes to terminate
        input_sockets: List of ZMQ input sockets to close
        output_sockets: List of ZMQ output sockets to close
        shutdown_paths: List of shutdown signal paths
        output_threads: List of output threads to join
        ctx: ZMQ context
        closed_flag: Single-element list used as mutable flag [is_closed]
    """
    if closed_flag[0]:
        return
    closed_flag[0] = True

    logger.info(f"{label}: Cleaning up resources (finalizer triggered)")

    # 1. Close input sockets
    for input_socket in input_sockets:
        try:
            if not input_socket.closed:
                input_socket.close()
        except Exception as e:
            logger.debug(f"{label}: Error closing input socket: {e}")

    # 2. Signal output threads to shutdown
    for shutdown_path in shutdown_paths:
        if shutdown_path:
            try:
                with ctx.socket(zmq.PAIR) as shutdown_sender:
                    shutdown_sender.connect(shutdown_path)
                    shutdown_sender.send(b"")
            except Exception as e:
                logger.debug(f"{label}: Error sending shutdown signal: {e}")

    # 3. Wait for output threads to finish (with short timeout)
    for thread in output_threads:
        if thread and thread.is_alive():
            thread.join(timeout=2.0)

    # 4. Force terminate any remaining engine core processes
    # This is the critical step to prevent zombie processes
    if engine_core_processes:
        shutdown_all_processes(engine_core_processes, allowed_seconds=3)

    logger.info(f"{label}: Resource cleanup completed")


class CoreManager:
    def __init__(self, config: Config):
        self.label = "Engine Core Mgr"
        # Use a mutable container so the static cleanup function can modify it
        self._closed_flag = [False]
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
        self.utility_response_queue = queue.Queue()
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

        # Register finalizer EARLY (before starting processes) to ensure cleanup
        # happens even if process startup fails. This follows vLLM's pattern.
        # Important: Use static function (not bound method) to avoid preventing GC.
        self._finalizer = weakref.finalize(
            self,
            _cleanup_core_manager_resources,
            self.label,
            self.engine_core_processes,
            self.input_sockets,
            self.output_sockets,
            self.shutdown_paths,
            self.output_threads,
            self.ctx,
            self._closed_flag,
        )

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
                # Check if any processes failed during startup
                finished = self.finished_procs()
                if finished:
                    logger.error(
                        f"{self.label}: Some processes failed to start: {finished}, shutting down all"
                    )
                    self.close()
                    raise RuntimeError(f"Failed to start EngineCore processes: {finished}")

        except Exception as e:
            logger.error(
                f"{self.label}: Failed to initialize all EngineCores, cleaning up: {e}"
            )
            self.close()
            raise

        logger.info(
            f"{self.label}: All {self.local_engine_count} EngineCores initialized and ready"
        )
        
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
                    # Engine core failed during initialization
                    raise RuntimeError(
                        f"{self.label}: DP rank {dp_rank} failed to initialize (received SHUTDOWN signal). "
                        "Check GPU memory availability and model configuration."
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
                    elif request_type == EngineCoreRequestType.UTILITY_RESPONSE:
                        self.utility_response_queue.put_nowait(data)
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
        if self._closed_flag[0]:
            return

        logger.info(
            f"{self.label}: Shutting down all {self.local_engine_count} EngineCores"
        )

        # Send graceful shutdown signals to all engine cores first
        # This gives them a chance to cleanup before we force terminate
        for dp_rank in range(self.local_engine_count):
            self._shutdown_engine_core_rank(dp_rank)

        # Trigger the finalizer to cleanup resources
        # This handles socket closing, thread joining, and process termination
        self._finalizer()
        
        # Clear the process list after cleanup
        self.engine_core_processes = []

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

    def broadcast_utility_command_sync(self, cmd: str, timeout: float = 300.0, **kwargs):
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
            # Send graceful shutdown message
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

    def finished_procs(self) -> dict[str, int]:
        """Returns dict of proc name -> exit code for any finished procs.
        
        This follows vLLM's pattern and provides better debugging info
        than a simple boolean.
        """
        return {
            proc.name: proc.exitcode
            for proc in self.engine_core_processes
            if proc is not None and proc.exitcode is not None
        }


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
