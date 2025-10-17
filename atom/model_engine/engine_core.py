import enum
import logging
import pickle
import queue
import signal
import threading
import weakref
from contextlib import ExitStack
from multiprocessing.shared_memory import SharedMemory
from typing import Any, List, Optional

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence
from atom.utils import (
    close_sockets,
    get_engine_client_zmq_addr,
    get_mp_context,
    init_exit_handler,
    make_zmq_socket,
    shutdown_all_processes,
    zmq_socket_ctx,
)

logger = logging.getLogger("atom")


class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used within EngineCore.
    SHUTDOWN = b"\x05"


class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[List[Sequence]]()
        self.input_address = input_address
        self.output_address = output_address
        self.input_thread = threading.Thread(
            target=self.process_input_sockets, args=(self.input_address,), daemon=True
        )
        self.input_thread.start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()

        self.profile_enbaled = config.torch_profiler_dir is not None
        init_exit_handler(self)

        # Initialize model runner processes
        try:
            good = False
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tensor_parallel_size,
                "atom.model_engine.model_runner.ModelRunner",
                config,
            )
            num_blocks = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            config.num_kvcache_blocks = num_blocks
            if not config.enforce_eager:
                cap_cost, bs = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{bs} cost: {cap_cost} seconds"
                )
            good = True
        finally:
            logger.info(
                f"{self.label}: load model runner {'success' if good else 'failed'}"
            )
            if not good:
                self._finalizer()

        self.scheduler = Scheduler(config)

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        self.runner_mgr.call_func("exit")
        self.runner_mgr.keep_monitoring = False
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait([get_exit_sequence()])
        self.output_thread.join(timeout=5.0)

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine: EngineCore = None
        try:
            engine = EngineCore(config, input_address, output_address)
            engine.start_profiler()
            engine.busy_loop()
            engine.stop_profiler()
        except Exception as e:
            logger.error(f"run_engine: exception: {e}", exc_info=True)
            raise e
        finally:
            if engine is not None:
                engine.exit()

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self.pull_and_process_input_queue()
            if not self.scheduler.is_finished():
                self._process_engine_step()
            elif shutdown:
                break

    def _process_engine_step(self):
        scheduled_batchs = self.scheduler.schedule()
        out = self.runner_mgr.call_func("forward", scheduled_batchs, wait_out=True)
        # scheduled_batchs.prev_seqs = prev_seqs
        self.scheduler.postprocess(scheduled_batchs.seqs, out)
        self.output_queue.put_nowait(
            [seq for seq in scheduled_batchs.seqs if seq.is_finished]
        )

    def pull_and_process_input_queue(self):
        recv_reqs = []
        while not self.input_queue.empty():
            seq = self.input_queue.get_nowait()
            if seq.status == SequenceStatus.EXIT_ENGINE:
                logger.debug(f"{self.label}: input_queue get {seq.status}")
                return True
            recv_reqs.append(seq)
        if len(recv_reqs) > 0:
            logger.info(f"{self.label}: put {len(recv_reqs)} reqs to scheduler")
            self.scheduler.extend(recv_reqs)
        return False

    def process_input_sockets(self, input_address: str):
        """Input socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(ctx, input_address, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
            input_socket.send(b"")
            poller.register(input_socket, zmq.POLLIN)
            logger.debug(f"{self.label}: input socket connected")
            alive = True

            while alive:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    serialized_obj = input_socket.recv(copy=False)
                    request_type, req = pickle.loads(serialized_obj)
                    if request_type == EngineCoreRequestType.ADD:
                        logger.debug(f"{self.label}: input get {request_type} {req.id}")
                        self.input_queue.put_nowait(req)
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(f"{self.label}: input get {request_type}")
                        self.input_queue.put_nowait(get_exit_sequence())
                        alive = False
                        reason = request_type
            logger.debug(f"{self.label}: input thread exit due to {reason}")

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            logger.debug(f"{self.label}: output socket connected")

            while True:
                seqs = self.output_queue.get()
                valid_seqs = [
                    seq for seq in seqs if seq.status != SequenceStatus.EXIT_ENGINE
                ]
                serialized_obj = pickle.dumps((EngineCoreRequestType.ADD, valid_seqs))
                socket.send(serialized_obj)
                num_valid = len(valid_seqs)
                if num_valid > 0:
                    logger.info(f"{self.label}: output send {num_valid} reqs")
                if len(valid_seqs) != len(seqs):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.debug(
                        f"{self.label}: output send {EngineCoreRequestType.SHUTDOWN}"
                    )
                    break

    def start_profiler(self):
        if self.profile_enbaled:
            self.runner_mgr.call_func("start_profiler")

    def stop_profiler(self):
        if self.profile_enbaled:
            print("Stopping profiler...")
            self.runner_mgr.call_func("stop_profiler", wait_out=True)
            print("Profiler stopped.")
