import enum
import logging
import pickle
import queue
import threading
import weakref
from contextlib import ExitStack
from typing import Any, Optional

import zmq
import zmq.asyncio

from atom.config import Config
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence

logger = logging.getLogger("atom")


from atom.model_engine.model_runner import ModelRunner
from atom.model_engine.scheduler import Scheduler
from atom.utils import (
    close_sockets,
    get_engine_client_zmq_addr,
    get_mp_context,
    make_zmq_socket,
    shutdown_all_processes,
    zmq_socket_ctx,
)


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
        self.ps = []
        self.events = []
        ctx = get_mp_context()
        if config.tensor_parallel_size > 1:
            for i in range(1, config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(
                    target=ModelRunner, name=f"ModelRunner{i}", args=(config, i, event)
                )
                process.start()
                self.ps.append(process)
                self.events.append(event)
        self.model_runner = ModelRunner(
            config, 0, self.events if config.tensor_parallel_size > 1 else []
        )
        logger.info("Engine core model runner loaded")
        self.scheduler = Scheduler(config)
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[Sequence]()
        self.input_address = input_address
        self.output_address = output_address
        logger.info(f"Engine core input address: {self.input_address}")
        logger.info(f"Engine core output address: {self.output_address}")
        self.input_thread = threading.Thread(
            target=self.process_input_sockets, args=(self.input_address,), daemon=True
        )
        self.input_thread.start()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()

        self.profile_enbaled = config.torch_profiler_dir is not None

    def exit(self):
        self._send_engine_dead()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        logger.info("Engine core model runner exit")

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine: EngineCore = None
        try:
            engine = EngineCore(config, input_address, output_address)
            engine.start_profiler()
            engine.busy_loop()
            engine.stop_profiler()
        finally:
            if engine is not None:
                engine.exit()

    def _send_engine_dead(self):
        logger.info("Send engine core dead signal")
        self.output_queue.put_nowait(get_exit_sequence())
        self.output_thread.join(timeout=5.0)

    def busy_loop(self):
        shutdown = False
        while True:
            shutdown = shutdown or self._process_input_queue()
            if not self.scheduler.is_finished():
                self._process_engine_step()
            elif shutdown:
                break

    def _process_engine_step(self):
        scheduled_batchs = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", scheduled_batchs)
        self.scheduler.postprocess(scheduled_batchs.seqs, token_ids)
        for seq in scheduled_batchs.seqs:
            if seq.is_finished:
                self.output_queue.put_nowait(seq)

    def _process_input_queue(self):
        while not self.input_queue.empty():
            seq = self.input_queue.get_nowait()
            print(f"Engine core processing sequence {seq.id} with status {seq.status}")
            if seq.status == SequenceStatus.EXIT_ENGINE:
                return True
            self.scheduler.add(seq)
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
            logger.info("Engine core input socket connected")

            while True:
                for input_socket, _ in poller.poll():
                    # (RequestType, RequestData)
                    serialized_obj = input_socket.recv(copy=False)
                    request_type, req = pickle.loads(serialized_obj)
                    if request_type == EngineCoreRequestType.ADD:
                        logger.info(f"Engine core input: {request_type} {req.id}")
                        self.input_queue.put_nowait(req)
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.info("Engine core input thread shutdown")
                        self.input_queue.put_nowait(get_exit_sequence())
                        break

    def process_output_sockets(self, output_address: str):
        """Output socket IO thread."""
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, output_address, zmq.PUSH, linger=4000)
            )
            logger.info("Engine core output socket connected")

            while True:
                seq = self.output_queue.get()
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.info("Engine core output thread closed")
                    break
                serialized_obj = pickle.dumps((EngineCoreRequestType.ADD, seq))
                socket.send(serialized_obj)
                logger.info(f"Engine core output: {seq.id}")

    def start_profiler(self):
        if self.profile_enbaled:
            self.model_runner.call("start_profiler")

    def stop_profiler(self):
        if self.profile_enbaled:
            print("Stopping profiler...")
            self.model_runner.call("stop_profiler")
            print("Profiler stopped.")
