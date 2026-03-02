# # SPDX-License-Identifier: MIT
# # Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import multiprocessing
import pickle
import queue
import threading
import weakref
from contextlib import ExitStack
from threading import Thread

import zmq
import zmq.asyncio
from aiter.dist.shm_broadcast import MessageQueue
from atom.disaggregation.kvoutput_aggregator import KVOutputAggregator
from atom.utils import (
    get_mp_context,
    get_open_zmq_ipc_path,
    init_exit_handler,
    make_zmq_socket,
    resolve_obj_by_qualname,
    shutdown_all_processes,
)

logger = logging.getLogger("atom")

class AsyncIOProc:

    def __init__(
        self,
        label: str,
        io_addrs: tuple[str, str],
        input_shm_handle: int,
        runner_qualname: str ,
        rank: int,
        kv_output_addr: str | None = None,  # KV aggregation output address
        *args,
        **kwargs,
    ):
        self.label = f"AsyncIOProc({label})"
        # io_addrs[0] for input, io_addrs[1] for original output
        self.io_addrs = io_addrs
        self.io_queues = queue.Queue(), queue.Queue()
        self.io_threads: list[threading.Thread] = []

        # KV aggregation related
        self.kv_func_name_list=["async_proc_aggregation"]
        self.kv_output_addr = kv_output_addr
        self.kv_queue: queue.Queue | None = None

        self.rpc_broadcast_mq = MessageQueue.create_from_handle(input_shm_handle, rank)
        # make sure exit handler is set before runner is created
        init_exit_handler(self)
        
        # Input/output threads
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

        # KV aggregation output thread
        if self.kv_output_addr is not None:
            self.kv_queue = queue.Queue()
            t = threading.Thread(
                target=self.send_output_to_socket,
                args=(self.kv_output_addr, self.kv_queue),
                daemon=True,
            )
            t.start()
            self.io_threads.append(t)

        runner_class = resolve_obj_by_qualname(runner_qualname)  # type: ignore
        self.runners: list[object] = [runner_class(rank, *args, **kwargs)]
        self.busy_loop()

    def exit(self):
        if not getattr(self, "still_running", True):
            return
        self.still_running = False
        logger.debug(f"{self.label}: Shutting down runner...")
        for el in self.runners:
            el.exit()
        for t in self.io_threads:
            t.join(timeout=0.5)

    def recv_input_from_socket(self, addr: str, input_queue: queue.Queue):
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(ctx, addr, zmq.DEALER, bind=False)
            )
            poller = zmq.Poller()
            # Send initial message to input socket - this is required
            # before the front-end ROUTER socket can send input messages
            # back to us.
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

    def busy_loop(self):
        while True:
            func_name, args = self.get_func()
            for runner in self.runners:
                func = getattr(runner, func_name, None)
                if func is None:
                    continue
                out = func(*args)
                if out is not None:
                    if func_name  not in self.kv_func_name_list:
                        self.io_queues[1].put_nowait(out)

                    # KV aggregation channel
                    if self.kv_queue is not None and func_name in self.kv_func_name_list:
                        self.kv_queue.put_nowait(out)

            if func_name == "exit":
                break
        logger.debug(f"{self.label}: exit busy_loop...")

    def get_func(self):
        method_name, *args = self.rpc_broadcast_mq.dequeue()
        return method_name, args


class AsyncIOProcManager:

    def __init__(self, finalizer, proc_num: int, runner: str, *args):
        self.parent_finalizer = finalizer
        self.proc_num = proc_num  # 新增：供 KVOutputAggregator 使用

        io_addrs = [get_open_zmq_ipc_path(), get_open_zmq_ipc_path()]
        self.procs: list[multiprocessing.Process] = []
        ctx = get_mp_context()
        self.mp_ctx = ctx
        self.runner_label = runner.split(".")[-1]
        self.label = f"AsyncIOProcManager({self.runner_label})"

        self.rpc_broadcast_mq = MessageQueue(
            proc_num, proc_num, max_chunk_bytes=16 * 1024 * 1024
        )
        scheduler_output_handle = self.rpc_broadcast_mq.export_handle()
        self.still_running = True
        init_exit_handler(self)

        # 新增：KV 聚合相关
        self.kv_output_aggregator: KVOutputAggregator | None = None
        self.kv_output_addrs = [get_open_zmq_ipc_path() for _ in range(proc_num)]
        self.kv_outputs_queues: list[queue.Queue] = [
            queue.Queue() for _ in range(proc_num)
        ]
        self.kv_output_threads: list[threading.Thread] = []

        for i in range(proc_num):
            label = f"ModelRunner{i}/{proc_num}"
            # 保持原行为：只有 rank0 有普通输出地址 io_addrs[1]
            addrs = ([None, io_addrs[1]] if i == 0 else [None, None])

            process = ctx.Process(
                target=AsyncIOProc,
                name=label,
                args=(
                    label,
                    addrs,
                    scheduler_output_handle,
                    runner,
                    i,
                    self.kv_output_addrs[i],  # 新增：传入 KV 聚合输出地址
                    *args,
                ),
            )
            process.start()
            self.procs.append(process)

        self.zmq_ctx = zmq.Context(io_threads=2)

        # 原有输出队列与线程：只从 rank0 通道读普通输出
        self.outputs_queue: queue.Queue = queue.Queue()
        self.output_thread = threading.Thread(
            target=self.process_output_sockets,
            name=f"{self.label}_output_thread",
            args=(io_addrs[1],),
            daemon=True,
        )
        self.output_thread.start()

        # 新增：每个 worker 一条 KV 输出通道
        for i, output_addr in enumerate(self.kv_output_addrs):
            t = threading.Thread(
                target=self.process_kv_output_sockets,
                name=f"{self.label}_kv_output_thread_{i}",
                args=(output_addr, i),
                daemon=True,
            )
            t.start()
            self.kv_output_threads.append(t)

        self.monitor_procs()

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # 1. kill all runners
        logger.info(f"{self.label}: shutdown all runners...")
        shutdown_all_processes(self.procs, allowed_seconds=1)
        self.procs = []
        # 2. 等待输出线程退出
        self.output_thread.join()
        for thread in self.kv_output_threads:
            thread.join(timeout=0.5)
        logger.info(f"{self.label}: All runners are shutdown.")
        # 3. put a SystemExit to unblock call_func
        self.outputs_queue.put_nowait(SystemExit())
        # 4. 调用上层 finalizer
        self.parent_finalizer()

    def process_output_sockets(self, output_address: str):
        """原有：从 rank0 的普通输出通道收结果"""
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)  # type: ignore
                self.outputs_queue.put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: output thread exit")

    def process_kv_output_sockets(self, output_address: str, worker_id: int):
        """新增：从每个 worker 的 KV 输出通道收结果"""
        output_socket = make_zmq_socket(self.zmq_ctx, output_address, zmq.PULL)
        try:
            poller = zmq.Poller()
            poller.register(output_socket, zmq.POLLIN)
            while self.still_running:
                socks = poller.poll(timeout=1000)
                if not socks:
                    continue
                obj = output_socket.recv(copy=False)
                obj = pickle.loads(obj)  # type: ignore
                self.kv_outputs_queues[worker_id].put_nowait(obj)
        finally:
            output_socket.close(linger=0)
            logger.debug(f"{self.label}: kv output thread {worker_id} exit")

    def call_func(self, func_name: str, *args, wait_out: bool = False):
        """保持原有接口，供非 KV 场景使用"""
        logger.debug(f"{self.label}: call_func {func_name} {args}")
        msg = (func_name, *args)
        self.rpc_broadcast_mq.enqueue(msg)
        if wait_out:
            ret = self.outputs_queue.get()
            if isinstance(ret, SystemExit):
                raise ret
            return ret

    def call_func_with_aggregation(
        self, func_name: str, *args, timeout: float = 10.0
    ):  
        
        """新增：支持 KV 输出聚合的方法"""
        if self.kv_output_aggregator is None:
            self.kv_output_aggregator = KVOutputAggregator(
                world_size=self.proc_num
            )

        logger.debug(f"{self.label}: call_func_with_aggregation {func_name} {args}")
        msg = (func_name, *args)
        self.rpc_broadcast_mq.enqueue(msg)

        # 收集所有 worker 的 KV 输出
        worker_outputs = []
        for i, output_queue in enumerate(self.kv_outputs_queues):
            try:
                output = output_queue.get(timeout=timeout)
                worker_outputs.append(output)
            except queue.Empty:
                logger.error(
                    f"{self.label}: Timeout waiting for KV output from worker {i}"
                )
                return None
        kv_output = None
        if worker_outputs:
            # res= self.kv_output_aggregator.aggregate(worker_outputs)
            # logger.info(f"!!!{worker_outputs=}")
            kv_output=self.kv_output_aggregator.aggregate(worker_outputs=worker_outputs)
            logger.debug(f" Aggregated KV output: {kv_output}")
        return kv_output
        # return worker_outputs,[set()]

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
            proc_name = next(proc.name for proc in procs if proc.sentinel == died[0])
            logger.error(
                f"{self.label}: [{proc_name}] proc died unexpectedly, shutting down.",
            )
            _self.exit()

        Thread(
            target=monitor_engine_cores, daemon=True, name=f"{self.runner_label}Monitor"
        ).start()