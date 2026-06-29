# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import enum
import logging
import pickle
import queue
import threading
import time
from contextlib import ExitStack
from typing import List

import torch
import zmq
from atom.config import Config, ParallelConfig
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.scheduler import DecodeScheduler, PrefillScheduler, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, get_exit_sequence
from atom.utils import envs, init_exit_handler, make_zmq_socket
from atom.utils.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)

from atom.kv_transfer.disaggregation import KVOutputAggregator

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
    # Stream output for callbacks
    STREAM = b"\x06"
    # Signal that EngineCore is fully initialized and ready
    READY = b"\x07"
    # Response to a synchronous utility command
    UTILITY_RESPONSE = b"\x08"


class EngineCore:
    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        self.input_queue = queue.Queue[Sequence]()
        self.output_queue = queue.Queue[List[Sequence]]()
        self.stream_output_queue = (
            queue.Queue()
        )  # Queue for streaming intermediate outputs
        # Queue for utility commands (processed in busy_loop to avoid thread contention)
        self.utility_queue = queue.Queue()
        self._has_pending_utility = (
            False  # Flag to avoid checking empty queue every loop
        )
        self._is_rl_weights_offloaded = (
            False  # True when weights are offloaded for RL training
        )
        self.input_address = input_address
        self.output_address = output_address
        self.output_thread = threading.Thread(
            target=self.process_output_sockets, args=(self.output_address,), daemon=True
        )
        self.output_thread.start()

        # Start input thread BEFORE _init_data_parallel so that CoreManager
        # can receive the input socket connection and proceed to start the
        # remaining DP ranks.  Without this, _init_data_parallel blocks on
        # rendezvous waiting for all DP ranks, but they haven't been spawned
        # yet because CoreManager is still waiting for *this* rank's socket.
        # The READY signal (sent at the end of __init__) gates actual request
        # processing, so starting the input thread early is safe.
        self.input_thread = threading.Thread(
            target=self.process_input_sockets, args=(self.input_address,), daemon=True
        )
        self.input_thread.start()

        self.profile_enbaled = config.torch_profiler_dir is not None
        self.mark_trace = getattr(config, "mark_trace", False)
        init_exit_handler(self)
        self._init_data_parallel(config)

        # Initialize model runner processes
        try:
            good = False
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tensor_parallel_size,
                config.runner_qualname,
                config,
            )
            self._post_model_load_hook()
            block_info = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            num_blocks = block_info["num_kvcache_blocks"]
            config.per_req_cache_equiv_blocks = block_info.get(
                "per_req_cache_equiv_blocks", 0
            )
            config.num_per_req_cache_groups = block_info.get(
                "num_per_req_cache_groups", 0
            )
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            config.num_kvcache_blocks = num_blocks
            # Decode in disagg mode defers CUDA graph capture until after kvcache
            # IPC import — the capture runs from DecodeEngineCore.__init__ instead.
            if not config.enforce_eager and not config.disagg_is_decode:
                # Start profiler before cudagraph capture only if mark-trace is enabled.
                if self.profile_enbaled and self.mark_trace:
                    self.runner_mgr.call_func(
                        "start_profiler", "capture_graph", wait_out=True
                    )
                cap_cost, bs = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{bs} cost: {cap_cost:.2f} seconds"
                )
                if self.profile_enbaled and self.mark_trace:
                    # Persist a dedicated capture-graph trace immediately.
                    self.runner_mgr.call_func("stop_profiler", wait_out=True)
            good = True
        finally:
            logger.info(
                f"{self.label}: load model runner {'success' if good else 'failed'}"
            )
            if not good:
                self._finalizer()

        # Decode in disagg mode defers Scheduler creation until after kvcache IPC
        # import sets config.num_kvcache_blocks (BlockManager asserts num_blocks > 0).
        # Keep the attribute defined (None) so EngineUtilityHandler below and other
        # consumers can reference it before DecodeEngineCore creates the real one.
        self.scheduler = None
        if not config.disagg_is_decode:
            self.scheduler = Scheduler(config)

        self.kv_transfer_enabled = bool(config.kv_transfer_config)
        if self.kv_transfer_enabled:
            self.kv_aggregator = KVOutputAggregator(
                world_size=config.tensor_parallel_size
            )

        self.utility_handler = EngineUtilityHandler(
            self.runner_mgr,
            self.output_queue,
            label=self.label,
            scheduler=self.scheduler,
        )

        self._send_ready_signal()
        logger.info(f"{self.label}: EngineCore fully initialized and ready")

    def _send_ready_signal(self):
        self.output_queue.put_nowait(("READY", None))

    def _post_model_load_hook(self):
        """Called after ModelRunner is initialized (model loaded) but before
        get_num_blocks/allocate_kv_cache.  Override in subclasses to inject
        inter-process synchronization at this point in the init sequence."""
        pass

    def _init_data_parallel(self, config: Config):
        pass

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        if not hasattr(self, "runner_mgr"):
            self._send_engine_dead()
            return
        self.runner_mgr.keep_monitoring = False
        try:
            self.runner_mgr.call_func("exit")
        except Exception:
            pass  # shared memory may already be freed
        for proc in self.runner_mgr.procs:
            try:
                alive = proc.is_alive()
            except ValueError:
                continue  # process object already closed by CoreManager
            if alive:
                proc.join(timeout=5)
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait([get_exit_sequence()])
        self.output_thread.join(timeout=0.5)

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine: EngineCore = None
        try:
            if config.parallel_config.data_parallel_size > 1:
                engine = DPEngineCoreProc(config, input_address, output_address)
            else:
                engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception as e:
            logger.error(f"run_engine: exception: {e}", exc_info=True)
            raise e
        finally:
            if engine is not None:
                engine.exit()

    def _is_idle_rl_weights_offloaded(self) -> bool:
        """Check if weights are offloaded for RL training.

        When offloaded, busy-wait with a short delay to avoid CPU spin.
        Returns True if the caller should skip model execution this tick.
        """
        if self._is_rl_weights_offloaded:
            time.sleep(0.01)
            return True
        return False

    def busy_loop(self):
        shutdown = False
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                if self._is_idle_rl_weights_offloaded():
                    continue
                if not self.scheduler.is_finished():
                    self._process_engine_step()
        finally:
            # Teardown runs even on exceptions so the sender thread/socket
            # don't leak. Isolate the final publish so a publisher hiccup
            # cannot skip shutdown_kv_events().
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during shutdown failed")
            self.scheduler.shutdown_kv_events()

    def _process_engine_step(self):
        try:
            return self._process_engine_step_inner()
        finally:
            # Swallow publisher errors so they cannot mask an exception from
            # the engine step itself.
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish in engine-step finally failed")

    def _process_engine_step_inner(self):
        result = self.scheduler.schedule()

        # Surface admit-rejected seqs (those `_unschedulable_reason` flags in
        # the scheduler) through the same finished-seq path as normal seqs.
        # Without this, `llm.generate()` blocks forever waiting for an output
        # the rejected seq will never produce.
        rejected = self.scheduler.take_rejected()
        if rejected:
            self.output_queue.put_nowait(rejected)

        if result is None:
            self._advance_idle_kv_transfer()
            return False
        scheduled_batch, seqs = result

        if scheduled_batch is None:
            logger.debug("%s: No sequences to schedule, skipping forward", self.label)
            self._advance_idle_kv_transfer()
            return False

        # Dispatch KV connector metadata to workers (triggers async KV load)
        if (
            self.kv_transfer_enabled
            and scheduled_batch.connector_meta_output is not None
        ):
            self.runner_mgr.call_func(
                "process_kvconnector_output", scheduled_batch.connector_meta_output
            )

        # Run the model forward pass if there are actual sequences
        has_seqs = len(scheduled_batch.req_ids) > 0
        if has_seqs:
            fwd_out = self.runner_mgr.call_func(
                "forward", scheduled_batch, wait_out=True
            )

        # Aggregate KV transfer status from all workers (only when PD disaggregation is active)
        self._poll_kv_transfer_progress()

        if not has_seqs:
            logger.debug("%s: Empty scheduled batch, skipping postprocess", self.label)
            return False

        seqs = seqs.values()
        # Pass stream_output_queue to postprocess for streaming callbacks
        finished_seqs = self.scheduler.postprocess(
            seqs,
            fwd_out,
            stream_output_queue=self.stream_output_queue,
            batch=scheduled_batch,
        )

        # Send stream outputs to main process via output_queue
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                # Send stream outputs as intermediate results
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass

        if finished_seqs:
            self.output_queue.put_nowait(finished_seqs)

        return True

    def _advance_idle_kv_transfer(self) -> None:
        # No forward batch will run this tick, but offload load/save work may
        # still need to be dispatched or reported back to the scheduler.
        self._dispatch_idle_offload_work()
        self._poll_kv_transfer_progress()

    def _poll_kv_transfer_progress(self) -> None:
        if not self.kv_transfer_enabled:
            return
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        self.scheduler._update_from_kv_xfer_finished(kvoutput)

    def _dispatch_idle_offload_work(self) -> None:
        if not self.kv_transfer_enabled:
            return
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not getattr(connector, "is_offload", False):
            return
        meta = connector.build_connector_meta()
        if meta is None or not getattr(meta, "requests", None):
            return
        self.runner_mgr.call_func("process_kvconnector_output", meta)

    def pull_and_process_input_queue(self):
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    logger.debug(f"{self.label}: input_queue get exit engine")
                    return True
                recv_reqs.append(seq)
        if len(recv_reqs) > 0:
            logger.debug(f"{self.label}: put {len(recv_reqs)} reqs to scheduler")
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
                    obj = input_socket.recv(copy=False)
                    request_type, reqs = pickle.loads(obj)
                    if request_type == EngineCoreRequestType.ADD:
                        req_ids = [req.id for req in reqs]
                        logger.debug(
                            f"{self.label}: input get {request_type} {req_ids}"
                        )
                        self.input_queue.put_nowait(reqs)
                    elif request_type == EngineCoreRequestType.UTILITY:
                        cmd = reqs.get("cmd") if isinstance(reqs, dict) else None
                        logger.debug(f"{self.label}: input get UTILITY command: {cmd}")
                        self.utility_queue.put_nowait((cmd, reqs))
                        self._has_pending_utility = True
                    elif request_type == EngineCoreRequestType.SHUTDOWN:
                        logger.debug(f"{self.label}: input get {request_type}")
                        self.input_queue.put_nowait([get_exit_sequence()])
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
                item = self.output_queue.get()
                if isinstance(item, tuple) and item[0] == "STREAM":
                    # Send stream outputs
                    stream_outputs = item[1]
                    obj = pickle.dumps((EngineCoreRequestType.STREAM, stream_outputs))
                    socket.send(obj)
                    continue

                if isinstance(item, tuple) and item[0] == "READY":
                    # Send READY signal to indicate EngineCore is fully initialized
                    obj = pickle.dumps((EngineCoreRequestType.READY, None))
                    socket.send(obj)
                    logger.debug(f"{self.label}: sent READY signal")
                    continue

                if isinstance(item, tuple) and item[0] == "UTILITY_RESPONSE":
                    # Send utility command response back to CoreManager
                    response_data = item[1]
                    serialized_obj = pickle.dumps(
                        (EngineCoreRequestType.UTILITY_RESPONSE, response_data)
                    )
                    socket.send(serialized_obj)
                    continue

                # Regular finished sequences
                seqs = item
                valid_seqs = [
                    seq for seq in seqs if seq.status != SequenceStatus.EXIT_ENGINE
                ]
                num_valid = len(valid_seqs)
                if num_valid > 0:
                    obj = pickle.dumps((EngineCoreRequestType.ADD, valid_seqs))
                    socket.send(obj)
                    logger.info(f"{self.label}: output send {num_valid} reqs")
                if len(valid_seqs) != len(seqs):
                    socket.send(pickle.dumps((EngineCoreRequestType.SHUTDOWN, None)))
                    logger.debug(
                        f"{self.label}: output send {EngineCoreRequestType.SHUTDOWN}"
                    )
                    break


class DPEngineCoreProc(EngineCore):
    def __init__(self, config: Config, input_address: str, output_address: str):
        # self.dp_group = config.parallel_config.dp_group
        self.dp_rank = config.parallel_config.data_parallel_rank
        # self.dp_group = config.parallel_config.stateless_init_dp_group()
        super().__init__(config, input_address, output_address)
        # Initialize to True so first iteration reaches all_reduce
        self.engines_running = True
        self._shutting_down = False

        if envs.ATOM_ENABLE_PREFILL_DELAYER:
            from atom.model_engine.prefill_delayer import PrefillDelayer

            self.scheduler.set_prefill_delayer(
                PrefillDelayer(
                    dp_size=config.parallel_config.data_parallel_size,
                    cpu_group=self.dp_group,
                    max_delay_passes=envs.ATOM_PREFILL_DELAYER_MAX_DELAY_PASSES,
                    max_delay_ms=envs.ATOM_PREFILL_DELAYER_MAX_DELAY_MS,
                    token_usage_low_watermark=envs.ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK,
                )
            )

    def _init_data_parallel(self, config: Config):
        dp_rank = config.parallel_config.data_parallel_rank
        dp_size = config.parallel_config.data_parallel_size
        local_dp_rank = config.parallel_config.data_parallel_rank_local

        assert dp_size > 1
        assert local_dp_rank is not None
        assert 0 <= local_dp_rank <= dp_rank < dp_size

        self.dp_rank = dp_rank
        self.dp_group = config.parallel_config.stateless_init_dp_group()
        # NOTE: PrefillDelayer attachment lives in __init__ (after
        # super().__init__ creates self.scheduler) — not here. This
        # function runs during super().__init__, before self.scheduler
        # exists.

    def exit(self):
        super().exit()
        if dp_group := getattr(self, "dp_group", None):
            stateless_destroy_torch_distributed_process_group(dp_group)

    def busy_loop(self):
        shutdown = False
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                shutdown = shutdown or self.pull_and_process_input_queue()
                local_unfinished = (
                    not self.scheduler.is_finished()
                    and not self._is_rl_weights_offloaded
                )

                global_has_unfinished, global_shutdown, global_offloaded = (
                    self._sync_dp_state(
                        local_unfinished, shutdown, self._is_rl_weights_offloaded
                    )
                )

                if global_shutdown and not global_has_unfinished:
                    logger.info(
                        f"{self.label}: All DP ranks agreed to shutdown, exiting busy_loop"
                    )
                    break

                if global_offloaded:
                    time.sleep(0.01)
                    continue

                if not global_has_unfinished and not self.engines_running:
                    self.engines_running = False
                    continue

                executed = self._process_engine_step()
                if not executed:
                    self._execute_dummy_batch()

                self.engines_running = global_has_unfinished
        finally:
            # Isolate the final publish so a publisher hiccup cannot skip
            # shutdown_kv_events() (which closes the sender thread/socket).
            try:
                self.scheduler.publish_kv_events()
            except Exception:
                logger.exception("KV event publish during DP shutdown failed")
            self.scheduler.shutdown_kv_events()

    def _execute_dummy_batch(self):
        return self.runner_mgr.call_func("dummy_execution", wait_out=True)

    def _sync_dp_state(
        self,
        local_has_unfinished: bool,
        local_shutdown: bool = False,
        local_offloaded: bool = False,
    ) -> tuple[bool, bool, bool]:
        if self._shutting_down:
            return local_has_unfinished, True, local_offloaded

        try:
            state_tensor = torch.tensor(
                [
                    1 if local_has_unfinished else 0,
                    1 if local_shutdown else 0,
                    1 if local_offloaded else 0,
                ],
                dtype=torch.int64,
                device="cpu",
            )
            torch.distributed.all_reduce(
                state_tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_has_unfinished = state_tensor[0].item() == 1
            global_shutdown = state_tensor[1].item() == 1
            global_offloaded = state_tensor[2].item() == 1
            return global_has_unfinished, global_shutdown, global_offloaded
        except RuntimeError as e:
            logger.warning(f"{self.label}: _sync_dp_state failed: {e}")
            self._shutting_down = True
            return local_has_unfinished, True, local_offloaded

    def _sync_shutdown_state(self, local_should_shutdown: bool) -> bool:
        try:
            tensor = torch.tensor(
                [local_should_shutdown], dtype=torch.int32, device="cpu"
            )
            torch.distributed.all_reduce(
                tensor, op=torch.distributed.ReduceOp.MAX, group=self.dp_group
            )
            global_should_shutdown = bool(tensor.item())
            return global_should_shutdown
        except RuntimeError as e:
            # If all_reduce fails, it means other ranks are shutting down
            logger.warning(
                f"{self.label}: Shutdown sync failed, assuming shutdown: {e}"
            )
            return True

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        if self._shutting_down:
            logger.info(f"{self.label}: Skipping DP sync during shutdown")
            return local_unfinished
        try:
            return ParallelConfig.has_unfinished_dp(self.dp_group, local_unfinished)
        except RuntimeError as e:
            # Handle case where other ranks have already shut down
            logger.warning(f"{self.label}: DP sync failed during shutdown: {e}")
            return local_unfinished


# ---------------------------------------------------------------------------
# Disaggregated prefill/decode engine cores
# ---------------------------------------------------------------------------


class PrefillEngineCore(EngineCore):
    """Disaggregated prefill instance.

    Responsibilities:
    - Runs only prefill forward passes (enforce_eager=True, no CUDA graphs).
    - Receives BlockAssignment messages from DecodeEngineCore over a direct
      ZMQ PULL socket (disagg_d2p_addr) and populates seq.block_table before
      calling schedule().
    - Runs each forward pass on a dedicated CUDA stream (_prefill_stream).
      Records a CUDA event immediately after each forward; a separate watcher
      thread polls the events and sends PrefillDone only after the GPU has
      committed all KV writes — ensuring decode never reads a partial cache.
    - Imports the KV cache tensor from decode via CUDA IPC at startup so that
      its forward pass writes directly into decode's GPU buffer.
    - Does NOT sample tokens; completed sequences are discarded here.
    """

    def __init__(self, config: Config, input_address: str, output_address: str):

        # Force eager mode — no CUDA graph capture on the prefill side.
        config.enforce_eager = True

        self._disagg_d2p_addr = config.disagg_d2p_addr  # PULL: receive BlockAssignment
        self._disagg_p2d_addr = config.disagg_p2d_addr  # PUSH: send PrefillDone
        self._disagg_weight_ipc_addr = config.disagg_weight_ipc_addr
        self._disagg_weight_ack_addr = config.disagg_weight_ack_addr
        self._disagg_kvcache_ipc_addr = config.disagg_kvcache_ipc_addr

        # Maps seq_id → BlockAssignment, populated by the receiver thread.
        self._pending_assignments: dict = {}
        # self._pending_lock = threading.Lock()

        # ZMQ context for disagg sockets (separate from the main engine sockets).
        self._disagg_ctx = zmq.Context()

        # Store config reference so _send_ready_signal can read num_kvcache_blocks
        # after EngineCore.__init__ sets it.
        self._config = config

        super().__init__(config, input_address, output_address)
        # Replace the base Scheduler created by EngineCore.__init__ with
        # PrefillScheduler, which has no BlockManager and only schedules
        # sequences that already have a block_table from decode.
        self.scheduler = PrefillScheduler(
            config, disagg_cu_shm_name=config.disagg_cu_shm_name
        )

    def _post_model_load_hook(self):
        """Round 1 bootstrap: export weights → send to decode → wait for ACK.

        Called after model is loaded but before get_num_blocks/allocate_kv_cache.
        Decode receives the handles, imports them, frees its own copy, and sends
        an ACK.  Only after the ACK does prefill proceed to measure free GPU memory
        and allocate KV cache — ensuring decode's weight tensors are already freed
        so prefill sees the full available memory budget.
        """
        logger.info("PrefillEngineCore: exporting weight IPC handles...")
        weight_handles = self.runner_mgr.call_func(
            "export_model_weight_ipc_handles", wait_out=True
        )
        weight_payload = pickle.dumps(weight_handles)
        logger.info(
            f"PrefillEngineCore: sending weight handles ({len(weight_payload)} bytes)..."
        )
        with self._disagg_ctx.socket(zmq.PUSH) as w_sock:
            w_sock.bind(self._disagg_weight_ipc_addr)
            w_sock.send(weight_payload)
        logger.info("PrefillEngineCore: weight handles sent, waiting for ACK...")
        with self._disagg_ctx.socket(zmq.PULL) as ack_sock:
            ack_sock.connect(self._disagg_weight_ack_addr)
            ack_sock.recv()
        logger.info("PrefillEngineCore: weight ACK received — decode weights freed")

    def _send_ready_signal(self):
        """Round 2 bootstrap: export kvcache handle → send to decode → emit READY."""
        logger.info("PrefillEngineCore: exporting kvcache IPC handle...")
        kvcache_args = self.runner_mgr.call_func(
            "export_kv_cache_ipc_handle", wait_out=True
        )
        kvcache_bundle = pickle.dumps(
            {
                "kvcache_args": kvcache_args,
                "num_kvcache_blocks": self._config.num_kvcache_blocks,
            }
        )
        logger.info(
            f"PrefillEngineCore: sending kvcache bundle ({len(kvcache_bundle)} bytes)..."
        )
        # Keep socket alive on self until exit() so linger doesn't block shutdown.
        self._bootstrap_push_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._bootstrap_push_sock.bind(self._disagg_kvcache_ipc_addr)
        self._bootstrap_push_sock.send(kvcache_bundle)
        logger.info("PrefillEngineCore: kvcache bundle sent")
        super()._send_ready_signal()

    def _init_disagg(self):
        """Start runtime coordination sockets and threads.

        Called from run_engine() after super().__init__().  The kvcache+weight
        bootstrap was already sent inside _send_ready_signal(), so _init_disagg
        only needs to set up the per-request BlockAssignment/PrefillDone channel.
        """
        # --- Create pool of CU-masked CUDA streams for prefill ---
        logger.info("PrefillEngineCore: creating prefill stream pool...")
        self.runner_mgr.call_func("create_prefill_stream_pool", wait_out=True)
        logger.info("PrefillEngineCore: prefill stream pool created")

        # --- Open the PUSH socket to send PrefillDone messages ---
        # Prefill connects (not binds) so decode's PULL bind is ready first,
        # preventing messages from being dropped before decode connects.
        self._p2d_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._p2d_sock.connect(self._disagg_p2d_addr)

        # --- Start thread to receive BlockAssignment from decode ---
        # Prefill binds so decode's PUSH connect finds a ready socket.
        self._assignment_sock = self._disagg_ctx.socket(zmq.PULL)
        self._assignment_sock.bind(self._disagg_d2p_addr)
        self._assignment_thread = threading.Thread(
            target=self._recv_block_assignments,
            daemon=True,
            name="PrefillEngineCore-AssignmentRecv",
        )
        self._assignment_thread.start()

    def _recv_block_assignments(self):
        """Background thread: pulls BlockAssignment messages from decode."""
        from atom.model_engine.disagg_types import BlockAssignment, DisaggMsgType

        sock = self._assignment_sock
        while True:
            try:
                raw = sock.recv()
            except zmq.error.ContextTerminated:
                break
            msg_type, payload = pickle.loads(raw)
            if msg_type == DisaggMsgType.BLOCK_ASSIGNMENT:
                assignment: BlockAssignment = payload
                with self.scheduler._pending_lock:
                    self._pending_assignments[assignment.seq_id] = assignment
            elif msg_type == DisaggMsgType.ABORT:
                seq_id = payload
                with self.scheduler._pending_lock:
                    self._pending_assignments.pop(seq_id, None)
                # Also remove from waiting queue if present.
                self.scheduler.waiting = type(self.scheduler.waiting)(
                    s for s in self.scheduler.waiting if s.id != seq_id
                )

    def _apply_pending_assignments(self):
        """Copy block assignments received from decode into waiting sequences.

        Must be called at the top of each engine step before schedule().
        """
        with self.scheduler._pending_lock:
            if not self._pending_assignments:
                return
            for seq in self.scheduler.waiting:
                if seq.id in self._pending_assignments:
                    assignment = self._pending_assignments.pop(seq.id)
                    seq.block_table = list(assignment.block_table)
                    seq.num_cached_tokens = assignment.num_cached_tokens

    def _process_engine_step(self):
        from atom.model_engine.disagg_types import DisaggMsgType, PrefillDone

        self._apply_pending_assignments()
        if not self.scheduler.has_requests():
            return False
        result = self.scheduler.schedule()
        if result is None:
            return False
        scheduled_batch, seqs = result
        if scheduled_batch is None:
            return False

        # Run on the dedicated prefill stream; returns sampled token IDs (one per seq).
        t0 = time.perf_counter()
        sampled_token_ids = self.runner_mgr.call_func(
            "prefill_forward", scheduled_batch, wait_out=True
        )
        iter_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"prefill iter {iter_ms:.2f}ms | "
            f"reqs={scheduled_batch.total_seqs_num} | "
            f"tokens={scheduled_batch.total_tokens_num}"
        )

        # Notify decode that prefill is done, including the first generated token
        # so decode can append it before its first decode step.
        for seq_id, num_tokens, token_id in zip(
            scheduled_batch.req_ids,
            scheduled_batch.num_scheduled_tokens,
            sampled_token_ids,
        ):
            done = PrefillDone(
                seq_id=seq_id,
                num_tokens_computed=int(num_tokens),
                sampled_token_id=int(token_id),
            )
            self._p2d_sock.send(pickle.dumps((DisaggMsgType.PREFILL_DONE, done)))

        # Remove completed sequences — prefill produces no output tokens.
        for seq in list(seqs.values()):
            seq.status = SequenceStatus.FINISHED
            try:
                self.scheduler.running.remove(seq)
            except ValueError:
                pass
        return True

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine = None
        try:
            engine = PrefillEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception as e:
            logger.error(f"PrefillEngineCore.run_engine: exception: {e}", exc_info=True)
            raise
        finally:
            if engine is not None:
                engine.exit()

    def exit(self):
        super().exit()
        try:
            if hasattr(self, "_bootstrap_push_sock"):
                self._bootstrap_push_sock.close(linger=0)
        except Exception:
            pass
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            pass


class DecodeEngineCore(EngineCore):
    """Disaggregated decode instance.

    Responsibilities:
    - Owns the BlockManager and the KV cache tensor.
    - On new request arrival: allocates KV blocks, sends BlockAssignment to
      prefill, and holds the sequence in prefill_pending until PrefillDone
      arrives.
    - PrefillDone signals move the sequence into the decode scheduler's
      running queue.
    - Exports the kv_cache IPC handle to prefill at startup.
    - Runs normal CUDA-graph-accelerated decode.
    """

    def __init__(self, config: Config, input_address: str, output_address: str):
        self._disagg_d2p_addr = config.disagg_d2p_addr  # PUSH: send BlockAssignment
        self._disagg_p2d_addr = config.disagg_p2d_addr  # PULL: receive PrefillDone
        self._disagg_weight_ipc_addr = config.disagg_weight_ipc_addr
        self._disagg_weight_ack_addr = config.disagg_weight_ack_addr
        self._disagg_kvcache_ipc_addr = config.disagg_kvcache_ipc_addr

        self._disagg_ctx = zmq.Context()

        # Store config so we can read/update num_kvcache_blocks after import.
        self._config = config

        # Suppress _send_ready_signal during super().__init__() — we send the
        # real READY only after kvcache IPC import and cudagraph capture.
        self._ready_deferred = True
        super().__init__(config, input_address, output_address)
        self._ready_deferred = False

        # --- Round 2: receive kvcache bundle from prefill ---
        logger.info("DecodeEngineCore: waiting for kvcache bundle from prefill...")
        with self._disagg_ctx.socket(zmq.PULL) as sock:
            sock.connect(self._disagg_kvcache_ipc_addr)
            raw = sock.recv()
        bundle = pickle.loads(raw)
        num_kvcache_blocks = bundle["num_kvcache_blocks"]
        logger.info(
            f"DecodeEngineCore: received kvcache bundle ({num_kvcache_blocks} blocks)"
        )

        # --- Import kvcache — sets self.kv_cache + binds to attention modules ---
        self.runner_mgr.call_func(
            "import_kv_cache_ipc_handle",
            bundle["kvcache_args"],
            num_kvcache_blocks,
            wait_out=True,
        )
        logger.info("DecodeEngineCore: kvcache IPC import complete")

        # --- Capture CUDA graphs now that kvcache is real ---
        config.num_kvcache_blocks = num_kvcache_blocks
        # if not config.enforce_eager:
        #     cap_cost, bs = self.runner_mgr.call_func("capture_cudagraph", wait_out=True)
        #     logger.info(
        #         f"DecodeEngineCore: cudagraph capture{bs} cost: {cap_cost:.2f}s"
        #     )

        if not config.enforce_eager:
            if self.profile_enbaled and self.mark_trace:
                self.runner_mgr.call_func(
                    "start_profiler", "capture_graph", wait_out=True
                )
            cap_cost, bs = self.runner_mgr.call_func("capture_cudagraph", wait_out=True)
            logger.info(
                f"DecodeEngineCore: cudagraph capture{bs} cost: {cap_cost:.2f}s"
            )
            if self.profile_enbaled and self.mark_trace:
                # Persist a dedicated capture-graph trace immediately.
                self.runner_mgr.call_func("stop_profiler", wait_out=True)

        # --- Create DecodeScheduler now that num_kvcache_blocks is set ---
        self.scheduler = DecodeScheduler(
            config, disagg_cu_shm_name=config.disagg_cu_shm_name
        )
        # EngineUtilityHandler was built in super().__init__() with scheduler=None
        # (decode defers scheduler creation); wire the real one in for MTP stats.
        self.utility_handler.scheduler = self.scheduler

        # --- Now truly ready ---
        super()._send_ready_signal()
        logger.info("DecodeEngineCore: fully initialized and ready")

    def _post_model_load_hook(self):
        """Round 1 bootstrap: receive weight IPC handles from prefill, import them,
        free decode's own weight copy, then send ACK so prefill can proceed to
        allocate KV cache with the full GPU memory budget.
        """
        logger.info("DecodeEngineCore: waiting for weight IPC handles from prefill...")
        with self._disagg_ctx.socket(zmq.PULL) as w_sock:
            w_sock.connect(self._disagg_weight_ipc_addr)
            raw = w_sock.recv()
        weight_handles = pickle.loads(raw)
        logger.info(
            f"DecodeEngineCore: received weight handles ({len(weight_handles)} params), importing..."
        )
        self.runner_mgr.call_func(
            "import_model_weight_ipc_handles", weight_handles, wait_out=True
        )
        # Give the GPU allocator a moment to reclaim the freed weight memory
        # before prefill measures free VRAM for KV cache sizing.  Without this,
        # the freed pages may not yet be visible and prefill can OOM on alloc.

        time.sleep(2)
        logger.info(
            "DecodeEngineCore: weight import complete, sending ACK to prefill..."
        )
        with self._disagg_ctx.socket(zmq.PUSH) as ack_sock:
            ack_sock.bind(self._disagg_weight_ack_addr)
            ack_sock.send(b"ok")
        logger.info("DecodeEngineCore: weight ACK sent")

    def _send_ready_signal(self):
        """No-op during super().__init__(): READY is deferred until after IPC import."""
        if getattr(self, "_ready_deferred", False):
            return
        super()._send_ready_signal()

    def _init_disagg(self):
        """Start runtime coordination sockets and threads.

        Called from run_engine() after __init__().  The kvcache+weight bootstrap
        was already handled in __init__(), so _init_disagg only sets up the
        per-request BlockAssignment/PrefillDone channel.
        """
        # Create pool of CU-masked CUDA streams for decode forward passes.
        self.runner_mgr.call_func("create_decode_stream_pool", wait_out=True)

        # Decode PUSH connects to prefill's bound PULL (d2p channel).
        self._d2p_sock = self._disagg_ctx.socket(zmq.PUSH)
        self._d2p_sock.connect(self._disagg_d2p_addr)

        # Decode PULL binds so prefill's connecting PUSH finds a ready socket (p2d channel).
        self._p2d_recv_sock = self._disagg_ctx.socket(zmq.PULL)
        self._p2d_recv_sock.bind(self._disagg_p2d_addr)

        # Start thread to receive PrefillDone from prefill.
        self._prefill_done_thread = threading.Thread(
            target=self._recv_prefill_done,
            daemon=True,
            name="DecodeEngineCore-PrefillDoneRecv",
        )
        self._prefill_done_thread.start()

    def _recv_prefill_done(self):
        """Background thread: receives PrefillDone messages from prefill."""
        from atom.model_engine.disagg_types import DisaggMsgType, PrefillDone

        sock = self._p2d_recv_sock
        while True:
            try:
                raw = sock.recv()
            except zmq.error.ContextTerminated:
                break
            msg_type, payload = pickle.loads(raw)
            if msg_type != DisaggMsgType.PREFILL_DONE:
                continue
            done: PrefillDone = payload
            self.scheduler.on_prefill_done(
                done.seq_id, done.num_tokens_computed, done.sampled_token_id
            )
            logger.info(
                f"DecodeEngineCore: seq {done.seq_id} prefill done "
                f"({done.num_tokens_computed} tokens cached), moved to running queue"
            )

    def _send_block_assignment(self, seq: Sequence):
        """Send BlockAssignment to prefill for a newly allocated sequence."""
        from atom.model_engine.disagg_types import BlockAssignment, DisaggMsgType

        assignment = BlockAssignment(
            seq_id=seq.id,
            block_table=list(seq.block_table),
            num_cached_tokens=seq.num_cached_tokens,
            context_len=seq.num_tokens,
        )
        self._d2p_sock.send(pickle.dumps((DisaggMsgType.BLOCK_ASSIGNMENT, assignment)))
        logger.info(
            f"DecodeEngineCore: seq {seq.id} blocks assigned "
            f"({len(assignment.block_table)} blocks), waiting for prefill"
        )

    def pull_and_process_input_queue(self):
        """Override: add new sequences to waiting queue, then allocate blocks
        and send BlockAssignment to prefill for any that fit."""
        recv_reqs = []
        while not self.input_queue.empty():
            seqs = self.input_queue.get_nowait()
            for seq in seqs:
                if seq.status == SequenceStatus.EXIT_ENGINE:
                    return True
                recv_reqs.append(seq)
        if recv_reqs:
            self.scheduler.extend(recv_reqs)
        for seq in self.scheduler.allocate_waiting():
            self._send_block_assignment(seq)
        return False

    def _process_engine_step(self):
        """Override: handle None from DecodeScheduler when prefill_waiting is
        non-empty but running is empty (sequences are still being prefilled)."""
        if not self.scheduler.has_requests():
            return False
        result = self.scheduler.schedule()
        if result is None:
            # Sequences exist but are still waiting for PrefillDone — spin.
            return False
        scheduled_batch, seqs = result
        if scheduled_batch is None:
            return False
        t0 = time.perf_counter()
        fwd_out = self.runner_mgr.call_func("forward", scheduled_batch, wait_out=True)
        iter_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"iter {iter_ms:.2f}ms | "
            f"reqs={scheduled_batch.total_seqs_num} "
            f"(prefill={scheduled_batch.total_seqs_num_prefill} "
            f"decode={scheduled_batch.total_seqs_num_decode}) | "
            f"tokens={scheduled_batch.total_tokens_num} "
            f"(prefill={scheduled_batch.total_tokens_num_prefill} "
            f"decode={scheduled_batch.total_tokens_num_decode})"
        )
        finished_seqs = self.scheduler.postprocess(
            seqs.values(), fwd_out, stream_output_queue=self.stream_output_queue
        )
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                self.output_queue.put_nowait(("STREAM", stream_outputs))
        except queue.Empty:
            pass
        if finished_seqs:
            self.output_queue.put_nowait(finished_seqs)
        return True

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        engine = None
        try:
            engine = DecodeEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception as e:
            logger.error(f"DecodeEngineCore.run_engine: exception: {e}", exc_info=True)
            raise
        finally:
            if engine is not None:
                engine.exit()

    def exit(self):
        super().exit()
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            pass
