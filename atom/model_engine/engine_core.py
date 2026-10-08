# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os
import pickle
import queue
import threading
import time
import uuid
from collections import deque
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace

import torch
import zmq

from atom.config import Config
from atom.kv_transfer.disaggregation import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import connector_metadata_has_work
from atom.metrics.scheduler import SchedulerMetrics
from atom.model_engine.async_proc import AsyncIOProcManager
from atom.model_engine.dynamic_chunking import (
    DYNAMIC_CHUNKING_POLL_STEPS,
    ChunkSizePredictor,
)
from atom.model_engine.engine_utility import EngineUtilityHandler
from atom.model_engine.ipc.buffer_types import (
    BufferDescriptor,
    HostIdentity,
    PromptHandle,
)
from atom.model_engine.ipc.control_codec import ControlCodec
from atom.model_engine.ipc.dto import AdmissionRequest, CompletionResult
from atom.model_engine.ipc.frame_utils import Fields, ProtocolError
from atom.model_engine.ipc.prompt_arena import PromptArenaReader
from atom.model_engine.ipc.request_codec import RequestCodec
from atom.model_engine.ipc.result_codec import ResultCodec
from atom.model_engine.scheduler import DecodeScheduler, PrefillScheduler, Scheduler
from atom.model_engine.sequence import (
    Sequence,
    SequenceStatus,
    new_block_table,
)
from atom.model_engine.state_runtime import StateRuntime
from atom.observability import tracing
from atom.utils import (
    engine_process_name,
    envs,
    init_exit_handler,
    make_zmq_socket,
    set_process_title,
)
from atom.utils.distributed.utils import (
    stateless_destroy_torch_distributed_process_group,
)
from atom.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
    tune_gc,
    unfreeze_gc_heap,
)

logger = logging.getLogger("atom")

# Pace of the idle KV drain. The busy loops never block, so an unpaced drain
# would fire one worker RPC round per spin; 1ms matches the PP head's existing
# idle token-poll timeout and is far below any transfer latency.
KV_IDLE_DRAIN_INTERVAL_S = 0.001

# Upper bound on the drain that runs after the loop exits. A peer that died
# mid-transfer leaves a completion that never arrives; exiting late beats
# never exiting.
KV_SHUTDOWN_DRAIN_TIMEOUT_S = 2.0


class _BoundedOutputQueue:
    def __init__(self, config, on_overflow):
        self._queue = queue.Queue(
            maxsize=config.ipc_max_pending_jobs * config.ipc_max_batch_size
        )
        self._limit = config.ipc_max_pending_bytes
        self._header_bytes = config.ipc_max_header_bytes
        self._bytes = 0
        self._lock = threading.Lock()
        self._on_overflow = on_overflow

    def put_nowait(self, item):
        size = self._header_bytes
        payload = item[1] if isinstance(item, tuple) else item
        if isinstance(item, tuple) and item[0] == "BUFFER_RELEASE":
            size = 512 + sum(
                len(value) for value in payload.values() if isinstance(value, str)
            )
        if isinstance(payload, (list, tuple)):
            for value in payload:
                value = value[1] if isinstance(value, tuple) else value
                tokens = getattr(value, "output_tokens", ())
                size += len(tokens) * 4 + len(getattr(value, "logprobs", ())) * 8
        with self._lock:
            overflow = self._bytes + size > self._limit
            if not overflow:
                self._bytes += size
        if not overflow:
            try:
                self._queue.put_nowait((item, size))
                return
            except queue.Full:
                with self._lock:
                    self._bytes -= size
        self._on_overflow("result queue capacity exceeded")

    def get(self, block=True, timeout=None):
        item, size = self._queue.get(block=block, timeout=timeout)
        with self._lock:
            self._bytes -= size
        return item

    def get_nowait(self):
        return self.get(block=False)

    def empty(self):
        return self._queue.empty()


class EngineCore:
    # This process's name, for the title and every GC log line. A class
    # attribute because it is per-process state and each engine is spawned into
    # its own interpreter; `_setup_engine_process` is the only writer.
    _process_name = "EngineCore"

    def __init__(self, config: Config, input_address: str, output_address: str):
        self.label = "Engine Core"
        self._engine_config = config
        self._ipc_engine_id = uuid.uuid4().hex
        self._ipc_session_id = None
        self._ipc_rank = config.parallel_config.data_parallel_rank
        self._ipc_reader = None
        self._ipc_lock = threading.RLock()
        self._ipc_stop = threading.Event()
        self._ipc_failure_reason = None
        self._ipc_cancelled = {}
        self._ipc_pending = set()
        self._ipc_live = set()
        self._ipc_pending_bytes = 0
        self._ipc_finished = set()
        self._ipc_finished_order = deque()
        self._ipc_registry_limit = (
            config.ipc_max_pending_jobs * config.ipc_max_batch_size
        )
        self.input_queue = queue.Queue(maxsize=config.ipc_max_pending_jobs)
        self.output_queue = _BoundedOutputQueue(config, self._fail_ipc)
        self.stream_output_queue = (
            queue.Queue()
        )  # Queue for streaming intermediate outputs
        # Queue for utility commands (processed in busy_loop to avoid thread contention)
        self.utility_queue = queue.Queue(maxsize=self._ipc_registry_limit)
        self._has_pending_utility = (
            False  # Flag to avoid checking empty queue every loop
        )
        self._is_rl_weights_offloaded = (
            False  # True when weights are offloaded for RL training
        )
        # Separate from memory residency: an update in progress, or one that
        # failed after an in-place write, must not serve.
        self._weight_sync_fenced = False
        self._rl_weights_inconsistent = False
        self.input_address = input_address
        self.output_address = output_address
        # Control traffic arrives on its own socket so CoreManager can keep the
        # request socket single-writer; see CoreManager._send_request.
        self.control_address = config.parallel_config.control_address
        assert self.control_address, (
            "parallel_config.control_address is unset -- an EngineCore must be "
            "launched through CoreManager, which allocates the control channel"
        )
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
            target=self.process_input_sockets,
            args=(self.input_address, self.control_address),
            daemon=True,
        )
        self.input_thread.start()

        self.mark_trace = getattr(config, "mark_trace", False)
        init_exit_handler(self)
        self._init_data_parallel(config)

        # Initialize model runner processes
        try:
            good = False
            # Number of worker processes for THIS EngineCore = one pipeline
            # stage's slice = tp x pcp. Pipeline parallelism spans *separate*
            # EngineCores (one per stage, spawned by CoreManager), not extra
            # workers inside a single EngineCore — so pp does NOT multiply here.
            # tp_world_size, not tensor_parallel_size: under simulated TP only
            # the first tp_world_size shards get a process.
            self.runner_mgr = AsyncIOProcManager(
                self._finalizer,
                config.tp_world_size * config.prefill_context_parallel_size,
                config.runner_qualname,
                config,
            )
            self._post_model_load_hook()
            block_info = self.runner_mgr.call_func("get_num_blocks", wait_out=True)
            num_blocks = block_info["num_kvcache_blocks"]
            # Sizing happens in the runner subprocess, so nothing it wrote to
            # its own `config` is visible here. Carry the per-class entry table
            # across; BlockManager (built in Scheduler below) and the
            # sliding-window pool each look up the class they declared, so
            # adding an architecture never touches this line.
            config.pool_entries = block_info.get("pool_entries", {})
            config.pool_entries_per_req = block_info.get("pool_entries_per_req", {})
            self.state_runtime = StateRuntime.from_wire(block_info["state_runtime"])
            ret = self.runner_mgr.call_func(
                "allocate_kv_cache", num_blocks, wait_out=True
            )
            assert ret, "Failed to allocate kv cache"

            config.num_kvcache_blocks = num_blocks
            # Config has already turned the feature off wherever it cannot act,
            # so every stage of a pipeline takes this branch or none does: the
            # sweep's dummy forwards cross stages and must run in lockstep.
            if config.enable_dynamic_chunking:
                # Startup profiling only measures the chunk cost that carries no
                # attention; the attention terms are calibrated from real
                # prefills while serving, so no chunk is rebalanced until then.
                profile = self.runner_mgr.call_func(
                    "profile_dynamic_chunking", wait_out=True
                )
                if not profile or "linear_coeff" not in profile:
                    config.enable_dynamic_chunking = False
                    reason = (profile or {}).get("error", "")
                    logger.warning(
                        "%s: disabling dynamic chunking because startup profiling "
                        "did not produce a chunk overhead baseline%s",
                        self.label,
                        f": {reason}" if reason else "",
                    )
                if (
                    config.enable_dynamic_chunking
                    and config.parallel_config.pipeline_parallel_rank != 0
                ):
                    # Downstream stages execute metadata from the PP head and
                    # never make chunking decisions themselves.
                    config.enable_dynamic_chunking = False
            if not config.enforce_eager and not config.disagg_is_decode:
                cap_cost, decode_sizes, pool_bytes = self.runner_mgr.call_func(
                    "capture_cudagraph", wait_out=True
                )
                logger.info(
                    f"{self.label}: cudagraph capture{decode_sizes} cost: {cap_cost:.2f} "
                    f"seconds, pool: {pool_bytes / (1 << 30):.2f}GB"
                )
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
            self.scheduler = Scheduler(
                config,
                state_runtime=self.state_runtime,
                prefill_token_graphs=self.runner_mgr.call_func(
                    "prefill_token_graphs", wait_out=True
                ),
            )
            if (
                config.parallel_config.data_parallel_size == 1
                and config.pipeline_parallel_size == 1
                and envs.ATOM_PREFILL_DECODE_INTERVAL > 0
            ):
                self._init_prefill_delayer(config)

        self._dynamic_chunking_enabled = config.enable_dynamic_chunking
        self._dynamic_chunking_poll_countdown = DYNAMIC_CHUNKING_POLL_STEPS

        self.kv_transfer_enabled = bool(config.kv_transfer_config)
        self._next_idle_kv_drain = 0.0
        if self.kv_transfer_enabled:
            # Physical: one output per launched worker, else this waits forever.
            self.kv_aggregator = KVOutputAggregator(world_size=config.tp_world_size)

        self.utility_handler = EngineUtilityHandler(
            self.runner_mgr,
            self.output_queue,
            label=self.label,
            scheduler=self.scheduler,
        )

        # KV cache allocated, graphs captured, BlockPool built: everything this
        # process holds for its lifetime exists, and the next thing is traffic.
        tracing.initialize()
        self._freeze_after_startup()

        self._send_ready_signal()
        logger.info(f"{self.label}: EngineCore fully initialized and ready")

    def _init_prefill_delayer(self, config: Config, cpu_group=None):
        if (
            not envs.ATOM_ENABLE_PREFILL_DELAYER
            or config.enable_rapidserve
            or self.scheduler is None
        ):
            return
        from atom.model_engine.prefill_delayer import PrefillDelayer

        self.scheduler.set_prefill_delayer(
            PrefillDelayer(
                dp_size=config.parallel_config.data_parallel_size,
                cpu_group=cpu_group,
                max_num_batched_tokens=config.max_num_batched_tokens,
                target_fill=envs.ATOM_PREFILL_DELAYER_TARGET_FILL,
                ttft_max_ticks=envs.ATOM_PREFILL_DELAYER_TTFT_MAX_TICKS,
                partial_max_ticks=envs.ATOM_PREFILL_DELAYER_PARTIAL_MAX_TICKS,
                stall_ticks=envs.ATOM_PREFILL_DELAYER_STALL_TICKS,
                kv_high_watermark=envs.ATOM_PREFILL_DELAYER_KV_HIGH_WATERMARK,
                token_usage_low_watermark=envs.ATOM_PREFILL_DELAYER_TOKEN_USAGE_LOW_WATERMARK,
                max_queue_ms=envs.ATOM_PREFILL_DELAYER_MAX_QUEUE_MS,
                prefill_decode_interval=envs.ATOM_PREFILL_DECODE_INTERVAL,
                exchange_prefill_costs=self.scheduler.cost_balance,
                cost_gate=envs.ATOM_PREFILL_DELAYER_COST_GATE,
            )
        )

    def _freeze_after_startup(self):
        """Freeze this process and its ModelRunner workers.

        The workers are driven from here because only the caller knows warmup
        is over -- weights, compile and capture all land via RPCs it sends, and
        `--enforce-eager` skips the capture step entirely.
        """
        freeze_gc_heap(self._process_name)
        try:
            self.runner_mgr.call_func("freeze_gc_heap", wait_out=True)
        except Exception as e:  # noqa: BLE001 - never fail startup over this
            logger.warning(f"{self._process_name}: worker heap freeze skipped: {e}")

    @staticmethod
    def _setup_engine_process(name: str):
        """Identity, orphan reaping and GC policy, for every `run_engine`.

        The disaggregated overrides do not call the base one, and each omission
        is silent: an unreaped orphan pins VRAM, an untitled process is
        `python` in `ps`, an unattached callback leaves ATOM_GC_DEBUG inert.
        """
        from atom.utils import enable_orphan_reaping

        EngineCore._process_name = name
        set_process_title(name)
        enable_orphan_reaping()  # orphans pin VRAM + IPC handles; see its docs
        tune_gc()
        maybe_attach_gc_debug_callback(name)

    def _send_ready_signal(self):
        self.output_queue.put_nowait(("READY", self._ready_payload()))

    def _ready_payload(self) -> dict:
        """Startup facts the frontend cannot read off its own Config.

        `num_kvcache_blocks` is measured in this subprocess, so the API server's
        Config still holds the placeholder. Publishing the derived prompt
        ceiling rather than the raw block count leaves the dcp arithmetic with
        its owner, `BlockManager`, and gives the frontend a single number to
        compare a prompt against.
        """
        payload = {
            "rank": self._ipc_rank,
            "role": getattr(self.scheduler, "_METRICS_ROLE", ""),
            "host_id": HostIdentity.current(),
            "buffer_modes": ["raw", "shm"],
        }
        if self.scheduler is not None:
            blocks = getattr(self.scheduler, "block_manager", None)
            if blocks is not None:
                payload["max_pool_tokens"] = blocks.max_pool_tokens
        return payload

    def _post_model_load_hook(self):
        """Called after ModelRunner is initialized (model loaded) but before
        get_num_blocks/allocate_kv_cache.  Override in subclasses to inject
        inter-process synchronization at this point in the init sequence."""

    def _init_data_parallel(self, config: Config):
        pass

    def exit(self):
        if not self.still_running:
            return
        self.still_running = False
        # Frozen weights and KV cache are unreachable *and* uncollectable, so
        # an engine destroyed in-process would read as a GPU memory leak.
        unfreeze_gc_heap()
        if not hasattr(self, "runner_mgr"):
            self._send_engine_dead()
            return
        self.runner_mgr.keep_monitoring = False
        try:
            self.runner_mgr.call_func("exit")
        except Exception:
            logger.debug(
                "runner exit call failed; shared memory may already be freed",
                exc_info=True,
            )
        # One deadline for all workers, so a long ATOM_SHUTDOWN_TIMEOUT_S is not
        # paid once per worker.
        timeout = envs.ATOM_SHUTDOWN_TIMEOUT_S
        deadline = time.monotonic() + timeout
        for proc in self.runner_mgr.procs:
            try:
                alive = proc.is_alive()
            except ValueError:
                continue  # process object already closed by CoreManager
            if alive:
                proc.join(timeout=max(deadline - time.monotonic(), 0))
                # The join above has a timeout; nothing after it did. A worker
                # that outlives it keeps its VRAM slice and its all-reduce IPC
                # handles, and `multiprocessing`'s atexit handler then joins the
                # same process again with NO timeout -- so an engine that is
                # already on its way out hangs there forever. Seen twice on the
                # k3-dev line: the MainThread parks in `_exit_function -> join`,
                # all TP workers stay alive, /metrics keeps answering 200, and
                # only a manual kill recovers the node. `enable_orphan_reaping`
                # cannot help, because PR_SET_PDEATHSIG fires when the parent
                # *dies* and this parent never does.
                try:
                    if proc.is_alive():
                        logger.warning(
                            "%s: worker pid=%s still alive after %gs "
                            "(ATOM_SHUTDOWN_TIMEOUT_S); terminating",
                            self.label,
                            getattr(proc, "pid", "?"),
                            timeout,
                        )
                        proc.terminate()
                        proc.join(timeout=5)
                    if proc.is_alive():
                        logger.error(
                            "%s: worker pid=%s ignored SIGTERM; killing",
                            self.label,
                            getattr(proc, "pid", "?"),
                        )
                        proc.kill()
                        proc.join(timeout=5)
                except (ValueError, OSError):
                    pass  # process object already closed / already reaped
        self._send_engine_dead()
        logger.debug(f"{self.label}: model runner exit")

    def _send_engine_dead(self):
        logger.debug(f"{self.label}: send SHUTDOWN request")
        self.output_queue.put_nowait(("STOPPED", {}))
        self.output_thread.join(timeout=0.5)
        if self._ipc_reader is not None:
            self._ipc_reader.close()

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        EngineCore._setup_engine_process(engine_process_name(config))

        engine: EngineCore = None
        try:
            if config.pipeline_parallel_size > 1:
                from atom.model_engine.pp_engine_core import PPEngineCoreProc

                engine = PPEngineCoreProc(config, input_address, output_address)
            elif config.parallel_config.data_parallel_size > 1:
                engine = DPEngineCoreProc(config, input_address, output_address)
            else:
                engine = EngineCore(config, input_address, output_address)
            engine.busy_loop()
        except Exception:
            logger.exception("run_engine failed")
            raise
        finally:
            if engine is not None:
                engine.exit()

    def _is_idle_rl_weights_offloaded(self) -> bool:
        """Check if weights are offloaded for RL training.

        When offloaded, busy-wait with a short delay to avoid CPU spin.
        Returns True if the caller should skip model execution this tick.
        """
        if self._is_rl_weights_offloaded or self._weight_sync_fenced:
            time.sleep(0.01)
            return True
        return False

    def busy_loop(self):
        shutdown = False
        metrics_interval = envs.ATOM_METRICS_UPDATE_INTERVAL_S
        next_metrics_push = 0.0
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                now = time.monotonic()
                if now >= next_metrics_push:
                    next_metrics_push = now + metrics_interval
                    self.utility_handler.push_metrics()
                self.scheduler.heartbeat_throughput(now)
                shutdown = shutdown or self.pull_and_process_input_queue()
                if shutdown:
                    break
                if self._is_idle_rl_weights_offloaded():
                    continue
                if not self.scheduler.is_finished():
                    self._process_engine_step()
                elif self.has_pending_kv_work():
                    self._advance_idle_kv_transfer()
        finally:
            try:
                # Teardown runs even on exceptions so the sender thread/socket
                # don't leak. Isolate the final publish so a publisher hiccup
                # cannot skip shutdown_kv_events().
                self._drain_kv_work_at_exit()
                try:
                    self.scheduler.publish_kv_events()
                except Exception:
                    logger.exception("KV event publish during shutdown failed")
                self.scheduler.shutdown_kv_events()
            finally:
                tracing.shutdown()

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

    def _release_multimodal_requests(self, sequences):
        request_ids = [
            seq.id for seq in sequences if getattr(seq, "cache_seed", -1) != -1
        ]
        if request_ids:
            # Ordered after the completed forward on every worker, including
            # final/aborted requests when there will be no subsequent batch.
            self.runner_mgr.call_func("release_multimodal_requests", request_ids)

    def _process_engine_step_inner(self):
        result = self.scheduler.schedule()

        # Surface admit-rejected seqs (those `_unschedulable_reason` flags in
        # the scheduler) through the same finished-seq path as normal seqs.
        # Without this, `llm.generate()` blocks forever waiting for an output
        # the rejected seq will never produce.
        rejected = self.scheduler.take_rejected()
        if rejected:
            self._release_multimodal_requests(rejected)
            self._emit_completions(rejected)

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
            self.scheduler.compute_detailed_aggregates(scheduled_batch, seqs)
            tracing.record_forward(scheduled_batch, seqs)
            self.scheduler.metrics.record_forward(scheduled_batch, seqs)
            fwd_out = self.runner_mgr.call_func(
                "forward", scheduled_batch, wait_out=True
            )
            if (
                self.scheduler.prefill_delayer is not None
                and scheduled_batch.total_seqs_num_prefill > 0
            ):
                # Arm post-prefill decode protection only after the prefill
                # forward really completed. A delayer FIRE merely grants
                # admission and can still result in a decode/empty batch.
                self.scheduler.prefill_delayer.notify_prefill_executed()

        # Aggregate KV transfer status from all workers (only when PD disaggregation is active)
        self._poll_kv_transfer_progress()

        if not has_seqs:
            logger.debug("%s: Empty scheduled batch, skipping postprocess", self.label)
            return False

        self._postprocess(seqs.values(), fwd_out, scheduled_batch)
        return True

    def _postprocess(self, seqs, fwd_out, batch):
        # Pass stream_output_queue to postprocess for streaming callbacks
        finished_seqs = self.scheduler.postprocess(
            seqs,
            fwd_out,
            stream_output_queue=self.stream_output_queue,
            batch=batch,
        )

        # Send stream outputs to main process via output_queue
        try:
            while not self.stream_output_queue.empty():
                stream_outputs = self.stream_output_queue.get_nowait()
                # Send stream outputs as intermediate results
                self._emit_streams(stream_outputs)
        except queue.Empty:
            pass

        if finished_seqs:
            self._release_multimodal_requests(finished_seqs)
            self._emit_completions(finished_seqs)

    def has_pending_kv_work(self) -> bool:
        """True while KV transfer work outlives the scheduler queues.

        ``postprocess`` parks a finished request in ``deferred_free_blocks``
        and drops it from ``running`` in the same pass, so
        ``Scheduler.is_finished()`` reads "idle" while that request's RDMA
        send or offload save is still in flight. Every busy loop ORs this
        predicate in next to ``is_finished()``; without it the last request's
        completion signals are never polled, its deferred blocks are never
        freed, and its save is never reported.

        Every liveness condition lives here. The loops call this and nothing
        else, so a new kind of pending work only has to be added once.
        """
        if not self.kv_transfer_enabled:
            return False
        if getattr(self.scheduler, "deferred_free_blocks", None):
            return True
        # Unit-store twin of `deferred_free_blocks`: a state store handed to the
        # worker keeps its PAGE units pinned out of the KV pool until its report
        # settles them or the reclaim ages them out. Both exits run only from
        # `_poll_kv_transfer_progress`, so the loop must keep polling while a
        # pin is outstanding or the units leak until the abandon window. Guarded
        # for the scheduler doubles that implement only the connector surface.
        bm = getattr(self.scheduler, "block_manager", None)
        pending_pins = getattr(bm, "has_pending_state_store_pins", None)
        if pending_pins is not None and pending_pins():
            return True
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not hasattr(connector, "has_pending_work"):
            return False
        return bool(connector.has_pending_work())

    def _advance_idle_kv_transfer(self, dispatch_new: bool = True) -> None:
        # No forward batch will run this tick, but offload load/save work may
        # still need to be dispatched or reported back to the scheduler.
        # `dispatch_new=False` on the shutdown drain: report in-flight work
        # back, but start no new transfers (see `_dispatch_idle_offload_work`).
        now = time.monotonic()
        if now < self._next_idle_kv_drain:
            return
        self._next_idle_kv_drain = now + KV_IDLE_DRAIN_INTERVAL_S
        self._dispatch_idle_offload_work(dispatch_new=dispatch_new)
        self._poll_kv_transfer_progress()

    def _drain_kv_work_at_exit(self) -> None:
        """Give in-flight KV transfers a bounded window to report back.

        The loop exits as soon as its queues are empty, so a save dispatched
        by the final batch would otherwise be abandoned with its completion
        unrecorded and its blocks still deferred.

        Drains with `dispatch_new=False`: it must let already-dispatched
        transfers finish and report, but must not publish new state loads/stores.
        A fresh store dispatched here spills bytes nothing will read back, and --
        worse -- keeps `has_pending_kv_work()` True, so this very loop could
        manufacture its own work and never converge before the deadline.
        """
        if not self.kv_transfer_enabled:
            return
        deadline = time.monotonic() + KV_SHUTDOWN_DRAIN_TIMEOUT_S
        try:
            while self.has_pending_kv_work():
                if time.monotonic() >= deadline:
                    logger.warning(
                        "%s: KV transfer still pending after %.1fs, exiting anyway",
                        self.label,
                        KV_SHUTDOWN_DRAIN_TIMEOUT_S,
                    )
                    break
                self._advance_idle_kv_transfer(dispatch_new=False)
                time.sleep(KV_IDLE_DRAIN_INTERVAL_S)
        except Exception:
            logger.exception("KV transfer drain during shutdown failed")

    def _poll_kv_transfer_progress(self) -> None:
        if not self.kv_transfer_enabled:
            return
        kvoutput = self.runner_mgr.call_func_with_aggregation("async_proc_aggregation")
        self.scheduler._update_from_kv_xfer_finished(kvoutput)
        # Reclaim any offload save whose completion report never came (LMCache
        # force-unpinned it upstream). Self-throttled, so calling it on every
        # poll is cheap; without it a stalled save hangs the engine forever.
        reconcile = getattr(self.scheduler, "_reconcile_stalled_deferred_saves", None)
        if callable(reconcile):
            reconcile()

    def _poll_dynamic_chunking_calibration(self) -> None:
        """Install a freshly calibrated chunk latency model, if one is ready.

        The workers time their own prefills, so this is one RPC round trip every
        `DYNAMIC_CHUNKING_POLL_STEPS` steps and nothing on the forward path.

        Called from the PP head's step loop rather than from here: dynamic
        chunking needs more than one stage, and every such deployment runs
        `PPEngineCoreProc`, whose head loop replaces `_process_engine_step`.
        Every other stage had the feature turned off during startup because it
        makes no chunking decisions.
        """
        if not self._dynamic_chunking_enabled:
            return
        self._dynamic_chunking_poll_countdown -= 1
        if self._dynamic_chunking_poll_countdown > 0:
            return
        self._dynamic_chunking_poll_countdown = DYNAMIC_CHUNKING_POLL_STEPS
        fit = self.runner_mgr.call_func("take_dynamic_chunking_fit", wait_out=True)
        coefficients = fit.get("coefficients") if fit else None
        if coefficients is None:
            if fit and fit.get("gave_up"):
                # The workers have stopped timing, so end the sweep too rather
                # than size every other request for a model that never lands.
                self._dynamic_chunking_enabled = False
                self.scheduler.abandon_chunk_latency_calibration()
            return
        # The workers stop timing prefills once they have answered with a fit, so
        # there is never a second one to collect - whether the scheduler takes
        # this model or rejects it as not worth acting on.
        self._dynamic_chunking_enabled = False
        try:
            predictor = ChunkSizePredictor.from_coefficients(coefficients)
        except (TypeError, ValueError):
            logger.exception(
                "%s: rejecting dynamic chunking coefficients %r",
                self.label,
                coefficients,
            )
            self.scheduler.abandon_chunk_latency_calibration()
            return
        self.scheduler.install_chunk_latency_model(predictor)

    def _dispatch_idle_offload_work(self, dispatch_new: bool = True) -> None:
        if not self.kv_transfer_enabled:
            return
        connector = getattr(self.scheduler, "kv_connector", None)
        if connector is None or not getattr(connector, "is_offload", False):
            return
        # getattr for the same reason as `kv_connector` above: this path is
        # reached with scheduler doubles that implement only the connector
        # surface.
        #
        # `dispatch_new=False` on the shutdown drain: publishing new state
        # loads/stores there hands the connector work whose bytes nothing will
        # read back, and -- worse -- each fresh store keeps `has_pending_kv_work`
        # True, so the drain loop that waits on it manufactures its own exit
        # condition and never converges. Still build/process the meta so
        # already-dispatched transfers finish and report.
        if dispatch_new:
            for name in ("_publish_state_loads", "_publish_state_stores"):
                publish = getattr(self.scheduler, name, None)
                if publish is not None:
                    publish()
        meta = connector.build_connector_meta()
        if not connector_metadata_has_work(meta):
            return
        self.runner_mgr.call_func("process_kvconnector_output", meta)

    def _remember_finished(self, request_id):
        if not hasattr(self, "_ipc_lock"):
            return
        with self._ipc_lock:
            self._ipc_cancelled.pop(request_id, None)
            self._ipc_live.discard(request_id)
            if request_id not in self._ipc_finished:
                self._ipc_finished.add(request_id)
                self._ipc_finished_order.append(request_id)
            while len(self._ipc_finished_order) > self._ipc_registry_limit:
                self._ipc_finished.discard(self._ipc_finished_order.popleft())

    def _fail_ipc(self, reason):
        self._ipc_failure_reason = reason
        self._ipc_stop.set()

    def _emit_completions(self, sequences):
        outputs = []
        for sequence in sequences:
            output = (
                sequence
                if isinstance(sequence, CompletionResult)
                else CompletionResult.from_sequence(sequence)
            )
            outputs.append(output)
            self._remember_finished(output.id)
        if outputs:
            self.output_queue.put_nowait(("COMPLETION", outputs))

    def _emit_streams(self, outputs):
        snapshots = [
            (
                request_id,
                replace(
                    output,
                    output_tokens=list(output.output_tokens),
                    kv_transfer_params_output=deepcopy(
                        output.kv_transfer_params_output
                    ),
                ),
            )
            for request_id, output in outputs
        ]
        if snapshots:
            self.output_queue.put_nowait(("STREAM", snapshots))

    def _take_admission(self, admission):
        if not isinstance(admission, AdmissionRequest):
            return admission
        with self._ipc_lock:
            cancelled = self._ipc_cancelled.pop(admission.id, None)
            self._ipc_pending.discard(admission.id)
            self._ipc_pending_bytes -= admission.prompt.nbytes
        try:
            if cancelled is not None:
                self._emit_completions(
                    [
                        CompletionResult(
                            id=admission.id,
                            num_prompt_tokens=admission.num_prompt_tokens,
                            arrive_time=admission.arrive_time,
                            leave_time=time.time(),
                            leave_reason="abort",
                        )
                    ]
                )
                return None
            sequence = Sequence.from_admission(admission, self._engine_config)
            tracing.enqueue(sequence, received_at=admission.received_at)
            SchedulerMetrics.enqueue(sequence, received_at=admission.received_at)
            return sequence
        except Exception as exc:
            logger.exception("%s: request %s was rejected", self.label, admission.id)
            self._remember_finished(admission.id)
            self.output_queue.put_nowait(
                (
                    "ERROR",
                    {
                        "code": "admission",
                        "message": str(exc),
                        "request_id": admission.id,
                    },
                )
            )
            return None
        finally:
            admission.prompt.release()

    def pull_and_process_input_queue(self):
        if getattr(self, "_ipc_stop", None) is not None and self._ipc_stop.is_set():
            return True
        recv_reqs = []
        shutdown = False
        while True:
            try:
                admissions = self.input_queue.get_nowait()
            except queue.Empty:
                break
            for admission in admissions:
                sequence = self._take_admission(admission)
                if sequence is None:
                    continue
                if sequence.status == SequenceStatus.EXIT_ENGINE:
                    shutdown = True
                else:
                    recv_reqs.append(sequence)
        if recv_reqs:
            self.scheduler.extend(recv_reqs)
        return shutdown

    def _release_prompt_consumer(self, descriptor, key):
        self.output_queue.put_nowait(
            (
                "BUFFER_RELEASE",
                {
                    "arena_id": descriptor.arena_id,
                    "arena_epoch": descriptor.epoch,
                    "allocation_id": descriptor.allocation_id,
                    "allocation_generation": descriptor.generation,
                    "admission_id": str(key[2]),
                    "request_id": key[2],
                    "rank": self._ipc_rank,
                    "reason": "last_reader",
                },
            )
        )

    def _handle_buffer_probe(self, message):
        session = message.metadata["session_id"]
        payload = message.payload
        identity = payload["probe_id"].rsplit(":", 1)
        if not session or len(identity) != 2 or identity[0] != session:
            raise ProtocolError("probe does not identify the manager session")
        try:
            rank = int(identity[1])
        except ValueError as exc:
            raise ProtocolError("probe rank is invalid") from exc
        if rank < 0:
            raise ProtocolError("probe rank is negative")
        self._ipc_session_id = session
        self._ipc_rank = rank
        result = {"probe_id": payload["probe_id"], "accessible": False, "error": ""}
        descriptor = payload.get("buffer")
        registration = payload.get("registration")
        if descriptor is not None:
            try:
                if registration is None or os.path.realpath(
                    registration["root_dir"]
                ) != os.path.realpath(self._engine_config.ipc_shm_dir):
                    raise ValueError(
                        "probe root is not registered in Core configuration"
                    )
                if self._ipc_reader is None:
                    self._ipc_reader = PromptArenaReader(
                        registration, on_release=self._release_prompt_consumer
                    )
                result["digest"] = self._ipc_reader.probe(descriptor)
                result["arena_epoch"] = descriptor.epoch
                result["accessible"] = True
            except (OSError, ValueError, BufferError) as exc:
                result["error"] = str(exc)
        self.output_queue.put_nowait(
            (
                "BUFFER_PROBE_RESULT",
                result,
                {"in_reply_to": message.metadata["message_id"]},
            )
        )

    def _queue_admissions(self, message, received_at):
        admissions = []
        accepted = False
        try:
            for values in message.payload:
                prompt = values["prompt"]
                if isinstance(prompt, BufferDescriptor):
                    if self._ipc_reader is None:
                        raise ProtocolError("shared prompt was not verified by a probe")
                    key = (
                        self._ipc_session_id,
                        self._ipc_engine_id,
                        values["id"],
                        "request",
                    )
                    prompt = self._ipc_reader.acquire(prompt, key)
                    values["prompt"] = prompt
                if not isinstance(prompt, PromptHandle):
                    raise ProtocolError("admission has no prompt owner")
                admission = AdmissionRequest(**values, received_at=received_at)
                admissions.append(admission)
            nbytes = sum(admission.prompt.nbytes for admission in admissions)
            with self._ipc_lock:
                ids = {admission.id for admission in admissions}
                if ids & (self._ipc_live | self._ipc_finished):
                    raise ProtocolError("duplicate or completed request ID")
                for admission in admissions:
                    cancelled = self._ipc_cancelled.get(admission.id, 0)
                    if (
                        cancelled
                        and admission.submission_sequence
                        and cancelled != admission.submission_sequence
                    ):
                        raise ProtocolError("cancel does not match admission sequence")
                if (
                    self._ipc_pending_bytes + nbytes
                    <= self._engine_config.ipc_max_pending_bytes
                ):
                    try:
                        self.input_queue.put_nowait(admissions)
                    except queue.Full:
                        pass
                    else:
                        self._ipc_pending.update(ids)
                        self._ipc_live.update(ids)
                        self._ipc_pending_bytes += nbytes
                        accepted = True
            if not accepted:
                self._emit_completions(
                    [
                        CompletionResult(
                            id=admission.id,
                            num_prompt_tokens=admission.num_prompt_tokens,
                            arrive_time=admission.arrive_time,
                            leave_time=time.time(),
                            leave_reason="overloaded",
                            external_request_id=admission.external_request_id,
                            parent_request_id=admission.parent_request_id,
                            sibling_index=admission.sibling_index,
                        )
                        for admission in admissions
                    ]
                )
        finally:
            if not accepted:
                for values in message.payload:
                    prompt = values.get("prompt")
                    if isinstance(prompt, PromptHandle):
                        prompt.release()
            message.owners.clear()

    def _queue_cancellation(self, message, request_ids=None, *, enqueue=True):
        watermark = message.metadata.get("submission_sequence", 0)
        with self._ipc_lock:
            for request_id in message.payload if request_ids is None else request_ids:
                if request_id in self._ipc_finished:
                    continue
                if (
                    len(self._ipc_cancelled) >= self._ipc_registry_limit
                    and request_id not in self._ipc_cancelled
                ):
                    raise ProtocolError("Core cancellation registry is full")
                self._ipc_cancelled[request_id] = watermark
                if enqueue:
                    self.utility_queue.put_nowait(
                        (
                            "abort_request",
                            {"req_id": request_id},
                            {},
                        )
                    )
                    self._has_pending_utility = True

    def process_input_sockets(self, input_address: str, control_address: str):
        """One input thread owns separate request and control decoders."""
        options = {
            "max_header_bytes": self._engine_config.ipc_max_header_bytes,
            "max_batch_size": self._engine_config.ipc_max_batch_size,
            "max_batch_bytes": getattr(
                self._engine_config, "ipc_max_batch_bytes", 512 << 20
            ),
        }
        requests, controls = RequestCodec(**options), ControlCodec(**options)
        with ExitStack() as stack, zmq.Context() as ctx:
            input_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    input_address,
                    zmq.DEALER,
                    bind=False,
                    hwm=self._engine_config.ipc_max_pending_jobs,
                )
            )
            control_socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    control_address,
                    zmq.DEALER,
                    bind=False,
                    hwm=self._engine_config.ipc_max_pending_jobs,
                )
            )
            poller = zmq.Poller()
            for sock in (input_socket, control_socket):
                sock.send(b"")
                poller.register(sock, zmq.POLLIN)
            alive = True
            while alive:
                for sock, _ in poller.poll():
                    frames = sock.recv_multipart(copy=False)
                    received_at = time.perf_counter()
                    try:
                        decoder = requests if sock is input_socket else controls
                        message = decoder.decode(
                            frames,
                            session_id=self._ipc_session_id,
                            engine_id=self._ipc_engine_id,
                        )
                        kind = Fields.kind(message.kind)
                        if self._ipc_session_id is None and kind != "BUFFER_PROBE":
                            raise ProtocolError(
                                "manager session has not completed its probe"
                            )
                        if kind == "ADD":
                            self._queue_admissions(message, received_at)
                        elif kind == "BUFFER_PROBE":
                            self._handle_buffer_probe(message)
                        elif kind == "UTILITY":
                            command, args = message.payload
                            if command == "abort_request":
                                self._queue_cancellation(
                                    message, [args["req_id"]], enqueue=False
                                )
                            self.utility_queue.put_nowait(
                                (command, args, message.metadata)
                            )
                            self._has_pending_utility = True
                        elif kind == "ABORT":
                            self._queue_cancellation(message)
                        elif kind == "SHUTDOWN":
                            self._ipc_stop.set()
                            alive = False
                        else:
                            raise ProtocolError(f"unsupported input kind {kind}")
                    except Exception as exc:
                        logger.exception("%s: IPC peer failed", self.label)
                        self._fail_ipc(str(exc))
                        alive = False
                    finally:
                        frames = None
                        message = None
                    if not alive:
                        break

    def process_output_sockets(self, output_address: str):
        """Serialize snapshots and lease events on the existing result socket."""
        config = self._engine_config
        codec = ResultCodec(
            max_header_bytes=config.ipc_max_header_bytes,
            max_batch_size=config.ipc_max_batch_size,
            max_batch_bytes=getattr(config, "ipc_max_batch_bytes", 512 << 20),
        )
        with ExitStack() as stack, zmq.Context() as ctx:
            socket = stack.enter_context(
                make_zmq_socket(
                    ctx,
                    output_address,
                    zmq.PUSH,
                    linger=4000,
                    hwm=config.ipc_max_pending_jobs,
                )
            )
            message_id = 0
            while True:
                failure = getattr(self, "_ipc_failure_reason", None)
                if failure is not None:
                    item = ("ERROR", {"code": "protocol", "message": failure})
                else:
                    try:
                        item = self.output_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue
                if isinstance(item, tuple):
                    kind, payload = item[:2]
                    metadata = item[2] if len(item) > 2 else {}
                else:
                    # Internal shutdown sentinels are never serialized as tokens.
                    finished = [
                        seq for seq in item if seq.status != SequenceStatus.EXIT_ENGINE
                    ]
                    if finished:
                        self._emit_completions(finished)
                    if len(finished) == len(item):
                        continue
                    kind, payload, metadata = "STOPPED", {}, {}
                name = Fields.kind(kind)
                batches = (
                    [
                        payload[start : start + config.ipc_max_batch_size]
                        for start in range(0, len(payload), config.ipc_max_batch_size)
                    ]
                    if name in ("STREAM", "COMPLETION", "ADD")
                    else [payload]
                )
                for batch in batches:
                    message_id += 1
                    context = {
                        "session_id": (
                            "" if name == "READY" else self._ipc_session_id or ""
                        ),
                        "engine_id": self._ipc_engine_id,
                        "message_id": message_id,
                        "in_reply_to": metadata.get("in_reply_to", 0),
                    }
                    try:
                        encoded = codec.encode(
                            kind, batch, metadata=metadata, **context
                        )
                    except Exception as exc:
                        logger.exception("%s: result encoding failed", self.label)
                        self._fail_ipc(str(exc))
                        encoded = codec.encode(
                            "ERROR", {"code": "result", "message": str(exc)}, **context
                        )
                    deadline = time.monotonic() + config.ipc_send_timeout
                    while True:
                        try:
                            tracker = socket.send_multipart(
                                encoded.frames,
                                flags=zmq.DONTWAIT,
                                copy=False,
                                track=True,
                            )
                            break
                        except zmq.Again:
                            if time.monotonic() >= deadline:
                                logger.error(
                                    "%s: result socket send timed out", self.label
                                )
                                self._fail_ipc("result socket send timed out")
                                return
                            socket.poll(50, zmq.POLLOUT)
                    if tracker is not None and not tracker.done:
                        try:
                            tracker.wait(timeout=config.ipc_send_timeout)
                        except zmq.NotDone:
                            self._fail_ipc("result buffer send did not complete")
                            return
                if failure is not None or name in ("STOPPED", "SHUTDOWN"):
                    return


class DPEngineCoreProc(EngineCore):
    def __init__(self, config: Config, input_address: str, output_address: str):
        # self.dp_group = config.parallel_config.dp_group
        self.dp_rank = config.parallel_config.data_parallel_rank
        # self.dp_group = config.parallel_config.stateless_init_dp_group()
        super().__init__(config, input_address, output_address)
        # Initialize to True so first iteration reaches all_reduce
        self.engines_running = True
        self._shutting_down = False

        self._init_prefill_delayer(config, self.dp_group)

    def _init_data_parallel(self, config: Config):
        dp_rank = config.parallel_config.data_parallel_rank
        dp_size = config.parallel_config.data_parallel_size
        local_dp_rank = config.parallel_config.data_parallel_rank_local
        local_dp_size = config.parallel_config.data_parallel_size_local

        assert dp_size > 1
        assert local_dp_rank is not None
        # Two independent bounds. The old chained form (local <= global) held
        # only by coincidence on one node: the second node of a 2x4 run has
        # local ranks 0..3 against global ranks 4..7.
        assert 0 <= dp_rank < dp_size, f"dp_rank={dp_rank} outside [0,{dp_size})"
        assert (
            0 <= local_dp_rank < local_dp_size
        ), f"local_dp_rank={local_dp_rank} outside [0,{local_dp_size})"

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
        metrics_interval = envs.ATOM_METRICS_UPDATE_INTERVAL_S
        next_metrics_push = 0.0
        try:
            while True:
                self.utility_handler.process_queue(self.utility_queue, self)
                now = time.monotonic()
                if now >= next_metrics_push:
                    next_metrics_push = now + metrics_interval
                    self.utility_handler.push_metrics()
                self.scheduler.heartbeat_throughput(now)
                shutdown = shutdown or self.pull_and_process_input_queue()
                local_unfinished = not self.scheduler.is_finished() and not (
                    self._is_rl_weights_offloaded or self._weight_sync_fenced
                )

                global_has_unfinished, global_shutdown, global_offloaded = (
                    self._sync_dp_state(
                        local_unfinished,
                        shutdown,
                        self._is_rl_weights_offloaded or self._weight_sync_fenced,
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
                    if self.scheduler.prefill_delayer is not None:
                        self.scheduler.prefill_delayer.notify_idle()
                    if self.has_pending_kv_work():
                        # Local RPCs only. Anything that reaches schedule()
                        # would run the delayer's cross-DP all_reduce off
                        # lockstep, so the idle drain must stay off that path.
                        self._advance_idle_kv_transfer()
                    continue

                executed = self._process_engine_step()
                if not executed:
                    self._execute_dummy_batch()

                self.engines_running = global_has_unfinished
        finally:
            try:
                # Isolate the final publish so a publisher hiccup cannot skip
                # shutdown_kv_events() (which closes the sender thread/socket).
                self._drain_kv_work_at_exit()
                try:
                    self.scheduler.publish_kv_events()
                except Exception:
                    logger.exception("KV event publish during DP shutdown failed")
                self.scheduler.shutdown_kv_events()
            finally:
                tracing.shutdown()

    def _execute_dummy_batch(self):
        prefill = self.scheduler.dummy_prefill
        fwd_out = self.runner_mgr.call_func("dummy_execution", prefill, wait_out=True)
        if prefill and self.scheduler.running:
            # This rank decoded last step and gave this tick to its peers'
            # prefill: the tokens that step sampled come back with this
            # forward (deferred output), and land like any step's.
            self._postprocess((), fwd_out, None)

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

        # ZMQ context for disagg sockets (separate from the main engine sockets).
        self._disagg_ctx = zmq.Context()

        # Store config reference so _send_ready_signal can read num_kvcache_blocks
        # after EngineCore.__init__ sets it.
        self._config = config

        super().__init__(config, input_address, output_address)
        # Replace the base Scheduler created by EngineCore.__init__ with
        # PrefillScheduler, which has no BlockManager and only schedules
        # sequences that already have a block_table from decode. The base
        # Scheduler may have started a KV event publisher with bound sockets;
        # prefill never publishes, so release it instead of leaking a thread
        # and an endpoint for the life of the process.
        if self.scheduler is not None:
            self.scheduler.shutdown_kv_events()
        self.scheduler = PrefillScheduler(
            config, disagg_cu_shm_name=config.disagg_cu_shm_name
        )
        self.utility_handler.scheduler = self.scheduler

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
        # # --- Create pool of CU-masked CUDA streams for prefill ---
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
                    seq.block_table = new_block_table(assignment.block_table)
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
        self.scheduler.metrics.record_forward(scheduled_batch, seqs)
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
            self._remember_finished(seq.id)
            try:
                self.scheduler.running.remove(seq)
            except ValueError:
                pass
        return True

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        EngineCore._setup_engine_process("PrefillEngineCore")
        engine = None
        try:
            engine = PrefillEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception:
            logger.exception("PrefillEngineCore.run_engine failed")
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
            logger.debug("bootstrap socket close failed", exc_info=True)
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            logger.debug("disagg context destroy failed", exc_info=True)


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

        if not config.enforce_eager:
            cap_cost, decode_sizes, _ = self.runner_mgr.call_func(
                "capture_cudagraph", wait_out=True
            )
            logger.info(
                f"DecodeEngineCore: cudagraph capture{decode_sizes} cost: "
                f"{cap_cost:.2f}s"
            )

        # --- Create DecodeScheduler now that num_kvcache_blocks is set ---
        self.scheduler = DecodeScheduler(
            config,
            disagg_cu_shm_name=config.disagg_cu_shm_name,
            state_runtime=self.state_runtime,
        )
        # The block pool exists only now, so the freeze inside
        # `super().__init__()` ran before there was one to take. Safe to repeat:
        # `_ready_deferred` held READY back, so nothing has been admitted yet
        # and `gc.freeze()` is additive.
        self._freeze_after_startup()

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
        if super().pull_and_process_input_queue():
            return True
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
        self.scheduler.metrics.record_forward(scheduled_batch, seqs)
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
                self._emit_streams(stream_outputs)
        except queue.Empty:
            pass
        if finished_seqs:
            self._emit_completions(finished_seqs)
        return True

    @staticmethod
    def run_engine(config: Config, input_address: str, output_address: str):
        EngineCore._setup_engine_process("DecodeEngineCore")
        engine = None
        try:
            engine = DecodeEngineCore(config, input_address, output_address)
            engine._init_disagg()
            engine.busy_loop()
        except Exception:
            logger.exception("DecodeEngineCore.run_engine failed")
            raise
        finally:
            if engine is not None:
                engine.exit()

    def exit(self):
        super().exit()
        try:
            self._disagg_ctx.destroy(linger=0)
        except Exception:
            logger.debug("disagg context destroy failed", exc_info=True)
