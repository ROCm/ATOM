# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import queue
from typing import ClassVar

from atom.model_engine.sequence import SequenceStatus

logger = logging.getLogger("atom")


class EngineUtilityHandler:
    """Centralised handler for all utility commands dispatched by EngineCore.

    Covers weight management, memory lifecycle, profiling, MTP statistics,
    and TorchSpec hidden-state extraction.  Every command is registered in
    ``_UTILITY_HANDLERS`` and executed in the main busy-loop thread so that
    ``runner_mgr.call_func`` calls are serialized.

    Parameters
    ----------
    runner_mgr : AsyncIOProcManager
        The model-runner process manager used to execute ``call_func``.
    output_queue : queue.Queue
        The EngineCore output queue for pushing ``UTILITY_RESPONSE`` messages
        back to ``CoreManager``.
    label : str, optional
        Label used in log messages (default ``"Engine Core"``).
    scheduler : Scheduler, optional
        The scheduler instance, needed by MTP statistics handlers.
    profiler_delay_iters : int, optional
        Engine steps to skip after ``start_profile`` before the profiler
        records (default 0, record immediately).
    profiler_max_iters : int, optional
        Recorded steps after which the profiler stops itself (default 0, run
        until ``stop_profile``).
    """

    # Utility command name  ->  handler method name
    _UTILITY_HANDLERS: ClassVar[dict[str, str]] = {
        "update_weights": "_handle_update_weights",
        "update_weights_shm": "_handle_update_weights_shm",
        "update_weights_ipc": "_handle_update_weights_ipc",
        "release_memory": "_handle_release_memory",
        "resume_memory": "_handle_resume_memory",
        "clear_kv_cache": "_handle_clear_kv_cache",
        "configure_hidden_states": "_handle_configure_hidden_states",
        "start_profile": "_handle_start_profile",
        "stop_profile": "_handle_stop_profile",
        "get_mtp_stats": "_handle_get_mtp_stats",
        "get_mtp_statistics": "_handle_get_mtp_statistics",
        "get_cache_statistics": "_handle_get_cache_statistics",
        "abort_request": "_handle_abort_request",
    }

    def __init__(
        self,
        runner_mgr,
        output_queue,
        label: str = "Engine Core",
        scheduler=None,
        profiler_delay_iters: int = 0,
        profiler_max_iters: int = 0,
    ):
        self.runner_mgr = runner_mgr
        self.output_queue = output_queue
        self.label = label
        self.scheduler = scheduler
        self.profiler_delay_iters = profiler_delay_iters
        self.profiler_max_iters = profiler_max_iters
        self._profiler_pending = 0
        self._profiler_recorded = 0
        self._profiler_active = False

    def process_queue(self, utility_queue, engine):
        """Drain *utility_queue* and execute each command.

        When the queue is empty, ``engine._has_pending_utility`` is set to
        ``False`` so that the next busy-loop iteration can skip the check.

        Sleep/wake state is tracked on *engine._is_rl_weights_offloaded* so that the
        busy-loop can skip model execution while the weights are offloaded.
        """
        if not engine._has_pending_utility:
            return

        while True:
            try:
                cmd, args = utility_queue.get_nowait()
                self._execute_utility_command(cmd, args)
                # Track sleep/wake transitions
                if cmd == "release_memory":
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if "weights" in tags:
                        engine._is_rl_weights_offloaded = True
                        logger.info(f"{self.label}: engine entered sleep mode")
                elif cmd in (
                    "resume_memory",
                    "update_weights_shm",
                    "update_weights_ipc",
                ):
                    tags = args.get("tags", []) if isinstance(args, dict) else []
                    if cmd == "resume_memory" and "weights" in tags:
                        engine._is_rl_weights_offloaded = False
                        logger.info(f"{self.label}: engine exited sleep mode")
                    elif cmd in ("update_weights_shm", "update_weights_ipc"):
                        is_last = (
                            args.get("is_last", True)
                            if isinstance(args, dict)
                            else True
                        )
                        if is_last:
                            engine._is_rl_weights_offloaded = False
                            logger.info(
                                f"{self.label}: engine exited sleep mode (weights updated)"
                            )
            except queue.Empty:
                engine._has_pending_utility = False
                break

    def _execute_utility_command(self, cmd: str, args: dict):
        import time as _time

        log = logger.info
        log(f"{self.label}: executing utility command: {cmd}")
        t0 = _time.monotonic()

        handler_name = self._UTILITY_HANDLERS.get(cmd)
        if handler_name:
            handler = getattr(self, handler_name)
            handler(args)
        else:
            logger.warning(f"{self.label}: Unknown utility command: {cmd}")

        elapsed = _time.monotonic() - t0
        log(f"{self.label}: utility command '{cmd}' finished in {elapsed:.2f}s")

    def _handle_update_weights(self, args: dict):
        """Handle direct weight update command."""
        named_tensors = args.get("named_tensors", [])
        flush_cache = args.get("flush_cache", True)
        result = self.runner_mgr.call_func(
            "update_weights", named_tensors, flush_cache, wait_out=True
        )
        logger.info(f"{self.label}: update_weights completed, updated={result}")

    def _handle_update_weights_shm(self, args: dict):
        """Handle shared-memory weight update command.

        Only lightweight metadata (shm_name, bucket_meta) travels through the
        control path.  The actual tensor data resides in POSIX shared memory and
        is read directly by each ModelRunner process.

        After all ModelRunners finish, a UTILITY_RESPONSE is pushed onto the
        output_queue so that the caller (LLMEngine) can synchronise.

        After completion, a ``UTILITY_RESPONSE`` is pushed so the caller
        (LLMEngine) can synchronise.
        """
        shm_name = args.get("shm_name", "")
        bucket_meta = args.get("bucket_meta", {})
        is_last = args.get("is_last", True)
        result = self.runner_mgr.call_func(
            "update_weights_from_shm", shm_name, bucket_meta, is_last, wait_out=True
        )
        logger.info(
            f"{self.label}: update_weights_shm completed, "
            f"updated={result}, is_last={is_last}"
        )
        # Signal completion back to CoreManager / LLMEngine
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "update_weights_shm", "result": result})
        )

    def _handle_update_weights_ipc(self, args: dict):
        """Handle CUDA IPC weight update command.

        The caller (LLMEngine) sends a CUDA IPC handle pointing to a GPU
        buffer that already contains the weight data. Each ModelRunner
        sub-process uses ``rebuild_ipc_handle()`` to map the same GPU memory
        and reads weights directly — no CPU round-trip.

        When ``ipc_handles`` (per-GPU dict) is present, each ModelRunner
        opens only its own GPU's handle — always same-GPU IPC, safe on ROCm.
        """
        ipc_handle = args.get("ipc_handle")
        bucket_meta = args.get("bucket_meta", {})
        is_last = args.get("is_last", True)
        ipc_handles = args.get("ipc_handles")
        result = self.runner_mgr.call_func(
            "update_weights_from_ipc",
            ipc_handle,
            bucket_meta,
            is_last,
            ipc_handles,
            wait_out=True,
        )
        logger.info(
            f"{self.label}: update_weights_ipc completed, "
            f"updated={result}, is_last={is_last}"
        )
        # Signal completion back to CoreManager / LLMEngine
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "update_weights_ipc", "result": result})
        )

    def _handle_release_memory(self, args: dict):
        """Handle memory release command (sleep mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        result = self.runner_mgr.call_func("release_memory", tags, wait_out=True)
        logger.info(f"{self.label}: release_memory completed, tags={tags}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "release_memory", "result": result})
        )

    def _handle_resume_memory(self, args: dict):
        """Handle memory resume command (wake up mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        result = self.runner_mgr.call_func("resume_memory", tags, wait_out=True)
        logger.info(f"{self.label}: resume_memory completed, tags={tags}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "resume_memory", "result": result})
        )

    def _handle_clear_kv_cache(self, args: dict):
        """Handle KV cache clear command."""
        # Use wait_out=True to ensure the GPU zero_() kernel completes before
        # any subsequent release_memory call can modify memory mappings.
        result = self.runner_mgr.call_func("clear_kv_cache", wait_out=True)
        logger.info(f"{self.label}: KV cache cleared")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "clear_kv_cache", "result": result})
        )

    def _handle_abort_request(self, args: dict):
        """Mark a sequence ABORTED (client disconnected) so the scheduler finishes
        it at the next step via the normal stop path (frees KV, drops it)."""
        req_id = args.get("req_id") if isinstance(args, dict) else None
        if req_id is None or self.scheduler is None:
            return
        found = False
        for seq in list(self.scheduler.running) + list(self.scheduler.waiting):
            if seq.id == req_id:
                seq.status = SequenceStatus.ABORTED
                found = True
        logger.info(f"{self.label}: abort_request req_id={req_id} found={found}")

    def _handle_configure_hidden_states(self, args: dict):
        """Configure hidden states extraction on all model runners (TorchSpec)."""
        aux_layer_ids = args.get("aux_layer_ids", [])
        mooncake_config = args.get("mooncake_config", {})
        result = self.runner_mgr.call_func(
            "configure_hidden_states", aux_layer_ids, mooncake_config, wait_out=True
        )
        logger.info(
            f"{self.label}: configure_hidden_states completed, "
            f"aux_layers={aux_layer_ids}"
        )
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "configure_hidden_states", "result": result})
        )

    # ------------------------------------------------------------------
    # Profiler
    # ------------------------------------------------------------------

    def _start_profiler_now(self):
        # start_profiler returns a bare True per rank, which the caller has no
        # use for: every branch here answers with a dict the endpoint can
        # inspect for "error" and "message".
        self.runner_mgr.call_func("start_profiler", wait_out=True)
        # Flip the scheduler flag so per-iteration detailed aggregates
        # (compute_detailed_aggregates) are emitted while profiling is active.
        if self.scheduler is not None:
            self.scheduler.profile_active = True
        self._profiler_active = True
        self._profiler_recorded = 0
        logger.info(f"{self.label}: profiler started")
        return {"message": "Profiling started"}

    def _stop_profiler_now(self):
        logger.info(f"{self.label}: stopping profiler...")
        result = self.runner_mgr.call_func("stop_profiler", wait_out=True)
        if self.scheduler is not None:
            self.scheduler.profile_active = False
        self._profiler_active = False
        self._profiler_pending = 0
        logger.info(f"{self.label}: profiler stopped, result={result}")
        return result

    def _handle_start_profile(self, args: dict):
        if self._profiler_active or self._profiler_pending:
            # Starting twice without an intervening stop would leave one trace
            # spanning both runs while the window counts from the second call.
            result = {
                "error": "Profiling is already in progress. Call /stop_profile first."
            }
            logger.warning(f"{self.label}: {result['error']}")
        elif self.profiler_delay_iters:
            self._profiler_pending = self.profiler_delay_iters
            result = {
                "armed_after_iters": self.profiler_delay_iters,
                "message": (
                    f"Profiling armed. Recording starts after "
                    f"{self.profiler_delay_iters} engine steps."
                ),
            }
            logger.info(f"{self.label}: {result['message']}")
        else:
            result = self._start_profiler_now()
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "start_profile", "result": result})
        )

    def _handle_stop_profile(self, args: dict):
        result = self._stop_profiler_now()
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "stop_profile", "result": result})
        )

    def profiler_step(self):
        """Advance the profiler window once per completed forward.

        Called after the forward RPC returns, so the trace export that closing
        a window triggers runs between engine steps rather than inside one.
        """
        if self._profiler_pending:
            self._profiler_pending -= 1
            if self._profiler_pending == 0:
                self._start_profiler_now()
        elif self._profiler_active and self.profiler_max_iters:
            self._profiler_recorded += 1
            if self._profiler_recorded >= self.profiler_max_iters:
                self._stop_profiler_now()

    # ------------------------------------------------------------------
    # MTP statistics
    # ------------------------------------------------------------------

    def _handle_get_mtp_stats(self, args: dict):
        """Print MTP statistics to log (fire-and-forget)."""
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is not None and stats.spec_enabled:
            stats.log_spec()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available "
                "(MTP not enabled or no tokens processed)\n"
            )

    def _handle_get_mtp_statistics(self, args: dict):
        """Return structured MTP statistics via UTILITY_RESPONSE."""
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.spec_enabled:
            result = {"enabled": False}
        else:
            result = stats.spec_statistics()
            result["enabled"] = True
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_mtp_statistics", "result": result})
        )

    # ------------------------------------------------------------------
    # Prefix cache statistics
    # ------------------------------------------------------------------

    def _handle_get_cache_statistics(self, args: dict):
        """Return structured prefix-cache statistics via UTILITY_RESPONSE.

        Same counters the periodic `[Cache Stats]` log line reports, on demand
        instead of every hundredth request — a client measuring reuse over a
        handful of requests cannot wait for that interval, and reading it out
        of a log is not something a client can do at all.
        """
        stats = None if self.scheduler is None else self.scheduler.engine_stats
        if stats is None or not stats.cache_enabled:
            result = {"enabled": False}
        else:
            result = stats.cache_statistics()
            result["enabled"] = True
            # The cache section counts the reuse a request wanted and did not
            # get; the funnel is where it was lost.
            result |= self.scheduler.block_manager.checkpoint_funnel()
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_cache_statistics", "result": result})
        )

    def push_metrics(self) -> None:
        """Publish this rank's metrics snapshot on the output socket.

        Pushed on the engine's own clock rather than answered on demand. The
        pull version was a synchronous round trip with a 5s deadline fired every
        5s from the API server; whenever the engine was busy -- a long prefill,
        a GEMM autotune, a large batch -- it could not answer in time, so under
        load it failed on essentially every attempt, buried the server log in
        tracebacks, and left late replies in the response queue for the *next*
        caller to mistake for its own. Pushing removes the deadline, and with it
        the last off-loop writer on the control socket.
        """
        self.output_queue.put_nowait(("METRICS", self.collect_metrics()))

    def collect_metrics(self) -> dict:
        """One rank's scheduler, KV, MTP, and cache metrics."""
        if self.scheduler is None:
            result = {"enabled": False}
        else:
            running, waiting = self.scheduler.get_request_counts()
            # None on the P/D prefill side, which owns no blocks — the decode
            # process does. Its snapshot then carries no kv_blocks_* keys at
            # all rather than a fabricated empty pool; the aggregator sums with
            # `.get(key, 0)`, so the decode rank's real figures come through
            # unchanged.
            block_manager = getattr(self.scheduler, "block_manager", None)
            kv_pool = None if block_manager is None else block_manager.kv
            kv_connector = getattr(self.scheduler, "kv_connector", None)

            engine_stats = self.scheduler.engine_stats
            if not engine_stats.spec_enabled:
                mtp = {"enabled": False}
            else:
                mtp = {"enabled": True, **engine_stats.spec_statistics()}

            if not engine_stats.cache_enabled:
                cache = {"enabled": False}
            else:
                cache = {
                    "enabled": True,
                    **engine_stats.cache_statistics(),
                    **self.scheduler.block_manager.checkpoint_funnel(),
                }

            offload = (
                kv_connector.get_statistics()
                if kv_connector is not None and hasattr(kv_connector, "get_statistics")
                else {}
            )
            result = {
                "enabled": True,
                # "prefill" / "decode" / "" — lets the aggregator recognise a
                # P/D pair, where one request is held by both ranks at once.
                "role": getattr(self.scheduler, "_METRICS_ROLE", ""),
                "requests_running": running,
                "requests_waiting": waiting,
                "requests_parked_kv_load": int(
                    getattr(self.scheduler, "_num_parked_remote_kv", 0)
                ),
                "requests_partial_prefill": int(
                    getattr(self.scheduler, "_partial_prefill_count", 0)
                ),
                "requests_finished": int(
                    getattr(self.scheduler, "total_finished_requests", 0)
                ),
                "prompt_tokens": int(getattr(self.scheduler, "total_prompt_tokens", 0)),
                "generation_tokens": int(
                    getattr(self.scheduler, "total_generation_tokens", 0)
                ),
                "preemptions": int(getattr(self.scheduler, "total_preemptions", 0)),
                "mtp": mtp,
                "cache": cache,
                "offload": offload,
            }
            if kv_pool is not None:
                result |= {
                    "kv_blocks_used": kv_pool.num_used,
                    "kv_blocks_free": kv_pool.num_free,
                    "kv_blocks_total": kv_pool.num_blocks,
                    "kv_blocks_indexed": kv_pool.num_indexed,
                }

        return result
