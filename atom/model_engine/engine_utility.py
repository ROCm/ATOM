# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import queue

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
    """

    # Utility command name  ->  handler method name
    _UTILITY_HANDLERS = {
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
        "abort_request": "_handle_abort_request",
    }

    def __init__(
        self, runner_mgr, output_queue, label: str = "Engine Core", scheduler=None
    ):
        self.runner_mgr = runner_mgr
        self.output_queue = output_queue
        self.label = label
        self.scheduler = scheduler

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

        logger.info(f"{self.label}: executing utility command: {cmd}")
        t0 = _time.monotonic()

        handler_name = self._UTILITY_HANDLERS.get(cmd)
        if handler_name:
            handler = getattr(self, handler_name)
            handler(args)
        else:
            logger.warning(f"{self.label}: Unknown utility command: {cmd}")

        elapsed = _time.monotonic() - t0
        logger.info(f"{self.label}: utility command '{cmd}' finished in {elapsed:.2f}s")

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

    def _handle_start_profile(self, args: dict):
        result = self.runner_mgr.call_func("start_profiler", wait_out=True)
        # Flip the scheduler flag so per-iteration detailed aggregates
        # (compute_detailed_aggregates) are emitted while profiling is active.
        if self.scheduler is not None:
            self.scheduler.profile_active = True
        logger.info(f"{self.label}: profiler started")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "start_profile", "result": result})
        )

    def _handle_stop_profile(self, args: dict):
        logger.info(f"{self.label}: stopping profiler...")
        result = self.runner_mgr.call_func("stop_profiler", wait_out=True)
        if self.scheduler is not None:
            self.scheduler.profile_active = False
        logger.info(f"{self.label}: profiler stopped, result={result}")
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "stop_profile", "result": result})
        )

    # ------------------------------------------------------------------
    # MTP statistics
    # ------------------------------------------------------------------

    def _handle_get_mtp_stats(self, args: dict):
        """Print MTP statistics to log (fire-and-forget)."""
        if self.scheduler is not None and self.scheduler.spec_stats is not None:
            self.scheduler.spec_stats._log()
        else:
            logger.info(
                "\n[MTP Stats] No MTP statistics available "
                "(MTP not enabled or no tokens processed)\n"
            )

    def _handle_get_mtp_statistics(self, args: dict):
        """Return structured MTP statistics via UTILITY_RESPONSE."""
        if self.scheduler is None or self.scheduler.spec_stats is None:
            result = {"enabled": False}
        else:
            result = self.scheduler.spec_stats.get_statistics()
            result["enabled"] = True
        self.output_queue.put_nowait(
            ("UTILITY_RESPONSE", {"cmd": "get_mtp_statistics", "result": result})
        )
