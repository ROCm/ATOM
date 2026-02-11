# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import queue

logger = logging.getLogger("atom")


class EngineUtilityHandler:
    """Handles utility commands dispatched by EngineCore.

    Parameters
    ----------
    runner_mgr : AsyncIOProcManager
        The model-runner process manager used to execute ``call_func``.
    output_queue : queue.Queue
        The EngineCore output queue, used by ``_handle_update_weights_shm``
        to push ``UTILITY_RESPONSE`` messages back to ``CoreManager``.
    label : str, optional
        Label used in log messages (default ``"Engine Core"``).
    """

    # Utility command name  ->  handler method name
    _UTILITY_HANDLERS = {
        "update_weights": "_handle_update_weights",
        "update_weights_shm": "_handle_update_weights_shm",
        "release_memory": "_handle_release_memory",
        "resume_memory": "_handle_resume_memory",
        "clear_kv_cache": "_handle_clear_kv_cache",
    }

    def __init__(self, runner_mgr, output_queue, label: str = "Engine Core"):
        self.runner_mgr = runner_mgr
        self.output_queue = output_queue
        self.label = label

    def process_queue(self, utility_queue, engine):
        """Drain *utility_queue* and execute each command.

        When the queue is empty, ``engine._has_pending_utility`` is set to
        ``False`` so that the next busy-loop iteration can skip the check.
        """
        if not engine._has_pending_utility:
            return

        while True:
            try:
                cmd, args = utility_queue.get_nowait()
                self._execute_utility_command(cmd, args)
            except queue.Empty:
                engine._has_pending_utility = False
                break

    def _execute_utility_command(self, cmd: str, args: dict):
        logger.debug(f"{self.label}: executing utility command: {cmd}")

        handler_name = self._UTILITY_HANDLERS.get(cmd)
        if handler_name:
            handler = getattr(self, handler_name)
            handler(args)
        else:
            logger.warning(f"{self.label}: Unknown utility command: {cmd}")

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

        NOTE: This is the **only** handler that writes to ``output_queue``.
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

    def _handle_release_memory(self, args: dict):
        """Handle memory release command (sleep mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        self.runner_mgr.call_func("release_memory", tags, wait_out=True)
        logger.info(f"{self.label}: release_memory completed, tags={tags}")

    def _handle_resume_memory(self, args: dict):
        """Handle memory resume command (wake up mode)."""
        tags = args.get("tags", ["weights", "kv_cache"])
        self.runner_mgr.call_func("resume_memory", tags, wait_out=True)
        logger.info(f"{self.label}: resume_memory completed, tags={tags}")

    def _handle_clear_kv_cache(self, args: dict):
        """Handle KV cache clear command."""
        # Use wait_out=True to ensure the GPU zero_() kernel completes before
        # any subsequent release_memory call can modify memory mappings.
        self.runner_mgr.call_func("clear_kv_cache", wait_out=True)
        logger.info(f"{self.label}: KV cache cleared")
