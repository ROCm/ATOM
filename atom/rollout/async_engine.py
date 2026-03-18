# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

from atom.model_engine.llm_engine import LLMEngine
from atom.rollout.weight_sync import load_weights_via_shm, load_weights_via_ipc

logger = logging.getLogger("atom")


class AsyncLLMEngine(LLMEngine):
    """Async LLM Engine with RLHF lifecycle management.

    Extends LLMEngine with GPU memory lifecycle APIs for RLHF training:
    - sleep(): release GPU resources
    - wake_up(): restore GPU resources
    - load_weights() / load_weights_ipc(): synchronize weights from training engine

    Automatically injects RLHFModelRunner (with weight sync + memory lifecycle
    extensions) into the EngineCore, keeping the base ModelRunner pure for
    inference-only deployments.
    """

    def __init__(self, model, **kwargs):
        # Inject RLHFModelRunner so that EngineCore creates runners with
        # weight sync + memory lifecycle extensions.
        kwargs.setdefault(
            "runner_qualname",
            "atom.rollout.model_runner_ext.RLHFModelRunner",
        )
        super().__init__(model, **kwargs)

    def sleep(self, level: int = 1):
        """Release GPU resources.

        Args:
            level: 1 = release KV cache only, 2 = release KV cache + weights
        """
        logger.info(f"AsyncLLMEngine sleep: level={level}")

        if level >= 1:
            self.core_mgr.broadcast_utility_command_sync("clear_kv_cache")
            logger.info("AsyncLLMEngine sleep: clear_kv_cache done")
            self.core_mgr.broadcast_utility_command_sync(
                "release_memory", tags=["kv_cache"]
            )
            logger.info("AsyncLLMEngine sleep: release_memory(kv_cache) done")
        if level >= 2:
            self.core_mgr.broadcast_utility_command_sync(
                "release_memory", tags=["weights"]
            )
            logger.info("AsyncLLMEngine sleep: release_memory(weights) done")

        logger.info("AsyncLLMEngine sleep: completed")

    def wake_up(self, tags: list[str] = None):
        """Restore GPU resources.

        Args:
            tags: Resource types to resume, default ["weights", "kv_cache"]
        """
        if tags is None:
            tags = ["weights", "kv_cache"]

        logger.info(f"AsyncLLMEngine wake_up: tags={tags}")
        self.core_mgr.broadcast_utility_command_sync("resume_memory", tags=tags)
        logger.info("AsyncLLMEngine wake_up: completed")

    def load_weights(self, weights, bucket_size_mb: int = 2048):
        """Load weights via POSIX shared memory."""
        load_weights_via_shm(self.core_mgr, weights, bucket_size_mb)

    def load_weights_ipc(self, weights, bucket_size_mb: int = 2048,
                         num_gpus: int = 1):
        """Load weights via CUDA IPC (zero-copy GPU-to-GPU).

        Args:
            weights: Iterator of (name, tensor) tuples.
            bucket_size_mb: Max bucket size in MiB.
            num_gpus: Total GPUs (TP * DP). When > 1, uses per-GPU IPC
                buffers to avoid cross-GPU hipIpcOpenMemHandle on ROCm.
        """
        load_weights_via_ipc(self.core_mgr, weights, bucket_size_mb,
                             num_gpus=num_gpus)

    def shutdown(self):
        """Shutdown engine and release all resources."""
        self.core_mgr.close()
