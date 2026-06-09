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

    def load_weights(
        self,
        weights,
        bucket_size_mb: int = 2048,
        num_gpus: int = 1,
        mode: str = "auto",
    ):
        """Load weights into the engine.

        Args:
            weights: Iterator of (name, tensor) tuples.
            bucket_size_mb: Max bucket size in MiB.
            num_gpus: Total GPUs (TP * DP). Only used in IPC mode.
            mode: "ipc" for CUDA IPC, "shm" for shared memory,
                  "auto" to pick IPC when CUDA is available.
        """
        import torch

        if mode == "auto":
            mode = "ipc" if torch.cuda.is_available() else "shm"
        if mode == "ipc":
            load_weights_via_ipc(
                self.core_mgr, weights, bucket_size_mb, num_gpus=num_gpus
            )
        else:
            load_weights_via_shm(self.core_mgr, weights, bucket_size_mb)

    def configure_hidden_states(self, aux_layer_ids, mooncake_config):
        """Configure hidden states extraction on all model runners.

        Args:
            aux_layer_ids: Layer indices to capture (e.g. ``[1, 15, 28]``).
            mooncake_config: Dict passed to ``EagleMooncakeStore.__init__``.
        """
        self.core_mgr.broadcast_utility_command_sync(
            "configure_hidden_states",
            aux_layer_ids=aux_layer_ids,
            mooncake_config=mooncake_config,
        )
        logger.info(
            f"AsyncLLMEngine: hidden states extraction configured, "
            f"aux_layers={aux_layer_ids}"
        )

    def generate_hidden_states(self, input_ids_list, data_ids):
        """Run prefill-only forward and extract hidden states to Mooncake.

        Each request's hidden states are written to Mooncake by the
        ``RLHFModelRunner`` during forward. This method drives the
        engine loop until all requests complete, then returns metadata.

        Args:
            input_ids_list: List of token id lists.
            data_ids: List of data IDs used as Mooncake keys.

        Returns:
            List of dicts with metadata for each completed request.
        """
        from atom.sampling_params import SamplingParams

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)

        prompts = [
            ids if isinstance(ids, list) else ids.tolist() for ids in input_ids_list
        ]

        self.core_mgr._rr_counter = 0
        self.add_request(prompts, sampling_params, request_ids=data_ids)

        outputs = {}
        while not self.is_finished() and (
            self.core_mgr.is_alive() or self.core_mgr.is_rest()
        ):
            seqs = self.step()
            outs = self.io_processor.postprocess(seqs)
            outputs.update(outs)

        results = []
        for data_id in data_ids:
            results.append(
                {
                    "mooncake_key": data_id,
                    "data_id": data_id,
                }
            )
        return results

    def shutdown(self):
        """Shutdown engine and release all resources."""
        self.core_mgr.close()
