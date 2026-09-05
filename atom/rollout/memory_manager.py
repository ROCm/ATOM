# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Optional

import torch
from atom.utils.forward_context import set_kv_cache_data

logger = logging.getLogger("atom")


class MemoryManagerMixin:
    """Mixin providing GPU memory lifecycle management for ModelRunner.

    Host class must provide:
      - self.model (nn.Module)
      - self.device (torch.device)
      - self.config (Config) — with num_kvcache_blocks
      - self.kv_cache — KV cache tensor
      - self.enforce_eager (bool)
      - self.label (str)
      - self.tokenID_processor — tokenIDProcessor instance
      - self.graphs (dict), self.graph_pool — CUDA graph state
      - self.allocate_kv_cache(num_blocks) — method
      - self.capture_cudagraph() — method
      - self.get_num_blocks() — method
    """

    def clear_kv_cache(self) -> bool:
        kv = self.kv_cache
        if kv is None:
            kv = getattr(self, "_kv_cache_backup", None)
        if kv is None:
            return True
        kv.zero_()
        torch.cuda.synchronize()
        logger.debug(f"{self.label}: KV cache cleared")
        return True

    def release_memory(self, tags: Optional[list[str]] = None) -> bool:

        if tags is None:
            tags = ["weights", "kv_cache"]

        # Synchronize ALL GPU streams before releasing memory to prevent
        # use-after-free: the tokenIDProcessor.async_copy_stream may have
        # pending async D2H copies, and clear_kv_cache's zero_() kernel
        # may still be running on the default stream.
        torch.cuda.synchronize()

        # Clean up tokenIDProcessor deferred output state to remove
        # stale GPU tensor references (prev_token_ids, etc.)
        if hasattr(self, "tokenID_processor"):
            self.tokenID_processor.clean()

        if "weights" in tags:
            self._release_weights()

        if "kv_cache" in tags:
            self._release_kv_cache()

        # Synchronize again and empty CUDA cache to return freed blocks
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        logger.info(f"{self.label}: GPU memory released, tags={tags}")
        return True

    def resume_memory(self, tags: Optional[list[str]] = None) -> bool:

        if tags is None:
            tags = ["weights", "kv_cache"]

        if "weights" in tags:
            self._resume_weights()

        if "kv_cache" in tags:
            self._resume_kv_cache()

        logger.info(f"{self.label}: GPU memory resumed, tags={tags}")
        return True

    def _release_weights(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            return
        # No-eager sleep policy: keep weights AND CUDA graphs resident so their
        # GPU addresses stay stable across sleep/wake. Online weight updates are
        # applied in-place (param.data.copy_ + in-place shuffle), so the graphs
        # remain valid and never need recapture — this avoids the GPU
        # memory-access fault that occurs when recapturing graphs on wake under
        # PYTORCH_CUDA_ALLOC_CONF=expandable_segments.
        if not self.enforce_eager:
            logger.info(
                f"{self.label}: no-eager sleep keeps weights + CUDA graphs resident"
            )
            return
        # Discard GPU weight data but keep shape/dtype metadata so that
        # weight sync (SHM or IPC) can do param.data.copy_() later.
        # The weights are always overwritten after resume, so offloading
        # to CPU wastes RAM.
        self._released_weight_meta = {}
        for name, param in self.model.named_parameters():
            self._released_weight_meta[name] = (param.shape, param.dtype)
            param.data = torch.empty(0, dtype=param.dtype, device="cpu")
        self._weights_discarded = True
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        logger.info(f"{self.label}: Weights discarded")

    def _resume_weights(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            return
        if getattr(self, "_weights_discarded", False):
            # Weights were discarded — allocate empty GPU tensors with the
            # correct shape so that weight sync (SHM or IPC) can copy_ into
            # them.  This avoids the CPU→GPU round-trip entirely.
            for name, param in self.model.named_parameters():
                if name in self._released_weight_meta:
                    shape, dtype = self._released_weight_meta[name]
                    param.data = torch.empty(shape, dtype=dtype, device=self.device)
            self._weights_discarded = False
            self._released_weight_meta = {}
            torch.cuda.synchronize()
            logger.info(f"{self.label}: Weight placeholders allocated on {self.device}")
        else:
            for param in self.model.parameters():
                param.data = param.data.to(self.device, non_blocking=False)
            torch.cuda.synchronize()
            logger.info(f"{self.label}: Weights restored to {self.device}")

    def _release_kv_cache(self) -> None:
        if not hasattr(self, "kv_cache") or self.kv_cache is None:
            return
        # No-eager: keep the KV cache resident so its GPU address stays stable —
        # decode CUDA graphs capture the KV cache base pointer, so freeing and
        # re-allocating it would invalidate the graphs and force a (fault-prone)
        # recapture on wake. Contents are still zeroed via clear_kv_cache().
        if not self.enforce_eager:
            logger.info(f"{self.label}: no-eager sleep keeps KV cache resident")
            return
        self._kv_cache_num_blocks = self.config.num_kvcache_blocks

        # Clear per-module KV cache views that share the underlying storage.
        # Without this, del self.kv_cache alone cannot free GPU memory.
        for model_obj in self._get_models_with_kv():
            for module in model_obj.modules():
                for attr in ("k_cache", "v_cache", "kv_cache", "kpool_tail_cache"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)

        set_kv_cache_data({})

        del self.kv_cache
        self.kv_cache = None
        for attr in (
            "kv_scale",
            "index_cache",
            "mamba_k_cache",
            "mamba_v_cache",
            "kpool_tail_cache",
            "_kv_cache_backup",
        ):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        torch.cuda.empty_cache()
        logger.info(f"{self.label}: KV cache released (GPU memory freed)")

    def _get_models_with_kv(self):
        models = [self.model]
        if hasattr(self, "drafter") and hasattr(self.drafter, "model"):
            models.append(self.drafter.model)
        return models

    def _resume_kv_cache(self) -> None:
        # No-eager sleep keeps the existing KV cache resident so CUDA graph
        # addresses remain stable. There is nothing to restore in this case.
        if not self.enforce_eager and getattr(self, "kv_cache", None) is not None:
            return

        if (
            not hasattr(self, "_kv_cache_num_blocks")
            or self._kv_cache_num_blocks is None
        ):
            logger.warning(f"{self.label}: No KV cache num_blocks to resume from")
            return
        saved_blocks = self._kv_cache_num_blocks
        self._kv_cache_num_blocks = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        available_blocks = self.get_num_blocks()["num_kvcache_blocks"]
        num_blocks = min(saved_blocks, available_blocks)
        if num_blocks < saved_blocks:
            logger.warning(
                f"{self.label}: KV cache blocks reduced from {saved_blocks} to "
                f"{num_blocks} due to changed GPU memory availability"
            )
        self.allocate_kv_cache(num_blocks)
        logger.info(
            f"{self.label}: KV cache re-allocated and bound ({num_blocks} blocks)"
        )
