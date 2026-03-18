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
            kv = getattr(self, '_kv_cache_backup', None)
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
        if hasattr(self, 'tokenID_processor'):
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
        if not hasattr(self, 'model') or self.model is None:
            return
        # Release CUDA graphs first — they hold references to weight memory
        # and prevent freeing GPU memory.
        if not self.enforce_eager and hasattr(self, 'graphs') and self.graphs:
            self._graphs_backup_keys = list(self.graphs.keys())
            self.graphs.clear()
            self.graph_pool = None
            logger.info(f"{self.label}: CUDA graphs released for sleep")
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
        if not hasattr(self, 'model') or self.model is None:
            return
        if getattr(self, '_weights_discarded', False):
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
        # Recapture CUDA graphs since weight GPU addresses have changed
        self._recapture_cudagraphs_if_needed()

    def _release_kv_cache(self) -> None:
        if not hasattr(self, 'kv_cache') or self.kv_cache is None:
            return
        self._kv_cache_num_blocks = self.config.num_kvcache_blocks

        # Clear per-module KV cache views that share the underlying storage.
        # Without this, del self.kv_cache alone cannot free GPU memory.
        for model_obj in self._get_models_with_kv():
            for module in model_obj.modules():
                for attr in ('k_cache', 'v_cache', 'kv_cache'):
                    if hasattr(module, attr):
                        setattr(module, attr, None)

        set_kv_cache_data({})

        del self.kv_cache
        self.kv_cache = None
        for attr in ('kv_scale', 'index_cache', 'mamba_k_cache',
                      'mamba_v_cache', '_kv_cache_backup'):
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        torch.cuda.empty_cache()
        logger.info(f"{self.label}: KV cache released (GPU memory freed)")

    def _get_models_with_kv(self):
        models = [self.model]
        if hasattr(self, 'drafter') and hasattr(self.drafter, 'model'):
            models.append(self.drafter.model)
        return models

    def _resume_kv_cache(self) -> None:
        if not hasattr(self, '_kv_cache_num_blocks') or self._kv_cache_num_blocks is None:
            logger.warning(f"{self.label}: No KV cache num_blocks to resume from")
            return
        saved_blocks = self._kv_cache_num_blocks
        self._kv_cache_num_blocks = None
        # Ensure CUDA allocator has released cached blocks before checking
        # available memory — otherwise stale cached blocks inflate 'used'.
        torch.cuda.empty_cache()
        # Cap at currently available blocks to prevent over-allocation after
        # sleep/wake cycles where the GPU memory layout may have changed
        # (e.g. allocator fragmentation, different weight tensor addresses).
        available_blocks = self.get_num_blocks()
        num_blocks = min(saved_blocks, available_blocks)
        if num_blocks < saved_blocks:
            logger.warning(
                f"{self.label}: KV cache blocks reduced from {saved_blocks} to "
                f"{num_blocks} due to changed GPU memory availability"
            )
        self.allocate_kv_cache(num_blocks)
        logger.info(f"{self.label}: KV cache re-allocated and bound ({num_blocks} blocks)")
        # Recapture CUDA graphs after KV cache re-allocation (addresses changed)
        self._recapture_cudagraphs_if_needed()

    def _recapture_cudagraphs_if_needed(self) -> None:
        """Recapture CUDA graphs if they were released during sleep.

        CUDA graphs capture GPU memory addresses at capture time.  After
        sleep/wake, weight and KV-cache tensors are at new addresses, so the
        old graphs are invalid and must be recaptured.

        We only recapture when **both** weights and KV cache are on GPU
        (i.e., the model is fully ready for inference).
        """
        if self.enforce_eager:
            return
        if not hasattr(self, '_graphs_backup_keys') or not self._graphs_backup_keys:
            return
        # Only recapture if both weights and KV cache are on GPU
        has_weights_on_gpu = any(
            p.is_cuda for p in self.model.parameters()
        )
        has_kv_cache = self.kv_cache is not None
        if not has_weights_on_gpu or not has_kv_cache:
            return
        logger.info(f"{self.label}: Recapturing CUDA graphs after sleep/wake cycle")
        try:
            self.capture_cudagraph()
            del self._graphs_backup_keys
            logger.info(f"{self.label}: CUDA graph recapture completed")
        except Exception as e:
            logger.error(
                f"{self.label}: CUDA graph recapture failed: {e}",
                exc_info=True,
            )
            # Fall back to eager mode rather than crashing
            self.enforce_eager = True
            self.graphs = {}
            self.graph_pool = None
            if hasattr(self, '_graphs_backup_keys'):
                del self._graphs_backup_keys
            logger.warning(
                f"{self.label}: Falling back to enforce_eager=True"
            )
