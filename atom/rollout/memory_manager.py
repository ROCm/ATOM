# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging

import torch

from atom.utils.forward_context import set_kv_cache_data

logger = logging.getLogger("atom")

# Every name a binder may have set to a view of the KV pool. Not just the cache
# ones: the pool is a single buffer now, so one surviving scale plane or
# indexer slice pins all of it -- what used to leak a scale tensor leaks the
# whole pool.
_POOL_VIEW_ATTRS = (
    "k_cache",
    "v_cache",
    "kv_cache",
    "kpool_tail_cache",
    "k_scale",
    "v_scale",
    "index_cache",
)


# Both of these take the runner rather than being methods on the mixin below,
# because its methods are called unbound on stand-ins that provide only the
# attributes they touch -- `tests/test_rollout_memory_manager.py` hands
# `_release_kv_cache` a `SimpleNamespace`.
def sleep_keeps_memory_resident(runner) -> bool:
    """Whether sleep should leave *runner*'s weights and KV pool allocated.

    Keeping them costs the whole footprint the caller went to sleep to
    reclaim, so it is opt-in (`Config.sleep_keeps_memory_resident`) rather
    than the default for every non-eager deployment. What it buys is that
    nothing a decode graph captured moves, so nothing has to be recaptured --
    recapture is what faults under
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments.

    Both reads go through `getattr`, and `enforce_eager` defaults to True,
    i.e. to releasing: a host that defines neither gets the behaviour every
    caller had before this option existed.
    """
    if getattr(runner, "enforce_eager", True):
        return False
    return bool(
        getattr(getattr(runner, "config", None), "sleep_keeps_memory_resident", False)
    )


def release_cudagraphs(runner) -> None:
    """Drop *runner*'s captured decode graphs and mark them for recapture.

    A graph replays the addresses it captured: every weight, and the base of
    the KV pool. Whichever of the two a release frees, the graphs that
    captured it can no longer be replayed, so both release paths come through
    here. Releasing the KV pool alone -- what `AsyncLLMEngine.sleep(level=1)`
    does, and the default level -- used to leave the graphs in place to be
    replayed against a pool that had since been freed and reallocated.
    """
    if getattr(runner, "enforce_eager", True) or not getattr(runner, "graphs", None):
        return
    runner._graphs_backup_keys = list(runner.graphs.keys())
    runner.graphs.clear()
    runner.graph_pool = None
    logger.info(f"{runner.label}: CUDA graphs released for sleep")


class MemoryManagerMixin:
    """Mixin providing GPU memory lifecycle management for ModelRunner.

    Host class must provide:
      - self.model (nn.Module)
      - self.device (torch.device)
      - self.config (Config) — with num_kvcache_blocks and
        sleep_keeps_memory_resident
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

    def release_memory(self, tags: list[str] | None = None) -> bool:

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

    def resume_memory(self, tags: list[str] | None = None) -> bool:

        if tags is None:
            tags = ["weights", "kv_cache"]

        if "weights" in tags:
            self._resume_weights()

        if "kv_cache" in tags:
            self._resume_kv_cache()

        self._recapture_cudagraphs_if_needed()

        logger.info(f"{self.label}: GPU memory resumed, tags={tags}")
        return True

    def _release_weights(self) -> None:
        if not hasattr(self, "model") or self.model is None:
            return
        if sleep_keeps_memory_resident(self):
            logger.info(
                f"{self.label}: sleep keeps the weights and the CUDA graphs "
                f"resident (Config.sleep_keeps_memory_resident)"
            )
            return
        # Release CUDA graphs first — they hold references to weight memory
        # and prevent freeing GPU memory.
        release_cudagraphs(self)
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
        if sleep_keeps_memory_resident(self) and not getattr(
            self, "_weights_discarded", False
        ):
            # Nothing was released, so there is nothing to restore -- and the
            # point of the option is that no parameter moves.
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
        if sleep_keeps_memory_resident(self):
            logger.info(
                f"{self.label}: sleep keeps the KV pool resident "
                f"(Config.sleep_keeps_memory_resident); clear_kv_cache() still "
                f"zeroes it"
            )
            return
        # The graphs captured the base of the pool this is about to free.
        release_cudagraphs(self)
        self._kv_cache_num_blocks = self.config.num_kvcache_blocks

        # Clear per-module KV cache views that share the underlying storage.
        # Without this, del self.kv_cache alone cannot free GPU memory.
        #
        # On the value and not the name: these names are not unique, and
        # `MiMoV2Attention.v_scale` is a float multiplier on V rather than a
        # dequant plane, which blanking would silently stop applying. Gating on
        # a sibling name instead would answer the wrong question -- and did:
        # `index_cache` lives on the `impl` that never holds a `k_cache`.
        for model_obj in self._get_models_with_kv():
            for module in model_obj.modules():
                for attr in _POOL_VIEW_ATTRS:
                    if isinstance(getattr(module, attr, None), torch.Tensor):
                        setattr(module, attr, None)
                # `DeepseekV32IndexerCache` holds its slice in a one-element
                # list the binder assigns *into*. Emptying the element and not
                # the list: waking rebinds with `kv_cache[0] = ...`, which
                # needs a list to still be there.
                if isinstance(getattr(module, "kv_cache", None), list):
                    module.kv_cache = [torch.tensor([])]

        set_kv_cache_data({})

        # A builder's pools hold views of the same buffer, so dropping only the
        # runner's reference frees nothing.
        for owner in (
            getattr(self, "attn_metadata_builder", None),
            getattr(self, "draft_kv_builder", None),
        ):
            if owner is not None:
                owner.release_kv_pools()

        del self.kv_cache
        self.kv_cache = None
        for attr in (
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
        # The matching half of the release guard: the pool was never freed, so
        # there is nothing to re-allocate and rebind.
        if (
            sleep_keeps_memory_resident(self)
            and getattr(self, "kv_cache", None) is not None
        ):
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

    def _recapture_cudagraphs_if_needed(self) -> None:
        """Recapture CUDA graphs if they were released during sleep.

        CUDA graphs capture GPU memory addresses at capture time.  After
        sleep/wake, weight and KV-cache tensors are at new addresses, so the
        old graphs are invalid and must be recaptured.

        We only recapture when **both** weights and KV cache are on GPU
        (i.e., the model is fully ready for inference).

        Nothing to do under `sleep_keeps_memory_resident`: nothing was
        released, so `_graphs_backup_keys` is absent and this returns below.
        """
        if getattr(self, "enforce_eager", True):
            return
        if not hasattr(self, "_graphs_backup_keys") or not self._graphs_backup_keys:
            return
        # Only recapture if both weights and KV cache are on GPU
        has_weights_on_gpu = any(p.is_cuda for p in self.model.parameters())
        has_kv_cache = self.kv_cache is not None
        if not has_weights_on_gpu or not has_kv_cache:
            return
        logger.info(f"{self.label}: Recapturing CUDA graphs after sleep/wake cycle")
        try:
            self.capture_cudagraph()
            del self._graphs_backup_keys
            logger.info(f"{self.label}: CUDA graph recapture completed")
        except Exception:
            logger.exception(f"{self.label}: CUDA graph recapture failed")
            # Fall back to eager mode rather than crashing
            self.enforce_eager = True
            self.graphs = {}
            self.graph_pool = None
            if hasattr(self, "_graphs_backup_keys"):
                del self._graphs_backup_keys
            logger.warning(f"{self.label}: Falling back to enforce_eager=True")
