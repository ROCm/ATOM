# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import os

import torch

from atom.utils.forward_context import set_kv_cache_data
from atom.utils.graph_holders import release_registered_graphs

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
    """Drop every captured graph *runner* holds, and mark them for recapture.

    A graph replays the addresses it captured: every weight, and the base of
    the KV pool. Whichever of the two a release frees, the graphs that
    captured it can no longer be replayed, so both release paths come through
    here. Releasing the KV pool alone -- what `AsyncLLMEngine.sleep(level=1)`
    does, and the default level -- used to leave the graphs in place to be
    replayed against a pool that had since been freed and reallocated.

    `runner.graphs` holds one of four stores, and not the one the default
    configuration fills: `--level 3` is PIECEWISE, where the graphs are one per
    compiled dense piece and that dict stays empty however much was captured.
    So each store is asked separately and the answers are OR-ed -- "was
    anything captured" is not a question `runner.graphs` can answer. Gating the
    whole function on it, which is what this used to do, made the release a
    no-op on precisely the configuration whose fault it exists to prevent.
    """
    if getattr(runner, "enforce_eager", True):
        return
    released = _release_tbo_graphs(runner)
    released |= _release_piecewise_graphs(runner)
    released |= _release_draft_graphs(runner)
    released |= _release_manual_graphs(runner)
    if not released:
        return
    # What `_recapture_cudagraphs_if_needed` gates on. Not `_graphs_backup_keys`:
    # that list belongs to the manual store and is empty under PIECEWISE, which
    # is how the recapture came to be skipped there along with the release.
    runner._graphs_released_for_sleep = True
    # Every graph that could have been sharing it is gone, so the next capture
    # makes its own. Under PIECEWISE the handle parked here is the DRAFT pool's
    # -- `warmup_draft_graphs` publishes it there -- since a piecewise capture
    # records no whole-forward graph to own one.
    runner.graph_pool = None
    logger.info(f"{runner.label}: CUDA graphs released for sleep")
    _warn_if_recapture_will_fault(runner)


def _release_tbo_graphs(runner) -> bool:
    """`UBatchWrapper`'s parallel store, one entry per shape.

    Under TBO the replayable handle is in `runner.graphs` like any other, but
    the wrapper keeps its own entry holding the graph, its per-ubatch contexts
    and the output tensor it captured. Left behind, that entry pins the graph's
    private memory pool -- the footprint the caller went to sleep to reclaim --
    until a recapture happens to overwrite the same key. `ModelRunner.exit()`
    has always cleared both stores; a release for sleep has to clear the same
    two.
    """
    tbo_graphs = getattr(getattr(runner, "model", None), "tbo_graphs", None)
    if not tbo_graphs:
        return False
    tbo_graphs.clear()
    logger.info(f"{runner.label}: TBO CUDA graphs released for sleep")
    return True


def _release_piecewise_graphs(runner) -> bool:
    """The per-piece store, and the dispatch that would replay out of it.

    A PIECEWISE capture puts nothing in `runner.graphs`: the compiled dense
    pieces self-capture into their own `CUDAGraphWrapper`s, and the capture loop
    in `ModelRunner.capture_cudagraph` moves on before the assignment. Those
    wrappers are reached through the registry they enter rather than by a walk
    from here -- `atom/utils/graph_holders.py` says why a walk cannot find them.
    `_piecewise_captured_tokens` is the runner's record of which shapes got a
    graph, and so its evidence that anything piecewise was captured at all.

    Clearing that record is not bookkeeping. It is what stops the next step
    dispatching PIECEWISE, which would either replay a graph holding the pool
    this release is freeing or -- its entry having just been dropped -- record a
    replacement mid-serve, uncoordinated, and hang on the first collective.
    """
    captured_tokens = getattr(runner, "_piecewise_captured_tokens", None)
    if not captured_tokens:
        return False
    captured_tokens.clear()
    runner._piecewise_sorted_tokens = []
    dropped = release_registered_graphs()
    logger.info(f"{runner.label}: {dropped} piecewise CUDA graphs released for sleep")
    return True


def _release_draft_graphs(runner) -> bool:
    """The drafter's recordings, one per captured batch.

    Walked rather than registered, because this store is reachable from here. A
    draft pass writes the KV it attends, so its recording holds the base of the
    pool the way a decode graph does, and `ATOM_DRAFT_CUDAGRAPH` is on by
    default. Wake recaptures them: `capture_cudagraph` ends in
    `warmup_draft_graphs` on both the manual and the piecewise path.
    """
    drafter = getattr(runner, "drafter", None)
    released = False
    for pass_ in getattr(drafter, "draft_graphs", None) or ():
        released |= bool(pass_.release_graphs())
    if released:
        logger.info(f"{runner.label}: draft CUDA graphs released for sleep")
    return released


def _release_manual_graphs(runner) -> bool:
    """The whole-forward store a FULL capture fills, and the logits with it."""
    released = False
    graphs = getattr(runner, "graphs", None)
    if graphs:
        runner._graphs_backup_keys = list(graphs.keys())
        graphs.clear()
        released = True
    # `graph_logits` holds the logits tensor each capture produced, allocated
    # from the graph's own pool. Same shape of leak as the TBO store: the
    # replay is already stopped by clearing `graphs`, but a live reference to
    # that tensor keeps the pool `empty_cache()` is about to be asked to
    # reclaim. Recapture refills it per key. Asked separately from `graphs`,
    # because a gate they shared is what skipped it wherever `graphs` was empty.
    graph_logits = getattr(runner, "graph_logits", None)
    if graph_logits:
        graph_logits.clear()
        released = True
    return released


def _warn_if_recapture_will_fault(runner) -> None:
    """Say so before the fault, not after.

    `expandable_segments` is what makes a recapture fault, and
    `sleep_keeps_memory_resident` is the way out -- but it is opt-in, so the
    default configuration walks into it. The failure handler in
    `_recapture_cudagraphs_if_needed` only gets to speak once the recapture has
    already gone wrong, and by then it has pinned the runner to eager.
    """
    if "expandable_segments" not in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        return
    if getattr(getattr(runner, "config", None), "sleep_keeps_memory_resident", False):
        return
    logger.warning(
        f"{runner.label}: released CUDA graphs for sleep with "
        f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments set. Recapture on wake "
        f"is what faults under expandable segments; set "
        f"Config.sleep_keeps_memory_resident=True to keep the weights and the "
        f"KV pool -- and so the graphs -- valid across sleep instead."
    )


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
        torch.cuda.empty_cache()
        # The size the pool slept at, not a fresh reading: `BlockManager`'s
        # `BlockPool` is sized in the engine process from the startup count and
        # nothing carries a wake-time one back, and the decode graphs were
        # captured against the original pool.
        free, total = torch.cuda.mem_get_info()
        logger.info(
            f"{self.label}: re-allocating {saved_blocks} KV blocks "
            f"({free / (1 << 30):.2f}GB free of {total / (1 << 30):.2f}GB)"
        )
        # After the allocation, which can now OOM: clearing it first would lose
        # the only record of the size, and the next wake would then take the
        # guard above and report success with no pool.
        self.allocate_kv_cache(saved_blocks)
        self._kv_cache_num_blocks = None
        logger.info(
            f"{self.label}: KV cache re-allocated and bound ({saved_blocks} blocks)"
        )

    def _recapture_cudagraphs_if_needed(self) -> None:
        """Recapture CUDA graphs if they were released during sleep.

        CUDA graphs capture GPU memory addresses at capture time.  After
        sleep/wake, weight and KV-cache tensors are at new addresses, so the
        old graphs are invalid and must be recaptured.

        We only recapture when **both** weights and KV cache are on GPU
        (i.e., the model is fully ready for inference).

        Nothing to do under `sleep_keeps_memory_resident`: nothing was
        released, so the flag `release_cudagraphs` sets is absent and this
        returns below. That flag, and not `_graphs_backup_keys`, because the
        manual store is empty under PIECEWISE whatever was captured -- keying on
        it left the default configuration releasing nothing and, when the
        release was fixed, would have left it recapturing nothing.
        """
        if getattr(self, "enforce_eager", True):
            return
        if not getattr(self, "_graphs_released_for_sleep", False):
            return
        # Only recapture if both weights and KV cache are on GPU
        has_weights_on_gpu = any(p.is_cuda for p in self.model.parameters())
        has_kv_cache = self.kv_cache is not None
        if not has_weights_on_gpu or not has_kv_cache:
            return
        logger.info(f"{self.label}: Recapturing CUDA graphs after sleep/wake cycle")
        try:
            self.capture_cudagraph()
            self._graphs_released_for_sleep = False
            # Absent when only the piecewise store had anything in it.
            if hasattr(self, "_graphs_backup_keys"):
                del self._graphs_backup_keys
            logger.info(f"{self.label}: CUDA graph recapture completed")
        except Exception:
            logger.exception(f"{self.label}: CUDA graph recapture failed")
            # Fall back to eager mode rather than crashing
            self.enforce_eager = True
            self.graphs = {}
            self.graph_pool = None
            self._graphs_released_for_sleep = False
            if hasattr(self, "_graphs_backup_keys"):
                del self._graphs_backup_keys
            logger.warning(f"{self.label}: Falling back to enforce_eager=True")
            # `sleep_keeps_memory_resident` reads `enforce_eager` first and
            # answers False for an eager runner, which is right -- there are no
            # graphs left to keep valid -- but it means an operator who set the
            # option now silently gets the release-and-restore behaviour it was
            # set to avoid. Say which of the two states we are in, because from
            # here the option's log lines never appear again.
            if getattr(
                getattr(self, "config", None), "sleep_keeps_memory_resident", False
            ):
                logger.warning(
                    f"{self.label}: Config.sleep_keeps_memory_resident is set "
                    f"but now inert: with no captured graphs there is nothing "
                    f"to keep valid, so sleep releases and restores the "
                    f"weights and the KV pool as it would without the option. "
                    f"Restart to recapture."
                )
