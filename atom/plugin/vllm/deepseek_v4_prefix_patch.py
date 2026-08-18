"""ATOM DeepSeek-V4 vLLM prefix-cache SWA-recompute patch.

V4's sliding-window (SWA) state is a per-request ring stored in a fixed
per-slot region of the ATOM proxy arena -- it is NOT keyed by a vLLM block, so
vLLM's block-level prefix cache never carries it. CSA/HCA compressed history,
by contrast, lives in the 128-token proxy pages and is reused for free on a
prefix-cache hit.

On a cross-request prefix hit the new request gets a fresh per-request state
slot whose SWA ring is empty; a non-block-aligned tail token whose SWA window
reaches back into the cached (not-re-forwarded) region would then read stale
ring data.

Fix (mirrors native ATOM scheduler "fix B'"): on a hit, drop the last
``ceil(win_with_spec / block_size)`` cached blocks so those tail tokens are
re-forwarded, repopulating the ring. The re-forwarded region is >= the ring
stride, so by the last prompt token ``prefix_swa_count`` collapses to 0 and its
whole window is served from the freshly computed extend KV. Compressed-KV reuse
is unaffected: ``n_committed = context_len // ratio`` and
``context_len = cached + scheduled`` is invariant under the shift.

In plugin mode vLLM owns the scheduler / KVCacheManager, so the block drop is
applied by wrapping ``KVCacheManager.get_computed_blocks`` -- the single point
where vLLM computes the local prefix-cache hit length. It is only called when
``request.num_computed_tokens == 0`` (a genuine cross-request hit), never on a
chunked-prefill resume, whose SWA ring is already populated by prior chunks.
"""

import functools
import logging
import math

logger = logging.getLogger("atom")


def _mark_v4_proxy_cache_mode(static_forward_context, is_profiling: bool) -> None:
    for layer in static_forward_context.values():
        if getattr(layer, "_atom_v4_proxy_layer", False):
            layer._atom_v4_profiling_kv_cache = is_profiling


_V4_PROXY_LAYER_MARKERS = (
    ".atom_deepseek_v4_proxy",
    ".atom_deepseek_v4_draft_proxy",
)


def _kv_cache_config_has_v4_proxy(kv_cache_config) -> bool:
    return any(
        any(
            marker in layer_name
            for marker in _V4_PROXY_LAYER_MARKERS
            for layer_name in group.layer_names
        )
        for group in kv_cache_config.kv_cache_groups
    )


def _kv_cache_config_needs_non_immediate_reuse(kv_cache_config) -> bool:
    return _kv_cache_config_has_v4_proxy(kv_cache_config) or bool(
        getattr(kv_cache_config, "has_mamba_layers", False)
    )


def apply_vllm_v4_block_reuse_patch() -> None:
    """Keep no-prefix-cache block reuse safe for ATOM stateful cache layouts.

    vLLM commit a82f1b388f changed non-caching pools to immediately reuse the
    blocks a request just freed. The V4 proxy allocation is a global arena: its
    fixed per-request SWA prefix and block-indexed CSA/HCA tails are carved
    across the physical vLLM page boundaries. Immediate block-id reuse therefore
    exposes stale compressed entries before the arena can safely recycle them.
    ATOM's GDN path likewise keeps recurrent state keyed by the Mamba block-table
    slots; immediate churn can recycle a slot while a mixed prefill/decode batch
    still references it.

    Mark only pools whose KV-cache groups contain an ATOM V4 proxy or Mamba/GDN
    state, then retain vLLM's pre-a82f free-queue ordering for those pools. Every
    ordinary MHA/MLA model keeps the upstream locality optimization.
    """
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    original_manager_init = KVCacheManager.__init__
    if not getattr(original_manager_init, "_atom_v4_block_reuse_patched", False):

        @functools.wraps(original_manager_init)
        def wrapped_manager_init(self, *args, **kwargs):
            original_manager_init(self, *args, **kwargs)
            kv_cache_config = kwargs.get("kv_cache_config")
            if kv_cache_config is None and args:
                kv_cache_config = args[0]
            if (
                kv_cache_config is not None
                and _kv_cache_config_needs_non_immediate_reuse(kv_cache_config)
            ):
                self.block_pool._atom_v4_proxy_arena = True
                logger.info(
                    "ATOM: using non-immediate KV block reuse for a packed V4 "
                    "or stateful Mamba/GDN cache"
                )

        wrapped_manager_init._atom_v4_block_reuse_patched = True
        KVCacheManager.__init__ = wrapped_manager_init

    original_free_blocks = BlockPool.free_blocks
    if getattr(original_free_blocks, "_atom_v4_block_reuse_patched", False):
        return

    @functools.wraps(original_free_blocks)
    def wrapped_free_blocks(self, ordered_blocks):
        if not getattr(self, "_atom_v4_proxy_arena", False) or self.enable_caching:
            return original_free_blocks(self, ordered_blocks)

        # a82f changed only the `enable_caching` branch inside free_blocks.
        # Temporarily select the old branch while preserving all other upstream
        # accounting/event logic and restore the real setting before returning.
        self.enable_caching = True
        try:
            return original_free_blocks(self, ordered_blocks)
        finally:
            self.enable_caching = False

    wrapped_free_blocks._atom_v4_block_reuse_patched = True
    BlockPool.free_blocks = wrapped_free_blocks
    logger.info("ATOM DeepSeek-V4: installed packed-proxy block reuse patch")


def apply_vllm_v4_profile_cache_patch() -> None:
    """Mark vLLM 0.26's temporary CUDA-graph profiling KV cache.

    The temporary cache intentionally contains only one block per captured
    request and cannot hold V4's fixed per-request SWA arena. The V4 forward
    must therefore stay on its existing dummy-attention path until vLLM
    installs the real cache.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original = GPUModelRunner.initialize_kv_cache
    if getattr(original, "_atom_v4_profile_cache_patched", False):
        return

    @functools.wraps(original)
    def wrapped_initialize_kv_cache(
        self,
        kv_cache_config,
        is_profiling: bool = False,
    ):
        result = original(
            self,
            kv_cache_config,
            is_profiling=is_profiling,
        )
        _mark_v4_proxy_cache_mode(
            self.compilation_config.static_forward_context,
            is_profiling,
        )
        return result

    wrapped_initialize_kv_cache._atom_v4_profile_cache_patched = True
    GPUModelRunner.initialize_kv_cache = wrapped_initialize_kv_cache


def _v4_sliding_window(vllm_config) -> int:
    hf = vllm_config.model_config.hf_config
    return int(getattr(hf, "sliding_window", 128) or 128)


def _drop_swa_warmup_blocks(
    manager,
    computed_blocks,
    num_computed_tokens: int,
    shared_prefix_boundary: int,
    *,
    warmup_blocks: int,
    block_size: int,
):
    if num_computed_tokens <= 0:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    # Drop the trailing warmup blocks from every KV cache group (V4 runs a
    # single proxy group). vLLM allocates fresh blocks for the dropped tail and
    # re-forwards those tokens, repopulating the SWA ring; the deep-prefix blocks
    # are still reused.
    dropped = 0
    new_groups = []
    for group in computed_blocks.blocks:
        block_list = list(group)
        keep = max(0, len(block_list) - warmup_blocks)
        dropped = max(dropped, len(block_list) - keep)
        new_groups.append(block_list[:keep])
    if dropped == 0:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    new_num_computed_tokens = max(0, num_computed_tokens - dropped * block_size)
    new_blocks = manager.create_kv_cache_blocks(tuple(new_groups))
    return new_blocks, new_num_computed_tokens, shared_prefix_boundary


def apply_vllm_v4_prefix_swa_patch(vllm_config) -> None:
    """Enable DeepSeek-V4 prefix caching by dropping the SWA warmup blocks.

    Call only for a DeepSeek-V4 deployment with prefix caching enabled. The
    number of blocks to drop is derived once from ``vllm_config`` and captured
    in the wrapper closure, so non-V4 deployments (which never install this
    patch) are unaffected.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    from atom.plugin.vllm.deepseek_v4_bridge import (
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        _v4_win_with_spec,
    )

    win_with_spec = _v4_win_with_spec(vllm_config, _v4_sliding_window(vllm_config))
    # The SWA ring's physical stride is win_with_spec = window + num_spec_tokens
    # (MTP draft tokens get their own ring slots). Rolling back ceil(stride /
    # block_size) whole blocks guarantees the re-forwarded region covers the full
    # ring, so the last prompt token reads its entire window from extend KV.
    warmup_blocks = math.ceil(win_with_spec / ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    if warmup_blocks <= 0:
        return

    original = KVCacheManager.get_computed_blocks
    if getattr(original, "_atom_v4_prefix_swa_patched", False):
        return

    @functools.wraps(original)
    def wrapped_get_computed_blocks(self, request):
        computed_blocks, num_computed_tokens, shared_prefix_boundary = original(
            self, request
        )
        return _drop_swa_warmup_blocks(
            self,
            computed_blocks,
            num_computed_tokens,
            shared_prefix_boundary,
            warmup_blocks=warmup_blocks,
            block_size=ATOM_DEEPSEEK_V4_BLOCK_SIZE,
        )

    wrapped_get_computed_blocks._atom_v4_prefix_swa_patched = True
    KVCacheManager.get_computed_blocks = wrapped_get_computed_blocks
    logger.info(
        "ATOM DeepSeek-V4: prefix caching enabled with SWA recompute "
        "(drop last %d cached block(s) per hit, win_with_spec=%d, block_size=%d).",
        warmup_blocks,
        win_with_spec,
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )
