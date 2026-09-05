"""ATOM DeepSeek-V4 vLLM prefix-cache tail-recompute patch.

V4's sliding-window state is a per-request ring in the ATOM proxy arena, not
keyed by a vLLM block, so vLLM's block-level prefix cache never carries it. On a
cross-request hit the new request gets an empty ring, and a tail token whose
window reaches into the cached region reads stale data.

Fix (mirrors native ATOM scheduler "fix B'"): roll every hit back by
``max(win_with_spec, index_topk)`` tokens so that tail is re-forwarded,
repopulating the ring. Compressed-KV reuse is unaffected -- ``n_committed =
context_len // ratio`` is invariant under the shift.

KNOWN ISSUE: the ``index_topk`` term is empirical (512 on V4-Flash vs a
128-token window) and the same floor appears with prefix caching off, so the
underlying defect is in the sparse indexer whenever fewer than ``index_topk``
rows are forwarded into a request's own state. This rollback is a workaround.
"""

import functools
import logging
import math

logger = logging.getLogger("atom")


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


def _v4_sliding_window(vllm_config) -> int:
    hf = vllm_config.model_config.hf_config
    return int(getattr(hf, "sliding_window", 128) or 128)


def _group_block_sizes(manager):
    """Real block size per KV cache group, in ``KVCacheBlocks.blocks`` order.

    Returns [] on an unexpected manager shape; the caller then falls back to the
    proxy block size.
    """
    try:
        return [m.block_size for m in manager.coordinator.single_type_managers]
    except AttributeError:
        return []


def _roll_back_prefix_hit(
    manager,
    computed_blocks,
    num_computed_tokens: int,
    shared_prefix_boundary: int,
    *,
    rollback_tokens: int,
):
    """Shorten a prefix hit by ``rollback_tokens`` so the tail is re-forwarded,
    repopulating the SWA ring and the sparse indexer's rows. Deep-prefix blocks
    are still reused.

    The rollback is expressed in **tokens**, not blocks, and converted per group
    using that group's own block size: DSpark adds a second group at block 64
    alongside the proxy's 128, and a shared block count would leave one group
    holding blocks past its declared ``num_computed_tokens``.
    """
    # Local import: deepseek_v4_bridge imports back into this package.
    from atom.plugin.vllm.deepseek_v4_bridge import ATOM_DEEPSEEK_V4_BLOCK_SIZE

    if num_computed_tokens <= 0 or rollback_tokens <= 0:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    # rollback_tokens is a multiple of every group's block size and
    # num_computed_tokens is block-aligned, so the difference stays aligned.
    new_num_computed_tokens = max(0, num_computed_tokens - rollback_tokens)

    block_sizes = _group_block_sizes(manager)
    groups = list(computed_blocks.blocks)
    new_groups = []
    dropped_any = False
    for idx, group in enumerate(groups):
        block_list = list(group)
        block_size = (
            block_sizes[idx] if idx < len(block_sizes) else ATOM_DEEPSEEK_V4_BLOCK_SIZE
        )
        keep = min(len(block_list), new_num_computed_tokens // block_size)
        if keep != len(block_list):
            dropped_any = True
        new_groups.append(block_list[:keep])
    if not dropped_any:
        return computed_blocks, num_computed_tokens, shared_prefix_boundary

    new_blocks = manager.create_kv_cache_blocks(tuple(new_groups))
    return new_blocks, new_num_computed_tokens, shared_prefix_boundary


def apply_vllm_v4_prefix_recompute_patch(vllm_config) -> None:
    """Enable DeepSeek-V4 prefix caching by recomputing the tail of every hit.

    Call only for a DeepSeek-V4 deployment with prefix caching enabled. The
    rollback length is derived once from ``vllm_config`` and captured in the
    wrapper closure, so non-V4 deployments (which never install this patch) are
    unaffected.
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
    # The ring is not the only state a hit fails to carry: leaving fewer than
    # `index_topk` freshly-forwarded tokens makes the sparse indexer emit another
    # request's content. Empirical on V4-Flash-0731 (TP4, greedy, long shared
    # prefix): 128/256/384 -> 1/6 correct, 512 -> 6/6, and 512 holds at 2K/6K/17K
    # prefixes, so it is a constant. Keep both terms.
    index_topk = int(getattr(vllm_config.model_config.hf_config, "index_topk", 0) or 0)
    rollback_tokens = max(win_with_spec, index_topk)
    # Round up to a whole proxy block: the rollback must be a multiple of every
    # group's block size, and the proxy's 128 is the coarsest in play (and a
    # multiple of the DSpark draft group's 64).
    rollback_blocks = math.ceil(rollback_tokens / ATOM_DEEPSEEK_V4_BLOCK_SIZE)
    if rollback_blocks <= 0:
        return
    rollback_tokens = rollback_blocks * ATOM_DEEPSEEK_V4_BLOCK_SIZE

    original = KVCacheManager.get_computed_blocks
    if getattr(original, "_atom_v4_prefix_recompute_patched", False):
        return

    @functools.wraps(original)
    def wrapped_get_computed_blocks(self, request):
        computed_blocks, num_computed_tokens, shared_prefix_boundary = original(
            self, request
        )
        return _roll_back_prefix_hit(
            self,
            computed_blocks,
            num_computed_tokens,
            shared_prefix_boundary,
            rollback_tokens=rollback_tokens,
        )

    wrapped_get_computed_blocks._atom_v4_prefix_recompute_patched = True
    KVCacheManager.get_computed_blocks = wrapped_get_computed_blocks
    logger.info(
        "ATOM DeepSeek-V4: prefix caching enabled with SWA recompute "
        "(roll back last %d token(s) per hit = %d proxy block(s), "
        "win_with_spec=%d, index_topk=%d).",
        rollback_tokens,
        rollback_blocks,
        win_with_spec,
        index_topk,
    )
