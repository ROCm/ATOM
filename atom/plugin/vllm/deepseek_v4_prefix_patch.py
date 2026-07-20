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


def _v4_sliding_window(vllm_config) -> int:
    hf = vllm_config.model_config.hf_config
    return int(getattr(hf, "sliding_window", 128) or 128)


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
        computed_blocks, num_computed_tokens = original(self, request)
        if num_computed_tokens <= 0:
            return computed_blocks, num_computed_tokens

        # Drop the trailing warmup blocks from every KV cache group (V4 runs a
        # single proxy group). vLLM allocates fresh blocks for the dropped
        # tail and re-forwards those tokens, repopulating the SWA ring; the
        # deep-prefix blocks are still reused.
        dropped = 0
        new_groups = []
        for group in computed_blocks.blocks:
            block_list = list(group)
            keep = max(0, len(block_list) - warmup_blocks)
            dropped = max(dropped, len(block_list) - keep)
            new_groups.append(block_list[:keep])
        if dropped == 0:
            return computed_blocks, num_computed_tokens

        new_num_computed_tokens = max(
            0, num_computed_tokens - dropped * ATOM_DEEPSEEK_V4_BLOCK_SIZE
        )
        new_blocks = self.create_kv_cache_blocks(tuple(new_groups))
        return new_blocks, new_num_computed_tokens

    wrapped_get_computed_blocks._atom_v4_prefix_swa_patched = True
    KVCacheManager.get_computed_blocks = wrapped_get_computed_blocks
    logger.info(
        "ATOM DeepSeek-V4: prefix caching enabled with SWA recompute "
        "(drop last %d cached block(s) per hit, win_with_spec=%d, block_size=%d).",
        warmup_blocks,
        win_with_spec,
        ATOM_DEEPSEEK_V4_BLOCK_SIZE,
    )
