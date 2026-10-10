# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MLA bind point drops 0.29's head slot, and only that.

vLLM 0.29 views each layer's cache as ``[B, H, N, C]`` (RFC #42082) where 0.28
handed MLA a three-dimensional page, and upstream absorbs the difference by
squeezing ``H`` at the bind point. ATOM's MLA layer inherited
``AttentionLayerBase``'s default, which binds the view as-is, so the extra axis
reached consumers that assert on rank or read ``size(1)`` as the block size.
``AttentionForVllmMLA.bind_kv_cache`` restores upstream's contract.

What is tested here is the *guard*, not the squeeze. Upstream squeezes
unconditionally; this layer cannot, because it can also be handed a 0.28-shaped
``[B, N, C]`` page whose ``N`` is 1 -- MLA's kernel block size -- and an
unconditional squeeze would eat the block dimension instead of a head slot and
silently reshape the page. That collision is the reason the guard exists and is
the case a reader is most likely to "simplify" away, so it gets its own test.
The third case, more than one head slot, is left for the consumer to interpret
rather than quietly flattened.

None of this needs a GPU to be true, but importing the layer pulls in vLLM,
which does.
"""

import pytest
import torch

try:  # `importorskip` only catches ImportError, and vLLM's platform probe
    import vllm  # noqa: F401  # raises RuntimeError on a host with no GPU.
except (ImportError, RuntimeError) as exc:
    pytest.skip(f"vLLM is not importable here: {exc}", allow_module_level=True)

from atom.plugin.vllm.attention.layer_mla import AttentionForVllmMLA

NUM_BLOCKS, BLOCK_SIZE, ENTRY = 4, 16, 8


def _bind(kv_cache: torch.Tensor) -> torch.Tensor:
    """Run only the bind point, with no engine underneath it.

    ``__init__`` builds a whole attention layer; the base this override defers
    to does nothing but ``self.kv_cache = kv_cache``, so an uninitialised
    instance is enough to observe what the override passes down.
    """
    layer = object.__new__(AttentionForVllmMLA)
    AttentionForVllmMLA.bind_kv_cache(layer, kv_cache)
    return layer.kv_cache


def test_029_head_slot_is_dropped():
    """``[B, H=1, N, C]`` binds as ``[B, N, C]``, and as the same memory."""
    page = torch.zeros(NUM_BLOCKS, 1, BLOCK_SIZE, ENTRY)
    bound = _bind(page)

    assert bound.shape == (NUM_BLOCKS, BLOCK_SIZE, ENTRY)
    # A copy would bind a page the engine does not write to.
    assert bound.data_ptr() == page.data_ptr()


def test_028_shaped_page_with_one_block_row_is_not_squeezed():
    """The collision the guard exists for: ``[B, N=1, C]`` must survive whole.

    ``N == 1`` is MLA's kernel block size, so this page is indistinguishable
    from a head slot by rank alone. Upstream's unconditional squeeze would turn
    it into ``[B, C]`` and every consumer below would read the entry dimension
    as the block dimension.
    """
    page = torch.zeros(NUM_BLOCKS, 1, ENTRY)
    bound = _bind(page)

    assert bound.shape == (NUM_BLOCKS, 1, ENTRY)
    assert bound.data_ptr() == page.data_ptr()


def test_more_than_one_head_slot_is_passed_through():
    """A future spec publishing ``H > 1`` is the consumer's to interpret."""
    page = torch.zeros(NUM_BLOCKS, 2, BLOCK_SIZE, ENTRY)
    bound = _bind(page)

    assert bound.shape == (NUM_BLOCKS, 2, BLOCK_SIZE, ENTRY)
    assert bound.data_ptr() == page.data_ptr()
