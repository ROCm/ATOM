# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""How many query heads an MLA decode hands aiter.

Kept free of triton and aiter imports so the rules are testable on CPU-only CI.
"""

_MLA_MIN_HEADS = 16  # AITER MLA kernels require at least 16 attention heads


def mla_min_query_heads(kv_cache_dtype: str, block_width: int, num_heads: int) -> int:
    """Query-head padding that lets aiter dispatch a NON-causal MLA decode.

    aiter picks its .co from (gqa, block width, causality), and on an fp8 cache
    a 2-wide non-causal block has no kernel at gqa=16: the asm dispatch hits
    AITER_CHECK and aborts the whole process. Padding to 32 instead reaches the
    4-wide non-causal kernel through aiter's own (gqa=32, width 2) remap.

    Only width 2 needs this. Widths 1, 3 and 4 have gqa=16 kernels, and aiter
    remaps anything wider onto gqa=32 itself. Padding rather than remapping
    width 2 to the 4-wide kernel at gqa=16 is deliberate: the persistent work
    descriptors are sized from the block width, so a 2-wide plan driving a
    4-wide kernel writes its partials past the reduce buffers (illegal access
    at some batch sizes, a non-terminating kernel at others).

    gfx1250 has none of those kernels: its page-size-1 decode is aiter's FlyDSL
    PS1 kernel alone, causal or not. Which head counts that kernel takes for a
    block wider than 1 depends on the aiter version: ROCm/aiter#6563 widened the
    set and added BF16 caches. 96 heads pass on every version, and so do 16
    heads up to 4 wide, so any other block wider than 1 is padded to 96.
    ``num_heads`` is the layer's own count, which decides between the two.
    """
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() == "gfx1250":
        if block_width > 1 and (num_heads > _MLA_MIN_HEADS or block_width > 4):
            return 96
        return _MLA_MIN_HEADS
    if block_width == 2 and kv_cache_dtype.startswith("fp8"):
        return 32
    return _MLA_MIN_HEADS
