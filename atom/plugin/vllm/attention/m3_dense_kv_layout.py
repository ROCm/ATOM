# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Address vLLM 0.28's K/V-separated dense KV cache with ATOM's page-16 kernels.

vLLM 0.28 (RFC #42082) hands every attention layer a 4-D ``[B, H, N, C]`` view.
Declaring ``num_head_slots=2`` puts K and V on the ``H`` axis instead of packing
them into the content dim, which is what the AITER shuffle kernels need: each
side must be its own run of whole 16-token pages.

Of the six ``KVCacheLayout`` permutations only the ones that keep ``H`` outside
``N`` preserve that, and of those only ``LBHNC`` is block-compact -- the property
vLLM demands once a model mixes page sizes (M3's key-only indexer spec does).
So ``LBHNC`` is what this module adapts:

    per-layer view   [B, 2, N, C]      C = num_kv_heads * head_size elements
    memory order     block-major, then the K plane, then the V plane

A block therefore spans ``2 * block_size / 16`` physical 16-token pages -- K in
the first half, V in the second -- rather than the ``block_size / 16`` a
dense-plane (``LHBNC``) layout would give. AITER's writer derives its page from
``slot // 16`` and assumes consecutive numbering, so the slot mapping has to be
rebased or every write after block 0 lands in the wrong page. That single factor
of two is the whole hazard this module exists to contain: it does not crash, it
silently attends to the wrong tokens.
"""

from __future__ import annotations

import torch

# AITER's shuffled KV kernels address memory in 16-token physical pages.
ASM_PAGE_SIZE = 16


def pages_per_side(block_size: int) -> int:
    """16-token pages one K (or V) plane of a block occupies."""
    if block_size <= 0 or block_size % ASM_PAGE_SIZE:
        raise ValueError(
            f"block_size must be a positive multiple of {ASM_PAGE_SIZE}, got {block_size}"
        )
    return block_size // ASM_PAGE_SIZE


def rebase_slots_to_page16(
    slot_mapping: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rebase a token slot mapping onto the packed layout's page numbering.

    Under ``LBHNC`` a block holds both sides, so it spans twice as many pages as
    the writer assumes. Only the block component of the slot doubles; the offset
    inside the block must not move::

        b * block_size + o   ->   2 * b * block_size + o

    Padding slots carry a negative sentinel that tells the writer to skip them.
    Clamping before the division keeps them negative instead of folding them
    onto a real page.
    """
    pages_per_side(block_size)  # validate
    if out is None:
        out = torch.empty_like(slot_mapping)
    torch.clamp(slot_mapping, min=0, out=out)
    return (
        out.div_(block_size, rounding_mode="floor").mul_(block_size).add_(slot_mapping)
    )


def page16_views(
    kv_cache: torch.Tensor,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reinterpret a K/V-separated ``LBHNC`` cache as AITER's page-16 K/V pair.

    Returns ``(key_view, value_view)`` sharing ``kv_cache``'s storage, shaped the
    way ``aiter``'s shuffled reader/writer expects. Both are indexed by the
    *rebased* page id (see :func:`rebase_slots_to_page16`); ``value_view`` is
    pre-shifted by one side so a single page id addresses K and V of the same
    token.

    Raises rather than returning a mis-addressed view: a wrong KV address here
    produces no error at runtime, only wrong output.
    """
    if kv_cache.dim() != 4 or kv_cache.shape[1] != 2:
        raise ValueError(
            "expected a K/V-separated [B, 2, N, C] cache (num_head_slots=2); got "
            f"shape {tuple(kv_cache.shape)}. Does the layer's KV spec declare "
            "num_head_slots=2 and state_content_bytes?"
        )
    if not kv_cache.is_contiguous():
        raise ValueError(
            "the K/V-separated cache is not contiguous, so its blocks are not "
            "whole pages. The resolved KV cache layout must be LBHNC; check the "
            "'Using ... KV cache layout' line in the server log."
        )

    num_blocks, _, block_size, content = kv_cache.shape
    if content != num_kv_heads * head_size:
        raise ValueError(
            f"content dim {content} != num_kv_heads * head_size "
            f"({num_kv_heads} * {head_size})"
        )
    per_side = pages_per_side(block_size)

    x = ASM_PAGE_SIZE // kv_cache.element_size()
    if head_size % x or ASM_PAGE_SIZE % x:
        raise ValueError(
            f"head_size {head_size} and page {ASM_PAGE_SIZE} must both be "
            f"multiples of {x} for a {kv_cache.element_size()}-byte cache"
        )

    total_pages = num_blocks * 2 * per_side
    key_view = kv_cache.view(
        total_pages, num_kv_heads, head_size // x, ASM_PAGE_SIZE, x
    )
    value_view = kv_cache.view(
        total_pages, num_kv_heads, ASM_PAGE_SIZE // x, head_size, x
    )[per_side:]
    return key_view, value_view


def expand_block_table_to_page16(
    block_table: torch.Tensor,
    block_size: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Restate a manager-block table in the packed layout's page-16 ids.

    ``page16_views`` hands the kernels a cache whose block dim is a 16-token
    page, so their block table has to be per page, not per manager block. Under
    the packed layout manager block ``b`` owns the ``pages_per_side`` K pages
    starting at ``2 * b * pages_per_side``::

        [b0, b1, ...]  ->  [16*b0, ..., 16*b0+7, 16*b1, ..., 16*b1+7]   (page 128)

    The V plane needs no separate table: ``page16_views`` pre-shifts its view by
    one side, so the same page id reaches both halves of the same block.
    """
    per_side = pages_per_side(block_size)
    if block_table.dim() != 2:
        raise ValueError(
            f"expected a [num_reqs, num_blocks] table, got {block_table.dim()}-D"
        )
    num_reqs, num_blocks = block_table.shape

    offsets = torch.arange(per_side, device=block_table.device, dtype=block_table.dtype)
    expanded = block_table.unsqueeze(-1) * (2 * per_side) + offsets
    if out is None:
        return expanded.reshape(num_reqs, num_blocks * per_side)
    if out.shape != (num_reqs, num_blocks * per_side):
        raise ValueError(
            f"out has shape {tuple(out.shape)}, expected "
            f"{(num_reqs, num_blocks * per_side)}"
        )
    return out.copy_(expanded.reshape(num_reqs, num_blocks * per_side))
