# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 page geometry of MiniMax-M3's sparse KV and index caches
(``--kv-cache-dtype nvfp4`` / ``--index-cache-dtype nvfp4``).

NVFP4 here is e2m1 data with one e4m3 scale per 16 elements, and (K, V only) one
fp32 *global* scale per layer that lives outside the page. A page is one buffer
holding data and scales, 128 tokens of head_dim 128.

Index page, one per block (9216 B)::

    [128 tokens x 64 B e2m1 data][128 tokens x 8 B e4m3 scales]

token-major, the low nibble the even element; what aiter's
``score_flydsl_nvfp4`` / ``score_prefill_flydsl_nvfp4`` read as the two views of
:func:`index_page_views`.

KV page, one per block, all kv heads of K then all of V (``2 * Hkv * 9216`` B)::

    [K: Hkv x 8192 B data | Hkv x 1024 B scales][V: same]

K scales and V data / scales are in the order the sparse-PA kernel's MFMA lanes
want them. aiter owns this layout (the writer, ``fused_qknorm_idxrqknorm`` in its NVFP4
mode, and the readers ``minimax_m3_sparse_pa_nvfp4_{decode,prefill}``); the offsets
below are what the tests' independent reader uses, and the sizes are what the pool and
the index cache allocate.
"""

from dataclasses import dataclass

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import EntryField

PAGE_TOKENS = 128
HEAD_DIM = 128
GROUP = 16  # elements one e4m3 scale covers
GROUPS = HEAD_DIM // GROUP
DATA_BYTES = HEAD_DIM // 2  # per token
SCALE_BYTES = GROUPS  # per token, index page
INDEX_DATA_PAGE_BYTES = PAGE_TOKENS * DATA_BYTES
INDEX_PAGE_BYTES = PAGE_TOKENS * (DATA_BYTES + SCALE_BYTES)  # 9216
HEAD_DATA_BYTES = PAGE_TOKENS * DATA_BYTES  # one head of one side, 8192
HEAD_SCALE_BYTES = PAGE_TOKENS * SCALE_BYTES  # 1024
HEAD_SIDE_BYTES = HEAD_DATA_BYTES + HEAD_SCALE_BYTES  # 9216 == INDEX_PAGE_BYTES


def kv_page_bytes(num_kv_heads: int) -> int:
    """Bytes of one KV page: K and V of every kv head."""
    return 2 * num_kv_heads * HEAD_SIDE_BYTES


def index_page_views(pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(data, scale) views of uint8 index pages ``[..., INDEX_PAGE_BYTES]``.

    data ``[..., 128, 64]`` and scale ``[..., 128, 8]`` keep the leading dims and
    their strides: the two tensors ``score_flydsl_nvfp4`` takes, each with its
    own page stride.
    """
    if pages.dtype != torch.uint8 or pages.shape[-1] != INDEX_PAGE_BYTES:
        raise ValueError(
            f"NVFP4 index pages are uint8 [..., {INDEX_PAGE_BYTES}], got "
            f"{pages.dtype} {tuple(pages.shape)}"
        )
    if pages.stride(-1) != 1:
        raise ValueError("NVFP4 pages need a contiguous byte axis")
    lead, lead_strides = tuple(pages.shape[:-1]), tuple(pages.stride()[:-1])
    off = pages.storage_offset()
    data = pages.as_strided(
        (*lead, PAGE_TOKENS, DATA_BYTES), (*lead_strides, DATA_BYTES, 1), off
    )
    scale = pages.as_strided(
        (*lead, PAGE_TOKENS, SCALE_BYTES),
        (*lead_strides, SCALE_BYTES, 1),
        off + INDEX_DATA_PAGE_BYTES,
    )
    return data, scale


# Byte offsets inside one head's region of a KV page side, token t of the page,
# element d (scale group g = d // 16). r = t % 32, n = 4 * (r // 8) + r % 4.
#   K data   t * 64 + d // 2                  (low nibble: even d)
#   K scale  ((((t // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2
#   V data   ((t // 8) * 128 + d) * 4 + (t % 8) // 2   (nibble t % 2)
#   V scale  g * 128 + t
# plus, per head h of side s (0 = K, 1 = V): data region s * Hkv * 9216 + h * 8192,
# scale region s * Hkv * 9216 + Hkv * 8192 + h * 1024.
def head_regions(side: int, head: int, num_kv_heads: int) -> tuple[int, int]:
    """Page byte offsets of one head's data and scale regions of side 0 (K) / 1 (V)."""
    base = side * num_kv_heads * HEAD_SIDE_BYTES
    return (
        base + head * HEAD_DATA_BYTES,
        base + num_kv_heads * HEAD_DATA_BYTES + head * HEAD_SCALE_BYTES,
    )


@dataclass(frozen=True)
class Nvfp4KvLayout:
    """MiniMax-M3's sparse layers' K/V under ``--kv-cache-dtype nvfp4``, as an
    ``MhaKvPool`` layout: per page ONE field of ``2 * Hkv * 9216`` B holding K and V
    (the sparse-PA kernels address both sides off one page stride), no scale planes.
    ``kv_views`` returns that field twice, as (k, v): both name the same pages."""

    tag: str = "nvfp4"

    def check(self, block_size: int, head_dim: int, blocks_per_entry: int) -> None:
        if block_size != PAGE_TOKENS or head_dim != HEAD_DIM or blocks_per_entry != 1:
            raise ValueError(
                f"an NVFP4 KV page is {PAGE_TOKENS} tokens of head_dim {HEAD_DIM}, "
                f"one block an entry; got block_size {block_size}, head_dim "
                f"{head_dim}, {blocks_per_entry} blocks an entry"
            )

    def cache_fields(self, layers, entry_tokens, num_kv_heads, head_dim):
        return [EntryField("kv", layers, (kv_page_bytes(num_kv_heads),), torch.uint8)]

    def scale_fields(self, layers, entry_tokens, num_kv_heads):
        return []

    def kv_views(self, pool, layer):
        pages = pool.field_view(
            "kv", layer, torch.uint8, (-1, kv_page_bytes(pool.num_kv_heads))
        )
        return pages, pages
