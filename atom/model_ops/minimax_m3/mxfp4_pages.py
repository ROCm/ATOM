# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 page geometry of MiniMax-M3's sparse KV and index caches
(``--kv-cache-dtype fp4`` / ``--index-cache-dtype fp4``).

One 128-token page of one head holds, for K (or V, or the index key), the
128-wide token rows as e2m1 data (two elements a byte, the low nibble the even
element) and one e8m0 scale per 32 elements::

    [128 tokens x 64 B data][128 tokens x 4 B scale]      = 8704 B

This is the page the aiter writer (``fused_qknorm_idxrqknorm`` with
``kv_cache_dtype="mxfp4"``) produces and ``pa_decode`` (MXFP4 K/V) /
``score_flydsl_fp4`` read.
"""

from dataclasses import dataclass

import torch

from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.minimax_m3.geometry import SPARSE_BLOCK_SIZE

HEAD_DIM = 128
DATA_BYTES = HEAD_DIM // 2
SCALE_BYTES = HEAD_DIM // 32
DATA_PAGE_BYTES = SPARSE_BLOCK_SIZE * DATA_BYTES
PAGE_BYTES = SPARSE_BLOCK_SIZE * (DATA_BYTES + SCALE_BYTES)


def page_views(pages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(data, scale) views of uint8 pages ``[..., PAGE_BYTES]``.

    data ``[..., 128, 64]`` and scale ``[..., 128, 4]`` keep the leading dims
    and their strides, so a ``[pages, heads, 8704]`` pool field becomes the
    ``[pages, heads, 128, 64]`` / ``[pages, heads, 128, 4]`` pair the kernels
    take, and a ``[pages, 8704]`` index cache the ``[pages, 128, 64]`` pair.
    """
    if pages.dtype != torch.uint8 or pages.shape[-1] != PAGE_BYTES:
        raise ValueError(
            f"MXFP4 pages are uint8 [..., {PAGE_BYTES}], got {pages.dtype} "
            f"{tuple(pages.shape)}"
        )
    if pages.stride(-1) != 1:
        raise ValueError("MXFP4 pages need a contiguous byte axis")
    lead, lead_strides = tuple(pages.shape[:-1]), tuple(pages.stride()[:-1])
    off = pages.storage_offset()
    data = pages.as_strided(
        (*lead, SPARSE_BLOCK_SIZE, DATA_BYTES), (*lead_strides, DATA_BYTES, 1), off
    )
    scale = pages.as_strided(
        (*lead, SPARSE_BLOCK_SIZE, SCALE_BYTES),
        (*lead_strides, SCALE_BYTES, 1),
        off + DATA_PAGE_BYTES,
    )
    return data, scale


@dataclass(frozen=True)
class Mxfp4KvLayout:
    """MiniMax-M3's sparse layers' K/V under ``--kv-cache-dtype fp4``, as an
    ``MhaKvPool`` layout: per (page, kv head) one 8704 B field of e2m1 data and e8m0
    scales, K and V a field each, no scale planes."""

    tag: str = "fp4"

    def check(self, block_size: int, head_dim: int, blocks_per_entry: int) -> None:
        if (
            block_size != SPARSE_BLOCK_SIZE
            or head_dim != HEAD_DIM
            or blocks_per_entry != 1
        ):
            raise ValueError(
                f"an MXFP4 page is {SPARSE_BLOCK_SIZE} tokens of head_dim {HEAD_DIM}, "
                f"one block an entry; got block_size {block_size}, head_dim "
                f"{head_dim}, {blocks_per_entry} blocks an entry"
            )

    def cache_fields(self, layers, entry_tokens, num_kv_heads, head_dim):
        per_entry = num_kv_heads * PAGE_BYTES
        return [
            EntryField("k", layers, (per_entry,), torch.uint8),
            EntryField("v", layers, (per_entry,), torch.uint8),
        ]

    def scale_fields(self, layers, entry_tokens, num_kv_heads):
        return []

    def kv_views(self, pool, layer):
        shape = (-1, pool.num_kv_heads, PAGE_BYTES)
        return (
            pool.field_view("k", layer, torch.uint8, shape),
            pool.field_view("v", layer, torch.uint8, shape),
        )
