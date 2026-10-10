# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode-side scatter of landed MLA rows into the paged KV cache.

A DCP producer packs the rows a decode rank owns into a landing slot in rank
order (see ``MooncakeConnector._execute_mla_regions``). Row ``j`` of
consumer region ``c`` belongs at token ``j % block_size`` of destination block
``dst_block_ids[j // block_size]``, so consecutive rows fill whole destination
pages. The scatter copies one segment -- a run of rows that is contiguous at
the destination, at most one page -- per 1-warp program. That page-granular
copy is about 2x faster than ``index_copy_`` and, because each program is short
and light on registers, slows concurrent decode kernels about 3x less.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# 256 int32 = 1 KiB per load/store iteration, one warp per segment.
_COPY_BLOCK = 256


@dataclass(frozen=True)
class DevicePointer:
    """A raw device address Triton accepts as an int32 pointer argument.

    Destination regions live in different tensors (the draft layer may sit in
    its own allocation), so the kernel addresses them as int32 offsets from one
    base: the lowest destination region address.
    """

    address: int
    device: torch.device
    dtype: torch.dtype = torch.int32

    def data_ptr(self) -> int:
        return self.address


try:  # Triton is only needed on GPU; unit tests may import without it.
    import triton
    import triton.language as tl

    @triton.jit
    def _segment_copy_kernel(
        src,
        dst,
        seg_src,
        seg_dst,
        seg_len,
        BLOCK: tl.constexpr,
    ):
        """Copy segment ``pid``: ``seg_len`` int32 words, ``seg_src -> seg_dst``.

        Landing slots are rewritten by the NIC between scatters, so loads skip
        the cache (``.cv``) instead of trusting lines a previous scatter of
        the same slot may have left behind.
        """
        s = tl.program_id(0)
        so = tl.load(seg_src + s)
        do = tl.load(seg_dst + s)
        n = tl.load(seg_len + s)
        offs = tl.arange(0, BLOCK)
        for off in range(0, n, BLOCK):
            o = off + offs
            m = o < n
            v = tl.load(src + so + o, mask=m, cache_modifier=".cv")
            tl.store(dst + do + o, v, mask=m, cache_modifier=".cs")

except ImportError:  # pragma: no cover - exercised only without Triton
    triton = None
    _segment_copy_kernel = None


def landing_segments(
    row_start: np.ndarray,
    row_count: np.ndarray,
    src_offset: np.ndarray,
    region_base: np.ndarray,
    block_bytes: np.ndarray,
    dst_block_ids: np.ndarray,
    block_size: int,
    dst_origin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Byte segments for the landed items of one request, vectorized.

    Item ``i`` holds rows ``[row_start[i], row_start[i] + row_count[i])`` of a
    region whose blocks are ``block_bytes[i]`` wide at ``region_base[i]``; the
    rows sit contiguously at byte ``src_offset[i]`` of the landing pool. Row
    ``j`` belongs at token ``j % block_size`` of block
    ``dst_block_ids[j // block_size]``. Returns ``(src, dst, nbytes)``, with
    ``dst`` relative to ``dst_origin``; a segment never crosses a destination
    page, because consecutive pages are not adjacent in general.
    """

    row_start = np.asarray(row_start, dtype=np.int64)
    row_count = np.asarray(row_count, dtype=np.int64)
    keep = row_count > 0
    if not keep.all():
        row_start, row_count = row_start[keep], row_count[keep]
        src_offset = np.asarray(src_offset, dtype=np.int64)[keep]
        region_base = np.asarray(region_base, dtype=np.int64)[keep]
        block_bytes = np.asarray(block_bytes, dtype=np.int64)[keep]
    if row_start.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    first_page = row_start // block_size
    row_stop = row_start + row_count
    pages_per_item = (row_stop - 1) // block_size - first_page + 1
    item = np.repeat(np.arange(row_start.size), pages_per_item)
    item_start = np.cumsum(pages_per_item) - pages_per_item
    page = first_page[item] + np.arange(item.size) - item_start[item]
    seg_row0 = np.maximum(page * block_size, row_start[item])
    seg_row1 = np.minimum((page + 1) * block_size, row_stop[item])
    token_bytes = np.asarray(block_bytes, dtype=np.int64)[item] // block_size
    src = (
        np.asarray(src_offset, dtype=np.int64)[item]
        + (seg_row0 - row_start[item]) * token_bytes
    )
    dst = (
        np.asarray(region_base, dtype=np.int64)[item]
        - dst_origin
        + np.asarray(dst_block_ids, dtype=np.int64)[page]
        * np.asarray(block_bytes, dtype=np.int64)[item]
        + (seg_row0 - page * block_size) * token_bytes
    )
    return src, dst, (seg_row1 - seg_row0) * token_bytes


def segment_table_capacity(num_segments: int) -> int:
    """Columns for a table of ``num_segments``, kept a multiple of 16.

    Each column of the ``[3, capacity]`` int64 table then starts 16-byte
    aligned, so every launch hits the same Triton specialization as warmup.
    """

    return -(-max(1, num_segments) // 16) * 16


def segment_table(
    src: np.ndarray, dst: np.ndarray, nbytes: np.ndarray, out: np.ndarray
) -> int:
    """Pack byte segments into ``out`` as int32-word offsets; returns the count.

    ``out`` is an int64 ``[3, capacity]`` array: sources, destinations and
    lengths in its rows, as ``scatter_segments`` reads them.
    """

    n = src.size
    if (src % 4).any() or (dst % 4).any() or (nbytes % 4).any():
        raise ValueError("Landing scatter segments must be 4-byte aligned")
    if out.shape[1] < n:
        raise ValueError(f"Segment table holds {out.shape[1]} segments, needs {n}")
    out[0, :n] = src // 4
    out[1, :n] = dst // 4
    out[2, :n] = nbytes // 4
    return n


def scatter_segments(
    landing: torch.Tensor,
    dst_origin: DevicePointer,
    table: torch.Tensor,
    num_segments: int,
) -> None:
    """Launch the segment copy on the current stream.

    ``landing`` is the uint8 landing pool; ``table`` is the int64 device
    ``[3, capacity]`` table ``segment_table`` fills, for ``num_segments``.
    """

    if num_segments == 0:
        return
    _segment_copy_kernel[(num_segments,)](
        landing.view(torch.int32),
        dst_origin,
        table[0],
        table[1],
        table[2],
        BLOCK=_COPY_BLOCK,
        num_warps=1,
    )
