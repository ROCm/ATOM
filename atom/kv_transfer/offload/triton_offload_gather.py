# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton byte gather/scatter for ATOM KV offload — one module, two layouts.

Both offload backends move paged GPU KV bytes into flat staging buffers via a
masked tiled copy. They differ only in how source blocks map to output offsets:

* **Region layout** (``gather_regions`` / ``scatter_regions``) — the general
  primitive. Each :class:`GatherRegion` is one source region (base addr +
  per-block ``unit_bytes``) with its own physical-id list and a destination byte
  offset; blocks land contiguously. Used by the DSV4 offload unit, whose
  compressed / SWA / CSA-state components have different strides and id lists.

* **Chunk-major layout** (``fused_pack_chunk_major`` / ``fused_unpack_chunk_major``)
  — an on-device-indexed *specialization* for the token-chunked MHA/MLA path,
  where every segment shares ONE ``block_ids`` list and the output is grouped
  ``[chunk0: seg0 blocks | seg1 blocks | ...]``. It computes the block lookup
  inside the kernel (no per-item host list), which is cheaper for the
  many-small-blocks-per-chunk case, so it is kept rather than re-expressed as
  regions.

Both share ``_BLOCK_BYTES``, the Triton guard, and the int64 plan helper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    _HAVE_TRITON = True
except Exception:  # pragma: no cover - triton always present in ATOM images
    _HAVE_TRITON = False

_BLOCK_BYTES = 1024


def _device_i64(values: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(list(values), dtype=torch.int64, device=device)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# =====================================================================
# Region layout (general primitive)
# =====================================================================
@dataclass(frozen=True)
class GatherRegion:
    """One source region and where its blocks land in the destination buffer.

    ``base_addr`` is a raw device address (``tensor.data_ptr()`` + region offset).
    Block ``i`` in ``physical_ids`` is copied between
    ``[base_addr + id*unit_bytes, +unit_bytes)`` and buffer
    ``[dst_offset + i*unit_bytes, +unit_bytes)``. Negative ids (window-freed SWA
    slots) are skipped.
    """

    base_addr: int
    unit_bytes: int
    physical_ids: Sequence[int]
    dst_offset: int


def _plan_regions(
    regions: Sequence[GatherRegion], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Flatten regions into per-item (region_ptr, buf_off, nbytes) int64 arrays."""
    region_ptrs: list[int] = []
    buf_offs: list[int] = []
    nbytes: list[int] = []
    max_nbytes = 0
    for r in regions:
        ub = int(r.unit_bytes)
        if ub <= 0:
            raise ValueError(f"GatherRegion.unit_bytes must be > 0, got {ub}")
        for i, phys in enumerate(r.physical_ids):
            phys = int(phys)
            if phys < 0:
                continue  # window-freed / absent slot
            region_ptrs.append(int(r.base_addr) + phys * ub)
            buf_offs.append(int(r.dst_offset) + i * ub)
            nbytes.append(ub)
            if ub > max_nbytes:
                max_nbytes = ub
    if not region_ptrs:
        empty = torch.empty(0, dtype=torch.int64, device=device)
        return empty, empty, empty, 0
    return (
        _device_i64(region_ptrs, device),
        _device_i64(buf_offs, device),
        _device_i64(nbytes, device),
        int(max_nbytes),
    )


if _HAVE_TRITON:

    @triton.jit
    def _gather_kernel(buf, region_ptrs, buf_offs, item_nbytes, BLOCK_BYTES: tl.constexpr):
        item = tl.program_id(0)
        tile = tl.program_id(1)
        nbytes = tl.load(item_nbytes + item).to(tl.int64)
        offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
        mask = offsets < nbytes
        src = (tl.load(region_ptrs + item).to(tl.int64) + offsets).to(
            tl.pointer_type(tl.uint8)
        )
        dst = buf + tl.load(buf_offs + item).to(tl.int64) + offsets
        tl.store(dst, tl.load(src, mask=mask), mask=mask)

    @triton.jit
    def _scatter_kernel(buf, region_ptrs, buf_offs, item_nbytes, BLOCK_BYTES: tl.constexpr):
        item = tl.program_id(0)
        tile = tl.program_id(1)
        nbytes = tl.load(item_nbytes + item).to(tl.int64)
        offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
        mask = offsets < nbytes
        src = buf + tl.load(buf_offs + item).to(tl.int64) + offsets
        dst = (tl.load(region_ptrs + item).to(tl.int64) + offsets).to(
            tl.pointer_type(tl.uint8)
        )
        tl.store(dst, tl.load(src, mask=mask), mask=mask)


def _validate_buf(buf: torch.Tensor) -> None:
    if not buf.is_cuda:
        raise ValueError("offload gather requires a CUDA/HIP buffer")
    if buf.dtype != torch.uint8:
        raise TypeError("offload gather buffer must be uint8")
    if not buf.is_contiguous():
        raise ValueError("offload gather buffer must be contiguous")


def gather_regions(
    buf: torch.Tensor,
    regions: Sequence[GatherRegion],
    *,
    stream: "torch.cuda.Stream | None" = None,
) -> None:
    """Copy KV source regions -> destination ``buf`` (save direction)."""
    _validate_buf(buf)
    if not _HAVE_TRITON:
        raise RuntimeError("offload gather requires Triton")
    region_ptrs, buf_offs, nbytes, max_nbytes = _plan_regions(regions, buf.device)
    n_items = int(region_ptrs.numel())
    if n_items == 0:
        return
    grid = (n_items, triton.cdiv(max_nbytes, _BLOCK_BYTES))
    ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
    with ctx:
        _gather_kernel[grid](
            buf, region_ptrs, buf_offs, nbytes, BLOCK_BYTES=_BLOCK_BYTES, num_warps=8
        )


def scatter_regions(
    buf: torch.Tensor,
    regions: Sequence[GatherRegion],
    *,
    stream: "torch.cuda.Stream | None" = None,
) -> None:
    """Copy source ``buf`` -> KV destination regions (load direction)."""
    _validate_buf(buf)
    if not _HAVE_TRITON:
        raise RuntimeError("offload gather requires Triton")
    region_ptrs, buf_offs, nbytes, max_nbytes = _plan_regions(regions, buf.device)
    n_items = int(region_ptrs.numel())
    if n_items == 0:
        return
    grid = (n_items, triton.cdiv(max_nbytes, _BLOCK_BYTES))
    ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
    with ctx:
        _scatter_kernel[grid](
            buf, region_ptrs, buf_offs, nbytes, BLOCK_BYTES=_BLOCK_BYTES, num_warps=8
        )


# =====================================================================
# Chunk-major layout (on-device-indexed specialization for MHA/MLA)
# =====================================================================
if _HAVE_TRITON:

    @triton.jit
    def _pack_chunk_major_kernel(
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        NUM_SEGMENTS: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        job = tl.program_id(0)
        tile = tl.program_id(1)
        chunk_id = job // NUM_SEGMENTS
        seg_id = job - chunk_id * NUM_SEGMENTS

        nblocks = tl.load(chunk_block_counts + chunk_id).to(tl.int64)
        seg_bytes = tl.load(segment_block_bytes + seg_id).to(tl.int64)
        nbytes = nblocks * seg_bytes
        offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
        mask = offsets < nbytes

        local_block = offsets // seg_bytes
        byte_in_block = offsets - local_block * seg_bytes
        block_offset = tl.load(chunk_block_offsets + chunk_id).to(tl.int64)
        physical_block = tl.load(
            block_ids + block_offset + local_block, mask=mask, other=0
        ).to(tl.int64)

        seg_addr = tl.load(segment_ptrs + seg_id)
        src = (seg_addr + physical_block * seg_bytes + byte_in_block).to(
            tl.pointer_type(tl.uint8)
        )
        dst = (
            device_buf
            + tl.load(chunk_output_bases + chunk_id).to(tl.int64)
            + tl.load(segment_prefix_bytes + seg_id).to(tl.int64) * nblocks
            + offsets
        )
        tl.store(dst, tl.load(src, mask=mask), mask=mask)

    @triton.jit
    def _unpack_chunk_major_kernel(
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        NUM_SEGMENTS: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
    ):
        job = tl.program_id(0)
        tile = tl.program_id(1)
        chunk_id = job // NUM_SEGMENTS
        seg_id = job - chunk_id * NUM_SEGMENTS

        nblocks = tl.load(chunk_block_counts + chunk_id).to(tl.int64)
        seg_bytes = tl.load(segment_block_bytes + seg_id).to(tl.int64)
        nbytes = nblocks * seg_bytes
        offsets = tile.to(tl.int64) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES).to(tl.int64)
        mask = offsets < nbytes

        local_block = offsets // seg_bytes
        byte_in_block = offsets - local_block * seg_bytes
        block_offset = tl.load(chunk_block_offsets + chunk_id).to(tl.int64)
        physical_block = tl.load(
            block_ids + block_offset + local_block, mask=mask, other=0
        ).to(tl.int64)

        src = (
            device_buf
            + tl.load(chunk_output_bases + chunk_id).to(tl.int64)
            + tl.load(segment_prefix_bytes + seg_id).to(tl.int64) * nblocks
            + offsets
        )
        seg_addr = tl.load(segment_ptrs + seg_id)
        dst = (seg_addr + physical_block * seg_bytes + byte_in_block).to(
            tl.pointer_type(tl.uint8)
        )
        tl.store(dst, tl.load(src, mask=mask), mask=mask)


def _build_chunk_major_meta(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    if not device_buf.is_cuda:
        raise ValueError("device_buf must be a CUDA/HIP tensor")
    if device_buf.dtype != torch.uint8:
        raise TypeError("device_buf must be uint8")
    if not device_buf.is_contiguous():
        raise ValueError("device_buf must be contiguous")
    if len(segment_tensors) != len(segment_block_bytes):
        raise ValueError("segment_tensors and segment_block_bytes size mismatch")
    if not segment_tensors:
        raise ValueError("at least one segment is required")

    device = device_buf.device
    segment_ptr_values: list[int] = []
    segment_prefix_values: list[int] = []
    bytes_per_block = 0
    for seg, nb in zip(segment_tensors, segment_block_bytes, strict=True):
        if not seg.is_cuda:
            raise ValueError("segment tensor must be CUDA/HIP")
        if seg.device != device:
            raise ValueError("segment/device mismatch")
        if not seg.is_contiguous():
            raise ValueError("segment tensor must be contiguous")
        nb = int(nb)
        if nb <= 0:
            raise ValueError("segment block bytes must be > 0")
        segment_ptr_values.append(int(seg.data_ptr()))
        segment_prefix_values.append(bytes_per_block)
        bytes_per_block += nb

    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    max_tile_nbytes = 0
    max_seg_bytes = max(int(nb) for nb in segment_block_bytes)
    for nblocks in chunk_block_counts:
        nblocks = int(nblocks)
        if nblocks < 0:
            raise ValueError("chunk block count must be non-negative")
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += nblocks
        byte_offset += nblocks * bytes_per_block
        max_tile_nbytes = max(max_tile_nbytes, nblocks * max_seg_bytes)

    if len(block_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    if int(device_buf.numel()) < byte_offset:
        raise ValueError("device_buf is smaller than chunk-major staging output")

    return (
        _device_i64(segment_ptr_values, device),
        _device_i64([int(x) for x in segment_block_bytes], device),
        _device_i64(segment_prefix_values, device),
        _device_i64([int(x) for x in chunk_block_counts], device),
        _device_i64(chunk_block_offsets, device),
        _device_i64(chunk_output_bases, device),
        _device_i64([int(x) for x in block_ids], device),
        torch.tensor([int(byte_offset), int(max_tile_nbytes)], dtype=torch.int64),
    )


def fused_pack_chunk_major(
    segment_tensors, segment_block_bytes, chunk_block_counts, block_ids, device_buf
) -> None:
    meta = _build_chunk_major_meta(
        segment_tensors, segment_block_bytes, chunk_block_counts, block_ids, device_buf
    )
    (
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        sizes,
    ) = meta
    if int(sizes[0].item()) == 0:
        return
    grid = (
        len(chunk_block_counts) * len(segment_tensors),
        triton.cdiv(int(sizes[1].item()), _BLOCK_BYTES),
    )
    _pack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        NUM_SEGMENTS=len(segment_tensors),
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
    )


def fused_unpack_chunk_major(
    device_buf, segment_tensors, segment_block_bytes, chunk_block_counts, block_ids
) -> None:
    meta = _build_chunk_major_meta(
        segment_tensors, segment_block_bytes, chunk_block_counts, block_ids, device_buf
    )
    (
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        sizes,
    ) = meta
    if int(sizes[0].item()) == 0:
        return
    grid = (
        len(chunk_block_counts) * len(segment_tensors),
        triton.cdiv(int(sizes[1].item()), _BLOCK_BYTES),
    )
    _unpack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        NUM_SEGMENTS=len(segment_tensors),
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
    )
