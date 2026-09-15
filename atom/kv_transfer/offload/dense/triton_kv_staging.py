# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton fused chunk-major staging for ATOM LMCache offload."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import triton
import triton.language as tl

_BLOCK_BYTES = 1024

# Warps per program.  A program moves BLOCK_BYTES of uint8, so this is really a
# choice of bytes per lane: two warps puts 8 on a lane at BLOCK_BYTES=1024.
# Packing the geometry the dense offload worker logs (296 segments, 23.0 MiB)
# runs at 943 GB/s on two warps, 782 on four and 586 on eight; one warp drops
# back to 720.  Nothing in these kernels reduces across lanes, so the warp count
# moves throughput only.
_NUM_WARPS = 2

# A tile table is a pure function of (chunk block counts, segment sizes), and
# both repeat across transfers, so the tables are memoised rather than rebuilt
# per launch: building one for the geometry above costs more than the kernel it
# feeds, a hit costs about a tenth of it.  Bounded because the entries are
# device tensors -- sixteen of that geometry is a few MiB.
_TILE_TABLE_CACHE_SIZE = 16
_TILE_TABLE_CACHE: OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor]] = OrderedDict()


@dataclass(frozen=True)
class _PreparedGroupMeta:
    chunk_counts: slice
    chunk_offsets: slice
    output_bases: slice
    block_ids: slice
    chunk_count: int
    total_bytes: int
    tile_job: torch.Tensor
    tile_pos: torch.Tensor


@dataclass(frozen=True)
class PreparedChunkMajorGroups:
    """One device metadata upload shared by all staging groups in a transfer."""

    device: torch.device
    metadata: torch.Tensor
    segment_ptrs: slice
    segment_block_bytes: slice
    segment_prefix_bytes: slice
    groups: tuple[_PreparedGroupMeta, ...]
    num_segments: int
    upload_count: int

    @property
    def group_count(self) -> int:
        return len(self.groups)


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
    tile_job,
    tile_pos,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.load(tile_job + pid)
    tile = tl.load(tile_pos + pid)
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
        block_ids + block_offset + local_block,
        mask=mask,
        other=0,
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
    data = tl.load(src, mask=mask)
    tl.store(dst, data, mask=mask)


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
    tile_job,
    tile_pos,
    NUM_SEGMENTS: tl.constexpr,
    BLOCK_BYTES: tl.constexpr,
):
    pid = tl.program_id(0)
    job = tl.load(tile_job + pid)
    tile = tl.load(tile_pos + pid)
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
        block_ids + block_offset + local_block,
        mask=mask,
        other=0,
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
    data = tl.load(src, mask=mask)
    tl.store(dst, data, mask=mask)


def _device_i64(values: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=device)


def _tile_table(
    chunk_block_counts: list[int],
    segment_block_bytes: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map each flat program id to the (job, tile within job) it moves.

    A job is one (chunk, segment) pair and owns ``count * segment_bytes`` bytes,
    so jobs do not all want the same number of tiles.  The rectangular grid this
    replaces gave every job the widest job's tile count and masked the surplus
    off, which priced a launch by ``max(segment_block_bytes)`` rather than by the
    bytes being staged -- segments registered alongside their scales are two
    orders of magnitude apart, so most of that grid moved nothing.

    Memoised on (counts, geometry, device); see ``_TILE_TABLE_CACHE_SIZE``.  A
    miss costs one H2D upload -- both columns travel as one ``2 x tiles`` tensor
    whose rows are contiguous -- so a new geometry adds one upload to the plan's
    one, and a repeat adds none.  Both sequences must already be normalized to
    ``int``: a hit is on the lookup path of every transfer, and coercing a
    few hundred segment sizes on the way in costs more than the kernel saves.
    """

    key = (tuple(chunk_block_counts), tuple(segment_block_bytes), device)
    cached = _TILE_TABLE_CACHE.get(key)
    if cached is not None:
        _TILE_TABLE_CACHE.move_to_end(key)
        return cached

    counts = np.asarray(key[0], dtype=np.int64)
    seg_bytes = np.asarray(key[1], dtype=np.int64)
    # Jobs in (chunk, segment) order, each wanting ceil(count * seg_bytes / tile).
    tiles = -(-np.outer(counts, seg_bytes).ravel() // _BLOCK_BYTES)
    starts = np.cumsum(tiles) - tiles
    total = int(tiles.sum())
    job_of_tile = np.repeat(np.arange(tiles.size, dtype=np.int64), tiles)
    pos_in_job = np.arange(total, dtype=np.int64) - np.repeat(starts, tiles)
    columns = torch.from_numpy(np.concatenate((job_of_tile, pos_in_job)))
    columns = columns.to(device=device).view(2, total)
    table = (columns[0], columns[1])
    _TILE_TABLE_CACHE[key] = table
    while len(_TILE_TABLE_CACHE) > _TILE_TABLE_CACHE_SIZE:
        _TILE_TABLE_CACHE.popitem(last=False)
    return table


def _segment_meta_values(
    segment_tensors: Sequence[torch.Tensor],
    segment_block_bytes: Sequence[int],
    device: torch.device,
) -> tuple[list[int], list[int], list[int], int]:
    if len(segment_tensors) != len(segment_block_bytes):
        raise ValueError("segment_tensors and segment_block_bytes size mismatch")
    if not segment_tensors:
        raise ValueError("at least one segment is required")

    segment_ptr_values: list[int] = []
    segment_prefix_values: list[int] = []
    normalized_block_bytes: list[int] = []
    bytes_per_block = 0
    for seg, nbytes in zip(segment_tensors, segment_block_bytes, strict=True):
        if not seg.is_cuda:
            raise ValueError("segment tensor must be CUDA/HIP")
        if seg.device != device:
            raise ValueError("segment/device mismatch")
        if not seg.is_contiguous():
            raise ValueError("segment tensor must be contiguous")
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise ValueError("segment block bytes must be > 0")
        segment_ptr_values.append(int(seg.data_ptr()))
        segment_prefix_values.append(bytes_per_block)
        normalized_block_bytes.append(nbytes)
        bytes_per_block += nbytes
    return (
        segment_ptr_values,
        normalized_block_bytes,
        segment_prefix_values,
        bytes_per_block,
    )


def _group_meta_values(
    chunk_block_counts: Sequence[int],
    block_ids: Sequence[int],
    *,
    bytes_per_block: int,
) -> tuple[list[int], list[int], list[int], list[int], int]:
    normalized_counts: list[int] = []
    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    for count in chunk_block_counts:
        count = int(count)
        if count < 0:
            raise ValueError("chunk block count must be non-negative")
        normalized_counts.append(count)
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += count
        byte_offset += count * bytes_per_block
    normalized_ids = [int(block_id) for block_id in block_ids]
    if len(normalized_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    return (
        normalized_counts,
        chunk_block_offsets,
        chunk_output_bases,
        normalized_ids,
        byte_offset,
    )


def prepare_chunk_major_groups(
    segment_tensors: Sequence[torch.Tensor],
    segment_block_bytes: Sequence[int],
    groups: Sequence[tuple[Sequence[int], Sequence[int]]],
    device: torch.device,
) -> PreparedChunkMajorGroups:
    """Build all static and dynamic Triton metadata with one H2D upload."""

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("prepared chunk-major metadata requires CUDA/HIP")
    (
        segment_ptr_values,
        normalized_block_bytes,
        segment_prefix_values,
        bytes_per_block,
    ) = _segment_meta_values(segment_tensors, segment_block_bytes, device)

    values = segment_ptr_values + normalized_block_bytes + segment_prefix_values
    num_segments = len(segment_ptr_values)
    segment_ptrs = slice(0, num_segments)
    segment_block_bytes_slice = slice(num_segments, 2 * num_segments)
    segment_prefix_bytes = slice(2 * num_segments, 3 * num_segments)
    prepared_groups: list[_PreparedGroupMeta] = []
    has_block_ids = False
    for chunk_block_counts, block_ids in groups:
        (
            counts,
            offsets,
            output_bases,
            normalized_ids,
            total_bytes,
        ) = _group_meta_values(
            chunk_block_counts,
            block_ids,
            bytes_per_block=bytes_per_block,
        )
        tile_job, tile_pos = _tile_table(counts, normalized_block_bytes, device)

        count_start = len(values)
        values.extend(counts)
        offset_start = len(values)
        values.extend(offsets)
        output_start = len(values)
        values.extend(output_bases)
        ids_start = len(values)
        values.extend(normalized_ids)
        has_block_ids = has_block_ids or bool(normalized_ids)
        prepared_groups.append(
            _PreparedGroupMeta(
                chunk_counts=slice(count_start, offset_start),
                chunk_offsets=slice(offset_start, output_start),
                output_bases=slice(output_start, ids_start),
                block_ids=slice(ids_start, len(values)),
                chunk_count=len(counts),
                total_bytes=total_bytes,
                tile_job=tile_job,
                tile_pos=tile_pos,
            )
        )

    metadata = torch.tensor(values, dtype=torch.int64).to(
        device=device, non_blocking=False
    )
    return PreparedChunkMajorGroups(
        device=device,
        metadata=metadata,
        segment_ptrs=segment_ptrs,
        segment_block_bytes=segment_block_bytes_slice,
        segment_prefix_bytes=segment_prefix_bytes,
        groups=tuple(prepared_groups),
        num_segments=num_segments,
        upload_count=int(has_block_ids),
    )


def _build_meta(
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

    normalized_block_bytes = [int(nb) for nb in segment_block_bytes]
    normalized_counts: list[int] = []
    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    for nblocks in chunk_block_counts:
        nblocks = int(nblocks)
        if nblocks < 0:
            raise ValueError("chunk block count must be non-negative")
        normalized_counts.append(nblocks)
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += nblocks
        byte_offset += nblocks * bytes_per_block

    if len(block_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    if int(device_buf.numel()) < byte_offset:
        raise ValueError("device_buf is smaller than chunk-major staging output")

    tile_job, tile_pos = _tile_table(normalized_counts, normalized_block_bytes, device)
    return (
        _device_i64(segment_ptr_values, device),
        _device_i64(normalized_block_bytes, device),
        _device_i64(segment_prefix_values, device),
        _device_i64(normalized_counts, device),
        _device_i64(chunk_block_offsets, device),
        _device_i64(chunk_output_bases, device),
        _device_i64([int(x) for x in block_ids], device),
        tile_job,
        tile_pos,
        int(byte_offset),
    )


def _prepared_launch_meta(
    prepared: PreparedChunkMajorGroups,
    group_index: int,
    device_buf: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    if not isinstance(prepared, PreparedChunkMajorGroups):
        raise TypeError("invalid prepared chunk-major metadata")
    if not device_buf.is_cuda:
        raise ValueError("device_buf must be a CUDA/HIP tensor")
    if device_buf.dtype != torch.uint8:
        raise TypeError("device_buf must be uint8")
    if not device_buf.is_contiguous():
        raise ValueError("device_buf must be contiguous")
    if device_buf.device != prepared.device:
        raise ValueError("prepared metadata/device mismatch")
    index = int(group_index)
    if index < 0 or index >= prepared.group_count:
        raise ValueError("prepared group_index is out of range")
    group = prepared.groups[index]
    if int(device_buf.numel()) < group.total_bytes:
        raise ValueError("device_buf is smaller than chunk-major staging output")
    if group.total_bytes == 0:
        return None

    metadata = prepared.metadata
    return (
        metadata[prepared.segment_ptrs],
        metadata[prepared.segment_block_bytes],
        metadata[prepared.segment_prefix_bytes],
        metadata[group.chunk_counts],
        metadata[group.chunk_offsets],
        metadata[group.output_bases],
        metadata[group.block_ids],
        group.tile_job,
        group.tile_pos,
    )


def fused_pack_chunk_major_prepared(
    prepared: PreparedChunkMajorGroups,
    group_index: int,
    device_buf: torch.Tensor,
) -> None:
    launch = _prepared_launch_meta(prepared, group_index, device_buf)
    if launch is None:
        return
    (
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        tile_job,
        tile_pos,
    ) = launch
    grid = (tile_job.numel(),)
    _pack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        tile_job,
        tile_pos,
        NUM_SEGMENTS=prepared.num_segments,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=_NUM_WARPS,
    )


def fused_unpack_chunk_major_prepared(
    prepared: PreparedChunkMajorGroups,
    group_index: int,
    device_buf: torch.Tensor,
) -> None:
    launch = _prepared_launch_meta(prepared, group_index, device_buf)
    if launch is None:
        return
    (
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        tile_job,
        tile_pos,
    ) = launch
    grid = (tile_job.numel(),)
    _unpack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        tile_job,
        tile_pos,
        NUM_SEGMENTS=prepared.num_segments,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=_NUM_WARPS,
    )


def fused_pack_chunk_major(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf,
) -> None:
    (
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        tile_job,
        tile_pos,
        total_bytes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
    if total_bytes == 0:
        return
    grid = (tile_job.numel(),)
    _pack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        tile_job,
        tile_pos,
        NUM_SEGMENTS=len(segment_tensors),
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=_NUM_WARPS,
    )


def fused_unpack_chunk_major(
    device_buf,
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
) -> None:
    (
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        tile_job,
        tile_pos,
        total_bytes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
    if total_bytes == 0:
        return
    grid = (tile_job.numel(),)
    _unpack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes_t,
        segment_prefix_bytes,
        chunk_block_counts_t,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids_t,
        tile_job,
        tile_pos,
        NUM_SEGMENTS=len(segment_tensors),
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=_NUM_WARPS,
    )
