# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton fused chunk-major staging for ATOM LMCache offload."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_BLOCK_BYTES = 1024
# 1024 bytes over two wavefronts is 8 bytes per lane. The tile used to be
# launched with eight warps, which the rectangular grid hid -- most programs
# had nothing to move, so what the ones that did cost per lane barely showed.
# Sized by the work, the shape is the whole cost: a sweep of tile x warps over
# the measured M3 geometry peaks flat along 8 bytes per lane (643/634/632 GB/s
# at 512x1, 1024x2, 2048x4) and falls off either side of it -- 448 GB/s at the
# eight warps this used to run, 45 GB/s at 8192x1. Two warps keeps the tile
# where every other part of this file already assumes it.
_NUM_WARPS = 2


@dataclass(frozen=True)
class _PreparedGroupMeta:
    chunk_counts: slice
    chunk_offsets: slice
    output_bases: slice
    block_ids: slice
    tile_job: slice
    tile_pos: slice
    chunk_count: int
    total_bytes: int
    num_tiles: int


@dataclass(frozen=True)
class PreparedChunkMajorGroups:
    """One device metadata upload shared by all staging groups in a transfer."""

    device: torch.device
    metadata: torch.Tensor
    tile_table: torch.Tensor
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
    counts,
    segment_block_bytes,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One grid entry per tile that has bytes to move, and nothing else.

    The grid used to be rectangular: ``(chunk * segment, tiles)``, where the
    tile count came from the largest segment. Segments are not the same size --
    M3 stages 16 KiB of K and of V per block next to 512-byte MXFP8 scales and
    one much larger cache -- so every small segment was launched with the tile
    count of the biggest one and masked off almost all of it. On the measured
    geometry that is 1,847,024 programs to move 23 MiB, about 12 payload bytes
    each, and it costs what it sounds like: 15.8 GB/s against 276.6 GB/s for
    the same bytes in segments of one size.

    So size the grid by the work instead. Each job's tile count is its own, the
    table maps a flat program id back to (job, tile within job), and the result
    depends on the total bytes rather than on the widest segment. It is built
    with tensor ops rather than Python loops: at production geometry a group is
    ~24k tiles, and unboxing that many Python ints per group is the same
    GIL-bound cost the rest of this path exists to avoid.

    ``device=None`` leaves the table on the host, for a caller that concatenates
    several groups' tables into one upload.
    """
    nseg = len(segment_block_bytes)
    blocks = torch.as_tensor([int(c) for c in counts], dtype=torch.int64)
    seg_bytes = torch.as_tensor(
        [int(nb) for nb in segment_block_bytes], dtype=torch.int64
    )
    # One job per (chunk, segment), chunk-major -- the order the kernel derives
    # from its job id.
    job_nbytes = blocks.repeat_interleave(nseg) * seg_bytes.repeat(len(counts))
    tiles = (job_nbytes + _BLOCK_BYTES - 1) // _BLOCK_BYTES
    jobs = torch.repeat_interleave(torch.arange(tiles.numel()), tiles)
    starts = torch.cumsum(tiles, 0) - tiles
    pos = torch.arange(int(tiles.sum())) - torch.repeat_interleave(starts, tiles)
    jobs = jobs.to(dtype=torch.int32)
    pos = pos.to(dtype=torch.int32)
    if device is not None:
        return jobs.to(device=device), pos.to(device=device)
    return jobs, pos


def _group_tile_tables(
    group_counts: Sequence[Sequence[int]],
    segment_block_bytes: Sequence[int],
) -> tuple[torch.Tensor, list[tuple[slice, slice, int]]]:
    """Every group's tile table in one host tensor, plus each group's slices.

    Laid out group by group as ``job`` then ``pos``, so one upload serves the
    whole transfer and a group's launch is two views into it.
    """
    tables: list[torch.Tensor] = []
    spans: list[tuple[slice, slice, int]] = []
    offset = 0
    for counts in group_counts:
        tile_job, tile_pos = _tile_table(counts, segment_block_bytes)
        num_tiles = int(tile_job.numel())
        tables.append(tile_job)
        tables.append(tile_pos)
        spans.append(
            (
                slice(offset, offset + num_tiles),
                slice(offset + num_tiles, offset + 2 * num_tiles),
                num_tiles,
            )
        )
        offset += 2 * num_tiles
    table = torch.cat(tables) if tables else torch.empty(0, dtype=torch.int32)
    return table, spans


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
    """Build all static and dynamic Triton metadata for a whole transfer.

    Two H2D uploads, whatever the group count: one int64 tensor of pointers,
    sizes and block ids, and one int32 tile table. ``upload_count`` counts only
    the first, because it is what the connector reports as block-id uploads.
    """

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
    group_slices: list[tuple[slice, slice, slice, slice, int, int]] = []
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

        count_start = len(values)
        values.extend(counts)
        offset_start = len(values)
        values.extend(offsets)
        output_start = len(values)
        values.extend(output_bases)
        ids_start = len(values)
        values.extend(normalized_ids)
        has_block_ids = has_block_ids or bool(normalized_ids)

        group_slices.append(
            (
                slice(count_start, offset_start),
                slice(offset_start, output_start),
                slice(output_start, ids_start),
                slice(ids_start, len(values)),
                len(counts),
                total_bytes,
            )
        )

    # Built only once every group has been validated above, and kept out of
    # ``values``: the tables are int32, and at production geometry one group is
    # ~24k tiles, so folding them into a Python list would unbox ~48k ints per
    # group -- the same GIL-bound cost the rest of this path exists to avoid.
    # They are concatenated and uploaded as one tensor instead.
    tile_table, tile_spans = _group_tile_tables(
        [
            [int(count) for count in chunk_block_counts]
            for chunk_block_counts, _ in groups
        ],
        normalized_block_bytes,
    )
    prepared_groups = [
        _PreparedGroupMeta(
            chunk_counts=chunk_counts,
            chunk_offsets=chunk_offsets,
            output_bases=output_bases,
            block_ids=ids,
            tile_job=job_span,
            tile_pos=pos_span,
            chunk_count=chunk_count,
            total_bytes=total_bytes,
            num_tiles=num_tiles,
        )
        for (
            chunk_counts,
            chunk_offsets,
            output_bases,
            ids,
            chunk_count,
            total_bytes,
        ), (job_span, pos_span, num_tiles) in zip(group_slices, tile_spans)
    ]

    metadata = torch.tensor(values, dtype=torch.int64).to(
        device=device, non_blocking=False
    )
    tile_table = tile_table.to(device=device, non_blocking=False)
    return PreparedChunkMajorGroups(
        device=device,
        metadata=metadata,
        tile_table=tile_table,
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

    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    counts = [int(n) for n in chunk_block_counts]
    for nblocks in counts:
        if nblocks < 0:
            raise ValueError("chunk block count must be non-negative")
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += nblocks
        byte_offset += nblocks * bytes_per_block

    if len(block_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    if int(device_buf.numel()) < byte_offset:
        raise ValueError("device_buf is smaller than chunk-major staging output")

    tile_job, tile_pos = _tile_table(counts, segment_block_bytes, device)

    return (
        _device_i64(segment_ptr_values, device),
        _device_i64([int(x) for x in segment_block_bytes], device),
        _device_i64(segment_prefix_values, device),
        _device_i64(counts, device),
        _device_i64(chunk_block_offsets, device),
        _device_i64(chunk_output_bases, device),
        _device_i64([int(x) for x in block_ids], device),
        tile_job,
        tile_pos,
        torch.tensor([int(byte_offset), int(tile_job.numel())], dtype=torch.int64),
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
        prepared.tile_table[group.tile_job],
        prepared.tile_table[group.tile_pos],
        group,
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
        group,
    ) = launch
    if group.num_tiles == 0:
        return
    grid = (group.num_tiles,)
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
        group,
    ) = launch
    if group.num_tiles == 0:
        return
    grid = (group.num_tiles,)
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
        sizes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
    num_tiles = int(sizes[1].item())
    if int(sizes[0].item()) == 0 or num_tiles == 0:
        return
    grid = (num_tiles,)
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
        sizes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
    num_tiles = int(sizes[1].item())
    if int(sizes[0].item()) == 0 or num_tiles == 0:
        return
    grid = (num_tiles,)
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
