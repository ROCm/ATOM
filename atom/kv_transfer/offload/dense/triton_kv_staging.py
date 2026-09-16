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


@dataclass(frozen=True)
class _PreparedGroupMeta:
    chunk_counts: slice
    chunk_offsets: slice
    output_bases: slice
    block_ids: slice
    chunk_count: int
    total_bytes: int
    max_tile_nbytes: int


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
    max_segment_block_bytes: int,
) -> tuple[list[int], list[int], list[int], list[int], int, int]:
    normalized_counts: list[int] = []
    chunk_block_offsets: list[int] = []
    chunk_output_bases: list[int] = []
    block_offset = 0
    byte_offset = 0
    max_tile_nbytes = 0
    for count in chunk_block_counts:
        count = int(count)
        if count < 0:
            raise ValueError("chunk block count must be non-negative")
        normalized_counts.append(count)
        chunk_block_offsets.append(block_offset)
        chunk_output_bases.append(byte_offset)
        block_offset += count
        byte_offset += count * bytes_per_block
        max_tile_nbytes = max(max_tile_nbytes, count * max_segment_block_bytes)
    normalized_ids = [int(block_id) for block_id in block_ids]
    if len(normalized_ids) != block_offset:
        raise ValueError("block_ids length does not match chunk block counts")
    return (
        normalized_counts,
        chunk_block_offsets,
        chunk_output_bases,
        normalized_ids,
        byte_offset,
        max_tile_nbytes,
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
    max_segment_block_bytes = max(normalized_block_bytes)
    for chunk_block_counts, block_ids in groups:
        (
            counts,
            offsets,
            output_bases,
            normalized_ids,
            total_bytes,
            max_tile_nbytes,
        ) = _group_meta_values(
            chunk_block_counts,
            block_ids,
            bytes_per_block=bytes_per_block,
            max_segment_block_bytes=max_segment_block_bytes,
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
        prepared_groups.append(
            _PreparedGroupMeta(
                chunk_counts=slice(count_start, offset_start),
                chunk_offsets=slice(offset_start, output_start),
                output_bases=slice(output_start, ids_start),
                block_ids=slice(ids_start, len(values)),
                chunk_count=len(counts),
                total_bytes=total_bytes,
                max_tile_nbytes=max_tile_nbytes,
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
        group,
    ) = launch
    grid = (
        group.chunk_count * prepared.num_segments,
        triton.cdiv(group.max_tile_nbytes, _BLOCK_BYTES),
    )
    _pack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        NUM_SEGMENTS=prepared.num_segments,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
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
        group,
    ) = launch
    grid = (
        group.chunk_count * prepared.num_segments,
        triton.cdiv(group.max_tile_nbytes, _BLOCK_BYTES),
    )
    _unpack_chunk_major_kernel[grid](
        device_buf,
        segment_ptrs,
        segment_block_bytes,
        segment_prefix_bytes,
        chunk_block_counts,
        chunk_block_offsets,
        chunk_output_bases,
        block_ids,
        NUM_SEGMENTS=prepared.num_segments,
        BLOCK_BYTES=_BLOCK_BYTES,
        num_warps=8,
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
        sizes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
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
        sizes,
    ) = _build_meta(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        block_ids,
        device_buf,
    )
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
