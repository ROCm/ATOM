# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton fused chunk-major staging for ATOM LMCache offload."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_BYTES = 1024


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
