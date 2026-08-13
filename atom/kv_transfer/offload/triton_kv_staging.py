# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Triton fused chunk-major staging for ATOM LMCache offload."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import triton
import triton.language as tl

from atom.kv_transfer.offload.copy_plan import build_copy_tiles

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


@triton.jit
def _gather_copy_plan_kernel(
    dst,
    source_ptrs,
    buffer_offsets,
    copy_sizes,
    BLOCK_BYTES: tl.constexpr,
):
    job_index = tl.program_id(0)
    nbytes = tl.load(copy_sizes + job_index).to(tl.int64)
    offsets = tl.arange(0, BLOCK_BYTES).to(tl.int64)
    mask = offsets < nbytes
    src = (tl.load(source_ptrs + job_index).to(tl.int64) + offsets).to(
        tl.pointer_type(tl.uint8)
    )
    dst_ptr = dst + tl.load(buffer_offsets + job_index).to(tl.int64) + offsets
    tl.store(dst_ptr, tl.load(src, mask=mask), mask=mask)


@triton.jit
def _scatter_copy_plan_kernel(
    src,
    destination_ptrs,
    buffer_offsets,
    copy_sizes,
    BLOCK_BYTES: tl.constexpr,
):
    job_index = tl.program_id(0)
    nbytes = tl.load(copy_sizes + job_index).to(tl.int64)
    offsets = tl.arange(0, BLOCK_BYTES).to(tl.int64)
    mask = offsets < nbytes
    src_ptr = src + tl.load(buffer_offsets + job_index).to(tl.int64) + offsets
    dst = (tl.load(destination_ptrs + job_index).to(tl.int64) + offsets).to(
        tl.pointer_type(tl.uint8)
    )
    tl.store(dst, tl.load(src_ptr, mask=mask), mask=mask)


class _NullCtx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


def _validate_copy_plan_buffer(buffer: torch.Tensor, name: str) -> None:
    if not isinstance(buffer, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not buffer.is_cuda:
        raise ValueError(f"{name} must be a CUDA/HIP tensor")
    if buffer.dtype != torch.uint8:
        raise TypeError(f"{name} must be uint8")
    if not buffer.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _build_copy_plan_meta(
    plan: Sequence[object],
    *,
    device: torch.device,
    buffer_numel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source_ptrs: list[int] = []
    buffer_offsets: list[int] = []
    copy_sizes: list[int] = []
    for tile_index, tile in enumerate(build_copy_tiles(plan, tile_bytes=_BLOCK_BYTES)):
        if tile.dst_offset + tile.nbytes > int(buffer_numel):
            raise ValueError(
                f"copy tile {tile_index} exceeds staging buffer: "
                f"end={tile.dst_offset + tile.nbytes}, size={int(buffer_numel)}"
            )
        source_ptrs.append(tile.src_addr)
        buffer_offsets.append(tile.dst_offset)
        copy_sizes.append(tile.nbytes)
    return (
        _device_i64(source_ptrs, device),
        _device_i64(buffer_offsets, device),
        _device_i64(copy_sizes, device),
    )


def gather_copy_plan(
    plan: Sequence[object],
    dst: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Gather raw device ranges from ``plan`` into ``dst``."""
    _validate_copy_plan_buffer(dst, "dst")
    stream_ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
    with stream_ctx:
        source_ptrs, buffer_offsets, copy_sizes = _build_copy_plan_meta(
            plan, device=dst.device, buffer_numel=int(dst.numel())
        )
        job_count = int(source_ptrs.numel())
        if job_count == 0:
            return
        grid = (job_count,)
        _gather_copy_plan_kernel[grid](
            dst,
            source_ptrs,
            buffer_offsets,
            copy_sizes,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
        )


def scatter_copy_plan(
    src: torch.Tensor,
    plan: Sequence[object],
    *,
    stream: torch.cuda.Stream | None = None,
) -> None:
    """Scatter bytes from ``src`` back to the raw device ranges in ``plan``."""
    _validate_copy_plan_buffer(src, "src")
    stream_ctx = torch.cuda.stream(stream) if stream is not None else _NullCtx()
    with stream_ctx:
        destination_ptrs, buffer_offsets, copy_sizes = _build_copy_plan_meta(
            plan,
            device=src.device,
            buffer_numel=int(src.numel()),
        )
        job_count = int(destination_ptrs.numel())
        if job_count == 0:
            return
        grid = (job_count,)
        _scatter_copy_plan_kernel[grid](
            src,
            destination_ptrs,
            buffer_offsets,
            copy_sizes,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
        )
