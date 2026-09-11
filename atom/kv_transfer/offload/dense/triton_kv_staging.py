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


def _validate_device_buf(device_buf: torch.Tensor) -> None:
    if not device_buf.is_cuda:
        raise ValueError("device_buf must be a CUDA/HIP tensor")
    if device_buf.dtype != torch.uint8:
        raise TypeError("device_buf must be uint8")
    if not device_buf.is_contiguous():
        raise ValueError("device_buf must be contiguous")


def _build_meta(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    _validate_device_buf(device_buf)
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


class _ChunkMajorPlan:
    """Reusable pack/unpack metadata for one segment geometry and chunk shape.

    ``_build_meta`` walks every segment in Python and issues seven blocking H2D
    copies.  Six of the seven tensors it returns depend only on the segment
    geometry -- fixed for the life of the KV cache -- and on
    ``chunk_block_counts``, which repeats across staging groups; only
    ``block_ids`` actually varies.  Rebuilding all seven per group put that work
    on the transfer's critical path once per group, and with the model
    co-resident it dominated the group: 24.33 ms of a 24.43 ms "pack" against
    0.10 ms for the Triton kernel itself.

    Building the six once and pushing ``block_ids`` through a reused pinned
    tensor takes one 256-chunk save from 9.5 GB/s to 21.9 GB/s under a
    co-resident model, and to 36.7 GB/s once the device-to-host leg is
    asynchronous too.
    """

    __slots__ = (
        "_block_ids_d",
        "_chunk_block_counts",
        "_chunk_block_offsets",
        "_chunk_output_bases",
        "_grid",
        "_host_events",
        "_host_slots",
        "_nblocks",
        "_num_segments",
        "_output_nbytes",
        "_segment_block_bytes",
        "_segment_prefix",
        "_segment_ptrs",
        "_slot",
    )

    # Host staging slots for block_ids. One would race: the pinned buffer is
    # handed to an asynchronous H2D, and the next group's host-side write could
    # land while that copy is still in flight. Waiting on the copy instead would
    # drain everything queued ahead of it on the stream -- the previous group's
    # pack kernel -- so rotate a few slots and only wait on wrap, which in
    # practice never blocks.
    _NUM_HOST_SLOTS = 4

    def __init__(self, meta, num_segments: int, nblocks: int) -> None:
        (
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            sizes,
        ) = meta
        self._num_segments = num_segments
        self._nblocks = nblocks
        self._output_nbytes = int(sizes[0].item())
        self._grid = (
            int(self._chunk_block_counts.numel()) * num_segments,
            triton.cdiv(int(sizes[1].item()), _BLOCK_BYTES),
        )
        self._host_slots = [
            torch.empty(nblocks, dtype=torch.int64, pin_memory=True)
            for _ in range(self._NUM_HOST_SLOTS)
        ]
        self._host_events = [torch.cuda.Event() for _ in range(self._NUM_HOST_SLOTS)]
        for event in self._host_events:
            event.record()
        self._slot = 0

    def _prepare(self, block_ids, device_buf: torch.Tensor) -> bool:
        # The cached metadata skips _build_meta, so device_buf still has to be
        # checked here; it is the one argument a caller can vary per call.
        _validate_device_buf(device_buf)
        if int(device_buf.numel()) < self._output_nbytes:
            raise ValueError("device_buf is smaller than chunk-major staging output")
        if len(block_ids) != self._nblocks:
            raise ValueError("block_ids length does not match chunk block counts")
        if self._output_nbytes == 0:
            return False
        slot = self._slot
        self._slot = (slot + 1) % self._NUM_HOST_SLOTS
        self._host_events[slot].synchronize()
        host = self._host_slots[slot]
        host.copy_(torch.as_tensor(block_ids, dtype=torch.int64))
        self._block_ids_d.copy_(host, non_blocking=True)
        self._host_events[slot].record()
        return True

    def pack(self, block_ids, device_buf: torch.Tensor) -> None:
        if not self._prepare(block_ids, device_buf):
            return
        _pack_chunk_major_kernel[self._grid](
            device_buf,
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            NUM_SEGMENTS=self._num_segments,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
        )

    def unpack(self, block_ids, device_buf: torch.Tensor) -> None:
        if not self._prepare(block_ids, device_buf):
            return
        _unpack_chunk_major_kernel[self._grid](
            device_buf,
            self._segment_ptrs,
            self._segment_block_bytes,
            self._segment_prefix,
            self._chunk_block_counts,
            self._chunk_block_offsets,
            self._chunk_output_bases,
            self._block_ids_d,
            NUM_SEGMENTS=self._num_segments,
            BLOCK_BYTES=_BLOCK_BYTES,
            num_warps=8,
        )


# Plans are keyed by everything _build_meta derives them from except block_ids.
# Segment data pointers are part of the key, so a reallocated KV cache misses
# rather than silently packing from freed storage. Only a couple of shapes occur
# in practice -- a full staging group plus whatever tail a store leaves -- but
# cap the cache anyway so an unusual traffic pattern cannot grow it without
# bound.
_PLAN_CACHE: dict[tuple, _ChunkMajorPlan] = {}
_MAX_PLANS = 64


def _get_plan(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    device_buf: torch.Tensor,
) -> _ChunkMajorPlan:
    counts = tuple(int(n) for n in chunk_block_counts)
    key = (
        tuple(int(t.data_ptr()) for t in segment_tensors),
        tuple(int(nb) for nb in segment_block_bytes),
        counts,
    )
    plan = _PLAN_CACHE.get(key)
    if plan is None:
        # block_ids is only length-checked by _build_meta, so a placeholder of
        # the right length is enough to get every other tensor built and every
        # argument validated; the tensor it returns becomes the plan's device
        # landing buffer for block_ids.
        nblocks = sum(counts)
        meta = _build_meta(
            segment_tensors,
            segment_block_bytes,
            counts,
            [0] * nblocks,
            device_buf,
        )
        if len(_PLAN_CACHE) >= _MAX_PLANS:
            _PLAN_CACHE.clear()
        plan = _ChunkMajorPlan(meta, len(segment_tensors), nblocks)
        _PLAN_CACHE[key] = plan
    return plan


def fused_pack_chunk_major(
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
    device_buf,
) -> None:
    _get_plan(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        device_buf,
    ).pack(block_ids, device_buf)


def fused_unpack_chunk_major(
    device_buf,
    segment_tensors,
    segment_block_bytes,
    chunk_block_counts,
    block_ids,
) -> None:
    _get_plan(
        segment_tensors,
        segment_block_bytes,
        chunk_block_counts,
        device_buf,
    ).unpack(block_ids, device_buf)
