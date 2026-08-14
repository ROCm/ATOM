# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Descriptor-driven bitwise copy between segmented GPU byte streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_TILE_BYTES = 4096


@dataclass(frozen=True)
class ByteSegment:
    ptr: int
    num_bytes: int


@dataclass(frozen=True)
class CopySpan:
    src_ptr: int
    dst_ptr: int
    num_bytes: int


def tensor_segment(tensor: torch.Tensor) -> ByteSegment:
    """Describe a contiguous tensor view as raw bytes without converting it."""
    if not tensor.is_contiguous():
        raise ValueError("paged state copy segments must be contiguous")
    return ByteSegment(int(tensor.data_ptr()), tensor.numel() * tensor.element_size())


def plan_segmented_copy(
    src: list[ByteSegment],
    dst: list[ByteSegment],
    total_bytes: int,
) -> list[CopySpan]:
    """Intersect two ordered logical streams into physical copy spans.

    Segment boundaries on the two sides need not agree.  This pure host planner
    is shared by scatter and gather, which prevents their wire orders from
    silently diverging.
    """
    total_bytes = int(total_bytes)
    if total_bytes < 0:
        raise ValueError("copy length must be non-negative")
    if sum(s.num_bytes for s in src) < total_bytes:
        raise ValueError("source segmented stream is shorter than the copy")
    if sum(s.num_bytes for s in dst) < total_bytes:
        raise ValueError("destination segmented stream is shorter than the copy")
    if any(s.num_bytes <= 0 for s in src + dst):
        raise ValueError("segmented streams cannot contain empty segments")
    if total_bytes == 0:
        return []

    spans: list[CopySpan] = []
    src_i = dst_i = 0
    src_off = dst_off = 0
    remaining = total_bytes
    while remaining:
        src_left = src[src_i].num_bytes - src_off
        dst_left = dst[dst_i].num_bytes - dst_off
        nbytes = min(src_left, dst_left, remaining)
        spans.append(
            CopySpan(
                src[src_i].ptr + src_off,
                dst[dst_i].ptr + dst_off,
                nbytes,
            )
        )
        remaining -= nbytes
        src_off += nbytes
        dst_off += nbytes
        if src_off == src[src_i].num_bytes:
            src_i += 1
            src_off = 0
        if dst_off == dst[dst_i].num_bytes:
            dst_i += 1
            dst_off = 0
    return spans


@triton.jit
def _copy_tiles_kernel(
    src_ptrs,
    dst_ptrs,
    valid_bytes,
    TILE_BYTES: tl.constexpr,
):
    tile = tl.program_id(0)
    offsets = tl.arange(0, TILE_BYTES)
    valid = tl.load(valid_bytes + tile)
    mask = offsets < valid
    src_addr = tl.load(src_ptrs + tile).to(tl.int64)
    dst_addr = tl.load(dst_ptrs + tile).to(tl.int64)
    src = (src_addr + offsets).to(tl.pointer_type(tl.uint8))
    dst = (dst_addr + offsets).to(tl.pointer_type(tl.uint8))
    value = tl.load(src, mask=mask)
    tl.store(dst, value, mask=mask)


def launch_copy_spans(spans: list[CopySpan], device: torch.device) -> None:
    """Copy all spans with one descriptor-driven Triton launch."""
    if not spans:
        return
    src_ptrs: list[int] = []
    dst_ptrs: list[int] = []
    valid_bytes: list[int] = []
    for span in spans:
        offset = 0
        while offset < span.num_bytes:
            nbytes = min(_TILE_BYTES, span.num_bytes - offset)
            src_ptrs.append(span.src_ptr + offset)
            dst_ptrs.append(span.dst_ptr + offset)
            valid_bytes.append(nbytes)
            offset += nbytes

    src_t = torch.tensor(src_ptrs, dtype=torch.int64, device=device)
    dst_t = torch.tensor(dst_ptrs, dtype=torch.int64, device=device)
    valid_t = torch.tensor(valid_bytes, dtype=torch.int32, device=device)
    _copy_tiles_kernel[(len(src_ptrs),)](
        src_t,
        dst_t,
        valid_t,
        TILE_BYTES=_TILE_BYTES,
        num_warps=8,
    )
