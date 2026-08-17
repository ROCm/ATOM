# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Descriptor-driven bitwise copy between segmented GPU byte streams."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

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
    """Intersect two ordered byte streams into physical copy spans."""
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


if triton is not None:

    @triton.jit
    def _copy_spans_kernel(descriptor, TILE_BYTES: tl.constexpr):
        # A row is `_descriptor`'s: source, destination, length. Triton cannot
        # read a plain module constant, so the three stay literal here and the
        # round-trip test is what holds the two ends to the same order.
        row = descriptor + tl.program_id(0) * 3
        src_ptr = tl.load(row)
        dst_ptr = tl.load(row + 1)
        length = tl.load(row + 2)
        start = tl.program_id(1) * TILE_BYTES
        if start < length:
            offsets = start + tl.arange(0, TILE_BYTES)
            mask = offsets < length
            src = (src_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
            dst = (dst_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
            tl.store(dst, tl.load(src, mask=mask), mask=mask)

else:
    _copy_spans_kernel = None


def _descriptor(spans: list[CopySpan], device: torch.device) -> torch.Tensor:
    """The spans as the kernel reads them: source, destination, length a row.

    Row-major, so the whole thing crosses in a single transfer. A tensor per
    column — three `torch.tensor(list, device=...)` calls — was four times
    this, being three allocations and three separate pageable copies. Built
    fresh each time: reusing one buffer measured the same and would have
    raised the question of when the last transfer out of it finished.
    """
    rows = np.empty((len(spans), 3), dtype=np.int64)
    rows[:, 0] = [s.src_ptr for s in spans]
    rows[:, 1] = [s.dst_ptr for s in spans]
    rows[:, 2] = [s.num_bytes for s in spans]
    return torch.from_numpy(rows).to(device)


def launch_copy_spans(spans: list[CopySpan], device: torch.device) -> None:
    """Copy all spans with one descriptor-driven Triton launch.

    The descriptor is one row per *span*, and the grid's second axis cuts each
    span into tiles on the device. Cutting them on the host instead — which
    this did — makes the descriptor as long as the tile count rather than the
    span count, and that was almost the whole cost: a DeepSeek-V4 checkpoint
    image is 2,632 tiles against 135 spans, and describing it took 0.60 ms
    against 0.018 ms of actually copying. It now takes 0.04 ms.

    The grid has to be as tall as the widest span, so where spans differ in
    size most programs find nothing to do — 94% of them on that same image,
    whose spans run from 8 KiB to 1.4 MB. Measured before being believed: an
    empty program is cheap enough that the trade is not close.
    """
    if not spans:
        return
    if _copy_spans_kernel is None:
        raise RuntimeError("paged state copy requires Triton")
    widest = max(s.num_bytes for s in spans)
    _copy_spans_kernel[(len(spans), -(-widest // _TILE_BYTES))](
        _descriptor(spans, device),
        TILE_BYTES=_TILE_BYTES,
        num_warps=8,
    )
