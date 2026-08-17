# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Descriptor-driven bitwise copy between segmented GPU byte streams."""

from __future__ import annotations

from collections.abc import Sequence
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
class SegmentedCopyPlan:
    """Where two ordered byte streams meet, in offsets rather than addresses.

    Which source segment meets which destination segment, at what offset into
    each and for how many bytes, follows from the two streams' *sizes* alone.
    Addresses enter only when a copy is issued.

    Holding them apart is what makes a finely segmented copy affordable. A
    caller whose geometry outlives its copies — a checkpoint image is the same
    shape for the life of the pool — walks the intersection once here and then
    spends a few vector adds per copy where it used to spend a Python loop per
    span. Measured on a DeepSeek-V4 image: 0.53 us a span against about none,
    which is the difference between an image cut fine enough to save PAGE
    units and one that costs more host time than the units are worth.

    The five arrays are parallel and one span long. `src` and `dst` name the
    roles the plan was built in, not a direction: an intersection is symmetric,
    so `write_descriptor` can read it either way and a restore reuses the plan
    its store was cut by.
    """

    src_seg: np.ndarray
    src_off: np.ndarray
    dst_seg: np.ndarray
    dst_off: np.ndarray
    length: np.ndarray

    @property
    def num_spans(self) -> int:
        return int(self.length.size)

    @property
    def widest(self) -> int:
        """Bytes in the longest span, which is what sets the kernel's grid."""
        return int(self.length.max()) if self.length.size else 0

    def write_descriptor(
        self,
        out: np.ndarray,
        src_bases: np.ndarray,
        dst_bases: np.ndarray,
        *,
        forward: bool = True,
    ) -> None:
        """Fill a `(num_spans, 3)` int64 block: source, destination, length.

        `src_bases` and `dst_bases` give one address per segment of each
        stream, which is where a caller's geometry enters — a slot's base plus
        a range's offset, a PAGE unit's base plus a region's. `forward=False`
        copies the destination stream back into the source instead.
        """
        src_col, dst_col = (0, 1) if forward else (1, 0)
        np.add(src_bases[self.src_seg], self.src_off, out=out[:, src_col])
        np.add(dst_bases[self.dst_seg], self.dst_off, out=out[:, dst_col])
        out[:, 2] = self.length


def plan_segmented_copy(
    src_sizes: Sequence[int],
    dst_sizes: Sequence[int],
    total_bytes: int,
) -> SegmentedCopyPlan:
    """Intersect two ordered byte streams into the spans a copy is made of."""
    total_bytes = int(total_bytes)
    if total_bytes < 0:
        raise ValueError("copy length must be non-negative")
    if sum(src_sizes) < total_bytes:
        raise ValueError("source segmented stream is shorter than the copy")
    if sum(dst_sizes) < total_bytes:
        raise ValueError("destination segmented stream is shorter than the copy")
    if any(size <= 0 for size in (*src_sizes, *dst_sizes)):
        raise ValueError("segmented streams cannot contain empty segments")

    src_seg: list[int] = []
    src_off: list[int] = []
    dst_seg: list[int] = []
    dst_off: list[int] = []
    length: list[int] = []
    src_i = dst_i = 0
    src_used = dst_used = 0
    remaining = total_bytes
    while remaining:
        src_left = src_sizes[src_i] - src_used
        dst_left = dst_sizes[dst_i] - dst_used
        nbytes = min(src_left, dst_left, remaining)
        src_seg.append(src_i)
        src_off.append(src_used)
        dst_seg.append(dst_i)
        dst_off.append(dst_used)
        length.append(nbytes)
        remaining -= nbytes
        src_used += nbytes
        dst_used += nbytes
        if src_used == src_sizes[src_i]:
            src_i += 1
            src_used = 0
        if dst_used == dst_sizes[dst_i]:
            dst_i += 1
            dst_used = 0
    i64 = np.int64
    return SegmentedCopyPlan(
        src_seg=np.array(src_seg, dtype=i64),
        src_off=np.array(src_off, dtype=i64),
        dst_seg=np.array(dst_seg, dtype=i64),
        dst_off=np.array(dst_off, dtype=i64),
        length=np.array(length, dtype=i64),
    )


if triton is not None:

    @triton.jit
    def _copy_spans_kernel(descriptor, TILE_BYTES: tl.constexpr):
        # A row is `write_descriptor`'s: source, destination, length. Triton
        # cannot read a plain module constant, so the three stay literal here
        # and the round-trip test is what holds the two ends to the same order.
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


def launch_copy_descriptor(
    descriptor: np.ndarray, widest: int, device: torch.device
) -> None:
    """Copy every span a descriptor names, in one launch.

    The descriptor is one row per *span*, row-major so the whole thing crosses
    in a single transfer, and the grid's second axis cuts each span into tiles
    on the device. Cutting them on the host instead — which this did — makes
    the descriptor as long as the tile count rather than the span count, and
    that was almost the whole cost: a DeepSeek-V4 checkpoint image is 2,632
    tiles against 135 spans, and describing it took 0.60 ms against 0.018 ms
    of actually copying.

    The grid has to be as tall as the widest span, so where spans differ in
    size most programs find nothing to do — 94% of them on that same image,
    whose spans run from 8 KiB to 1.4 MB. Measured before being believed: an
    empty program is cheap enough that the trade is not close.

    It does leave a residual cost per span, on the device rather than the
    host. A PAGE unit's plane region is 1.4 MB whatever the source ranges look
    like, so cutting the image finer adds grid rows without shortening them:
    134 spans measured 0.042 ms an op against 363 spans at 0.076, about
    0.15 us a span. That is a third of what describing one used to cost, and
    it is what a prefix-sum grid would go after if a much finer image ever
    made it worth the device-side search.

    Pageable on purpose. A 3 KB transfer measured the same from pinned memory,
    and pinning would make the descriptor's lifetime the caller's problem:
    `.to()` from pageable memory stages synchronously, so the buffer is free
    to be refilled the moment this returns.
    """
    if descriptor.shape[0] == 0:
        return
    if _copy_spans_kernel is None:
        raise RuntimeError("paged state copy requires Triton")
    _copy_spans_kernel[(descriptor.shape[0], -(-widest // _TILE_BYTES))](
        torch.from_numpy(descriptor).to(device),
        TILE_BYTES=_TILE_BYTES,
        num_warps=8,
    )
