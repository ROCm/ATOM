# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Descriptor-driven bitwise copy between segmented GPU byte streams."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

_TILE_BYTES = 4096


@dataclass(frozen=True, eq=False)
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

    The first five arrays are parallel and one span long. `src` and `dst` name
    the roles the plan was built in, not a direction: an intersection is
    symmetric, so `write_descriptor` can read it either way and a restore
    reuses the plan its store was cut by.

    The last two are the same spans cut into tiles, which is the unit the
    kernel actually runs on. Compared by identity (`eq=False`): the fields are
    arrays, so a generated `__eq__` would raise rather than answer.
    """

    src_seg: np.ndarray
    src_off: np.ndarray
    dst_seg: np.ndarray
    dst_off: np.ndarray
    length: np.ndarray
    # Which span a tile belongs to, and its byte offset inside that span.
    span_of_tile: np.ndarray
    tile_start: np.ndarray
    # The two above, per device, uploaded on first use. Fixed geometry, so one
    # upload serves every copy for the life of the plan.
    _resident: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def num_spans(self) -> int:
        return int(self.length.size)

    @property
    def num_tiles(self) -> int:
        return int(self.span_of_tile.size)

    def tiling_on(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(span_of_tile, tile_start)` resident on `device`.

        Uploaded pageably, which synchronizes the stream — acceptable only
        because it happens once per plan and the first copy is a warmup, not a
        request. A caller that reaches this from a live batch pays the whole
        outstanding queue for it.
        """
        tables = self._resident.get(device)
        if tables is None:
            tables = (
                torch.from_numpy(self.span_of_tile).to(device),
                torch.from_numpy(self.tile_start).to(device),
            )
            self._resident[device] = tables
        return tables

    def write_descriptor(
        self,
        out: np.ndarray,
        src_bases: np.ndarray,
        dst_bases: np.ndarray,
        *,
        forward: bool = True,
    ) -> None:
        """Fill `(copies * num_spans, 3)` int64 rows: source, destination, length.

        `src_bases` and `dst_bases` are `(copies, segments)` — one row of
        segment addresses per copy, which is where a caller's geometry enters:
        a slot's base plus a range's offset, a PAGE unit's base plus a
        region's. `out` holds the copies back to back in the same order.

        Every copy in one call goes the same way, so a caller with both
        directions makes two calls. `forward=False` copies the destination
        stream back into the source instead.

        The copies are filled in one pass rather than one at a time. At these
        sizes each of the three fills below is nearly all numpy call overhead
        — a span table is a few hundred entries — so paying it per copy is
        what made describing a batch a quarter of the whole copy path, once
        the kernel stopped being the bottleneck. Batched it is about 7x
        cheaper, and it is the same three lines with one more axis.
        """
        # Numpy would broadcast a short `dst_bases` rather than complain, and
        # every copy in the batch would then be aimed at the first image's
        # addresses -- silent cross-request corruption, at raw pointers, with
        # the other images' checkpoint records still claiming them. This is
        # where the pointers are made, so it is where the shapes are checked;
        # `launch_copy_descriptor` only inherits them. Both sides are asked
        # the same way, down to the rank: checking only the leading axis of
        # `dst_bases` would let a 1-D one reach `dst_bases[:, dst_seg]`, which
        # raises an index error about an array the caller did not pass.
        if (
            src_bases.ndim != 2
            or dst_bases.ndim != 2
            or len(dst_bases) != len(src_bases)
        ):
            raise ValueError(
                f"a copy needs one row of bases per copy on both sides, got "
                f"{src_bases.shape} and {dst_bases.shape}"
            )
        copies = len(src_bases)
        if out.shape != (copies * self.num_spans, 3):
            raise ValueError(
                f"a descriptor for {copies} copies of {self.num_spans} spans "
                f"must be {(copies * self.num_spans, 3)}, got {out.shape}"
            )
        src_col, dst_col = (0, 1) if forward else (1, 0)
        rows = out.reshape(copies, self.num_spans, 3)
        np.add(src_bases[:, self.src_seg], self.src_off, out=rows[:, :, src_col])
        np.add(dst_bases[:, self.dst_seg], self.dst_off, out=rows[:, :, dst_col])
        rows[:, :, 2] = self.length


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
    lengths = np.array(length, dtype=i64)
    span_of_tile, tile_start = _tiling(lengths)
    return SegmentedCopyPlan(
        src_seg=np.array(src_seg, dtype=i64),
        src_off=np.array(src_off, dtype=i64),
        dst_seg=np.array(dst_seg, dtype=i64),
        dst_off=np.array(dst_off, dtype=i64),
        length=lengths,
        span_of_tile=span_of_tile,
        tile_start=tile_start,
    )


def _tiling(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cut the spans into tiles: which span each is in, and where it starts.

    A copy kernel wants one program per tile *that exists*. Deriving that on
    the device would need a search; deriving it here needs none, because a
    plan's spans do not change. The result is two small arrays that go to the
    device once and serve every copy the plan describes.
    """
    counts = -(-lengths // _TILE_BYTES)
    total = int(counts.sum())
    span_of_tile = np.repeat(np.arange(lengths.size, dtype=np.int32), counts)
    # Exclusive prefix sum. Written as cumsum-minus-self rather than by
    # prepending a zero and dropping the last, which has no answer for a plan
    # with no spans at all: the prepended zero survives, and the subtraction
    # below then fails to broadcast against an empty `counts`.
    first = np.cumsum(counts) - counts
    within = np.arange(total, dtype=np.int64) - np.repeat(first, counts)
    return span_of_tile, within * _TILE_BYTES


if triton is not None:

    @triton.jit
    def _copy_tiles_kernel(
        descriptor,
        span_of_tile,
        tile_start,
        num_tiles,
        num_spans,
        TILE: tl.constexpr,
    ):
        # One program per tile that exists. The tiling is the plan's, resident
        # on the device, so which span this tile belongs to and where it
        # starts are two loads rather than a search.
        #
        # The two counts are ordinary arguments, not `tl.constexpr`: only
        # `TILE` has to be one, for `tl.arange`. Specialising on the other two
        # would key the compiled kernel to a pool geometry, so every image
        # shape would miss the on-disk cache and pay the JIT again -- for a
        # divide the copy does not notice.
        #
        # A row is `write_descriptor`'s: source, destination, length. Triton
        # cannot read a plain module constant, so the three stay literal here
        # and the round-trip test is what holds the two ends to the same order.
        pid = tl.program_id(0)
        op = pid // num_tiles
        tile = pid % num_tiles
        span = tl.load(span_of_tile + tile)
        start = tl.load(tile_start + tile)
        row = descriptor + (op * num_spans + span) * 3
        src_ptr = tl.load(row)
        dst_ptr = tl.load(row + 1)
        length = tl.load(row + 2)
        offsets = start + tl.arange(0, TILE)
        mask = offsets < length
        src = (src_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
        dst = (dst_ptr.to(tl.int64) + offsets).to(tl.pointer_type(tl.uint8))
        tl.store(dst, tl.load(src, mask=mask), mask=mask)

else:
    _copy_tiles_kernel = None


def launch_copy_descriptor(descriptor: torch.Tensor, plan: SegmentedCopyPlan) -> None:
    """Copy every span a resident descriptor names, in one launch.

    One row per span, row-major, holding several copies back to back — every
    op of a step goes out together.

    The upload is the caller's, and `descriptor` arrives on the device
    already. That is not tidiness: a pageable `torch.from_numpy(x).to(dev)`
    synchronizes the current stream, and this runs from `build()` with the
    previous forward still enqueued, so the host waits out the whole queue
    rather than the 800 KB. Measured behind 4 ms of work it cost 2.9 ms
    against 0.1 ms staged through pinned memory — a cost the transfer's own
    size says nothing about. `CpuGpuBuffer` is what the caller stages with,
    and what the rest of this repo already uses to avoid exactly this.

    The grid is one program per tile that exists. It used to be rectangular,
    `(spans, ceil(widest / TILE))`, which gives every span as many programs as
    the *widest* one needs: on a DeepSeek-V4 image, whose spans run 8 KiB to
    1.4 MB, that is 46,364 programs to do 2,631 tiles of work. One op could
    afford the waste; a batch cannot, and this path now always batches. At 256
    ops the rectangular grid measured 5.14 ms against 1.19 ms for this one.

    What makes the dense grid cheap is that the tiling is not a function of
    the copy — the plan's spans are fixed, so `span_of_tile` and `tile_start`
    are computed once and stay resident. No device-side search, and no
    `widest` for a caller to get wrong: passing one too small used to truncate
    every longer span silently, byte-correct on its prefix and stale on its
    tail.
    """
    rows = descriptor.shape[0]
    if rows == 0:
        return
    if _copy_tiles_kernel is None:
        raise RuntimeError("paged state copy requires Triton")
    # Checked because this is where host arithmetic becomes device pointers:
    # the kernel reads `descriptor` as a contiguous int64 (rows, 3) and indexes
    # it by op, so a wrong shape or dtype is a wrong address rather than an
    # error.
    if descriptor.dtype != torch.int64 or descriptor.ndim != 2:
        raise ValueError("a copy descriptor must be a 2-D int64 tensor")
    if descriptor.shape[1] != 3 or not descriptor.is_contiguous():
        raise ValueError("a copy descriptor must be contiguous with 3 columns")
    if rows % plan.num_spans:
        raise ValueError(
            f"descriptor of {rows} rows is not a whole number of "
            f"{plan.num_spans}-span copies"
        )
    span_of_tile, tile_start = plan.tiling_on(descriptor.device)
    _copy_tiles_kernel[(rows // plan.num_spans * plan.num_tiles,)](
        descriptor,
        span_of_tile,
        tile_start,
        plan.num_tiles,
        plan.num_spans,
        TILE=_TILE_BYTES,
        # Four warps, not more: this kernel is nothing but load and store, and
        # its speed turns out to be set by the width of one lane's access,
        # `TILE / (num_warps * 64)`. Sixteen bytes is the fast point -- a
        # 128-bit access -- and three unrelated (TILE, warps) pairs that land
        # on it measured within 0.5% of each other, while eight warps halves
        # the width and costs 12%. Raising this to fill more of the machine
        # makes it slower, so it is not a knob to turn up.
        num_warps=4,
    )
