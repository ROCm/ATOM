# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Zero the same entries of several `[layers, entries, ...]` views in one launch.

`view.index_fill_(1, idx, 0)` over a pool field is one fill per field, and on a
pool past 2 GiB PyTorch's 32-bit indexing splits each fill again -- one kernel
per layer for a paged K or V field. Every one of those kernels also writes the
field's own element type a lane at a time, which for the pools' `uint8` byte
views is one byte a lane. A PAGE checkpoint's released units cost 26 such
kernels on Qwen3.5 (11 MHA layers, four fields), 1.2 us a unit, where the bytes
they write are worth 0.05 us at HBM speed.

What each of those fills writes is a set of contiguous chunks: element
`[layer, entry]` of a view whose trailing dims are dense is `chunk` contiguous
bytes at `layer_base + entry * entry_stride`. So the views are described once,
as spans -- one per (view, layer): that base, that stride, that length, all in
bytes -- and the spans are cut into tiles. A launch is then one program per
(entry, tile), storing 16 B a lane.

`SpanZeroer` is the eager entry point a step's maintenance calls: it stages the
plan's tables and the entries through pinned memory in ONE upload and launches
once, so it never synchronizes the stream (a pageable index upload, or a
first-use upload of resident tables, would wait out the queued forward). The
kernel itself reads one device table with a grid set by the entry count, so
`launch_zero_spans` over a device-resident table is CUDA-graph safe.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

# Bytes one program zeroes, at `_NUM_WARPS` warps. Measured on MI455X over the
# Qwen3.5 c1024 pool (705 / 1786 units, 516 MB / 1.3 GB): 16 KiB x 4 warps is
# 38.8 / 81.1 us, 13.3 / 16.1 TB/s; 4 KiB tiles lose at the mean (44.6 us),
# 8 warps at 4 KiB fall to 8.5 TB/s, and 8-32 KiB at 2-8 warps are within 5%.
_TILE_BYTES = 16384
_NUM_WARPS = 4
# Every span base, stride and length is a multiple of this, which is what lets
# the kernel store 16 B at a time; the pools align fields to 256 B.
_ALIGN_BYTES = 16


def _dense_trailing(view: torch.Tensor) -> bool:
    """Whether `view[l, i]` is one contiguous chunk for every `l` and `i`."""
    expected = 1
    for size, stride in zip(reversed(view.shape[2:]), reversed(view.stride()[2:])):
        if size != 1 and stride != expected:
            return False
        expected *= size
    return True


def views_key(views: Sequence[torch.Tensor]) -> tuple:
    """What a plan was built from: it stays valid while this is unchanged."""
    return tuple((v.data_ptr(), tuple(v.shape), v.stride()) for v in views)


class ZeroSpanPlan:
    """The chunks an entry index addresses in each of `views`, as int64 tables.

    Every view is `[layers, entries, ...]` with dense trailing dims, the entry
    axis being dim 1 as `index_fill_(1, ...)` takes it. Built from the views'
    addresses, so a plan is valid while their storage is (`key`).

    Tables, all int64: `spans` (S, 3) = (address of the entry-0 chunk, entry
    stride, chunk length) in bytes, and per tile `span_of_tile` and
    `tile_start` (its byte offset inside the span).
    """

    def __init__(self, views: Sequence[torch.Tensor]):
        if not views:
            raise ValueError("a zero plan needs at least one view")
        devices = {v.device for v in views}
        if len(devices) != 1:
            raise ValueError(f"views span devices {sorted(map(str, devices))}")
        self.device = next(iter(devices))
        self.num_entries = min(int(v.shape[1]) for v in views)
        bases: list[int] = []
        strides: list[int] = []
        lengths: list[int] = []
        for view in views:
            if view.dim() < 2:
                raise ValueError(
                    f"a view needs [layers, entries, ...], got {view.shape}"
                )
            if not _dense_trailing(view):
                raise ValueError(
                    f"view {tuple(view.shape)} / {view.stride()} is not dense past "
                    "dim 1, so one of its elements is not a single chunk"
                )
            item = view.element_size()
            chunk = item * int(np.prod(view.shape[2:], dtype=np.int64))
            if chunk == 0:
                continue
            for layer in range(int(view.shape[0])):
                bases.append(view.data_ptr() + layer * view.stride(0) * item)
                strides.append(view.stride(1) * item)
                lengths.append(chunk)
        for name, values in (("base", bases), ("stride", strides), ("length", lengths)):
            bad = [v for v in values if v % _ALIGN_BYTES]
            if bad:
                raise ValueError(
                    f"span {name}s must be multiples of {_ALIGN_BYTES} B, got {bad[:4]}"
                )
        self.key = views_key(views)
        i64 = np.int64
        self.spans = np.array(list(zip(bases, strides, lengths)), dtype=i64).reshape(
            -1, 3
        )
        counts = -(-np.array(lengths, dtype=i64) // _TILE_BYTES)
        self.span_of_tile = np.repeat(np.arange(len(lengths), dtype=i64), counts)
        first = np.cumsum(counts) - counts
        within = np.arange(int(counts.sum()), dtype=i64) - np.repeat(first, counts)
        self.tile_start = within * _TILE_BYTES
        self.bytes_per_entry = int(sum(lengths))
        # The three tables back to back, each padded to 16 B, as `SpanZeroer`
        # stages them ahead of the entries (`header_offsets` says where).
        self.header = np.concatenate(
            [
                np.pad(p, (0, p.size % 2))
                for p in (self.spans.reshape(-1), self.span_of_tile, self.tile_start)
            ]
        )

    @property
    def num_tiles(self) -> int:
        return int(self.span_of_tile.size)

    def header_offsets(self) -> tuple[int, int, int]:
        """Where `header`'s three tables start, in int64 elements."""
        sot = _even(self.spans.size)
        return 0, sot, sot + _even(self.span_of_tile.size)


def _even(n: int) -> int:
    return n + n % 2


if triton is not None:

    @triton.jit
    def _zero_spans_kernel(
        table,
        tile_span_at,
        tile_start_at,
        entries_at,
        num_tiles,
        TILE: tl.constexpr,
    ):
        # One program per (entry, tile). `table` is a plan's header followed by
        # the entries (`ZeroSpanPlan.header`), the offsets in int64 elements. A
        # span row is a (view, layer): the address of its entry-0 chunk, the
        # entry stride, the chunk length, in bytes and multiples of 16, so the
        # store is 16 B a lane (rows of four int32) under a row mask for a
        # span's short last tile.
        pid = tl.program_id(0)
        j = pid // num_tiles
        tile = pid % num_tiles
        span = tl.load(table + tile_span_at + tile)
        start = tl.load(table + tile_start_at + tile)
        base = tl.load(table + span * 3)
        stride = tl.load(table + span * 3 + 1)
        length = tl.load(table + span * 3 + 2)
        entry = tl.load(table + entries_at + j)
        dst = tl.multiple_of(
            (base + entry * stride + start).to(tl.pointer_type(tl.int32)), 16
        )
        r = tl.arange(0, TILE // 16)
        c = tl.arange(0, 4)
        mask = (start + r[:, None] * 16 + c[None, :] * 0) < length
        tl.store(
            dst + r[:, None] * 4 + c[None, :],
            tl.zeros([TILE // 16, 4], dtype=tl.int32),
            mask=mask,
        )

else:
    _zero_spans_kernel = None


def launch_zero_spans(table: torch.Tensor, plan: ZeroSpanPlan, count: int) -> None:
    """Zero entry `table[header:][j]` of every span of `plan`, for j < `count`.

    `table` is int64 on the views' device: `plan.header` followed by the
    entries. Nothing is read back to the host and the grid is
    `count * plan.num_tiles`, so a launch over a resident table is CUDA-graph
    safe. The kernel neither wraps negative entries the way `index_fill_`
    does nor bounds-checks them: validate where the values come from.
    """
    if count == 0 or plan.num_tiles == 0:
        return
    if _zero_spans_kernel is None:
        raise RuntimeError("zero_spans requires Triton")
    _, tile_span_at, tile_start_at = plan.header_offsets()
    _zero_spans_kernel[(count * plan.num_tiles,)](
        table,
        tile_span_at,
        tile_start_at,
        plan.header.size,
        plan.num_tiles,
        TILE=_TILE_BYTES,
        num_warps=_NUM_WARPS,
    )


class SpanZeroer:
    """`view.index_fill_(1, entries, 0)` for every view at once, eagerly.

    One pinned staging upload (the plan's tables and the entries together) and
    one launch a call. Staging is a ring of two pinned/device buffer pairs, each
    fenced on its last upload, because a non-blocking H2D reads its pinned
    source when the stream reaches it. Plans are cached by `views_key`, so a
    reallocated pool gets a new one rather than a stale address.

    Entries come from the host -- that is where the callers hold them -- and are
    checked against `[0, entries)` there, before any address is formed.
    """

    _SLOTS = 2

    def __init__(self) -> None:
        self._plans: dict[tuple, ZeroSpanPlan] = {}
        self._ring: list[tuple[torch.Tensor, torch.Tensor, torch.cuda.Event] | None]
        self._ring = [None] * self._SLOTS
        self._next = 0

    def plan(self, views: Sequence[torch.Tensor]) -> ZeroSpanPlan:
        key = views_key(views)
        plan = self._plans.get(key)
        if plan is None:
            if len(self._plans) >= 8:
                self._plans.clear()
            plan = self._plans[key] = ZeroSpanPlan(views)
        return plan

    def _slot(self, need: int, device: torch.device):
        i = self._next
        self._next = (i + 1) % self._SLOTS
        entry = self._ring[i]
        if entry is not None:
            # The slot's last upload may still be reading its pinned source.
            entry[2].synchronize()
            host, dev, _ = entry
            if host.numel() >= need and dev.device == device:
                return i, host, dev
        cap = 1 << max(12, (need - 1).bit_length())
        host = torch.empty(cap, dtype=torch.int64, pin_memory=True)
        dev = torch.empty(cap, dtype=torch.int64, device=device)
        self._ring[i] = (host, dev, torch.cuda.Event())
        return i, host, dev

    def __call__(
        self, views: Sequence[torch.Tensor], entries: Sequence[int] | np.ndarray
    ) -> None:
        if isinstance(entries, np.ndarray):
            idx = entries.astype(np.int64, copy=False).reshape(-1)
        else:
            idx = np.fromiter(entries, dtype=np.int64, count=len(entries))
        if idx.size == 0:
            return
        plan = self.plan(views)
        lo, hi = int(idx.min()), int(idx.max())
        if lo < 0 or hi >= plan.num_entries:
            raise IndexError(
                f"entries span [{lo}, {hi}], outside the views' [0, {plan.num_entries})"
            )
        header = plan.header
        n_head = header.size
        total = n_head + idx.size
        i, host, dev = self._slot(total, plan.device)
        hnp = host.numpy()
        hnp[:n_head] = header
        hnp[n_head:total] = idx
        dev[:total].copy_(host[:total], non_blocking=True)
        self._ring[i][2].record()
        launch_zero_spans(dev, plan, idx.size)
