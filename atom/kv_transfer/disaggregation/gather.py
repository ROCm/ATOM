# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GatherLayout:
    """One source unit's strided slices covering a complete destination unit.

    Units may be contiguous token rows or physical PAGEs. Ownership is supplied
    separately as the source index tensor; this describes only physical bytes.
    ``first``, ``src_stride`` and ``width`` describe strided slices, such as one
    rank's heads of a PAGE; ``gather_units`` copies whole units only and
    refuses the rest.
    """

    source_bytes: int
    unit_bytes: int
    first: int
    src_stride: int
    width: int
    # `(src offset, bytes)` slices laid end to end, where one stride cannot
    # describe them (NVFP4 pages: data and scale runs of different widths).
    segments: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        if self.segments:
            self._check_segments()
            return
        if min(self.source_bytes, self.unit_bytes, self.src_stride, self.width) <= 0:
            raise ValueError("Invalid gather byte geometry")
        if (
            self.first < 0
            or any(
                x % 4
                for x in (
                    self.source_bytes,
                    self.unit_bytes,
                    self.first,
                    self.src_stride,
                    self.width,
                )
            )
            or self.unit_bytes % self.width
        ):
            raise ValueError("Gather slices must cover aligned 4-byte units")
        end = (
            self.first
            + (self.unit_bytes // self.width - 1) * self.src_stride
            + self.width
        )
        if end > self.source_bytes:
            raise ValueError("Gather slices exceed the source unit")

    def _check_segments(self) -> None:
        if len(self.segments) > MAX_SEGMENTS:
            raise ValueError(f"At most {MAX_SEGMENTS} gather segments")
        if sum(w for _, w in self.segments) != self.unit_bytes or any(
            w <= 0 or off < 0 or off % 4 or w % 4 or off + w > self.source_bytes
            for off, w in self.segments
        ):
            raise ValueError("Gather segments must tile the unit in 4-byte words")

    @classmethod
    def from_slices(cls, source_bytes: int, unit_bytes: int, slices) -> GatherLayout:
        """The layout of `[src offset, dst offset, bytes]` rows filling a unit in
        order: one stride where the rows share one, segments otherwise."""
        rows = [tuple(int(x) for x in row) for row in slices]
        if [dst for _, dst, _ in rows] != [
            sum(w for *_, w in rows[:i]) for i in range(len(rows))
        ]:
            raise ValueError("Gather slices must fill the destination unit in order")
        first, _, width = rows[0]
        stride = rows[1][0] - first if len(rows) > 1 else source_bytes
        if all(r == (first + i * stride, i * width, width) for i, r in enumerate(rows)):
            return cls(source_bytes, unit_bytes, first, stride, width)
        segments = tuple((src, w) for src, _, w in rows)
        return cls(source_bytes, unit_bytes, first, source_bytes, width, segments)

    @property
    def whole_unit(self) -> bool:
        return (
            not self.segments
            and self.first == 0
            and self.width == self.source_bytes == self.unit_bytes
        )


MAX_SEGMENTS = 4


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _gather_strided_units(
        src,
        indices,
        dst,
        SRC_WORDS: tl.constexpr,
        DST_WORDS: tl.constexpr,
        FIRST: tl.constexpr,
        STRIDE: tl.constexpr,
        WIDTH: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        unit = tl.program_id(0)
        word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        block_id = tl.load(indices + unit).to(tl.int64)
        src_offset = (
            block_id * SRC_WORDS + FIRST + word // WIDTH * STRIDE + word % WIDTH
        )
        value = tl.load(src + src_offset, mask=word < DST_WORDS)
        tl.store(dst + unit * DST_WORDS + word, value, mask=word < DST_WORDS)

    @triton.jit
    def _gather_segment_units(
        src,
        indices,
        dst,
        SRC_WORDS: tl.constexpr,
        DST_WORDS: tl.constexpr,
        O0: tl.constexpr,
        O1: tl.constexpr,
        O2: tl.constexpr,
        O3: tl.constexpr,
        E0: tl.constexpr,
        E1: tl.constexpr,
        E2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        unit = tl.program_id(0)
        word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        block_id = tl.load(indices + unit).to(tl.int64)
        # segment k covers destination words [E(k-1), Ek); Ok is its source start
        offset = tl.where(
            word < E0,
            O0 + word,
            tl.where(
                word < E1,
                O1 + word - E0,
                tl.where(word < E2, O2 + word - E1, O3 + word - E2),
            ),
        )
        value = tl.load(src + block_id * SRC_WORDS + offset, mask=word < DST_WORDS)
        tl.store(dst + unit * DST_WORDS + word, value, mask=word < DST_WORDS)

except ImportError:  # CPU layout/protocol tests do not require Triton.
    _gather_strided_units = None
    _gather_segment_units = None


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy only this rank's byte slices; no dequantization or KV format change."""
    if source.dtype != torch.uint8 or out.dtype != torch.uint8:
        raise ValueError("KV gather requires byte views")
    if not source.is_contiguous() or not out.is_contiguous():
        raise ValueError("KV gather requires contiguous PAGE and staging views")
    if (
        source.numel() % plan.source_bytes
        or out.numel() != indices.numel() * plan.unit_bytes
    ):
        raise ValueError("KV gather unit/staging size mismatch")
    if not indices.numel():
        return
    if plan.whole_unit:
        torch.index_select(
            source.view(-1, plan.source_bytes),
            0,
            indices,
            out=out.view(-1, plan.unit_bytes),
        )
    elif plan.segments:
        _gather_segments(source, indices, out, plan)
    elif source.is_cuda:
        _gather_strided_units[
            (indices.numel(), triton.cdiv(plan.unit_bytes // 4, 256))
        ](
            source.view(torch.int32),
            indices,
            out.view(torch.int32),
            SRC_WORDS=plan.source_bytes // 4,
            DST_WORDS=plan.unit_bytes // 4,
            FIRST=plan.first // 4,
            STRIDE=plan.src_stride // 4,
            WIDTH=plan.width // 4,
            BLOCK=256,
            num_warps=1,
        )
    else:
        # Same layout on CPU for byte-exact protocol tests with a synchronous NIC.
        src = source.reshape(-1, plan.source_bytes)
        dst = out.view(indices.numel(), plan.unit_bytes)
        for offset in range(0, plan.unit_bytes, plan.width):
            first = plan.first + offset // plan.width * plan.src_stride
            dst[:, offset : offset + plan.width] = src[
                indices, first : first + plan.width
            ]


def _gather_segments(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """`plan.segments` of each indexed source unit, end to end in `out`."""
    if source.is_cuda:
        offsets = [off // 4 for off, _ in plan.segments]
        ends = list(itertools.accumulate(w // 4 for _, w in plan.segments))
        # pad to four segments: an empty one starts where the last ended
        offsets += [offsets[-1]] * (MAX_SEGMENTS - len(offsets))
        ends += [ends[-1]] * (MAX_SEGMENTS - len(ends))
        _gather_segment_units[
            (indices.numel(), triton.cdiv(plan.unit_bytes // 4, 256))
        ](
            source.view(torch.int32),
            indices,
            out.view(torch.int32),
            SRC_WORDS=plan.source_bytes // 4,
            DST_WORDS=plan.unit_bytes // 4,
            O0=offsets[0],
            O1=offsets[1],
            O2=offsets[2],
            O3=offsets[3],
            E0=ends[0],
            E1=ends[1],
            E2=ends[2],
            BLOCK=256,
            num_warps=1,
        )
        return
    src = source.reshape(-1, plan.source_bytes)
    dst = out.view(indices.numel(), plan.unit_bytes)
    at = 0
    for off, width in plan.segments:
        dst[:, at : at + width] = src[indices, off : off + width]
        at += width
