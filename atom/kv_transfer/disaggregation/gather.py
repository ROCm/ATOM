# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

# A segment gather passes each segment's bounds to its kernel as constants; four fit
# the widest named-run page, NVFP4 K/V (K data, K scales, V data, V scales).
MAX_SEGMENTS = 4


@dataclass(frozen=True)
class GatherLayout:
    """One source unit's strided slices covering a complete destination unit.

    Units may be contiguous token rows or physical PAGEs. Ownership is supplied
    separately as the source index tensor; this describes only physical bytes.
    ``first``, ``src_stride`` and ``width`` describe strided slices, such as one
    rank's heads of a PAGE.
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
            raise ValueError(
                f"Gather segments {self.segments}: the segment kernel takes at "
                f"most {MAX_SEGMENTS}"
            )
        if (
            self.source_bytes % 4
            or sum(size for _, size in self.segments) != self.unit_bytes
            or any(
                size <= 0
                or offset < 0
                or offset % 4
                or size % 4
                or offset + size > self.source_bytes
                for offset, size in self.segments
            )
        ):
            raise ValueError(
                f"Gather segments {self.segments} must tile a {self.unit_bytes} B "
                f"unit in 4-byte words of a {self.source_bytes} B source unit"
            )

    @classmethod
    def from_slices(cls, source_bytes: int, unit_bytes: int, slices) -> GatherLayout:
        """The layout of ``[src offset, dst offset, bytes]`` rows that fill a unit in
        order, such as ``consumer_head_slices`` cuts: one stride, else segments."""
        rows = [tuple(int(x) for x in row) for row in slices]
        ends = list(itertools.accumulate(size for *_, size in rows))
        if ends[-1] != unit_bytes or [dst for _, dst, _ in rows] != [0, *ends[:-1]]:
            raise ValueError("Gather slices must fill the destination unit in order")
        first, _, width = rows[0]
        stride = rows[1][0] - first if len(rows) > 1 else source_bytes
        if all(
            (src, size) == (first + i * stride, width)
            for i, (src, _, size) in enumerate(rows)
        ):
            return cls(source_bytes, unit_bytes, first, stride, width)
        segments = tuple((src, size) for src, _, size in rows)
        return cls(source_bytes, unit_bytes, first, source_bytes, width, segments)

    @property
    def whole_unit(self) -> bool:
        return (
            not self.segments
            and self.first == 0
            and self.width == self.source_bytes == self.unit_bytes
        )


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy this rank's byte slices of the units ``indices`` names into ``out``.

    ``source`` and ``out`` are contiguous byte views, ``out`` exactly one unit
    per index; a strided or segment ``plan`` needs them on GPU. No dequantization
    or KV format change.
    """
    if not indices.numel():
        return
    if plan.whole_unit:
        torch.index_select(
            source.view(-1, plan.source_bytes),
            0,
            indices,
            out=out.view(-1, plan.unit_bytes),
        )
        return
    # Imported here, so the whole-unit path runs without Triton.
    from atom.kv_transfer.disaggregation import gather_kernels

    if plan.segments:
        kernel, constants = gather_kernels.gather_segment_units, _segment_bounds(plan)
    else:
        kernel = gather_kernels.gather_strided_units
        constants = {
            "FIRST": plan.first // 4,
            "STRIDE": plan.src_stride // 4,
            "WIDTH": plan.width // 4,
        }
    words = plan.unit_bytes // 4
    kernel[(indices.numel(), -(-words // 256))](
        source.view(torch.int32),
        indices,
        out.view(torch.int32),
        SRC_WORDS=plan.source_bytes // 4,
        DST_WORDS=words,
        BLOCK=256,
        num_warps=1,
        **constants,
    )


def _segment_bounds(plan: GatherLayout) -> dict[str, int]:
    """``gather_segment_units``' word bounds: where each segment starts in the source
    unit (``O0``-``O3``) and ends in the destination (``E0``-``E2``)."""
    starts = [offset // 4 for offset, _ in plan.segments]
    ends = list(itertools.accumulate(size // 4 for _, size in plan.segments))
    # Pad to four with empty segments, each ending where the last one does.
    pad = MAX_SEGMENTS - len(starts)
    starts += starts[-1:] * pad
    ends += ends[-1:] * pad
    return {
        "O0": starts[0],
        "O1": starts[1],
        "O2": starts[2],
        "O3": starts[3],
        "E0": ends[0],
        "E1": ends[1],
        "E2": ends[2],
    }
