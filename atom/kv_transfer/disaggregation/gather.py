# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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

    def __post_init__(self):
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

    @classmethod
    def from_slices(cls, source_bytes: int, unit_bytes: int, slices) -> GatherLayout:
        """The layout of ``[src offset, dst offset, bytes]`` rows, such as
        ``consumer_head_slices`` cuts, that fill a unit in order at one stride."""
        rows = [tuple(int(x) for x in row) for row in slices]
        first, _, width = rows[0]
        stride = rows[1][0] - first if len(rows) > 1 else source_bytes
        if unit_bytes != width * len(rows) or any(
            row != (first + i * stride, i * width, width) for i, row in enumerate(rows)
        ):
            raise ValueError("Nonuniform head runs need a physical layout adapter")
        return cls(source_bytes, unit_bytes, first, stride, width)

    @property
    def whole_unit(self) -> bool:
        return self.first == 0 and self.width == self.source_bytes == self.unit_bytes


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy this rank's byte slices of the units ``indices`` names into ``out``.

    ``source`` and ``out`` are contiguous byte views, ``out`` exactly one unit
    per index; a strided ``plan`` needs them on GPU. No dequantization or KV
    format change.
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
    from atom.kv_transfer.disaggregation.gather_kernels import gather_strided_units

    words = plan.unit_bytes // 4
    gather_strided_units[(indices.numel(), -(-words // 256))](
        source.view(torch.int32),
        indices,
        out.view(torch.int32),
        SRC_WORDS=plan.source_bytes // 4,
        DST_WORDS=words,
        FIRST=plan.first // 4,
        STRIDE=plan.src_stride // 4,
        WIDTH=plan.width // 4,
        BLOCK=256,
        num_warps=1,
    )
