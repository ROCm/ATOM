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
    rank's heads of a PAGE; ``gather_units`` copies whole units only.
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

    @property
    def whole_unit(self) -> bool:
        return self.first == 0 and self.width == self.source_bytes == self.unit_bytes


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy the whole source units ``indices`` names into ``out``, in order.

    ``source`` and ``out`` are contiguous byte views, ``out`` exactly one unit
    per index, and ``plan`` a whole-unit layout. No dequantization or KV
    format change.
    """
    if not indices.numel():
        return
    torch.index_select(
        source.view(-1, plan.source_bytes),
        0,
        indices,
        out=out.view(-1, plan.unit_bytes),
    )
