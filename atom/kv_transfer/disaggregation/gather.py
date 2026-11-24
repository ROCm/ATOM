# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import coalesce_contiguous

# A gather program copies this many destination words of a unit, its BLOCK.
_BLOCK_WORDS = 256


@dataclass(frozen=True)
class GatherLayout:
    """The ``(src offset, bytes)`` segments of a source unit that, laid end to
    end, make one destination unit.

    Units may be contiguous token rows or physical PAGEs. Ownership is supplied
    separately as the source index tensor; this describes only physical bytes,
    such as one rank's heads of a PAGE.
    """

    source_bytes: int
    unit_bytes: int
    segments: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if (
            not self.segments
            or self.source_bytes % 4
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
        """The segments of ``[src offset, dst offset, bytes]`` rows that fill a unit
        in order, such as ``consumer_head_slices`` cuts; source-adjacent rows merge."""
        src, dst, size = np.asarray(slices, dtype=np.int64).reshape(-1, 3).T
        if (
            not size.size
            or size.sum() != unit_bytes
            or (dst != size.cumsum() - size).any()
        ):
            raise ValueError(
                f"{size.size} gather slices must fill a {unit_bytes} B unit in order"
            )
        src, _, size = coalesce_contiguous(src, dst, size)
        return cls(source_bytes, unit_bytes, tuple(zip(src.tolist(), size.tolist())))

    @property
    def whole_unit(self) -> bool:
        return self.segments == ((0, self.source_bytes),)


def gather_units(
    source: torch.Tensor, indices: torch.Tensor, out: torch.Tensor, plan: GatherLayout
):
    """Copy this rank's byte slices of the units ``indices`` names into ``out``.

    ``source`` and ``out`` are contiguous byte views, ``out`` exactly one unit
    per index; a plan that is not the whole unit needs them on GPU. No
    dequantization or KV format change.
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
    from atom.kv_transfer.disaggregation.gather_kernels import gather_unit_words

    words = plan.unit_bytes // 4
    gather_unit_words[(indices.numel(), -(-words // _BLOCK_WORDS))](
        source.view(torch.int32),
        indices,
        out.view(torch.int32),
        _device_src_offsets(plan, source.device),
        plan.source_bytes // 4,
        words,
        BLOCK=_BLOCK_WORDS,
        num_warps=1,
    )


def _src_offsets(plan: GatherLayout) -> np.ndarray:
    """The int32 source word of each destination word, ``plan.segments`` laid end
    to end, so one program's block of words may span many segments."""
    return np.concatenate(
        [
            np.arange(offset // 4, (offset + size) // 4, dtype=np.int32)
            for offset, size in plan.segments
        ]
    )


@functools.cache
def _device_src_offsets(plan: GatherLayout, device: torch.device) -> torch.Tensor:
    # A blocking copy, so the table is whole before any worker's stream reads it.
    return torch.from_numpy(_src_offsets(plan)).to(device)
