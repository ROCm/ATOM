# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Byte gather primitives, independent of attention family and rank ownership."""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import torch

from atom.kv_transfer.disaggregation.sharded_transfer import coalesce_contiguous

# A gather program copies one table row of at most this many words, its BLOCK.
_PIECE_WORDS = 256


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
    from atom.kv_transfer.disaggregation.gather_kernels import gather_unit_pieces

    pieces = _device_pieces(plan, source.device)
    gather_unit_pieces[(indices.numel(), pieces.shape[0])](
        source.view(torch.int32),
        indices,
        out.view(torch.int32),
        pieces,
        plan.source_bytes // 4,
        plan.unit_bytes // 4,
        BLOCK=_PIECE_WORDS,
        num_warps=1,
    )


def _pieces(plan: GatherLayout) -> np.ndarray:
    """``plan.segments`` as int32 ``[src word, dst word, words]`` rows of at most
    ``_PIECE_WORDS`` words, one gather program each."""
    rows, at = [], 0
    for offset, size in plan.segments:
        words = size // 4
        for start in range(0, words, _PIECE_WORDS):
            rows.append(
                (offset // 4 + start, at + start, min(_PIECE_WORDS, words - start))
            )
        at += words
    return np.array(rows, dtype=np.int32)


@functools.cache
def _device_pieces(plan: GatherLayout, device: torch.device) -> torch.Tensor:
    # A blocking copy, so the table is whole before any worker's stream reads it.
    return torch.from_numpy(_pieces(plan)).to(device)
