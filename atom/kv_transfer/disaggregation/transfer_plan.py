# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Map P/D rank ownership to physical gather units and destination PAGEs.

Topology selects token ownership. Region metadata selects byte addressing.
The transport consumes TransferPlan without knowing whether the model uses
MLA or another attention family.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from atom.kv_transfer.disaggregation.gather import GatherLayout
from atom.kv_transfer.disaggregation.landing_scatter import landing_segments
from atom.kv_transfer.disaggregation.sharded_transfer import (
    DCPShardPlan,
    build_dcp_shard_plan,
    coalesce_contiguous,
    pack_slots,
)

TOKEN_UNITS = "token_rows_v1"


@dataclass(frozen=True)
class TransferTopology:
    source_tp: int = 1
    source_dcp: int = 1
    target_tp: int = 1
    target_tp_rank: int = 0
    target_dcp: int = 1
    target_dcp_rank: int = 0
    interleave: int = 1


@dataclass(frozen=True)
class RegionLayout:
    source_region: int
    target_region: int
    target_base: int
    source_page_bytes: int
    target_page_bytes: int
    # Declared physical capability, not inferred from a byte width alone.
    token_contiguous: bool = False


@dataclass(frozen=True)
class PlannedRegion:
    source_region: int
    target_region: int
    target_base: int
    gather: GatherLayout


@dataclass(frozen=True)
class TransferPlan:
    regions: tuple[PlannedRegion, ...]
    source_units: np.ndarray
    destination_pages: np.ndarray
    units_per_page: int
    wire_layout: str

    @property
    def minimum_slot_bytes(self) -> int:
        """Space for one destination PAGE after ownership mapping."""
        return max(
            (r.gather.unit_bytes * self.units_per_page for r in self.regions),
            default=0,
        )

    def pack(self, slot_bytes: int) -> list[list[tuple[int, int, int, int]]]:
        return pack_slots(
            [r.gather.unit_bytes for r in self.regions],
            self.source_units.size,
            slot_bytes,
            self.units_per_page,
        )

    def ready_items(
        self, items: Sequence[tuple[int, int, int, int]], page_offset: int
    ) -> list[list[int]]:
        return [
            [
                self.regions[pos].target_region,
                page_offset * self.units_per_page + start,
                stop - start,
                offset,
            ]
            for pos, start, stop, offset in items
        ]

    def direct_descriptors(
        self,
        items: Sequence[tuple[int, int, int, int]],
        staging_base: int,
        pages_per_mr: Callable[[int], int],
    ) -> tuple[list[int], list[int], list[int]]:
        """Coalesce destination-adjacent PAGEs within one region and one MR."""
        runs = []
        for pos, start, stop, offset in items:
            region = self.regions[pos]
            page_bytes = region.gather.unit_bytes * self.units_per_page
            src, dst, size = landing_segments(
                np.array([start]),
                np.array([stop - start]),
                np.array([staging_base + offset]),
                np.array([region.target_base]),
                np.array([page_bytes]),
                self.destination_pages,
                self.units_per_page,
                0,
            )
            mr = (dst - region.target_base) // (pages_per_mr(page_bytes) * page_bytes)
            boundary = np.concatenate(([True], mr[1:] != mr[:-1]))
            runs.append(coalesce_contiguous(src, dst, size, boundary))
        return tuple(np.concatenate(part).tolist() for part in zip(*runs))


def build_transfer_plan(
    topology: TransferTopology,
    layouts: Sequence[RegionLayout],
    source_pages: Sequence[int],
    destination_pages: Sequence[int],
    block_size: int,
    *,
    token_ownership: DCPShardPlan | None = None,
) -> TransferPlan:
    """Plan supported topology/layout combinations; never guess a token layout.

    PP layer ownership is already expressed by each region's source/target map.
    DCP ownership is calculated before addressing bytes; a TP-replicated plane
    is owned by token alone, whatever the two TP sizes. token_ownership may
    reuse the same topology's plan from the direct/index path, so a
    mixed-region transfer does not calculate DCP ownership twice.
    """
    t = topology
    if min(t.source_tp, t.source_dcp, t.target_tp, t.target_dcp, block_size) <= 0:
        raise ValueError("Parallel sizes and block size must be positive")
    if (
        not 0 <= t.target_tp_rank < t.target_tp
        or not 0 <= t.target_dcp_rank < t.target_dcp
    ):
        raise ValueError("Target rank is outside its parallel group")
    if t.source_dcp != 1:
        raise ValueError("Staging requires a non-DCP producer")
    if t.target_dcp == 1:
        raise ValueError("Staging without DCP token ownership needs a PAGE adapter")
    source_pages = np.asarray(source_pages, dtype=np.int64)
    destination_pages = np.asarray(destination_pages, dtype=np.int64)
    if (source_pages < 0).any() or (destination_pages < 0).any():
        raise ValueError("Negative physical PAGE index")
    ownership = (
        token_ownership
        if token_ownership is not None
        else build_dcp_shard_plan(
            source_pages,
            block_size=block_size,
            dcp_size=t.target_dcp,
            dcp_rank=t.target_dcp_rank,
            interleave_size=t.interleave,
            dst_pages=destination_pages.size,
        )
    )
    indices = ownership.source_token_per_dst_token()[: ownership.landing_row_count()]
    regions = []
    for layout in layouts:
        src, dst = layout.source_page_bytes, layout.target_page_bytes
        if not layout.token_contiguous:
            raise ValueError(
                "Token ownership needs an explicit token-contiguous layout adapter"
            )
        if src != dst:
            raise ValueError("Token ownership needs equal source and target PAGEs")
        if src % block_size:
            raise ValueError("PAGE width is not divisible by token count")
        width = src // block_size
        regions.append(
            PlannedRegion(
                layout.source_region,
                layout.target_region,
                layout.target_base,
                GatherLayout(width, width, 0, width, width),
            )
        )
    return TransferPlan(
        tuple(regions), indices, destination_pages, block_size, TOKEN_UNITS
    )
