# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Neutral contiguous-copy plan types shared by PAGE, SLOT, and Triton."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PageCopy:
    """One contiguous source unit and its staging-buffer destination."""

    block_id: int
    region_index: int
    src_addr: int
    dst_offset: int
    nbytes: int


@dataclass(frozen=True)
class CopyTile:
    """One valid Triton job from a possibly larger copy-plan entry."""

    src_addr: int
    dst_offset: int
    nbytes: int


def build_copy_tiles(
    plan: Sequence[object],
    *,
    tile_bytes: int,
) -> list[CopyTile]:
    """Expand copy ranges into only the tiles that contain valid bytes."""

    try:
        tile_bytes = operator.index(tile_bytes)
    except TypeError as exc:
        raise ValueError("copy tile bytes must be an integer") from exc
    if tile_bytes <= 0:
        raise ValueError(f"copy tile bytes must be > 0, got {tile_bytes}")

    tiles: list[CopyTile] = []
    for copy_index, copy in enumerate(plan):
        try:
            src_addr = operator.index(copy.src_addr)
            dst_offset = operator.index(copy.dst_offset)
            nbytes = operator.index(copy.nbytes)
        except (AttributeError, TypeError) as exc:
            raise TypeError(
                "copy plan entries must define integer src_addr, "
                "dst_offset, and nbytes"
            ) from exc
        if src_addr <= 0:
            raise ValueError(
                f"copy plan entry {copy_index} has invalid src_addr={src_addr}"
            )
        if dst_offset < 0:
            raise ValueError(
                f"copy plan entry {copy_index} has negative dst_offset={dst_offset}"
            )
        if nbytes <= 0:
            raise ValueError(
                f"copy plan entry {copy_index} has nonpositive nbytes={nbytes}"
            )
        for tile_offset in range(0, nbytes, tile_bytes):
            tiles.append(
                CopyTile(
                    src_addr=src_addr + tile_offset,
                    dst_offset=dst_offset + tile_offset,
                    nbytes=min(tile_bytes, nbytes - tile_offset),
                )
            )
    return tiles


__all__ = ["CopyTile", "PageCopy", "build_copy_tiles"]
