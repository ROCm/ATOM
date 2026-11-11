# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 token-row tiling shared by Kimi K1 and K2 substrates.

gfx1250 WMMA owns sixteen M rows per instruction.  Kimi's gfx1250 substrate
therefore treats S=17..48 as two or three independent row tiles; it never
allocates an S-by-K activation slab in LDS.  This module is pure Python so
builders and tests can prove complete ownership before device emission.
"""

from __future__ import annotations

from dataclasses import dataclass

WMMA_TOKEN_ROWS = 16
GFX1250_MAX_TOKENS = 48


@dataclass(frozen=True)
class TokenTile:
    index: int
    row_start: int
    rows: int

    @property
    def row_stop(self) -> int:
        return self.row_start + self.rows

    @property
    def row_range(self) -> tuple[int, ...]:
        return tuple(range(self.row_start, self.row_stop))

    def global_row(self, local_row: int) -> int:
        if not 0 <= local_row < self.rows:
            raise ValueError(f"local row {local_row} outside tile {self.index}")
        return self.row_start + local_row


@dataclass(frozen=True)
class Gfx1250TokenTiling:
    """Complete, non-overlapping S-row partition into M16 WMMA tiles."""

    tokens: int

    def __post_init__(self) -> None:
        if not 1 <= self.tokens <= GFX1250_MAX_TOKENS:
            raise ValueError(
                f"gfx1250 tokens must be in [1, {GFX1250_MAX_TOKENS}], "
                f"got {self.tokens}"
            )

    @property
    def tiles(self) -> tuple[TokenTile, ...]:
        return tuple(
            TokenTile(
                index,
                row_start,
                min(WMMA_TOKEN_ROWS, self.tokens - row_start),
            )
            for index, row_start in enumerate(range(0, self.tokens, WMMA_TOKEN_ROWS))
        )

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def lds_rows(self) -> int:
        """Maximum rows any reusable tile-local LDS allocation needs."""

        return min(self.tokens, WMMA_TOKEN_ROWS)

    def tile(self, index: int) -> TokenTile:
        if not 0 <= index < self.tile_count:
            raise ValueError(f"tile {index} outside [0, {self.tile_count})")
        return self.tiles[index]

    def owner(self, row: int) -> tuple[int, int]:
        """Return ``(tile, local_row)`` for one global token row."""

        if not 0 <= row < self.tokens:
            raise ValueError(f"token row {row} outside [0, {self.tokens})")
        return row // WMMA_TOKEN_ROWS, row % WMMA_TOKEN_ROWS
