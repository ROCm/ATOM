# SPDX-License-Identifier: MIT
"""Owner-only PAGE fields and complete request STATE geometry for CSA2."""

from dataclasses import dataclass

import torch

from .entry_arena import EntryField, entry_bytes_for, field_extents
from .v4_pool_geometry import WindowParams


@dataclass(frozen=True)
class V41PoolGeometry:
    layers: int
    owners: tuple[tuple[int, int], ...]
    block_size: int
    window_size: int
    head_dim: int
    index_dim: int
    history_size: int = 3
    packed: bool = False

    def __post_init__(self):
        if (
            min(
                self.layers,
                self.block_size,
                self.window_size,
                self.head_dim,
                self.index_dim,
                self.history_size,
            )
            <= 0
        ):
            raise ValueError("Cache dimensions must be positive")
        if self.head_dim % 32 or self.index_dim % 32:
            raise ValueError("Cache dimensions must be divisible by 32")
        if not self.packed and self.row_bytes % 256:
            raise ValueError("BF16 attention rows must align to 256 bytes")
        if len({owner for owner, _ in self.owners}) != len(self.owners):
            raise ValueError("Each global owner must be declared once")
        for owner, ratio in self.owners:
            if not 0 <= owner < self.layers or ratio not in (1, 2):
                raise ValueError("Invalid global owner or compression ratio")
            if self.block_size % ratio:
                raise ValueError("PAGE token count must divide every compression group")

    @property
    def row_bytes(self):
        return self.head_dim * 2

    @property
    def alignment(self):
        return 256 if self.packed else self.row_bytes

    @property
    def main_row_bytes(self):
        return (
            self.head_dim // 2 + self.head_dim // 16 if self.packed else self.row_bytes
        )

    @property
    def index_row_bytes(self):
        return (
            self.index_dim // 2 + self.index_dim // 32
            if self.packed
            else self.index_dim * 2
        )

    @property
    def window_row_bytes(self):
        return self.head_dim + self.head_dim // 32 if self.packed else self.row_bytes

    @property
    def tail_owners(self):
        return tuple(owner for owner, ratio in self.owners if ratio == 2)

    @property
    def page_fields(self):
        return [
            EntryField(
                f"{kind}_{owner}",
                1,
                (self.block_size // ratio, dim),
                torch.uint8 if self.packed else torch.bfloat16,
                align=self.alignment,
            )
            for owner, ratio in self.owners
            for kind, dim in (
                ("main", self.main_row_bytes if self.packed else self.head_dim),
                ("index", self.index_row_bytes if self.packed else self.index_dim),
            )
        ]

    @property
    def state_fields(self):
        return [
            EntryField(
                "window",
                self.layers,
                (
                    self.window_size,
                    self.window_row_bytes if self.packed else self.head_dim,
                ),
                torch.uint8 if self.packed else torch.bfloat16,
                align=self.alignment,
            ),
            EntryField(
                "tail_values",
                len(self.tail_owners),
                (1, self.head_dim),
                torch.float32,
                align=self.alignment,
            ),
            EntryField(
                "tail_scores",
                len(self.tail_owners),
                (1, self.head_dim),
                torch.float32,
                align=self.alignment,
            ),
            # Position followed by compressed IDs/DEAD, oldest first.
            EntryField(
                "cursor", 1, (self.history_size + 1,), torch.int64, align=self.alignment
            ),
        ]

    def _aligned_bytes(self, fields):
        return -(-entry_bytes_for(fields) // self.alignment) * self.alignment

    @property
    def page_bytes(self):
        return self._aligned_bytes(self.page_fields)

    @property
    def state_bytes(self):
        return self._aligned_bytes(self.state_fields)

    def main_offset(self, owner):
        return next(
            start // (1 if self.packed else self.row_bytes)
            for field, start, _ in field_extents(self.page_fields)
            if field.name == f"main_{owner}"
        )

    def window(self, layer, pages):
        if not 0 <= layer < self.layers:
            raise IndexError("Window layer is outside the cache")
        if self.packed:
            return WindowParams(
                ring_start=pages * self.page_bytes
                + layer * self.window_size * self.window_row_bytes,
                slot_rows=self.state_bytes,
                ring_slots=self.window_size,
                ring_stride=1,
                run_rows=self.window_row_bytes,
            )
        return WindowParams(
            ring_start=pages * self.page_bytes // self.row_bytes
            + layer * self.window_size,
            slot_rows=self.state_bytes // self.row_bytes,
            ring_slots=self.window_size,
            ring_stride=self.window_size,
            run_rows=self.window_size,
        )

    @property
    def layout_id(self):
        return (
            f"dsv41-{'packed' if self.packed else 'bf16'}-state-v1:layers={self.layers}:owners={self.owners}"
            f":block={self.block_size}:window={self.window_size}"
            f":dims={self.head_dim},{self.index_dim}:history={self.history_size}"
        )
