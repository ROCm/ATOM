# SPDX-License-Identifier: MIT
"""Owner-only PAGE fields and complete request STATE geometry for CSA2."""

from dataclasses import dataclass

import torch

from .entry_arena import EntryField, entry_bytes_for, field_extents
from .v4_pool_geometry import WindowParams

# The packed main pool's FP4 grid. Finer than the index and window planes,
# which take `quantize_fp4`'s group-32 E8M0 default; a scatter that disagrees
# with `main_row_bytes` writes rows the readers cannot decode.
MAIN_FP4 = {"group_size": 16, "scale_dtype": torch.float8_e4m3fn}


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
    speculative_tokens: int = 0

    def __post_init__(self):
        if self.speculative_tokens < 0:
            raise ValueError("Speculative window slack cannot be negative")
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
        if not self.packed:
            return self.row_bytes
        return self.head_dim // 2 + self.head_dim // MAIN_FP4["group_size"]

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
    def ring_slots(self):
        # Verification includes one guaranteed input plus speculative tokens.
        # Slack retains the window behind every possible accepted prefix.
        return self.window_size + self.speculative_tokens

    @property
    def compress_owners(self):
        return tuple(owner for owner, _ in self.owners)

    @property
    def compress_ratios(self):
        """`(ratio, overlap)` per distinct ratio: CSA2 never overlaps."""
        return tuple(sorted({(ratio, False) for _, ratio in self.owners}))

    @property
    def compress_ring_slots(self):
        """V4's `STATE_SIZE`: the pool window plus rejected-draft slack.

        One width for every owner rather than one per ratio. The ring only has
        to be at least `K_pool = ratio` (no overlap in CSA2) and the widest
        owner sets that; a ratio-1 owner spending one extra row is cheaper than
        a second field and a second modulus to keep in step with the kernels.

        The slack is why a rejected draft cannot corrupt the next round: round
        R's discarded writes sit at most `speculative_tokens` ids past R+1's
        commit head, so they fall outside the `K_pool`-wide window R+1 reads.
        """
        widest = max((ratio for _, ratio in self.owners), default=1)
        return widest + self.speculative_tokens

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
        window = EntryField(
            "window",
            self.layers,
            (self.ring_slots, self.window_row_bytes if self.packed else self.head_dim),
            torch.uint8 if self.packed else torch.bfloat16,
            align=self.alignment,
        )
        # The compressor's own ring, V4's `kv_state` / `score_state`: the last
        # `K_pool` raw projections per owner, so a pool window reaching back
        # before this forward reads them from here instead of the caller
        # carrying an incomplete group across the boundary.
        rings = [
            EntryField(
                name,
                len(self.owners),
                (self.compress_ring_slots, self.head_dim),
                torch.float32,
                align=self.alignment,
            )
            for name in ("compress_kv", "compress_score")
        ]
        # Position followed by compressed IDs/DEAD, oldest first.
        cursor = EntryField(
            "cursor", 1, (self.history_size + 1,), torch.int64, align=self.alignment
        )
        return [window, *rings, cursor]

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
                + layer * self.ring_slots * self.window_row_bytes,
                slot_rows=self.state_bytes,
                ring_slots=self.ring_slots,
                ring_stride=1,
                run_rows=self.window_row_bytes,
            )
        return WindowParams(
            ring_start=pages * self.page_bytes // self.row_bytes
            + layer * self.ring_slots,
            slot_rows=self.state_bytes // self.row_bytes,
            ring_slots=self.ring_slots,
            ring_stride=self.ring_slots,
            run_rows=self.ring_slots,
        )

    @property
    def layout_id(self):
        identity = (
            # v2: the compressor's incomplete-group tail became a K_pool ring.
            # Every other term below was already the same in v1, so without the
            # bump a v1 image would read as compatible and restore into fields
            # that no longer mean what it holds.
            f"dsv41-{'packed' if self.packed else 'bf16'}-state-v2:layers={self.layers}:owners={self.owners}"
            f":block={self.block_size}:window={self.window_size}"
            f":dims={self.head_dim},{self.index_dim}:history={self.history_size}"
        )
        return identity + (
            f":spec={self.speculative_tokens}" if self.speculative_tokens else ""
        )
