# SPDX-License-Identifier: MIT
"""Owner-only PAGE fields and complete request STATE geometry for CSA2.

A PAGE holds the main latent of every owner; its index rows live in a region
of their own, one per owner, addressed by the same block id. The two planes
are bought together (`paged_bytes`) and read by the same block table, which is
the arrangement DeepSeek-V4's indexer already has -- see `sub_pool_spec`'s
"rides the same block table as its main compressed KV".

Keeping the index rows inside the PAGE instead would tie the finest
addressable group of them to the page: a paged scorer takes one block stride
and computes `id * stride`, so an index plane strided by the PAGE can only be
addressed a whole page's rows at a time. Region-major makes that stride the
rows' own, so a block id can name a group of rows finer than a page.

Everything here scales with cached history except the STATE entry, so the
regions are laid out main-pages, index planes, STATE -- one boundary between
the two currencies rather than one per plane.

What sets `block_size` is neither of the paddings this file rounds up. Both
are negligible: a field's own alignment comes out at zero for every shape CSA2
runs (a plane is `rows * row_bytes` and both factors are already coarse), and
an index plane is aligned once over the whole pool rather than per page, so
four owners cost at most four alignments. The two terms that do move are the
internal fragmentation of each request's last page -- `(block_size - 1)` tokens
at 3200 B (BF16 KV and index), so 816 KB per in-flight request at 256, ~0.1% of
a pool -- and the block table, `[max_num_seqs, max_model_len / block_size]`
int32 held twice and copied to the device every forward: 134 MB at block 16
against 8.4 MB at 256, for a million-token context. The block table is the
larger of the two and it is per-forward traffic, which is why CSA2 takes a
coarse page, as DeepSeek-V4 does. What a coarse page costs is prefix reuse
below its own size, which no longer matches at all.
"""

from dataclasses import dataclass

import torch

from .entry_arena import EntryField, entry_bytes_for, field_extents, plan_regions
from .v4_pool_fields import (
    MQA_LOGITS_PRESHUFFLE_ROWS,
    fp8_indexer_block_fields,
    indexer_block_regions,
)
from .v4_pool_geometry import WindowParams

# The packed main pool's FP4 grid. Finer than the index and window planes,
# which take `quantize_fp4`'s group-32 E8M0 default; a scatter that disagrees
# with `main_row_bytes` writes rows the readers cannot decode.
MAIN_FP4 = {"group_size": 16, "scale_dtype": torch.float8_e4m3fn}

# The FP8 index plane's scale format, in `indexer_k_quant_and_cache`'s spelling:
# a power-of-two scale, which is what V4's compressor writes for the same plane
# and what the published indexer quantizes with. The readers take the scale as
# fp32 either way, so this is free to change -- but a writer and an oracle that
# disagree about it differ by up to 2x per row and nothing about the bytes says
# which one meant what.
INDEX_FP8_SCALE_FMT = "ue8m0"


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
    # The distinct compression ratios this configuration's layers run, read
    # off the topology rather than enumerated, so a ratio no layer wants gets
    # no buffer. Every layer at a ratio reserves its rows the same way, so the
    # indptr is the ratio's and not the layer's -- the same thing V4's three
    # `kv_indptr_{swa,csa,hca}` are, at one fixed address each.
    layer_ratios: tuple[int, ...] = ()
    # The width the scorer emits, whatever a row can see: a row short on
    # history is `-1` padded, not narrowed.
    index_topk: int = 0
    # Rows one index block id names: the `KVBlockSize` every reader of this
    # plane passes. A layout choice and not the kernel's tile -- 8 is what lets
    # a candidate list be a block table, the length candidates are picked in.
    index_block_rows: int = MQA_LOGITS_PRESHUFFLE_ROWS

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
        rows = self.index_block_rows
        if rows % MQA_LOGITS_PRESHUFFLE_ROWS and rows != 8:
            raise ValueError(
                f"An index block holds whole {MQA_LOGITS_PRESHUFFLE_ROWS}-row MFMA "
                f"tiles, or exactly 8 rows shuffled in groups of 8; got {rows}"
            )
        for owner, ratio in self.owners:
            if not 0 <= owner < self.layers or ratio not in (1, 2):
                raise ValueError("Invalid global owner or compression ratio")
            if self.block_size % ratio:
                raise ValueError("PAGE token count must divide every compression group")
            # A block id names `index_block_rows` rows, so a PAGE's rows have to
            # be a whole number of them -- the ratio-2 owners are what makes
            # that a statement about twice the PAGE.
            if self.rows_per_page(ratio) % rows:
                raise ValueError(
                    f"An FP8 index plane needs whole {rows}-row blocks: "
                    f"ratio {ratio} gives a PAGE {self.rows_per_page(ratio)} rows, "
                    f"so raise the PAGE token count to a multiple of "
                    f"{rows * max(r for _, r in self.owners)}"
                )

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
        """What one index row costs, which is not what one row *is*.

        Under preshuffle a row's bytes are interleaved across its tile, so this
        is the block over its rows -- the same convention
        `fp8_indexer_block_fields` states, and the reason a scale sits past the
        block's data rather than beside its own row.
        """
        block = indexer_block_regions(
            fp8_indexer_block_fields(
                self.index_block_rows, self.index_dim, torch.float8_e4m3fn
            )
        )[1]
        return block // self.index_block_rows

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

    def batch_topk(self, ratio):
        """The selection width a batch gets at `ratio`, before a scorer runs.

        The scorer emits `index_topk` and pads what a row cannot see, so this
        is a constant rather than a batch's shape -- which is what lets the
        indptr reserve its rows up front, and keeps the scan it feeds to one
        compiled variant.
        """
        return self.index_topk if ratio else 0

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
        """The main latent of every owner. The index rows are a region apart."""
        return [
            EntryField(
                f"main_{owner}",
                1,
                (
                    self.rows_per_page(ratio),
                    self.main_row_bytes if self.packed else self.head_dim,
                ),
                torch.uint8 if self.packed else torch.bfloat16,
                align=self.alignment,
            )
            for owner, ratio in self.owners
        ]

    def rows_per_page(self, ratio):
        """Rows one PAGE holds for an owner at this ratio, main and index alike."""
        return self.block_size // ratio

    @property
    def paged_bytes(self):
        """One PAGE's whole cost: its main rows and its index rows.

        What the pool budget is declared in. The two planes are one block id
        and one budget, laid out apart -- `paged_extents` says where.
        """
        return self.page_bytes + sum(
            self.rows_per_page(ratio) * self.index_row_bytes for _, ratio in self.owners
        )

    def paged_extents(self, pages):
        """`({owner: index plane offset}, bytes the PAGE currency spans)`.

        The offset is from the pool's first PAGE and the total is where STATE
        begins, so this is the one walk that places everything scaling with
        history. Within a plane the pages are dense -- `index_rows *
        index_row_bytes` apart -- which is the stride a paged reader is handed;
        only the plane's own start is aligned.

        The total is rounded to `alignment` rather than to `plan_regions`' own,
        because the window ring behind it is addressed in whole rows.
        """
        offsets, total = plan_regions(
            [pages * self.page_bytes]
            + [
                pages * self.rows_per_page(ratio) * self.index_row_bytes
                for _, ratio in self.owners
            ]
        )
        return (
            dict(zip(self.compress_owners, offsets[1:])),
            -(-total // self.alignment) * self.alignment,
        )

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
        # Behind every PAGE-scaled region, main and index alike -- not just the
        # main pages.
        state_start = self.paged_extents(pages)[1]
        if self.packed:
            return WindowParams(
                ring_start=state_start
                + layer * self.ring_slots * self.window_row_bytes,
                slot_rows=self.state_bytes,
                ring_slots=self.ring_slots,
                ring_stride=1,
                run_rows=self.window_row_bytes,
            )
        return WindowParams(
            ring_start=state_start // self.row_bytes + layer * self.ring_slots,
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
            #
            # The index plane's format is here even though no STATE field holds
            # an index row: this string is what a resumed prefix is matched on,
            # and a plane at a different precision selects different rows, so
            # the prefix is a different computation. It is spelled out rather
            # than read off a field because FP8 is the only plane the scorer
            # takes -- an image from a build that had another must not match.
            f"dsv41-{'packed' if self.packed else 'bf16'}-index-fp8"
            f"-state-v2:layers={self.layers}:owners={self.owners}"
            f":block={self.block_size}:window={self.window_size}"
            f":dims={self.head_dim},{self.index_dim}:history={self.history_size}"
        )
        return identity + (
            f":spec={self.speculative_tokens}" if self.speculative_tokens else ""
        )
