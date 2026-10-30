# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A DSpark draft's KV, kept only as far back as its sliding window reads.

A draft that attends a sliding window (MiniMax-M3.1's DSpark: 4096 keys) reads
nothing of a long context but its tail, yet a pool riding the target's block ids
holds a draft entry for every target block. This pool holds the tail alone:

  * Its blocks have their own ids. A sequence's `draft_block_table` is aligned
    with its `block_table` -- entry `i` holds the draft rows of positions
    `[i * block_size, (i + 1) * block_size)` -- and is -1 below the window.
    What is held is always one contiguous run, `[draft_window_start, len)`.
  * A block falls out of the window once no later block pass can read it. Every
    pass anchors at or past `finalized - 1` (a sequence never rolls back below
    what it finalized), so `DraftWindowGeometry.first_live_block` is a bound
    that only grows.
  * Content-addressed like the target's blocks and under the same hashes, so a
    block a finished request hands back stays claimable until something else
    needs its slot. A prefix hit is only as long as the draft rows its first
    block pass reads are cached (`gate`); the hit is cut at the first missing
    one, which bounds the recompute to one window.

A prompt's prefill allocates only the window its first block pass reads, so the
context rows below it are neither written nor -- since the draft's projection
runs on the kept rows alone -- computed.
"""

from __future__ import annotations

from collections.abc import Sequence as SequenceT
from dataclasses import dataclass

from atom.model_engine.block_pool import BlockPool

# The Pool.STATE entry class the draft window is sized in. Owned here, by the
# consumer of its count, like `STATE_SLOT_CLASS` in `kv_block.py`.
DRAFT_WINDOW_CLASS = "draft_window"


@dataclass(frozen=True)
class DraftWindowGeometry:
    """Where a windowed draft's block pass reads, in the scheduler's blocks.

    `window` keys a block row attends back from its own position; `read_page`
    the page the per-row decode starts a row at; `block_size` the scheduler
    block both pools are paged in; `spec_tokens` the verify width, which bounds
    how far a sequence's scheduled length runs ahead of what it finalized.
    """

    window: int
    read_page: int
    block_size: int
    spec_tokens: int

    def __post_init__(self) -> None:
        for name in ("window", "read_page", "block_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"draft window {name} must be positive, got {value}")
        if self.block_size % self.read_page:
            raise ValueError(
                f"the draft reads pages of {self.read_page} tokens, which do not "
                f"tile its {self.block_size}-token blocks"
            )
        if self.spec_tokens < 0:
            raise ValueError(f"spec_tokens must be >= 0, got {self.spec_tokens}")

    def first_read_pos(self, anchor: int) -> int:
        """Lowest key position a block pass anchored at `anchor` reads.

        The pass's first row sits at `anchor + 1` and attends back `window`
        keys, from `anchor + 2 - window`; the per-row decode starts that row
        on the page holding that key (`_window_rows` in the MiniMax-M3 draft).
        """
        start = max(0, anchor + 2 - self.window)
        return start - start % self.read_page

    def first_live_block(self, finalized: int) -> int:
        """First block a sequence that finalized `finalized` tokens can read.

        Every later block pass anchors at or past `finalized - 1`, and the
        read start only moves right with the anchor.
        """
        return self.first_read_pos(max(0, finalized - 1)) // self.block_size

    @property
    def scheduled_slack(self) -> int:
        """Tokens a scheduled length may run past the finalized one.

        A decode step appends a placeholder per draft token before the step
        that verifies them reports back, and deferred output holds one more
        step's worth: `2 * spec_tokens + 1` at steady state, one more for the
        bonus token.
        """
        return 2 * self.spec_tokens + 2

    @property
    def blocks_per_req(self) -> int:
        """Most blocks one sequence holds at once.

        A sequence holds `[first_live_block(finalized), table_end)`, where the
        table reaches the draft round's lookahead (`spec_tokens - 1`) past a
        length up to `scheduled_slack` beyond `finalized`. Taken as the maximum
        over `finalized` rather than a closed form: the read start moves in
        pages and the table in blocks, and the expression is periodic in
        `block_size` once the window is full.
        """
        bs = self.block_size
        reach = self.scheduled_slack + max(0, self.spec_tokens - 1)
        return max(
            -(-(finalized + reach) // bs) - self.first_live_block(finalized)
            for finalized in range(1, self.window + 2 * bs + 1)
        )

    def to_wire(self) -> dict[str, int]:
        return {
            "window": self.window,
            "read_page": self.read_page,
            "block_size": self.block_size,
            "spec_tokens": self.spec_tokens,
        }

    @classmethod
    def from_wire(cls, wire: dict) -> DraftWindowGeometry:
        return cls(
            window=int(wire["window"]),
            read_page=int(wire["read_page"]),
            block_size=int(wire["block_size"]),
            spec_tokens=int(wire["spec_tokens"]),
        )


class DraftWindowPool:
    """The scheduler's half of a windowed draft KV pool.

    Disabled (every method a no-op) when the runner sized no draft window, so
    `BlockManager` calls it unconditionally.
    """

    def __init__(
        self,
        num_blocks: int = 0,
        geometry: DraftWindowGeometry | None = None,
        *,
        enable_prefix_caching: bool = True,
        cache_policy: str = "lru",
        protected_ratio: float = 0.5,
    ):
        if bool(num_blocks) != (geometry is not None):
            raise ValueError(
                f"a draft window needs both a pool and its geometry; got "
                f"{num_blocks} blocks and geometry {geometry}"
            )
        self.enabled = num_blocks > 0
        self.geometry = geometry
        self.prefix_caching = enable_prefix_caching
        self.pool = BlockPool(
            num_blocks, cache_policy=cache_policy, protected_ratio=protected_ratio
        )

    @classmethod
    def from_config(cls, config, **kwargs) -> DraftWindowPool:
        entries = (getattr(config, "pool_entries", None) or {}).get(
            DRAFT_WINDOW_CLASS, 0
        )
        wire = getattr(config, "draft_window_geometry", None)
        geometry = DraftWindowGeometry.from_wire(wire) if wire else None
        return cls(
            int(entries),
            geometry,
            enable_prefix_caching=config.enable_prefix_caching,
            **kwargs,
        )

    # ------------------------------ geometry ------------------------------- #
    def _block_tokens(self, seq, i: int):
        bs = self.geometry.block_size
        return seq.token_ids[i * bs : (i + 1) * bs]

    def first_block(self, seq, finalized: int | None = None) -> int:
        """First block `seq` holds: the window of a pass anchored at the last
        of `finalized` tokens (all of them by default, which is a prefill's)."""
        return self.geometry.first_live_block(
            len(seq) if finalized is None else finalized
        )

    # ------------------------------ admission ------------------------------ #
    def gate(self, seq, hit: int, hashes: SequenceT[int]) -> int:
        """Largest boundary `<= hit` whose window rows the cache still holds.

        A prefill resuming at `hit` reads the draft rows of
        `[first_block(seq), hit)` from the cache and writes the rest itself. The
        first of them that is gone is where it has to resume instead, which is
        never below the window's first block. What it cuts shows up where every
        gate's does, as `num_compressed_hit_blocks` minus the admitted hit.
        """
        if not self.enabled or not self.prefix_caching or hit <= 0:
            return hit
        for i in range(self.first_block(seq), hit):
            if i >= len(hashes) or not self._holds(seq, i, hashes[i]):
                return i
        return hit

    def _holds(self, seq, i: int, h: int) -> bool:
        block_id = self.pool.lookup(h)
        return block_id != -1 and self.pool.block(block_id).token_ids == (
            self._block_tokens(seq, i)
        )

    def admission_blocks(self, seq, table_blocks: int) -> int:
        """Draft blocks admitting `seq` with a `table_blocks`-long table takes,
        cached ones included: claiming one takes it off the free list too."""
        if not self.enabled:
            return 0
        return max(0, table_blocks - self.first_block(seq))

    def has_free(self, count: int) -> bool:
        return not self.enabled or self.pool.has_free(count)

    # ----------------------------- allocation ------------------------------ #
    def allocate(
        self, seq, hit: int, hashes: SequenceT[int], table_blocks: int
    ) -> None:
        """Fill `seq.draft_block_table` for a prefill resuming at `hit`.

        -1 below the window, the cache's blocks for `[first, hit)`, fresh ones
        from there to `table_blocks`. The caller ran `gate` over the same
        hashes in the same admission and nothing in between takes a draft
        block, so a cached block that is not there is a bug, not a miss.
        """
        if not self.enabled:
            return
        if seq.draft_block_table:
            raise RuntimeError(f"seq {seq.id} already holds a draft window")
        first = min(self.first_block(seq), table_blocks)
        table = seq.draft_block_table
        table.extend([-1] * first)
        seq.draft_window_start = first
        for i in range(first, min(hit, table_blocks)):
            block_id = self.pool.lookup(hashes[i])
            if block_id == -1:
                raise RuntimeError(
                    f"seq {seq.id}: draft block {i} passed the window gate but "
                    "is no longer cached"
                )
            self.pool.claim(block_id)
            table.append(block_id)
        self._extend(seq, table_blocks)

    def _extend(self, seq, table_blocks: int) -> None:
        table = seq.draft_block_table
        while len(table) < table_blocks:
            block_id = self.pool.pop()
            self.pool.allocate(block_id)
            table.append(block_id)

    def sync(self, seq, table_blocks: int, finalized: int) -> None:
        """Follow the target's table to `table_blocks` and hand back what no
        pass anchored at or past `finalized - 1` reads.

        Oldest first, so the oldest is evicted first: a later prompt over the
        same prefix is longer, and its window starts further right.
        """
        if not self.enabled:
            return
        table = seq.draft_block_table
        dead = min(self.first_block(seq, finalized), len(table))
        for i in range(seq.draft_window_start, dead):
            self.pool.free(table[i])
            table[i] = -1
        seq.draft_window_start = max(seq.draft_window_start, dead)
        self._extend(seq, table_blocks)

    def release(self, seq) -> None:
        """Hand back every block `seq` holds, keeping their hashes claimable."""
        if not self.enabled:
            return
        table = seq.draft_block_table
        for i in range(seq.draft_window_start, len(table)):
            self.pool.free(table[i])
        del table[:]
        seq.draft_window_start = 0

    # ------------------------------ publishing ----------------------------- #
    def publish(self, seq, i: int, h: int, token_ids) -> None:
        """Index `seq`'s block `i` under the target's hash for it.

        Called where the target publishes block `i`, which is once every row of
        it is computed and final. A held draft block holds the same rows: it is
        either claimed whole from the cache or allocated before any forward
        over it. Below the window there is nothing to publish.
        """
        if not self.enabled or not self.prefix_caching:
            return
        table = seq.draft_block_table
        if i < len(table) and table[i] >= 0:
            self.pool.publish(table[i], h, token_ids)

    def window(self, seq) -> tuple[int, list[int]]:
        """`(first block held, the held ids)`: what a forward reads and writes."""
        start = seq.draft_window_start
        return start, list(seq.draft_block_table[start:])

    def clear_index(self) -> None:
        if self.enabled:
            self.pool.clear_index()

    def stats(self) -> dict[str, int]:
        return self.pool.eviction_stats()
