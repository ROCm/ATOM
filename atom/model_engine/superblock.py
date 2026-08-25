# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which physical superblock a paged block sits in, and whether it is whole.

A superblock is a **physical memory concept only**: `blocks_per_super`
consecutive block ids that share one contiguous byte range. It exists so a
hybrid attention backend can be handed one contiguous range for a per-request
state slot, and it is the unit of *allocation and reclamation* and of nothing
else.

Everything else stays at block granularity, where it is already validated:
prefix hashing, `ref_count`, cross-request sharing, the vacant/cached split and
its eviction order, offload, remote KV transfer, block tables, `slot_mapping`.
`BlockPool` stays in charge of all of it. This class only answers two
questions — *which superblock is this block in* and *is that superblock wholly
unreferenced* — and offers a packing preference for the next free block.

An earlier attempt made this a parallel allocator with its own free lists. That
was wrong twice over: every block-level policy would have had to be re-derived
on top of it, and the copy that existed had already drifted (one free set with
no cached tier, and a state claim that never evicted). The lesson is the shape:
a constraint on an allocator, not a second allocator.

The failure mode this design must avoid is one live block per superblock across
the whole pool -- every superblock pinned, nothing reclaimable, while the pool
is nearly empty. `preferred_free` is what prevents it: a request's fresh blocks
pack into one superblock before opening the next, so live blocks cluster by
request rather than scattering. Shared blocks break the clustering, but a shared
block is exactly the one with high reuse value, so its superblock has earned
staying resident.

With packing holding, starvation is structural rather than accidental: a
superblock with no live block is always claimable, because its cached content is
spendable exactly as `BlockPool` already spends `_cached`. A claim can only fail
when every superblock holds a genuinely-live block, which `max_num_seqs` bounds.
"""

from __future__ import annotations

UNTYPED = "untyped"
KV = "kv"
STATE = "state"


class SuperblockMap:
    """Per-superblock liveness and typing over a `BlockPool`'s block ids.

    Bookkeeping only: it allocates nothing and frees nothing. `BlockPool` tells
    it when a block becomes live or stops being live, and it answers questions
    about the superblock that block belongs to.
    """

    def __init__(self, num_blocks: int, blocks_per_super: int) -> None:
        if blocks_per_super < 1:
            raise ValueError(f"blocks_per_super must be >= 1, got {blocks_per_super}")
        if num_blocks % blocks_per_super:
            raise ValueError(
                f"{num_blocks} blocks is not a whole number of "
                f"{blocks_per_super}-block superblocks"
            )
        self.blocks_per_super = blocks_per_super
        self.num_supers = num_blocks // blocks_per_super
        #: Blocks with `ref_count > 0`. A superblock is reclaimable at zero —
        #: its remaining blocks may still hold cached content, which is
        #: spendable, and that is the distinction a live count captures and a
        #: "is anything in here" flag would not.
        self.live_count: list[int] = [0] * self.num_supers
        self.kind: list[str] = [UNTYPED] * self.num_supers
        #: The KV superblock fresh allocations pack into. -1 when none is open.
        self._open = -1

    # ------------------------------ mapping -------------------------------- #
    def super_of(self, block_id: int) -> int:
        return block_id // self.blocks_per_super

    def block_range(self, index: int) -> range:
        start = index * self.blocks_per_super
        return range(start, start + self.blocks_per_super)

    # ---------------------------- liveness --------------------------------- #
    def on_block_live(self, block_id: int) -> None:
        """A block gained its first reference."""
        index = self.super_of(block_id)
        self.live_count[index] += 1
        if self.kind[index] == UNTYPED:
            self.kind[index] = KV

    def on_block_free(self, block_id: int) -> None:
        """A block's last reference went. Its content may still be cached."""
        index = self.super_of(block_id)
        if self.live_count[index] == 0:
            raise AssertionError(
                f"superblock {index} freed below zero live blocks "
                f"(block {block_id})"
            )
        self.live_count[index] -= 1
        if self.live_count[index] == 0 and self._open == index:
            # Nothing live to pack alongside; let the next allocation pick
            # afresh rather than keeping a claim on this one.
            self._open = -1

    def is_reclaimable(self, index: int) -> bool:
        """Whether this superblock could be taken whole, evicting if needed."""
        return self.live_count[index] == 0 and self.kind[index] != STATE

    # ---------------------------- packing ---------------------------------- #
    def preferred_free(self, free: set[int]) -> int:
        """A free block in the open superblock, or -1 to let the pool choose.

        Deliberately a *preference*, not a decision. The caller applies it only
        after its own vacant/cached ordering has selected a tier, so packing
        chooses among equally-eligible candidates and never promotes a cached
        block over a vacant one. Getting that backwards would spend reusable
        content while empty blocks waited.
        """
        if self._open < 0 or self.kind[self._open] != KV:
            return -1
        for block_id in self.block_range(self._open):
            if block_id in free:
                return block_id
        return -1

    def note_allocation(self, block_id: int) -> None:
        """Record where fresh content landed, so the next one packs beside it."""
        index = self.super_of(block_id)
        if self.kind[index] != STATE:
            self._open = index

    # ------------------------------ typing --------------------------------- #
    def take_state(self, index: int) -> None:
        if self.live_count[index]:
            raise AssertionError(
                f"superblock {index} has {self.live_count[index]} live blocks"
            )
        self.kind[index] = STATE
        if self._open == index:
            self._open = -1

    def release_state(self, index: int) -> None:
        if self.kind[index] != STATE:
            raise ValueError(f"superblock {index} is not a state slot")
        self.kind[index] = UNTYPED

    def untype(self, index: int) -> None:
        """Forget a superblock's typing, once nothing of its content is worth keeping.

        Deliberately NOT called from `BlockPool.free`. A KV superblock whose
        last block is released still holds cached content a prefix could hit,
        and untyping it there would erase the only thing distinguishing "empty,
        free for the taking" from "reclaimable but you would be spending
        reuse" — which is exactly the preference `claim_superblock` needs to
        make. The caller untypes when it has actually dropped the content.
        """
        if self.live_count[index]:
            return
        if self.kind[index] != STATE:
            self.kind[index] = UNTYPED

    # ---------------------------- diagnostics ------------------------------ #
    def occupancy(self) -> dict[str, int]:
        free = sum(
            1
            for i in range(self.num_supers)
            if self.live_count[i] == 0 and self.kind[i] != STATE
        )
        state = sum(1 for k in self.kind if k == STATE)
        pinned = sum(
            1
            for i in range(self.num_supers)
            if 0 < self.live_count[i] < self.blocks_per_super
        )
        return {
            "supers_total": self.num_supers,
            "supers_reclaimable": free,
            "supers_state": state,
            # The accumulation risk: superblocks holding a live block and so
            # unavailable as a contiguous slot, however much of them is free.
            # A rising standing count here — not the per-event drain rate — is
            # what says the packing preference has stopped working.
            "supers_partially_pinned": pinned,
        }
