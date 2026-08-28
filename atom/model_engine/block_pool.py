# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from atom.model_engine.kv_block import Block


@dataclass(frozen=True)
class BlockRetirement:
    """What taking the highest block out of the pool cost.

    `moved_to` is -1 when the block was free and could simply be dropped.
    Otherwise it is where the block's contents now live, and every holder's
    block table has to follow — which is why this is reported rather than
    handled here: the pool does not know who holds what.
    """

    retired: int
    moved_to: int


class BlockPool:
    """Paged blocks with ref counts and a content-addressed index.

    `BlockManager` owns the one instance, over the compressed KV blocks. This
    was split out when a second pool existed — the sliding window was its own
    content-addressed block pool, driven in lockstep with this one — and is
    kept separate because the class it serves is a whole mechanism (free list,
    hash index, lazy eviction) rather than a helper of the manager. The window
    is a per-request ring sharing a slot with the compressor state now; see
    `v4_pool_geometry.py`.

    Eviction is lazy: a freed block keeps its hash and contents until its slot
    is handed out for something else, so a later request can still claim a
    freed-but-not-overwritten block. `allocate` — not `free` — is therefore the
    eviction event, and `on_evict` fires there.

    Free blocks sit in one of two containers, by whether they still hold
    reusable content:

      `_vacant`    no hash. Nothing is lost by taking one, so they go first,
                   lowest id first — which also drains the top of the pool,
                   where a shrinking boundary eats.
      `_cached`    still reachable by hash. Least-recently-freed first, so
                   handing one out evicts the coldest content.

    One queue for both would evict a cached block while a vacant one waited
    behind it, purely on release order.

    `_cached` is an insertion-ordered mapping rather than a queue because
    `claim` takes a *named* block off the free list on every prefix hit: left
    in place, its entry would put the block back at its old position when it is
    freed again, which is the LRU order inverted for exactly the blocks being
    reused most. Removing it costs O(1) here and O(n) from a deque, on a path
    that runs once per hit block.
    """

    def __init__(
        self,
        num_blocks: int,
        on_evict: Callable[[int], None] | None = None,
        max_blocks: int | None = None,
    ):
        # `max_blocks` is how far `extend` may go, and so how many Block
        # objects exist. It is the pool's share of a fixed plane rather than
        # its current size; a pool with a pinned boundary passes neither and
        # gets a maximum equal to its size.
        self.max_blocks: int = num_blocks if max_blocks is None else max_blocks
        if not 0 <= num_blocks <= self.max_blocks:
            raise ValueError(f"{num_blocks} blocks outside 0..{self.max_blocks}")
        self.num_blocks: int = num_blocks
        self._on_evict = on_evict
        self.blocks: list[Block] = [Block(i) for i in range(self.max_blocks)]
        self._hash_to_block_id: dict[int, int] = {}
        # Both containers may hold ids that were re-claimed straight off the
        # free list (`claim`) or retired, so membership in the set — not
        # presence in a container — is what makes an id free. `pop` skips the
        # stale entries.
        self._vacant: list[int] = list(range(num_blocks))
        self._cached: OrderedDict[int, None] = OrderedDict()
        self._free: set[int] = set(range(num_blocks))
        self._used: set[int] = set()
        # Raw PAGE units reserved by multi-unit objects such as state checkpoints.
        self._raw_unit_owner: dict[int, tuple[Hashable, int]] = {}
        # Reusable content this pool destroyed, split by what destroyed it.
        # Both are evictions in the sense that a later prefix hit is now
        # impossible, and they read the same in a hit rate, but they want
        # opposite fixes — the same reason `StateSlotPool` keeps `evicted`
        # and `orphaned` apart:
        #   `blocks_evicted`  the pool was out of vacant blocks and spent a
        #                     cached one. Says the paged pool is too small.
        #   `blocks_retired`  the boundary moved down over cached content.
        #                     Says the split is wrong, not the total.
        # Counted here rather than derived from `on_evict` because that hook
        # also fires for relocation (`_adopt`), which destroys nothing: the
        # hash moves to the block that adopted it.
        self.blocks_evicted: int = 0
        self.blocks_retired: int = 0

    # ------------------------------- counts -------------------------------- #
    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._used)

    @property
    def num_indexed(self) -> int:
        """Blocks reachable by content hash, live or merely not-yet-overwritten."""
        return len(self._hash_to_block_id)

    def has_free(self, n: int) -> bool:
        return len(self._free) >= n

    def is_used(self, block_id: int) -> bool:
        return block_id in self._used

    def block(self, block_id: int) -> Block:
        return self.blocks[block_id]

    @property
    def num_reusable_free(self) -> int:
        """Free blocks still holding content a prefix hit could claim.

        The pool's headroom before the *next* allocation has to evict: while
        vacant blocks remain this is slack, and once they are gone every
        allocation spends one of these. `num_free - num_reusable_free` is the
        vacant count, which is the number that actually has to reach zero
        before `blocks_evicted` can start moving.
        """
        return sum(1 for b in self._free if self.blocks[b].hash != -1)

    def eviction_stats(self) -> dict[str, int]:
        """Content this pool destroyed, and the headroom it has left.

        Counters, not rates, for the same reason `CacheStats.get_statistics`
        hands back counts: a rate cannot be summed across DP ranks that saw
        different traffic.
        """
        return {
            "blocks_evicted": self.blocks_evicted,
            "blocks_retired": self.blocks_retired,
            "blocks_total": self.num_blocks,
            "blocks_used": self.num_used,
            "blocks_free": self.num_free,
            "blocks_free_reusable": self.num_reusable_free,
            "blocks_indexed": self.num_indexed,
        }

    # ------------------------------- index --------------------------------- #
    def lookup(self, h: int) -> int:
        """Block id indexed under content hash `h`, or -1."""
        return self._hash_to_block_id.get(h, -1)

    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        """Index `block_id` under the content hash of the tokens it now holds."""
        block = self.blocks[block_id]
        block.update(h, token_ids)
        self._hash_to_block_id[h] = block_id

    def clear_index(self) -> None:
        """Drop every content-hash entry, keeping blocks live sequences hold.

        Those stay valid through their block_table refs; they are simply no
        longer reachable by hash, so no future request can claim them.
        """
        self._hash_to_block_id.clear()
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = array.array("i")
        # Every free block is vacant now, and which container an id is in is
        # only ever decided from its hash — so the split has to be redrawn
        # here rather than left to drift.
        self._cached.clear()
        self._vacant = sorted(self._free)
        heapify(self._vacant)

    def _unindex(self, block_id: int) -> bool:
        """Drop `block_id`'s index entry and forget what it held.

        Returns whether an index entry actually went — i.e. whether reusable
        content was destroyed. A block with no hash, or one whose hash the
        index has since re-pointed elsewhere, costs nothing to drop, and the
        callers that count evictions must not count those.
        """
        block = self.blocks[block_id]
        dropped = False
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
            dropped = True
            if self._on_evict is not None:
                self._on_evict(block.hash)
        block.hash = -1
        block.token_ids = array.array("i")
        return dropped

    # ---------------------------- allocation ------------------------------- #
    def _take_free(self) -> int:
        """Next free block id, or -1. Vacant before cached; see the class doc.

        An entry is stale when the block has since been taken, or when it has
        gained or lost a hash and so belongs in the other container — an id can
        sit in both at once, and testing only that it is free would hand a
        cached block out of the vacant half. Both conditions are the ones that
        decide which half it belongs to anyway, so the test is the definition
        rather than a guard bolted on top.
        """
        while self._vacant:
            block_id = heappop(self._vacant)
            if block_id in self._free and self.blocks[block_id].hash == -1:
                self._free.discard(block_id)
                return block_id
        while self._cached:
            block_id, _ = self._cached.popitem(last=False)
            if block_id in self._free and self.blocks[block_id].hash != -1:
                self._free.discard(block_id)
                return block_id
        return -1

    def pop(self) -> int:
        block_id = self._take_free()
        if block_id < 0:
            raise AssertionError("No free blocks available")
        return block_id

    def _take_named(self, block_id: int) -> None:
        """Take one specific block off the free list, content and all.

        The cached half is ordered by when a block was released, so an id left
        in it after being taken would come back at its old position rather than
        the end — see the class doc. The vacant half is ordered by id, where a
        leftover entry is only a wasted pop.
        """
        self._free.discard(block_id)
        self._cached.pop(block_id, None)

    def allocate(self, block_id: int) -> Block:
        """Take `block_id` for fresh content, evicting whatever it held."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        if self._unindex(block_id):
            self.blocks_evicted += 1
        block.reset()
        self._take_named(block_id)
        self._used.add(block_id)
        return block

    def claim(self, block_id: int) -> Block:
        """Take a share of `block_id` for the content it already holds.

        The cache-hit counterpart of `allocate`, and deliberately not built on
        it: `allocate`'s reset would drop the hash and destroy the entry for
        every other request that could still hit it.
        """
        block = self.blocks[block_id]
        if block_id in self._used:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self._take_named(block_id)
            self._used.add(block_id)
        return block

    def free(self, block_id: int) -> None:
        if block_id in self._raw_unit_owner:
            raise AssertionError(
                f"block {block_id} is a reserved raw unit; use release_units"
            )
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count:
            return
        self._used.remove(block_id)
        self._free.add(block_id)
        if block.hash != -1:
            self._cached[block_id] = None
            return
        heappush(self._vacant, block_id)
        # Stale entries are skipped, not removed, so the heap can outgrow the
        # pool under churn. Rebuilding costs one pass and buys at least
        # `num_blocks` pushes.
        if len(self._vacant) > 2 * self.num_blocks + 2:
            self._vacant = [b for b in self._free if self.blocks[b].hash == -1]
            heapify(self._vacant)

    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None:
        """Reserve arbitrary PAGE-sized units for raw storage."""
        if count < 0:
            raise ValueError(f"unit count must be non-negative, got {count}")
        if owner is None:
            raise ValueError("a raw-unit reservation needs an owner")
        if not self.has_free(count):
            return None
        unit_ids: list[int] = []
        for piece_index in range(count):
            block_id = self.pop()
            self.allocate(block_id)
            self._raw_unit_owner[block_id] = (owner, piece_index)
            unit_ids.append(block_id)
        return unit_ids

    def release_units(self, unit_ids: Iterable[int], owner: Hashable) -> None:
        """Release a complete raw-unit reservation back to the PAGE pool."""
        ids = list(unit_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("a raw-unit release contains duplicate ids")
        # Validate ownership before releasing any unit.
        for piece_index, block_id in enumerate(ids):
            actual = self._raw_unit_owner.get(block_id)
            expected = (owner, piece_index)
            if actual != expected:
                raise AssertionError(
                    f"raw unit {block_id} belongs to {actual!r}, not {expected!r}"
                )
        for block_id in ids:
            del self._raw_unit_owner[block_id]
            self.free(block_id)

    # ------------------------------ resizing ------------------------------- #
    def extend(self, count: int) -> int:
        """Grow the pool by up to `count` blocks; returns how many it took.

        Capped at `max_blocks`, which is how many blocks the plane can address
        — beyond it there is nothing to hand out even if the caller has bytes.
        """
        taken = min(count, self.max_blocks - self.num_blocks)
        for block_id in range(self.num_blocks, self.num_blocks + taken):
            self._free.add(block_id)
            heappush(self._vacant, block_id)
        self.num_blocks += taken
        return taken

    def retire_top(self) -> BlockRetirement | None:
        """Take the highest block id out of the pool, or None if it cannot.

        The highest specifically, because the ids the pool gives up have to be
        the ones the boundary is about to cover — any free block will not do.
        A block that is merely cached costs nothing to retire; the index entry
        goes and the content with it. One that a sequence still holds has to
        move, and its new id is reported so its holders can follow.

        Only fails when the top block is in use and nothing is free to move it
        into, which is the same condition that would block admitting the
        request in the first place.
        """
        top = self.num_blocks - 1
        if top < 0:
            return None
        # Raw units cannot move without updating their owning record.
        if top in self._raw_unit_owner:
            return None
        if top in self._free:
            self._take_named(top)
            if self._unindex(top):
                self.blocks_retired += 1
            destination = -1
        else:
            destination = self._take_free()
            if destination < 0:
                return None
            self._adopt(destination, top)
        self.num_blocks -= 1
        return BlockRetirement(top, destination)

    def _adopt(self, destination: int, source: int) -> None:
        """Give `source`'s identity to `destination`, leaving source empty.

        Ref count, hash and tokens all move: a relocation is invisible to the
        sequences holding the block except for the id itself, which the caller
        rewrites. The bytes are the caller's to move too — this is the
        bookkeeping half, and the two have to happen in the same pass.
        """
        # The destination may have come off the cached half of the free list,
        # in which case making room for the relocation destroyed its content.
        # `retire_top` is the only caller, so the boundary is what spent it.
        if self._unindex(destination):
            self.blocks_retired += 1
        src, dst = self.blocks[source], self.blocks[destination]
        dst.ref_count, dst.hash, dst.token_ids = src.ref_count, src.hash, src.token_ids
        if src.hash != -1 and self._hash_to_block_id.get(src.hash) == source:
            self._hash_to_block_id[src.hash] = destination
        self._used.discard(source)
        self._used.add(destination)
        src.ref_count, src.hash, src.token_ids = 0, -1, array.array("i")
