# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import OrderedDict
from collections.abc import Callable
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
        # PAGE-sized units temporarily owned by something other than a token
        # block.  DeepSeek-V4 paged state checkpoints use this to keep one
        # logical state image in several arbitrary (not necessarily adjacent)
        # physical blocks.  A reserved unit remains in `_used`, so none of the
        # ordinary PAGE allocation paths can hand out one fragment behind the
        # checkpoint owner's back.
        self._raw_unit_owner: dict[int, object] = {}

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

    def is_reserved_unit(self, block_id: int) -> bool:
        """Whether `block_id` is raw storage rather than a live PAGE."""
        return block_id in self._raw_unit_owner

    def reserved_unit_owner(self, block_id: int) -> object | None:
        return self._raw_unit_owner.get(block_id)

    # ------------------------------- index --------------------------------- #
    def lookup(self, h: int) -> int:
        """Block id indexed under content hash `h`, or -1."""
        return self._hash_to_block_id.get(h, -1)

    def publish(self, block_id: int, h: int, token_ids: list[int]) -> None:
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
                block.token_ids = []
        # Every free block is vacant now, and which container an id is in is
        # only ever decided from its hash — so the split has to be redrawn
        # here rather than left to drift.
        self._cached.clear()
        self._vacant = sorted(self._free)
        heapify(self._vacant)

    def _unindex(self, block_id: int) -> None:
        """Drop `block_id`'s index entry and forget what it held."""
        block = self.blocks[block_id]
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
            if self._on_evict is not None:
                self._on_evict(block.hash)
        block.hash = -1
        block.token_ids = []

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
        self._unindex(block_id)
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

    def reserve_units(self, count: int, owner: object) -> list[int] | None:
        """Atomically reserve any `count` PAGE-sized units for raw storage.

        The ids deliberately carry no contiguity promise.  Cached PAGE content
        is evicted through the normal `allocate` path, including `on_evict`, so
        the PAGE hash and the state-checkpoint hash cannot drift apart.

        Returns ``None`` without changing the pool when the total free-unit
        count is insufficient.  Callers that can evict a multi-unit state
        checkpoint do that first and retry.
        """
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
            # The reverse map names both the atomic allocation and this
            # fragment's ordered position in its logical byte stream.
            self._raw_unit_owner[block_id] = (owner, piece_index)
            unit_ids.append(block_id)
        return unit_ids

    def release_units(self, unit_ids, owner: object) -> None:
        """Release a complete raw-unit reservation back to the PAGE pool."""
        ids = list(unit_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("a raw-unit release contains duplicate ids")
        for piece_index, block_id in enumerate(ids):
            actual = self._raw_unit_owner.get(block_id)
            expected = (owner, piece_index)
            if actual != expected:
                raise AssertionError(
                    f"raw unit {block_id} belongs to {actual!r}, not {expected!r}"
                )
        # Validate the complete set before releasing any member: a checkpoint
        # image is an atomic allocation and a partial release corrupts it.
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
        # A raw unit is one fragment of a larger atomic object.  Relocating it
        # here would require updating that object's ordered unit table and
        # copying bytes before the old address disappeared.  The first paged
        # checkpoint implementation therefore refuses the resize explicitly.
        if top in self._raw_unit_owner:
            return None
        if top in self._free:
            self._take_named(top)
            self._unindex(top)
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
        self._unindex(destination)
        src, dst = self.blocks[source], self.blocks[destination]
        dst.ref_count, dst.hash, dst.token_ids = src.ref_count, src.hash, src.token_ids
        if src.hash != -1 and self._hash_to_block_id.get(src.hash) == source:
            self._hash_to_block_id[src.hash] = destination
        self._used.discard(source)
        self._used.add(destination)
        src.ref_count, src.hash, src.token_ids = 0, -1, []
