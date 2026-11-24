# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from itertools import chain, islice, zip_longest

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

    Optional ``slru`` puts blocks claimed for reuse into a bounded protected
    queue on release. New content remains probationary until claimed, so a
    stream of one-off prefixes cannot evict the whole hot set. Vacant blocks
    still go first; referenced blocks are never candidates in either policy.

    ``residues`` > 1 splits the free blocks into that many lanes by
    ``block_id % residues``, each with its own vacant / cached / protected
    containers, and `pop` takes from the lane the caller names. MiniMax-M3's
    index sharding hands logical block j a block of lane ``j % residues``, so a
    block's id says which rank keeps its index (`minimax_m3/index_shard.py`).
    One lane is the pool as it always was. A lane is a fixed share of the ids,
    so a pool with lanes does not move its boundary (`extend`, `retire_top`)
    and keeps no raw units.

    Optional write-back (`set_write_back`): a block a lower tier has not stored
    yet ("dirty") is recomputed if evicted, one it holds is only reloaded. An
    allocation that reaches a dirty block moves it to its lane's `_held`, still
    free and claimable, and evicts the next clean one; held blocks go only once
    no clean block with content is left in the lane, coldest first, counted in
    `blocks_evicted_dirty`. The writer stores them meanwhile (`held_blocks`).
    """

    def __init__(
        self,
        num_blocks: int,
        on_evict: Callable[[int], None] | None = None,
        max_blocks: int | None = None,
        cache_policy: str = "lru",
        protected_ratio: float = 0.5,
        on_remap: Callable[[int], None] | None = None,
        residues: int = 1,
    ):
        cache_policy = cache_policy.strip().lower()
        if cache_policy not in {"lru", "slru"}:
            raise ValueError(f"unknown prefix cache policy: {cache_policy!r}")
        if not 0 < protected_ratio < 1:
            raise ValueError("protected_ratio must be between 0 and 1")
        self.cache_policy = cache_policy
        self.protected_ratio = protected_ratio
        if residues < 1:
            raise ValueError(f"a pool has at least one lane, not {residues}")
        if residues > 1 and (
            num_blocks % residues or max_blocks not in (None, num_blocks)
        ):
            raise ValueError(
                f"{residues} lanes need a fixed pool of a multiple of {residues} "
                f"blocks, not {num_blocks} (max {max_blocks})"
            )
        self.residues = residues
        # Per lane (`block_id % residues`), as every free-list container below.
        self._protected: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(residues)
        ]
        self._reused: set[int] = set()
        # `max_blocks` is how far `extend` may go, and so how many Block
        # objects exist. It is the pool's share of a fixed plane rather than
        # its current size; a pool with a pinned boundary passes neither and
        # gets a maximum equal to its size.
        self.max_blocks: int = num_blocks if max_blocks is None else max_blocks
        if not 0 <= num_blocks <= self.max_blocks:
            raise ValueError(f"{num_blocks} blocks outside 0..{self.max_blocks}")
        self.num_blocks: int = num_blocks
        self._on_evict = on_evict
        # Called with a hash whose index entry moved to another block id
        # (republished elsewhere, `_adopt`): ids resolved before then are stale.
        self._on_remap = on_remap
        self.blocks: list[Block] = [Block(i) for i in range(self.max_blocks)]
        self._hash_to_block_id: dict[int, int] = {}
        # Bumped when a hash leaves the index or moves to another id (not when
        # one is added): ids resolved at version v still hold while it is v.
        self.index_version: int = 0
        # Both containers may hold ids that were re-claimed straight off the
        # free list (`claim`) or retired, so membership in the set — not
        # presence in a container — is what makes an id free. `pop` skips the
        # stale entries. A lane's ascending ids are already a heap.
        self._vacant: list[list[int]] = [
            list(range(lane, num_blocks, residues)) for lane in range(residues)
        ]
        self._cached: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(residues)
        ]
        # Dirty free blocks an allocation passed over, coldest first (see the
        # class doc); empty without `set_write_back`.
        self._held: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(residues)
        ]
        self._is_dirty: Callable[[int], bool] | None = None
        self._free: set[int] = set(range(num_blocks))
        self._free_count: list[int] = [len(v) for v in self._vacant]
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
        # Of `blocks_evicted`, held blocks evicted for want of a clean one.
        self.blocks_evicted_dirty: int = 0

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

    def has_free_per_residue(self, demand: Sequence[int]) -> bool:
        """Whether lane r holds `demand[r]` free blocks, for every lane."""
        return all(have >= want for have, want in zip(self._free_count, demand))

    def next_evicted(self, count: int):
        """The free blocks with content that the next `count` allocations
        would evict, coldest first (`_take_free`'s order). Vacant blocks go
        first and lose nothing, so they shorten the list rather than appear in
        it. Allocations draw on the lanes evenly, so each lane gives its next
        ``ceil(count / residues)`` victims, interleaved. Lazy: consume it
        before changing the pool. Held blocks are not in it (`held_blocks`)."""
        share = -(-count // self.residues)
        lanes = [
            islice(
                chain(cached, protected),
                max(0, share - (free - len(cached) - len(protected) - len(held))),
            )
            for free, cached, protected, held in zip(
                self._free_count, self._cached, self._protected, self._held
            )
        ]
        if len(lanes) == 1:
            return lanes[0]
        return (b for group in zip_longest(*lanes) for b in group if b is not None)

    def set_write_back(self, is_dirty: Callable[[int], bool] | None) -> None:
        """Evict clean blocks before dirty ones (class doc). `is_dirty(block_id)`
        runs on each free block with content an allocation reaches: keep it
        cheap. None turns it off; blocks held by then stay last."""
        self._is_dirty = is_dirty

    def held_blocks(self):
        """The free blocks an allocation passed over as dirty, coldest first
        in each lane, the lanes interleaved. Lazy: consume it before changing
        the pool."""
        if len(self._held) == 1:
            return iter(self._held[0])
        return (
            b for group in zip_longest(*self._held) for b in group if b is not None
        )

    @property
    def num_held(self) -> int:
        return sum(map(len, self._held))

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
        return (
            sum(map(len, self._cached))
            + sum(map(len, self._protected))
            + self.num_held
        )

    def eviction_stats(self) -> dict[str, int]:
        """Content this pool destroyed, and the headroom it has left.

        Counters, not rates, for the same reason `EngineStats.cache_statistics`
        hands back counts: a rate cannot be summed across DP ranks that saw
        different traffic.
        """
        return {
            "blocks_evicted": self.blocks_evicted,
            "blocks_evicted_dirty": self.blocks_evicted_dirty,
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

    @property
    def index_get(self):
        """The index's own `dict.get`: block id or None, for C-speed lookups
        of a whole chain with `map`. Read it per walk, not once per pool."""
        return self._hash_to_block_id.get

    def used_among(self, block_ids: list[int], n: int) -> set[int]:
        """Which of the first `n` of `block_ids` (distinct) some live sequence
        holds. No slice; the set holds only the used ids, and a waiting head's
        hit blocks are mostly free."""
        return self._used.intersection(islice(block_ids, n))

    def claim_many(self, block_ids) -> None:
        """`claim` each of `block_ids`, in order, with the lookups hoisted."""
        blocks, used, take = self.blocks, self._used, self._take_named
        slru = self.cache_policy == "slru"
        for block_id in block_ids:
            block = blocks[block_id]
            if slru and block.hash != -1:
                self._reused.add(block_id)
            if block_id in used:
                block.ref_count += 1
            else:
                assert block.ref_count == 0
                block.ref_count = 1
                take(block_id)
                used.add(block_id)

    def publish(self, block_id: int, h: int, token_ids: array.array) -> None:
        """Index `block_id` under the content hash of the tokens it now holds.
        A different hash it was indexed under goes, as on eviction: indexed
        content never changes."""
        block = self.blocks[block_id]
        index = self._hash_to_block_id
        old = block.hash
        if old not in (-1, h) and index.get(old) == block_id:
            del index[old]
            self.index_version += 1
            if self._on_evict is not None:
                self._on_evict(old)
        block.update(h, token_ids)
        prev = index.get(h)
        if prev == block_id:
            return
        index[h] = block_id
        if prev is not None:
            self.index_version += 1
            if self._on_remap is not None:
                self._on_remap(h)

    def clear_index(self) -> None:
        """Drop every content-hash entry, keeping blocks live sequences hold.

        Those stay valid through their block_table refs; they are simply no
        longer reachable by hash, so no future request can claim them.
        """
        self._hash_to_block_id.clear()
        self.index_version += 1
        for block in self.blocks:
            if block.ref_count == 0:
                block.hash = -1
                block.token_ids = array.array("i")
        # Every free block is vacant now, and which container an id is in is
        # only ever decided from its hash — so the split has to be redrawn
        # here rather than left to drift.
        for cached, protected, held in zip(self._cached, self._protected, self._held):
            cached.clear()
            protected.clear()
            held.clear()
        self._reused.clear()
        self._vacant = [[] for _ in range(self.residues)]
        for block_id in sorted(self._free):
            self._vacant[block_id % self.residues].append(block_id)

    def _unindex(self, block_id: int) -> bool:
        """Drop `block_id`'s index entry and forget what it held.

        Returns whether an index entry actually went — i.e. whether reusable
        content was destroyed. A block with no hash, or one whose hash the
        index has since re-pointed elsewhere, costs nothing to drop, and the
        callers that count evictions must not count those.
        """
        block = self.blocks[block_id]
        self._protected[block_id % self.residues].pop(block_id, None)
        self._held[block_id % self.residues].pop(block_id, None)
        self._reused.discard(block_id)
        dropped = False
        if block.hash != -1 and self._hash_to_block_id.get(block.hash) == block_id:
            del self._hash_to_block_id[block.hash]
            self.index_version += 1
            dropped = True
            if self._on_evict is not None:
                self._on_evict(block.hash)
        block.hash = -1
        block.token_ids = array.array("i")
        return dropped

    # ---------------------------- allocation ------------------------------- #
    def _take_free(self, lane: int = 0) -> int:
        """Next free block id of `lane`, or -1. Vacant before cached; see the
        class doc.

        An entry is stale when the block has since been taken, or when it has
        gained or lost a hash and so belongs in the other container — an id can
        sit in both at once, and testing only that it is free would hand a
        cached block out of the vacant half. Both conditions are the ones that
        decide which half it belongs to anyway, so the test is the definition
        rather than a guard bolted on top.
        """
        vacant = self._vacant[lane]
        while vacant:
            block_id = heappop(vacant)
            if block_id in self._free and self.blocks[block_id].hash == -1:
                self._free.discard(block_id)
                self._free_count[lane] -= 1
                return block_id
        is_dirty = self._is_dirty
        held = self._held[lane]
        for queue in (self._cached[lane], self._protected[lane]):
            while queue:
                block_id, _ = queue.popitem(last=False)
                if block_id in self._free and self.blocks[block_id].hash != -1:
                    if is_dirty is not None and is_dirty(block_id):
                        held[block_id] = None
                        continue
                    self._free.discard(block_id)
                    self._free_count[lane] -= 1
                    return block_id
        while held:
            block_id, _ = held.popitem(last=False)
            if block_id in self._free and self.blocks[block_id].hash != -1:
                self._free.discard(block_id)
                self._free_count[lane] -= 1
                if is_dirty is not None and is_dirty(block_id):
                    self.blocks_evicted_dirty += 1
                return block_id
        return -1

    def pop(self, lane: int = 0) -> int:
        """A free block of `lane` (`block_id % residues`), see `_take_free`."""
        block_id = self._take_free(lane)
        if block_id < 0:
            raise AssertionError(f"No free blocks available in lane {lane}")
        return block_id

    def _take_named(self, block_id: int) -> None:
        """Take one specific block off the free list, content and all.

        The cached half is ordered by when a block was released, so an id left
        in it after being taken would come back at its old position rather than
        the end — see the class doc. The vacant half is ordered by id, where a
        leftover entry is only a wasted pop.
        """
        lane = block_id % self.residues
        if block_id in self._free:
            self._free.remove(block_id)
            self._free_count[lane] -= 1
        self._cached[lane].pop(block_id, None)
        self._protected[lane].pop(block_id, None)
        self._held[lane].pop(block_id, None)

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
        if self.cache_policy == "slru" and block.hash != -1:
            self._reused.add(block_id)
        if block_id in self._used:
            block.ref_count += 1
        else:
            assert block.ref_count == 0
            block.ref_count = 1
            self._take_named(block_id)
            self._used.add(block_id)
        return block

    def claim_unreused(self, block_id: int) -> Block:
        """`claim` without counting a reuse, for a holder that only copies the
        block out (a lazy offload demotion): under SLRU, freeing it must not
        promote it into the protected queue. Not a `claim` flag, which every
        prefix hit would pay to test."""
        reused = block_id in self._reused
        block = self.claim(block_id)
        if not reused:
            self._reused.discard(block_id)
        return block

    def free(self, block_id: int, coldest: bool = False) -> None:
        """Drop a share of `block_id`; the last one returns it to the free list.

        `coldest` puts a block with content back at the head of its queue
        instead of the tail, to be evicted first, for a holder that took it
        from there only to copy it out (a lazy offload demotion).
        """
        if block_id in self._raw_unit_owner:
            raise AssertionError(
                f"block {block_id} is a reserved raw unit; use release_units"
            )
        block = self.blocks[block_id]
        block.ref_count -= 1
        if block.ref_count:
            return
        lane = block_id % self.residues
        self._used.remove(block_id)
        self._free.add(block_id)
        self._free_count[lane] += 1
        if block.hash != -1:
            protected = self._protected[lane]
            if coldest:
                # At the head before the trim: a full protected queue then
                # demotes this block rather than a hotter one.
                queue = protected if block_id in self._reused else self._cached[lane]
                queue[block_id] = None
                queue.move_to_end(block_id, last=False)
                if queue is protected:
                    self._trim_protected(lane)
            elif block_id in self._reused:
                protected[block_id] = None
                self._trim_protected(lane)
            else:
                self._cached[lane][block_id] = None
            return
        vacant = self._vacant[lane]
        heappush(vacant, block_id)
        # Stale entries are skipped, not removed, so the heap can outgrow the
        # pool under churn. Rebuilding costs one pass and buys at least
        # `num_blocks` pushes.
        if len(vacant) > 2 * self.num_blocks // self.residues + 2:
            vacant = [
                b
                for b in self._free
                if b % self.residues == lane and self.blocks[b].hash == -1
            ]
            heapify(vacant)
            self._vacant[lane] = vacant

    def _trim_protected(self, lane: int) -> None:
        # Bound protection so new prefixes can compete for cache space: a
        # lane's share of the bound.
        limit = int(self.num_blocks // self.residues * self.protected_ratio)
        protected = self._protected[lane]
        while len(protected) > limit:
            block_id, _ = protected.popitem(last=False)
            self._reused.discard(block_id)
            self._cached[lane][block_id] = None

    def reserve_units(self, count: int, owner: Hashable) -> list[int] | None:
        """Reserve arbitrary PAGE-sized units for raw storage."""
        if count < 0:
            raise ValueError(f"unit count must be non-negative, got {count}")
        if owner is None:
            raise ValueError("a raw-unit reservation needs an owner")
        self._require_one_lane("raw units")
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

    def rekey_units(
        self,
        unit_ids: Iterable[int],
        old_owner: Hashable,
        new_owner: Hashable,
    ) -> None:
        """Atomically transfer a raw reservation between logical owners.

        The units remain allocated throughout the hand-off.  This is required
        when an external-transfer scratch image becomes a reusable checkpoint:
        releasing and re-reserving would briefly expose the units through the
        free list, allowing another allocation to overwrite bytes that the new
        checkpoint index already advertises.
        """
        ids = list(unit_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("a raw-unit ownership transfer contains duplicate ids")
        if old_owner is None or new_owner is None:
            raise ValueError("raw-unit ownership transfer needs both owners")
        for piece_index, block_id in enumerate(ids):
            actual = self._raw_unit_owner.get(block_id)
            expected = (old_owner, piece_index)
            if actual != expected:
                raise AssertionError(
                    f"raw unit {block_id} belongs to {actual!r}, not {expected!r}"
                )
        for piece_index, block_id in enumerate(ids):
            self._raw_unit_owner[block_id] = (new_owner, piece_index)

    # ------------------------------ resizing ------------------------------- #
    def extend(self, count: int) -> int:
        """Grow the pool by up to `count` blocks; returns how many it took.

        Capped at `max_blocks`, which is how many blocks the plane can address
        — beyond it there is nothing to hand out even if the caller has bytes.
        """
        self._require_one_lane("growing")
        taken = min(count, self.max_blocks - self.num_blocks)
        for block_id in range(self.num_blocks, self.num_blocks + taken):
            self._free.add(block_id)
            heappush(self._vacant[0], block_id)
        self._free_count[0] += taken
        self.num_blocks += taken
        return taken

    def _require_one_lane(self, what: str) -> None:
        if self.residues > 1:
            raise NotImplementedError(f"{what}: a pool with lanes is fixed")

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
        self._require_one_lane("retiring")
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
        self._trim_protected(0)
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
        if source in self._reused:
            self._reused.discard(source)
            self._reused.add(destination)
        dst.ref_count, dst.hash, dst.token_ids = src.ref_count, src.hash, src.token_ids
        if src.hash != -1 and self._hash_to_block_id.get(src.hash) == source:
            self._hash_to_block_id[src.hash] = destination
            self.index_version += 1
            if self._on_remap is not None:
                self._on_remap(src.hash)
        self._used.discard(source)
        self._used.add(destination)
        src.ref_count, src.hash, src.token_ids = 0, -1, array.array("i")
