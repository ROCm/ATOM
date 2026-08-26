# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import array
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from atom.model_engine.kv_block import Block
from atom.model_engine.superblock import SuperblockMap


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
        superblocks: "SuperblockMap | None" = None,
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
        # Superblock claims that found nothing whole, and claims that had to
        # spend cached content to get it. The first is a state slot that could
        # not be created; the second is reuse destroyed to create one. They
        # want opposite fixes -- more superblocks vs fewer state slots -- and
        # read identically in a hit rate, so they are counted apart.
        self.superblock_claims_refused: int = 0
        self.superblocks_evicted_cached: int = 0
        # Physical grouping, when a hybrid backend needs contiguous ranges for
        # its per-request state. `None` is the whole story for every model that
        # does not: no mapping, no packing preference, and every path below
        # behaves exactly as it did before superblocks existed.
        self.superblocks = superblocks

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
        stats = {}
        if self.superblocks is not None:
            stats = dict(self.superblocks.occupancy())
            stats["superblock_claims_refused"] = self.superblock_claims_refused
            stats["superblocks_evicted_cached"] = self.superblocks_evicted_cached
        return stats | {
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
        # Nothing is reachable by hash any more, so no superblock is holding
        # content worth preferring away from — see `SuperblockMap.untype`, which
        # `free` deliberately does not call.
        if self.superblocks is not None:
            for index in range(self.superblocks.num_supers):
                self.superblocks.untype(index)

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
        # Packing, when superblocks are in play: prefer a free block in the
        # superblock fresh content is already landing in, so a request's live
        # blocks cluster instead of scattering one per superblock and leaving
        # none reclaimable whole. Applied per tier, *inside* the vacant-then-
        # cached order rather than ahead of it — preferring a cached block in
        # the open superblock over a vacant one anywhere would spend reusable
        # content while empty blocks waited.
        if self.superblocks is not None:
            vacant = {b for b in self._free if self.blocks[b].hash == -1}
            block_id = self.superblocks.preferred_free(vacant)
            if block_id >= 0:
                self._take_named(block_id)
                return block_id
        while self._vacant:
            block_id = heappop(self._vacant)
            if block_id in self._free and self.blocks[block_id].hash == -1:
                self._free.discard(block_id)
                return block_id
        if self.superblocks is not None:
            cached = {b for b in self._free if self.blocks[b].hash != -1}
            block_id = self.superblocks.preferred_free(cached)
            if block_id >= 0:
                self._take_named(block_id)
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
        block.reset()  # ref_count = 1: this is where the block becomes live
        self._take_named(block_id)
        self._used.add(block_id)
        if self.superblocks is not None:
            self.superblocks.on_block_live(block_id)
            # Fresh content, so this is the superblock the next fresh block
            # should pack beside. `claim` deliberately does not say this: a
            # cache hit lands wherever the content already is, and following it
            # would scatter the writer's frontier across the pool.
            self.superblocks.note_allocation(block_id)
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
            if self.superblocks is not None:
                self.superblocks.on_block_live(block_id)
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
        # Before the tier split below, which returns early: the block stopped
        # being live whether its content stayed cached or not, and a superblock
        # counts live blocks, not occupied ones.
        if self.superblocks is not None:
            self.superblocks.on_block_free(block_id)
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

    # ---------------------------- superblocks ------------------------------ #
    def can_claim_superblock(self) -> bool:
        """Whether a contiguous superblock could be had, evicting if it came to that."""
        return self._next_claimable_superblock() >= 0

    def can_claim_superblocks(self, count: int) -> bool:
        """Whether `count` superblocks could be had, one after another.

        Counted rather than found: asking `_next_claimable_superblock` `count`
        times would name the same one every time, since nothing is taken until
        a claim happens. A KV floor is held back — spending the pool down to
        nothing for state slots would leave no room to prefill the request that
        wanted them, and admission would then refuse for want of blocks it had
        just given away.
        """
        sb = self.superblocks
        if sb is None or count <= 0:
            return False
        reclaimable = sum(1 for i in range(sb.num_supers) if sb.is_reclaimable(i))
        return reclaimable - count >= self._kv_superblock_floor()

    def _kv_superblock_floor(self) -> int:
        """Superblocks kept out of reach of state slots, for live KV.

        A tenth of the pool, floor 1. Arbitrary in the way a floor has to be:
        the right number depends on prompt lengths nobody knows at sizing time,
        and being wrong low costs an admission deferred a step where being
        wrong high costs checkpoints that had somewhere to go.
        """
        sb = self.superblocks
        return max(1, sb.num_supers // 10) if sb is not None else 0

    def _next_claimable_superblock(self) -> int:
        """A superblock with no live block, preferring one that costs nothing.

        What a claim costs is decided by hashes, not by typing. A superblock
        whose blocks carry no hash holds nothing any request could resume
        from, so taking it destroys nothing — whether it is UNTYPED (never
        used) or KV (used, released, and never checkpointed, which is what a
        request's own working blocks become when they end). Keying on
        `kind == UNTYPED` alone, as this first did, made the second case
        invisible and refused claims against a pool that was entirely free.

        Only once none of those remain is a cached superblock spent, and then
        by LRU read out of `_cached` — the pool's own release order over
        blocks — rather than from a second order kept per superblock, which
        could drift from it. A superblock inherits the coldness of its coldest
        block, which is the block `_take_free` would have spent next anyway.
        Scanning by index instead spends whichever sits lowest in the pool
        however hot its content is.
        """
        sb = self.superblocks
        if sb is None:
            return -1
        for index in range(sb.num_supers):
            if sb.is_reclaimable(index) and self._superblock_is_free(index):
                return index
        # `_cached` is insertion-ordered by release, coldest first, so the
        # first fully-reclaimable superblock it names is the LRU one.
        for block_id in self._cached:
            index = sb.super_of(block_id)
            if sb.is_reclaimable(index):
                return index
        return -1

    def _superblock_is_free(self, index: int) -> bool:
        """Whether taking this superblock would destroy no reusable content."""
        sb = self.superblocks
        if sb is None:
            return False
        return all(self.blocks[b].hash == -1 for b in sb.block_range(index))

    def claim_superblock(self) -> int:
        """Take one superblock whole, for a contiguous per-request state slot.

        -1 rather than a raise when none can be had: the caller's answer is to
        defer the admission a step, which is what it already does when the
        state pool is full, and turning a scheduling decision into a crash
        would be a regression on a path that has always been able to say no.

        Cached content in the superblock is spent through `allocate`, so
        `blocks_evicted` and `on_evict` fire exactly as they do for any other
        eviction — a checkpoint whose KV blocks went this way has to learn its
        prefix is gone by the same route it always did.
        """
        sb = self.superblocks
        if sb is None:
            raise AssertionError("no superblock map: this pool is block-only")
        index = self._next_claimable_superblock()
        if index < 0:
            # Nothing whole to give. A deferred admission rather than a fault,
            # but counted: it is the difference between "the pool is busy" and
            # "state slots cannot be created at all", which the hit rate alone
            # cannot tell apart.
            self.superblock_claims_refused += 1
            return -1
        if not self._superblock_is_free(index):
            # Spending a superblock whose blocks still held reusable content.
            # Keyed on hashes, not on `kind`: a KV superblock that was used,
            # released and never checkpointed holds nothing findable, so
            # taking it costs no reuse and must not be counted as if it did.
            #
            # Counted apart from `blocks_evicted`, which the loop below also
            # moves: this says the claim had to reach into the cache, i.e. the
            # pool had nothing spare, where `blocks_evicted` alone cannot say
            # whether a KV allocation or a state slot destroyed the content.
            self.superblocks_evicted_cached += 1
        for block_id in sb.block_range(index):
            if self.blocks[block_id].hash != -1:
                # Destroys reusable content: counted and announced through the
                # ordinary path, so a checkpoint filed under this hash learns
                # its prefix is gone by the route it always did.
                #
                self.allocate(block_id)
                self.free(block_id)
            # After `free`, not instead of it: `free` returns the block to the
            # free list, and leaving it there would let `pop` hand out an id
            # this superblock now owns.
            self._take_named(block_id)
        sb.take_state(index)
        return index

    def release_superblock(self, index: int) -> None:
        """Give a state slot's superblock back to the block pool."""
        sb = self.superblocks
        if sb is None:
            raise AssertionError("no superblock map: this pool is block-only")
        sb.release_state(index)
        owned = set(sb.block_range(index))
        # Purge before re-adding. `claim_superblock` took these ids off `_free`
        # but could not purge `_vacant`, which tolerates stale entries by
        # design — `_take_free` skips an id that is no longer free. Pushing on
        # top would leave each id in the heap twice, and once the block is
        # freed again *both* copies pass the staleness test, so `pop` hands the
        # same id to two callers and the second `allocate` trips its ref-count
        # assertion. The trace replay found this; no unit test reached it.
        self._vacant = [b for b in self._vacant if b not in owned]
        heapify(self._vacant)
        for block_id in sb.block_range(index):
            # `_used` too, not only `_free`. A block left in `_used` while
            # sitting on the free list is handed out by `pop` and then fails
            # `allocate`'s ref-count assertion — and `claim` would treat it as
            # a live share rather than a fresh take. The trace replay caught
            # this several hundred requests in; the unit tests did not, because
            # none of them claimed a superblock whose blocks had been live.
            self._used.discard(block_id)
            self.blocks[block_id].ref_count = 0
            self._free.add(block_id)
            heappush(self._vacant, block_id)
        sb.untype(index)

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
