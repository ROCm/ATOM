# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import TYPE_CHECKING

from atom.model_engine.state_runtime import StateTransfer

if TYPE_CHECKING:
    from atom.model_engine.state_offload import StateOffloadIndex

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class GroupRetirement:
    """What `retire_top` did, and what the caller still owes.

    `relocated_to` of -1 means the top group was empty or was itself the thing
    worth spending, so no bytes move. Otherwise the caller must copy
    `retired` → `relocated_to` before the next forward, and then either re-point
    the owning sequence (`held_checkpoint` false) or nothing at all — the index
    has already been re-filed here, because it is this class's to keep
    consistent.
    """

    retired: int
    relocated_to: int
    held_checkpoint: bool


class StateGroupPool:
    """Own Active Slot allocation and the fork-checkpoint index."""

    def __init__(
        self,
        num_groups: int,
        transfer: StateTransfer | None = None,
        hash_block_size: int = 1,
        enabled: bool = True,
    ):
        self.enabled: bool = enabled and num_groups > 0
        self.num_groups: int = num_groups
        self.transfer: StateTransfer = transfer or StateTransfer.none()
        # Committed tokens needed to make a fork destination self-contained.
        self.min_fork_tokens: int = self.transfer.fork_tokens
        self.successor_room: float = self.transfer.successor_room
        self.hash_block_size: int = hash_block_size
        if self.transfer.copies:
            raise ValueError("PAGE-copy checkpoints do not belong to StateGroupPool")
        # The free list, split by whether the group still carries content worth
        # something. Two containers rather than one queue because the two halves
        # want opposite orders and mixing them serves neither:
        #
        #   `_vacant`       nothing to lose, so order is free and it is chosen
        #                   to pack allocation towards index 0 — which is what
        #                   lets the pool shrink from the top without having to
        #                   move anybody.
        #   `_checkpointed` content is a resumable boundary, so order is LRU
        #                   and the head is what a shortage spends first.
        #
        # Vacant is always drawn from first, so a checkpoint can only be evicted
        # once there is nothing free left to take. A single release-ordered FIFO
        # cannot express that: a checkpoint released before a never-used group
        # sits ahead of it and is spent first.
        #
        # Which half a free group belongs to is a function of `group_hash`, not
        # a third piece of state, so `_set_hash` is the one place that moves a
        # group across and nobody else has to remember to. `_free` is membership
        # for both; `_vacant`'s heap keeps entries a `claim` took out and
        # `_pop_vacant` drops them when they surface.
        self._free: set[int] = set(range(num_groups))
        self._vacant: list[int] = list(range(num_groups))
        self._checkpointed: deque[int] = deque()
        # hash -> group holding the state as of that block boundary.
        self.hash_to_group: dict[int, int] = {}
        # Reverse map, for lazy eviction when a group is handed out. -1 = the
        # group carries no checkpoint.
        self.group_hash: list[int] = [-1] * num_groups
        # Backing store beneath the pool, or None. Not a member of
        # `BlockManager.state_caches`: every one of those is a veto, and this
        # tier does the opposite -- it makes more boundaries reachable.
        self.offload: StateOffloadIndex | None = None
        # (group, staging_slot) pairs the next forward must copy out of HBM.
        self._spill_copies: list[tuple[int, int]] = []
        # One-shot latch so the undrained-consumer diagnosis cannot flood the
        # log from `pop()`, which is itself a hot path.
        self._warned_spill_copies_undrained = False
        # Groups serving as a fork source in the in-flight step, counted by how
        # many requests fork off each. They stay off the free list until
        # `release_pins`, so the step that reads them cannot race a request that
        # was handed the same group. The count matters because a checkpoint is
        # read-only and several requests may share one: the group goes back only
        # once, and no reader may take it over while another still reads it.
        self._pinned: dict[int, int] = {}
        # Of those, the ones whose reader is the batch the *next* pass builds
        # rather than the one just built. `checkpoint` runs in postprocess,
        # after its batch went out, so the forward that reads the source it
        # hands over is one pass further off than a resume's. Carried as a set
        # rather than a second count because the depth is only ever one pass
        # more — see `release_pins`.
        self._deferred: set[int] = set()
        self._relocations: list[tuple[int, int]] = []
        # `dropped` had no group to go to; `evicted` landed and was later
        # spent on an allocation. Counted apart because they read the same in a
        # hit rate and want opposite fixes — the first says the pool is too
        # small for the rate of keeping, the second for how long a checkpoint
        # has to last.
        self.checkpoints_kept: int = 0
        self.checkpoints_dropped: int = 0
        self.checkpoints_evicted: int = 0
        # Died because the prefix it was filed under left the KV index first,
        # so nothing could ever have resumed off it. Apart from `evicted`
        # because they want opposite fixes: `evicted` says this pool is too
        # small for how long a checkpoint has to last, `orphaned` says the KV
        # pool is too small for the same span.
        self.checkpoints_orphaned: int = 0

    # ------------------------------ free list ------------------------------ #
    def has_free(self) -> bool:
        return bool(self._free)

    def num_free(self) -> int:
        return len(self._free)

    def is_free(self, group: int) -> bool:
        return group in self._free

    def holds_checkpoint(self, group: int) -> bool:
        """Whether `group` is on the free list *and* still worth something."""
        return group in self._free and self.group_hash[group] != -1

    def pop(self, *, spill: bool = True) -> int:
        """Hand out a group, evicting its checkpoint if it carried one.

        Vacant first, lowest index first; only when nothing is vacant does a
        checkpoint get spent, and then it is the least recently used one.

        `spill=False` gives up the evicted checkpoint's bytes instead of
        staging them, and exists for one caller: a group about to be filled by
        an offload *load*. The spill copy is issued by the next forward while
        the load runs on the tier's own thread ahead of it, so the two would
        race over the same group and the spill would store the loaded state
        under the evicted checkpoint's hash. Nothing downstream could tell.
        Forgoing the spill costs one checkpoint -- exactly the pre-tier
        behaviour, counted as `spills_forgone`.

        The state twin of `BlockManager._pop_free_block`: groups sit on the free
        list carrying whatever the last owner left in them, and re-allocation —
        not the free — is the eviction event.

        Lowest-index-first is not a fairness choice, it is what keeps the top of
        the pool cold: a group high in the range is only reached at a
        high-water-mark of concurrency, so after the peak passes the top holds
        the least recently touched things — exactly what shrinking should spend.
        """
        group = self._pop_vacant()
        if group < 0:
            group = self._checkpointed.popleft()
            self._free.discard(group)
            self.checkpoints_evicted += 1
            if spill:
                # Before invalidate() clears group_hash[group]: the hash is the
                # key the tier stores under.
                self._spill(group)
            elif self.offload is not None and self.offload.enabled:
                self.offload.spills_forgone += 1
        self.invalidate(group)
        return group

    def _pop_vacant(self) -> int:
        """Lowest vacant index, or -1, dropping entries that are stale.

        An entry is stale when the group has since been claimed or has taken a
        hash, both of which leave the heap untouched — cheaper than an exact
        removal, and the two conditions below are the same ones that decide
        which half a group belongs to anyway.
        """
        while self._vacant:
            group = heappop(self._vacant)
            if group in self._free and self.group_hash[group] == -1:
                self._free.discard(group)
                return group
        return -1

    def claim(self, group: int) -> None:
        """Take one specific free group off the list, content and all.

        Linear in `_checkpointed` when the group holds a checkpoint, which is
        the case the resume path takes. That is deliberate: the free list stays
        the single source of truth for "how many groups are free", which
        admission and every caller of `has_free` rely on. The scan is bounded by
        max_num_seqs and runs once per resuming request, against a
        `can_allocate` that already hashed every block of the prompt.
        """
        self._free.discard(group)
        if self.group_hash[group] != -1:
            self._checkpointed.remove(group)

    def release(self, group: int) -> None:
        """Hand a group back, to whichever half its content puts it in.

        A group still carrying a checkpoint goes to the LRU tail, so being
        resumed from refreshes it — `claim` deliberately leaves the hash in
        place, which is what makes reuse count as use.
        """
        self._free.add(group)
        if self.group_hash[group] != -1:
            self._checkpointed.append(group)
            return
        heappush(self._vacant, group)
        # Every group that took a hash while vacant left an entry behind, so on
        # a long-lived server the stale ones outnumber the live ones without
        # this. Rebuilding costs one pass and buys at least `num_groups` pushes.
        if len(self._vacant) > 2 * self.num_groups + 2:
            self._vacant = [g for g in self._free if self.group_hash[g] == -1]
            heapify(self._vacant)

    def _set_hash(self, group: int, h: int) -> None:
        """Change what an existing group backs, re-filing it if it is free.

        Going through here is what lets the two halves of the free list be a
        function of `group_hash` rather than a third thing to keep in step.
        `h` of -1 means the group backs nothing.
        """
        if group not in self._free:
            self.group_hash[group] = h
            return
        self.claim(group)
        self.group_hash[group] = h
        self.release(group)

    # ----------------------------- resizing -------------------------------- #
    def extend(self, count: int) -> None:
        """Add `count` group indices at the top. The caller freed the bytes.

        `group_hash` is grown but never shrunk, so an index that was retired and
        then handed back out reuses its slot rather than appending a second one.
        """
        for group in range(self.num_groups, self.num_groups + count):
            if group < len(self.group_hash):
                self.group_hash[group] = -1
            else:
                self.group_hash.append(-1)
            self.release(group)
        self.num_groups += count

    def retire_top(self) -> "GroupRetirement | None":
        """Give up the highest group index, relocating whatever sits on it.

        Shrinking is index-forced — the bytes being handed back are the ones the
        top group occupies — but the policy is not: what gets *spent* is the
        least recently used checkpoint, whichever index that is. So the top
        group's content moves to a target taken in the same order `pop` uses
        (vacant first, LRU checkpoint only if nothing is vacant) and the caller
        issues one copy.

        Without that move, shrinking would be anti-LRU: a group's index records
        the concurrency high-water mark when it was handed out and is never
        refreshed by use, so a checkpoint resumed from every second could sit at
        the top and be spent ahead of one nothing has touched in minutes.

        Returns `None` when the top group cannot be given up this pass — it is
        pinned as a fork source, or it is live and there is nowhere to move it.
        Both clear on their own, so the caller retries rather than blocks.
        """
        top = self.num_groups - 1
        if top < 0 or self.is_pinned(top):
            return None
        held = self.holds_checkpoint(top)
        if self.is_free(top) and not held:
            self.claim(top)
            self.num_groups -= 1
            return GroupRetirement(top, -1, False)

        dst = self._take_relocation_target(exclude=top)
        if dst < 0 and not held:
            return None
        if dst < 0:
            # Nothing free anywhere, so the top group is by elimination the only
            # checkpoint left — spending it is what LRU asks for regardless.
            self.claim(top)
            self.invalidate(top)
        elif held:
            self._rehome_checkpoint(top, dst)
        self.num_groups -= 1
        return GroupRetirement(top, dst, held)

    def _take_relocation_target(self, exclude: int) -> int:
        """A group to move content into, in `pop`'s order but skipping `exclude`."""
        group = self._pop_vacant()
        if group >= 0:
            return group
        for group in self._checkpointed:
            if group != exclude:
                self.claim(group)
                self.invalidate(group)
                return group
        return -1

    def _rehome_checkpoint(self, src: int, dst: int) -> None:
        """Give a checkpoint a different group index, LRU position included.

        Written against the containers rather than through `claim`/`release`
        because keeping the position is the whole point: released groups go to
        the tail, and a checkpoint that only changed address has not been used.
        """
        h = self.group_hash[src]
        self._checkpointed[self._checkpointed.index(src)] = dst
        self._free.discard(src)
        self._free.add(dst)
        self.group_hash[src] = -1
        self.group_hash[dst] = h
        self.hash_to_group[h] = dst

    # ---------------------------- applicability ---------------------------- #
    def applies(self, seq) -> bool:
        """Whether this class gates or checkpoints anything for `seq`.

        Per-request state is declared by the attention type, so a seq on a model
        without one carries no group and this class has no say over its hits.
        """
        return self.enabled and seq.has_per_req_cache

    # ------------------------------- lookup -------------------------------- #
    def resumable_hit(
        self,
        seq,
        hit: int,
        block_hashes: list[int],
        assume_checkpointed: bool = False,
    ) -> int:
        """Shrink a compressed-prefix hit to the nearest recoverable boundary.

        Returns the largest `L <= hit` such that a checkpoint exists for the
        prefix of `L` blocks AND the forward resuming there can leave its own
        group self-contained, scanning right-to-left so the hit is cut as little
        as possible. 0 is always valid — a request starting from scratch needs
        no prior state.

        The fork test is what `min_fork_tokens` buys: resuming reads the
        checkpoint and writes a fresh group, and that forward has to leave the
        fresh group whole. A boundary too close to the end of the prompt fails
        it and the scan keeps walking back.

        Without this a hit hands the resumed forward a group freshly popped off
        the free list and it reads the previous occupant's state.

        `assume_checkpointed` drops the index lookup and keeps the fork test,
        which is the counterfactual the protocol describes: this class's ladder
        made dense, leaving only what a checkpoint could not have fixed.
        """
        if not self.applies(seq):
            return hit
        hbs = self.hash_block_size
        for i in range(hit - 1, -1, -1):
            if not assume_checkpointed and not self._resumable_from(
                block_hashes[i], (i + 1) * hbs
            ):
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup_group(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
        return self.hash_to_group.get(h, -1)

    # ------------------------------- offload ------------------------------- #
    def _resumable_from(self, h: int, tokens: int) -> bool:
        """Whether a checkpoint for `h` can be *reached*, in HBM or beneath it.

        Reached, not merely indexed. `resumable_hit` scans right to left and
        stops at the first boundary this accepts, so accepting one whose bytes
        nothing can deliver does not cost a wasted lookup — it costs the whole
        walk-back, hiding every shorter checkpoint still resident in HBM.

        HBM wins without a preference rule: both tiers are indexed by the same
        hash, so the scan takes the rightmost boundary wherever it lives.
        """
        if h in self.hash_to_group:
            return True
        # Invariant, relied on to keep this hot path free of an `enabled` check:
        # a disabled (depth-0) tier never reaches `request_spill`, because
        # `_spill` returns on `not enabled` first, so `hashes` stays empty and
        # the membership test is false anyway. Anything that adds to `hashes`
        # outside the confirm-after-spill path must re-check `enabled` here.
        return self.offload is not None and h in self.offload.hashes and tokens > 0

    def _spill(self, group: int) -> None:
        """Stage `group`'s bytes for the tier, if there is room. Never blocks.

        Beyond the staging depth the spill is simply dropped and
        `checkpoints_evicted` still counts it — identical to the behaviour
        without a tier, so there is no regression to reason about.
        """
        if self.offload is None or not self.offload.enabled:
            return
        h = self.group_hash[group]
        if h == -1:
            return
        slot = self.offload.request_spill(h, group)
        if slot >= 0:
            self._spill_copies.append((group, slot))
            # Exact leak detector, not a heuristic: a slot must be released
            # before it can be handed out again, so at most `staging_depth`
            # copies can be outstanding if the consumer drains every step.
            # More than that is proof `take_spill_copies` is not being called.
            if (
                len(self._spill_copies) > self.offload.staging_depth
                and not self._warned_spill_copies_undrained
            ):
                self._warned_spill_copies_undrained = True
                logger.warning(
                    "StateGroupPool has %d undrained spill copies for a staging "
                    "ring of depth %d. take_spill_copies() must be called every "
                    "step and each slot released with release_staging(); until "
                    "it is, the staging ring leaks and every spill is dropped.",
                    len(self._spill_copies),
                    self.offload.staging_depth,
                )

    def has_pending_spill(self, group: int) -> bool:
        """Whether a queued spill copy still has to read `group`.

        The window a load must not enter. It closes when the copy is issued on
        the compute stream (`take_spill_copies` -> the next forward), after
        which only the staging entry is read and the group is free to be
        overwritten. Normally empty for a group `pop(spill=False)` just handed
        out; a copy left undrained by a batch that was never forwarded is the
        case this catches.
        """
        return any(g == group for g, _slot in self._spill_copies)

    def take_spill_copies(self) -> list[tuple[int, int]]:
        """`(group, staging_slot)` pairs the next forward must copy.

        The consumer turns each into `relocate_state_slots([(group,
        num_groups + slot)])` -- the ring is K groups appended past the pool's
        own range, so a staging slot needs no second addressing scheme.

        Not optional: whoever attaches a tier MUST drain this every step and
        MUST call `release_staging(slot)` once the bytes are safe. Skip the
        drain and the list grows without bound; skip the release and
        `request_spill` returns -1 forever, silently evicting every checkpoint.
        Both are warned about once, but a warning is a diagnosis, not a repair.
        """
        out, self._spill_copies = self._spill_copies, []
        return out

    # ---------------------------- checkpointing ---------------------------- #
    def checkpoint(self, seq, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of this boundary, filed under hash `h`.

        The owner moves to a fresh group and the old group becomes read-only.
        """
        if not self.applies(seq) or not self.transfer.forks:
            return
        old = seq.per_req_cache_group
        if old < 0:
            return
        if not self.has_free():
            self.checkpoints_dropped += 1
            return
        seq.per_req_cache_group = self.pop()
        seq.state_fork_src = old
        # Not released: `state_fork_src` names `old` as what this request's
        # NEXT forward reads, and that forward is two passes off — one to build
        # the batch carrying the fork, one to issue it. Handing it back now put
        # it on the free list during the pass that admits the requests which
        # could pop it, and then one kernel would read and write it at once.
        # `_index` files it either way; `_set_hash` leaves a non-free group be.
        self._index(h, old)
        self.pin(old, reader_is_next_batch=True)
        self.checkpoints_kept += 1

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of the checkpoints the ladder asked this pool to keep.

        Folds in the tier's counters when one is attached, which is what makes
        `spills_dropped` visible -- a ring dropping one spill in three never
        trips the 256-consecutive starvation warning and would otherwise say
        nothing. Keys are prefixed `state_offload_` because the aggregation is
        by key across every state class.
        """
        fates = {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.checkpoints_evicted,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }
        if self.offload is not None:
            fates.update(
                (f"state_offload_{k}", v) for k, v in self.offload.stats().items()
            )
        return fates

    def record_relocation(self, src: int, dst: int) -> None:
        """Schedule an Active Slot relocation for the next batch."""
        self._relocations.append((src, dst))

    def take_relocations(self) -> tuple[tuple[int, int], ...]:
        relocations, self._relocations = self._relocations, []
        return tuple(relocations)

    def _index(self, h: int, group: int) -> None:
        """File `group` as the checkpoint for hash `h`.

        Caller guarantees `group` holds the state as of that boundary and has
        been handed back to the free list — a checkpointed group is never
        written again, so the request that produced it must have moved on.
        """
        if not self.enabled:
            return
        self.invalidate(group)
        prev = self.hash_to_group.get(h, -1)
        self.hash_to_group[h] = group
        self._set_hash(group, h)
        # Re-filing a hash onto a different group orphans the old one. Drop its
        # back-pointer directly rather than through `invalidate`, which would
        # take the entry just written with it.
        if prev != -1 and prev != group:
            self._set_hash(prev, -1)

    def unindex(self, h: int) -> None:
        """Drop the checkpoint filed under `h`. The dual of `_index`.

        Called when the KV block of that hash leaves the block index. The two
        pools are addressed by the same chained content hash and a prefix hit
        is a joint claim on both, so a checkpoint whose block is gone can never
        be reached again: `_gated_hit` caps at the last block still indexed.
        Until it is dropped it holds a group and sits in the LRU queue ahead of
        checkpoints that are still worth something, so the pool spends a live
        one to make room for a dead one.

        Necessary but not sufficient for reachability — an *earlier* block in
        the chain may be the one that went — so this reclaims a subset. Making
        it exact would have the state pool watch every block of every prefix,
        which costs more than the tail it would catch.

        """
        group = self.hash_to_group.get(h, -1)
        if group < 0:
            return
        if self.offload is not None:
            self._spill(group)
        self.invalidate(group)
        self.checkpoints_orphaned += 1

    def clear_index(self) -> None:
        """Drop all checkpoint hashes, preserving only in-flight readers."""
        for group in list(self.hash_to_group.values()):
            self.invalidate(group)

    def invalidate(self, group: int) -> None:
        """Drop `group`'s checkpoint. Called when the group is handed out."""
        if not self.enabled:
            return
        h = self.group_hash[group]
        if h == -1:
            return
        self._set_hash(group, -1)
        if self.hash_to_group.get(h) == group:
            del self.hash_to_group[h]

    # -------------------------------- pins --------------------------------- #
    def pin(self, group: int, *, reader_is_next_batch: bool = False) -> None:
        """Hold `group` off the free list until the forward that reads it ran.

        The caller states which batch that is, not when to let go: a resume
        pins while its own batch is being built, `checkpoint` pins after its
        batch went out and the forward that reads what it handed over is the
        one the next pass builds. `release_pins` turns the two into passes.
        """
        self._pinned[group] = self._pinned.get(group, 0) + 1
        if reader_is_next_batch:
            self._deferred.add(group)

    def is_pinned(self, group: int) -> bool:
        return group in self._pinned

    def pin_count(self, group: int) -> int:
        """Requests forking off `group` in the in-flight step."""
        return self._pinned.get(group, 0)

    def unpin(self, group: int) -> None:
        """Drop one reference without freeing — the caller took the group over.

        Only legal on the last reference; a group another request still reads
        cannot be taken over. `BlockManager.cancel_state_fork` enforces that.
        """
        count = self._pinned.get(group, 0)
        if count <= 1:
            self._pinned.pop(group, None)
            self._deferred.discard(group)
        else:
            self._pinned[group] = count - 1

    def drop_reader(self, group: int) -> None:
        """A reader of `group` is gone before it read. Free it if it was last.

        A pin says some forward still has to read this group. When the request
        that owed that forward is deallocated the obligation goes with it, and
        holding the group to the clock would cost admission a group for
        nothing — which is what keeps `checkpoint` capacity-neutral for a
        request that finishes or is preempted on the boundary it published.
        `release_pins` stays the backstop, for readers still to come.
        """
        if group < 0 or not self.is_pinned(group):
            return
        self.unpin(group)
        if not self.is_pinned(group):
            self.release(group)

    def release_pins(self) -> None:
        """Return the sources whose reading forward has been issued.

        They go back to the free list, once each however many requests read
        them. Safe once that forward is out: a request handed one of these
        groups next pass runs its own forward after it, on the same stream.

        A pin taken while a batch was being built is read by that batch, so it
        clears here, one pass later. A pin marked `reader_is_next_batch` is
        read by the batch this pass is about to build, so it survives one more
        — and `checkpoint`'s source has to, because it is on nobody's free
        list to protect it during the very pass that admits the requests which
        could otherwise be handed it.

        Every pin clears within two passes whatever else happens. Nothing here
        waits on the obligation being consumed, so a request preempted between
        taking a checkpoint and its next batch cannot strand a group — which
        is why this stays a clock rather than moving to `_consume_state_forks`,
        where it would be tighter and leak.
        """
        if not self._pinned:
            return
        held, self._deferred = self._deferred, set()
        for group in sorted(self._pinned):
            if group in held:
                continue
            self.release(group)
            del self._pinned[group]
