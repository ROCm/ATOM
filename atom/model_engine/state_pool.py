# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush

from atom.model_engine.state_runtime import StateTransfer


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
        # Whether a checkpoint may sit anywhere inside a forward rather than
        # only at its last token. False is the conservative answer and the
        # reason `BlockManager` still cuts prefill chunks onto rungs; see
        # `StateTransfer.readable_midstep`.
        self.readable_midstep: bool = self.transfer.readable_midstep
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
        # Groups whose checkpoint is a guess rather than this prompt's own end;
        # `mark_speculative` explains what the distinction is worth. Cleared
        # when the guess pays off (`promote`) or the content goes (`release` of
        # a group carrying nothing) — both of which end the group's claim to be
        # spent first.
        self._speculative: set[int] = set()
        # hash -> group holding the state as of that block boundary.
        self.hash_to_group: dict[int, int] = {}
        # Reverse map, for lazy eviction when a group is handed out. -1 = the
        # group carries no checkpoint.
        self.group_hash: list[int] = [-1] * num_groups
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

    def pop(self) -> int:
        """Hand out a group, evicting its checkpoint if it carried one.

        Vacant first, lowest index first; only when nothing is vacant does a
        checkpoint get spent, and then it is the least recently used one.

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
        # Whatever guess this group carried is spent with it. Not folded into
        # `invalidate`, which also runs on the re-file inside `_index` and would
        # clear a mark `_commit_pending` had only just made.
        self._speculative.discard(group)
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

    def mark_speculative(self, group: int) -> None:
        """File `group` at the LRU *head* when it is released, not the tail.

        The next shortage then spends it before every checkpoint already held.
        For a placement that guesses where reuse will resume rather than
        knowing: the ladder rung and the demand guess, the prompt-end anchor
        knows.

        The guess and the knowledge are not close in value. Measured on the
        cc-traces at conc 4, prompt-end anchors are read back 85.2% of the time
        and demand rungs 2.8%, while the demand is 47% of all writes (1,370 of
        2,919) — so under plain LRU the placement almost never read evicts the
        one almost always read. Demoting rather than dropping keeps the demand's
        own purpose: it exists to fill a gap once, and one spent before it is
        read cost nothing a checkpoint never taken would have saved.

        A mark, not an argument to `release`, because the group that carries a
        checkpoint under `fork` is *pinned* rather than released — it goes back
        through `release_pins` two passes later, which knows only the group
        index. Marking is the one mechanism both paths can reach.
        """
        if self.enabled:
            self._speculative.add(group)

    def promote(self, group: int) -> None:
        """A guess paid off: stop spending this one first.

        Called from the resume path when a request actually reads `group`. The
        group may be on the free list right now, so move it in `_checkpointed`
        too rather than only clearing the mark — otherwise the promotion would
        not take effect until something released it again.
        """
        if not self.enabled or group not in self._speculative:
            return
        self._speculative.discard(group)
        if group in self._free and self.group_hash[group] != -1:
            self._checkpointed.remove(group)
            self._checkpointed.append(group)

    def release(self, group: int) -> None:
        """Hand a group back, to whichever half its content puts it in.

        A group still carrying a checkpoint goes to the LRU tail, so being
        resumed from refreshes it — `claim` deliberately leaves the hash in
        place, which is what makes reuse count as use. One marked by
        `mark_speculative` goes to the head instead.
        """
        self._free.add(group)
        if self.group_hash[group] != -1:
            if group in self._speculative:
                self._checkpointed.appendleft(group)
            else:
                self._checkpointed.append(group)
            return
        self._speculative.discard(group)
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
            checkpointed = block_hashes[i] in self.hash_to_group
            if not assume_checkpointed and not checkpointed:
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup_group(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
        return self.hash_to_group.get(h, -1)

    # ------------------------- midstep checkpointing ------------------------ #
    #
    # `checkpoint` below keeps state at the position a forward *ended* at, which
    # is the only position a backend can be assumed to expose. A backend that
    # declares `readable_midstep` exposes more: its chunk kernel already
    # materializes the recurrent state at every interior chunk boundary, so a
    # snapshot at any of them is a copy rather than a forward.
    #
    # That splits keeping a checkpoint into two moments that `checkpoint` runs
    # as one:
    #
    #   `reserve_midstep`  before the forward, take the destination group and
    #                      name the position. The bytes do not exist yet.
    #   `publish_midstep`  after it, once the runner has copied them out, file
    #                      the hash so a resuming request can find it.
    #
    # Publishing at reservation time would index a group over bytes the forward
    # had not written, and a request resuming there would read whatever the
    # previous tenant left — the same failure `copy`'s deferral avoids, for the
    # same reason. Reserving at publish time is the other way wrong: the free
    # list has to be committed before the batch is built, or the group the
    # runner is told to write may have been handed to an admission in between.

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        """Take a destination group for each `(position, hash)` this step covers.

        Returns `(group, position, hash)` per reservation actually made — the
        runner needs `group` and `position`, `publish_midstep` needs the hash.
        Best-effort and order-preserving: reservations stop at the first one the
        free list cannot fill, so the earliest position (the one an earlier
        forward reaches first) is the one that survives a shortage.

        Nothing is indexed here. `pop` takes the group off the free list, so an
        admission in the same pass cannot be handed it, and it carries no hash
        until `publish_midstep`, so a resuming request cannot find it either.

        A reservation therefore holds capacity that a checkpoint under `fork` or
        `copy` does not — those hand a group back the moment they take one. It
        is held for exactly one forward, and `cancel_midstep` returns it if that
        forward never runs. `has_free` is still the admission gate, so the worst
        case is an admission deferred a step, never one starved: the same
        best-effort contract `checkpoint` keeps.
        """
        if not self.applies(seq) or not self.readable_midstep:
            return []
        out = []
        for pos, h in positions:
            if not self.has_free():
                self.checkpoints_dropped += 1
                break
            out.append((self.pop(), pos, h))
        return out

    def publish_midstep(self, reservations: list[tuple], seq=None) -> None:
        """File each reserved group under its hash, now that its bytes exist.

        The group goes back on the free list *as a checkpoint* — the same
        capacity-neutral move `checkpoint` makes, covered by the same lazy
        eviction: whoever pops it next invalidates the hash on the way out.

        A position that is not this prompt's own end is marked speculative and
        goes to the LRU head — the ladder rung and the demand guess where the
        next turn will resume, the anchor knows; `mark_speculative` gives what
        the difference measures. `seq` is optional so a caller with no sequence
        in hand still publishes, as the anchor, which is the conservative half
        of the choice.
        """
        anchor = getattr(seq, "checkpoint_end_pos", 0) if seq is not None else 0
        for group, pos, h in reservations:
            if anchor and pos != anchor:
                self.mark_speculative(group)
            self._index(h, group)
            self.release(group)
            self.checkpoints_kept += 1

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Hand back reservations whose forward never ran (preempt, abort).

        Released vacant, not indexed: the bytes were never written, so the group
        holds nothing anybody should be able to find.
        """
        for group, _pos, _h in reservations:
            self.release(group)

    # ---------------------------- checkpointing ---------------------------- #
    def checkpoint(self, seq, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of this boundary, filed under hash `h`.

        The group cannot be shared while its owner still writes it, so the owner
        moves to a fresh group and the old one, never written again, becomes the
        checkpoint. The next forward reads it and fills the replacement, which is
        the whole reason `min_fork_tokens` gates the position.

        `boundary_blocks` names the position, which is all this needs it for: a
        group is a single entry, not a span of them, but whether the position
        is this prompt's own end decides where the checkpoint sits in the LRU
        order — see `mark_speculative`.

        Best-effort: with no free group the seq simply keeps writing its own and
        no checkpoint is taken.
        """
        if not self.applies(seq) or not self.transfer.forks:
            return
        old = seq.per_req_cache_group
        if old < 0:
            return
        anchor = getattr(seq, "checkpoint_end_pos", 0)
        guess = bool(anchor) and boundary_blocks * self.hash_block_size != anchor
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
        if guess:
            self.mark_speculative(old)
        self.pin(old, reader_is_next_batch=True)
        self.checkpoints_kept += 1

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of the checkpoints the ladder asked this pool to keep."""
        return {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.checkpoints_evicted,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }

    def record_relocation(self, src: int, dst: int) -> None:
        """Schedule an Active Slot relocation for the next batch."""
        self._relocations.append((src, dst))

    def occupancy(self) -> dict[str, int]:
        """How much of this pool is live, held by checkpoints, or spare.

        Separate from `checkpoint_fates` because those are cumulative events
        and these are an instantaneous reading.

        `groups_vacant` at 0 is not a shortage. A group with no live owner sits
        on the free list either way; vacant just means it also has no content
        left to lose. A steady state of all-held-none-vacant is this pool
        working, and admission never consults the split — `has_free` counts the
        whole free list.

        The reading that says "too small" is `checkpoints_dropped`: the ladder
        asked to keep a checkpoint and there was no free group to put it in.
        Not `checkpoints_evicted`, which only says more checkpoints were taken
        than a pool this size can hold at once, and which is therefore roughly
        `kept - num_groups` for any workload that outlives the pool.
        """
        held = sum(1 for g in self._free if self.group_hash[g] != -1)
        return {
            "groups_total": self.num_groups,
            "groups_used": self.num_groups - len(self._free),
            "groups_held": held,
            "groups_vacant": len(self._free) - held,
        }

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
