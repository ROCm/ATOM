# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from math import inf

# `StateTransfer.kind` values. Plain strings rather than an enum because the
# choice crosses a process boundary inside a dict (ModelRunner's `block_info`),
# where a scalar survives and a class does not.
FORK = "fork"
COPY = "copy"
NONE = "none"


@dataclass(frozen=True)
class StateTransfer:
    """How a backend hands one request's state over to another slot.

    Three answers, and every checkpoint decision downstream follows from which:

      `none()`    no per-request state, or none that can be handed over at all.
                  Nothing is ever checkpointed and prefix hits shrink to 0.
      `fork(n)`   the state rolls. The owner gives its slot to the index and
                  takes a fresh one, reading the old and writing the new for
                  exactly one forward — which has to leave the new slot
                  self-contained, and that takes `n` committed tokens.
      `copy()`    one request's state is a contiguous byte range another slot
                  can be handed a duplicate of. Nothing is given away, so
                  nothing downstream has to cooperate: no successor forward, and
                  the resuming side is handed a duplicate too.

    The two mechanisms are not interchangeable, and which one a backend can
    offer decides where it may checkpoint. A fork's contract binds the *next*
    forward, so it can only be taken where that forward is known to be long
    enough — true on a prompt, false during generation, where a step commits
    `1 + accepted_drafts` and acceptance is not knowable in advance. That is why
    DeepSeek-V4 copies: it is the only way to checkpoint at a decode boundary.
    See `/app/logs_claude/verify_v4_min_fork.py` for the arithmetic.

    These used to be one integer, `min_fork_tokens`, with 0 spelling `none()` —
    which is exactly the value `copy()` has to report, so the two were
    indistinguishable. Splitting the kind out is what lets a backend say "no
    successor needed" without saying "no state".
    """

    kind: str
    fork_tokens: int = 0
    #: Whether this backend can extract a state snapshot at a position *inside*
    #: a forward, rather than only at the forward's last token.
    #:
    #: A separate axis from `kind`, and it must stay one. `kind` answers how a
    #: slot is handed to *another request*; this answers where within one
    #: forward a snapshot can be taken at all. GDN is `fork(1)` and — once its
    #: chunk kernel's per-chunk intermediates are copied out — midstep-readable;
    #: DeepSeek-V4 is `copy()` and is not, because its compressor ring is not
    #: materialized at interior boundaries the way a chunk kernel's `h` is.
    #: Deriving either from the other would silently turn the prefill chunk cut
    #: off for a backend that still needs it.
    #:
    #: Default False, so every backend keeps today's behavior until it has
    #: actually ported the copy-out.
    readable_midstep: bool = False

    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def fork(cls, tokens: int, readable_midstep: bool = False) -> "StateTransfer":
        assert tokens > 0, "a fork binds its successor forward; use none()"
        return cls(FORK, tokens, readable_midstep)

    @classmethod
    def copy(cls, readable_midstep: bool = False) -> "StateTransfer":
        return cls(COPY, readable_midstep=readable_midstep)

    @classmethod
    def from_config(
        cls, kind: str, fork_tokens: int, readable_midstep: bool = False
    ) -> "StateTransfer":
        """Rebuild from the scalars that crossed the process boundary."""
        if kind == FORK:
            return cls.fork(fork_tokens, readable_midstep)
        assert kind in (COPY, NONE), f"unknown state transfer kind {kind!r}"
        return cls(kind, readable_midstep=readable_midstep)

    @property
    def copies(self) -> bool:
        return self.kind == COPY

    @property
    def forks(self) -> bool:
        return self.kind == FORK

    @property
    def successor_room(self) -> float:
        """`StateCache.successor_room` for a class transferred this way."""
        return inf if self.kind == NONE else float(self.fork_tokens)


@dataclass(frozen=True)
class SlotRetirement:
    """What `retire_top` did, and what the caller still owes.

    `relocated_to` of -1 means the top slot was empty or was itself the thing
    worth spending, so no bytes move. Otherwise the caller must copy
    `retired` → `relocated_to` before the next forward, and then either re-point
    the owning sequence (`held_checkpoint` false) or nothing at all — the index
    has already been re-filed here, because it is this class's to keep
    consistent.
    """

    retired: int
    relocated_to: int
    held_checkpoint: bool


class StateSlotPool:
    """Per-request state slots, plus a content index over the free ones.

    A *slot* is one index of the pre-allocated state tensor: one complete
    recurrent state, across every layer (GDN conv+ssm, the DeepSeek-V4
    compressor ring and sliding window). This pool owns the free list, so it is
    the single answer to "can one more request be admitted".

    Slots are handed out *per need*, not in fixed-width groups. A live request
    under speculative decoding holds `1 + num_spec` of them — one committed
    state plus a rollback slot per speculated token — while a checkpoint holds
    exactly one, because the rollback slots are scratch a resumed prefix has no
    use for. That asymmetry is the point: this pool used to allocate the wide
    shape unconditionally, so at `--num-speculative-tokens 2` every checkpoint
    wasted two thirds of its bytes, and those bytes come out of the same budget
    as the KV cache.

    The slots one request holds need not be adjacent. Nothing downstream reads
    them by arithmetic on a base: the ssm kernel gathers each index out of the
    indices tensor and the conv path is handed column 0 alone, so a request's
    slots reach the kernels as a list and scattered is as good as contiguous.

    A per-request state cannot be rebuilt from cached KV blocks: the cache holds
    the compressor's *output*, the state is its rolling *input* window. So a
    prefix-cache hit is only recoverable up to a boundary where somebody
    checkpointed the state — which is what the index over the free list
    answers.

    Capacity model: a checkpoint is a slot sitting on the free list with its
    content still valid, filed under the content hash of the last block it
    covers. This is the KV block pool's lazy-eviction model (`pop` drops the
    hash at hand-out time, not at free time) applied to state slots. The index
    therefore holds nothing back — `pop` invalidates whatever it hands out, so a
    checkpoint can never shrink admission, and under full concurrency the
    checkpoint set drains to empty on its own.

    *How* a slot reaches the index is the backend's `StateTransfer`, and it is
    the only thing that differs between the two mechanisms this class runs:

      `fork`  the owner gives its slot away and takes a fresh one, so the
              checkpoint costs no bytes but binds the very next forward, which
              has to leave the replacement self-contained (`min_fork_tokens`).
      `copy`  the state is a byte range, so a duplicate goes to the index and
              the owner is not disturbed at all. Nothing is bound: no successor
              forward, and the resuming side copies rather than forking too.

    Both meet the same index and the same free list. Under `copy` the bytes are
    moved by a forward, so this class only schedules the pairs (`take_copies`)
    and the next batch issues them.

    The count of slots is not fixed for life: `extend` and `retire_top` move it
    when the state pool's share of the byte budget changes. Retiring is
    index-forced but its cost is not — see `retire_top`.

    Vocabulary: *checkpoint* is the state sense throughout — a boundary this
    class kept resumable. *Publish* is reserved for a block entering the
    content-addressed KV index (`BlockManager.hash_blocks`, the KV events).

    `enabled` covers the *index* only. The free list stays live either way:
    admission needs a slot whether or not anything is ever checkpointed.
    """

    def __init__(
        self,
        num_slots: int,
        transfer: StateTransfer | None = None,
        hash_block_size: int = 1,
        enabled: bool = True,
    ):
        self.enabled: bool = enabled and num_slots > 0
        self.num_slots: int = num_slots
        self.transfer: StateTransfer = transfer or StateTransfer.none()
        # Committed tokens the forward after a fork must cover for the new slot
        # to come out self-contained. 0 under `copy`, where the destination is
        # complete the moment the copy lands and no forward is involved.
        self.min_fork_tokens: int = self.transfer.fork_tokens
        self.successor_room: float = self.transfer.successor_room
        # Whether a checkpoint may sit anywhere inside a forward rather than
        # only at its last token. False is the conservative answer and the
        # reason `BlockManager` still cuts prefill chunks onto rungs; see
        # `StateTransfer.readable_midstep`.
        self.readable_midstep: bool = self.transfer.readable_midstep
        self.hash_block_size: int = hash_block_size
        # The free list, split by whether the slot still carries content worth
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
        # cannot express that: a checkpoint released before a never-used slot
        # sits ahead of it and is spent first.
        #
        # Which half a free slot belongs to is a function of `slot_hash`, not
        # a third piece of state, so `_set_hash` is the one place that moves a
        # slot across and nobody else has to remember to. `_free` is membership
        # for both; `_vacant`'s heap keeps entries a `claim` took out and
        # `_pop_vacant` drops them when they surface.
        self._free: set[int] = set(range(num_slots))
        self._vacant: list[int] = list(range(num_slots))
        self._checkpointed: deque[int] = deque()
        # Slots whose checkpoint is a guess rather than this prompt's own end;
        # `mark_speculative` explains what the distinction is worth. Cleared
        # when the guess pays off (`promote`) or the content goes (`release` of
        # a slot carrying nothing) — both of which end the slot's claim to be
        # spent first.
        self._speculative: set[int] = set()
        # hash -> slot holding the state as of that block boundary.
        self.hash_to_slot: dict[int, int] = {}
        # Reverse map, for lazy eviction when a slot is handed out. -1 = the
        # slot carries no checkpoint.
        self.slot_hash: list[int] = [-1] * num_slots
        # Slots serving as a fork source in the in-flight step, counted by how
        # many requests fork off each. They stay off the free list until
        # `release_pins`, so the step that reads them cannot race a request that
        # was handed the same slot. The count matters because a checkpoint is
        # read-only and several requests may share one: the slot goes back only
        # once, and no reader may take it over while another still reads it.
        self._pinned: dict[int, int] = {}
        # Of those, the ones whose reader is the batch the *next* pass builds
        # rather than the one just built. `checkpoint` runs in postprocess,
        # after its batch went out, so the forward that reads the source it
        # hands over is one pass further off than a resume's. Carried as a set
        # rather than a second count because the depth is only ever one pass
        # more — see `release_pins`.
        self._deferred: set[int] = set()
        # `copy` only. Seqs whose last forward left their state on a boundary
        # worth keeping. `take_copies` turns each into a copy pair when the next
        # batch is built, which is the latest moment the owner is still known to
        # hold the slot being duplicated.
        self._checkpoint_pending: list = []
        # (src, dst) slot pairs the next batch must copy before its forward.
        # Both halves of the protocol feed this under `copy`: keeping a
        # checkpoint copies the owner's state out, resuming from one copies it
        # back in.
        self._copies: list[tuple[int, int]] = []
        # `dropped` had no slot to go to; `evicted` landed and was later
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
    def has_free(self, count: int = 1) -> bool:
        """Whether `count` slots can be handed out.

        A checkpoint on the free list counts: `pop` spends it rather than
        refusing, so it holds no capacity back. This is the admission gate, and
        the caller asks for what it will actually take — `1 + num_spec` for a
        request that will speculate, 1 for one that will not.
        """
        return len(self._free) >= count

    def num_free(self) -> int:
        return len(self._free)

    def is_free(self, slot: int) -> bool:
        return slot in self._free

    def holds_checkpoint(self, slot: int) -> bool:
        """Whether `slot` is on the free list *and* still worth something."""
        return slot in self._free and self.slot_hash[slot] != -1

    def pop(self) -> int:
        """Hand out a slot, evicting its checkpoint if it carried one.

        Vacant first, lowest index first; only when nothing is vacant does a
        checkpoint get spent, and then it is the least recently used one.

        The state twin of `BlockManager._pop_free_block`: slots sit on the free
        list carrying whatever the last owner left in them, and re-allocation —
        not the free — is the eviction event.

        Lowest-index-first is not a fairness choice, it is what keeps the top of
        the pool cold: a slot high in the range is only reached at a
        high-water-mark of concurrency, so after the peak passes the top holds
        the least recently touched things — exactly what shrinking should spend.

        One slot. A caller needing a request's whole set asks `pop_many`.
        """
        slot = self._pop_vacant()
        if slot < 0:
            slot = self._checkpointed.popleft()
            self._free.discard(slot)
            self.checkpoints_evicted += 1
        # Whatever guess this slot carried is spent with it. Not folded into
        # `invalidate`, which also runs on the re-file inside `_index` and would
        # clear a mark `_commit_pending` had only just made.
        self._speculative.discard(slot)
        self.invalidate(slot)
        return slot

    def pop_many(self, count: int) -> list[int]:
        """Hand out `count` slots as one request's set. Caller checked `has_free`.

        Returned in allocation order, and the caller keeps that order: element 0
        is the committed state every path reads and writes, and elements 1..n
        are speculation rollback, which only the spec-decode path touches. The
        set is not adjacent and nothing may assume it is.

        Not atomic, because it does not need to be: `has_free(count)` was
        checked against a free list only this thread mutates, and `pop` never
        refuses — with nothing vacant it spends the LRU checkpoint. The worst
        case is a wide request evicting checkpoints for slots it then keeps,
        which is what admitting it means.
        """
        return [self.pop() for _ in range(count)]

    def _pop_vacant(self) -> int:
        """Lowest vacant index, or -1, dropping entries that are stale.

        An entry is stale when the slot has since been claimed or has taken a
        hash, both of which leave the heap untouched — cheaper than an exact
        removal, and the two conditions below are the same ones that decide
        which half a slot belongs to anyway.
        """
        while self._vacant:
            slot = heappop(self._vacant)
            if slot in self._free and self.slot_hash[slot] == -1:
                self._free.discard(slot)
                return slot
        return -1

    def claim(self, slot: int) -> None:
        """Take one specific free slot off the list, content and all.

        Linear in `_checkpointed` when the slot holds a checkpoint, which is
        the case the resume path takes. That is deliberate: the free list stays
        the single source of truth for "how many slots are free", which
        admission and every caller of `has_free` rely on. The scan is bounded by
        max_num_seqs and runs once per resuming request, against a
        `can_allocate` that already hashed every block of the prompt.
        """
        self._free.discard(slot)
        if self.slot_hash[slot] != -1:
            self._checkpointed.remove(slot)

    def mark_speculative(self, slot: int) -> None:
        """File `slot` at the LRU *head* when it is released, not the tail.

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

        A mark, not an argument to `release`, because the slot that carries a
        checkpoint under `fork` is *pinned* rather than released — it goes back
        through `release_pins` two passes later, which knows only the slot
        index. Marking is the one mechanism both paths can reach.
        """
        if self.enabled:
            self._speculative.add(slot)

    def promote(self, slot: int) -> None:
        """A guess paid off: stop spending this one first.

        Called from the resume path when a request actually reads `slot`. The
        slot may be on the free list right now, so move it in `_checkpointed`
        too rather than only clearing the mark — otherwise the promotion would
        not take effect until something released it again.
        """
        if not self.enabled or slot not in self._speculative:
            return
        self._speculative.discard(slot)
        if slot in self._free and self.slot_hash[slot] != -1:
            self._checkpointed.remove(slot)
            self._checkpointed.append(slot)

    def release(self, slot: int) -> None:
        """Hand a slot back, to whichever half its content puts it in.

        A slot still carrying a checkpoint goes to the LRU tail, so being
        resumed from refreshes it — `claim` deliberately leaves the hash in
        place, which is what makes reuse count as use. One marked by
        `mark_speculative` goes to the head instead.
        """
        self._free.add(slot)
        if self.slot_hash[slot] != -1:
            if slot in self._speculative:
                self._checkpointed.appendleft(slot)
            else:
                self._checkpointed.append(slot)
            return
        self._speculative.discard(slot)
        heappush(self._vacant, slot)
        # Every slot that took a hash while vacant left an entry behind, so on
        # a long-lived server the stale ones outnumber the live ones without
        # this. Rebuilding costs one pass and buys at least `num_slots` pushes.
        if len(self._vacant) > 2 * self.num_slots + 2:
            self._vacant = [g for g in self._free if self.slot_hash[g] == -1]
            heapify(self._vacant)

    def release_many(self, slots) -> None:
        """Hand back a whole set. Tolerates -1 and an empty list.

        The -1 tolerance is what lets a caller release a request's slots without
        first asking whether it ever got any: a seq that was never admitted, or
        one whose set was already taken over by an adopt, carries the sentinel.
        """
        for slot in slots or ():
            if slot >= 0:
                self.release(slot)

    def _set_hash(self, slot: int, h: int) -> None:
        """Change what an existing slot backs, re-filing it if it is free.

        Going through here is what lets the two halves of the free list be a
        function of `slot_hash` rather than a third thing to keep in step.
        `h` of -1 means the slot backs nothing.
        """
        if slot not in self._free:
            self.slot_hash[slot] = h
            return
        self.claim(slot)
        self.slot_hash[slot] = h
        self.release(slot)

    # ----------------------------- resizing -------------------------------- #
    def extend(self, count: int) -> None:
        """Add `count` slot indices at the top. The caller freed the bytes.

        `slot_hash` is grown but never shrunk, so an index that was retired and
        then handed back out reuses its slot rather than appending a second one.
        """
        for slot in range(self.num_slots, self.num_slots + count):
            if slot < len(self.slot_hash):
                self.slot_hash[slot] = -1
            else:
                self.slot_hash.append(-1)
            self.release(slot)
        self.num_slots += count

    def retire_top(self) -> "SlotRetirement | None":
        """Give up the highest slot index, relocating whatever sits on it.

        Shrinking is index-forced — the bytes being handed back are the ones the
        top slot occupies — but the policy is not: what gets *spent* is the
        least recently used checkpoint, whichever index that is. So the top
        slot's content moves to a target taken in the same order `pop` uses
        (vacant first, LRU checkpoint only if nothing is vacant) and the caller
        issues one copy.

        Without that move, shrinking would be anti-LRU: a slot's index records
        the concurrency high-water mark when it was handed out and is never
        refreshed by use, so a checkpoint resumed from every second could sit at
        the top and be spent ahead of one nothing has touched in minutes.

        Returns `None` when the top slot cannot be given up this pass — it is
        pinned as a fork source, or it is live and there is nowhere to move it.
        Both clear on their own, so the caller retries rather than blocks.
        """
        top = self.num_slots - 1
        if top < 0 or self.is_pinned(top):
            return None
        held = self.holds_checkpoint(top)
        if self.is_free(top) and not held:
            self.claim(top)
            self.num_slots -= 1
            return SlotRetirement(top, -1, False)

        dst = self._take_relocation_target(exclude=top)
        if dst < 0 and not held:
            return None
        if dst < 0:
            # Nothing free anywhere, so the top slot is by elimination the only
            # checkpoint left — spending it is what LRU asks for regardless.
            self.claim(top)
            self.invalidate(top)
        elif held:
            self._rehome_checkpoint(top, dst)
        self.num_slots -= 1
        return SlotRetirement(top, dst, held)

    def _take_relocation_target(self, exclude: int) -> int:
        """A slot to move content into, in `pop`'s order but skipping `exclude`."""
        slot = self._pop_vacant()
        if slot >= 0:
            return slot
        for slot in self._checkpointed:
            if slot != exclude:
                self.claim(slot)
                self.invalidate(slot)
                return slot
        return -1

    def _rehome_checkpoint(self, src: int, dst: int) -> None:
        """Give a checkpoint a different slot index, LRU position included.

        Written against the containers rather than through `claim`/`release`
        because keeping the position is the whole point: released slots go to
        the tail, and a checkpoint that only changed address has not been used.
        """
        h = self.slot_hash[src]
        self._checkpointed[self._checkpointed.index(src)] = dst
        self._free.discard(src)
        self._free.add(dst)
        self.slot_hash[src] = -1
        self.slot_hash[dst] = h
        self.hash_to_slot[h] = dst

    # ---------------------------- applicability ---------------------------- #
    def applies(self, seq) -> bool:
        """Whether this class gates or checkpoints anything for `seq`.

        Per-request state is declared by the attention type, so a seq on a model
        without one carries no slot and this class has no say over its hits.
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
        slot self-contained, scanning right-to-left so the hit is cut as little
        as possible. 0 is always valid — a request starting from scratch needs
        no prior state.

        The fork test is what `min_fork_tokens` buys: resuming reads the
        checkpoint and writes a fresh slot, and that forward has to leave the
        fresh slot whole. A boundary too close to the end of the prompt fails
        it and the scan keeps walking back. Under `copy` the resumer is handed
        the bytes instead of reading across two slots, so `min_fork_tokens` is
        0 and the test is vacuous — one expression covers both.

        Without this a hit hands the resumed forward a slot freshly popped off
        the free list and it reads the previous occupant's state.

        `assume_checkpointed` drops the index lookup and keeps the fork test,
        which is the counterfactual the protocol describes: this class's ladder
        made dense, leaving only what a checkpoint could not have fixed.
        """
        if not self.applies(seq):
            return hit
        hbs = self.hash_block_size
        for i in range(hit - 1, -1, -1):
            if not assume_checkpointed and block_hashes[i] not in self.hash_to_slot:
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup(self, h: int) -> int:
        """Slot holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
        return self.hash_to_slot.get(h, -1)

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
    #   `reserve_midstep`  before the forward, take the destination slot and
    #                      name the position. The bytes do not exist yet.
    #   `publish_midstep`  after it, once the runner has copied them out, file
    #                      the hash so a resuming request can find it.
    #
    # Publishing at reservation time would index a slot over bytes the forward
    # had not written, and a request resuming there would read whatever the
    # previous tenant left — the same failure `copy`'s deferral avoids, for the
    # same reason. Reserving at publish time is the other way wrong: the free
    # list has to be committed before the batch is built, or the slot the
    # runner is told to write may have been handed to an admission in between.

    def reserve_midstep(self, seq, positions: list[tuple[int, int]]) -> list[tuple]:
        """Take a destination slot for each `(position, hash)` this step covers.

        Returns `(slot, position, hash)` per reservation actually made — the
        runner needs `slot` and `position`, `publish_midstep` needs the hash.
        Best-effort and order-preserving: reservations stop at the first one the
        free list cannot fill, so the earliest position (the one an earlier
        forward reaches first) is the one that survives a shortage.

        Nothing is indexed here. `pop` takes the slot off the free list, so an
        admission in the same pass cannot be handed it, and it carries no hash
        until `publish_midstep`, so a resuming request cannot find it either.

        A reservation therefore holds capacity that a checkpoint under `fork` or
        `copy` does not — those hand a slot back the moment they take one. It
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
        """File each reserved slot under its hash, now that its bytes exist.

        The slot goes back on the free list *as a checkpoint* — the same
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
        for slot, pos, h in reservations:
            if anchor and pos != anchor:
                self.mark_speculative(slot)
            self._index(h, slot)
            self.release(slot)
            self.checkpoints_kept += 1

    def cancel_midstep(self, reservations: list[tuple]) -> None:
        """Hand back reservations whose forward never ran (preempt, abort).

        Released vacant, not indexed: the bytes were never written, so the slot
        holds nothing anybody should be able to find.
        """
        for slot, _pos, _h in reservations:
            self.release(slot)

    # ---------------------------- checkpointing ---------------------------- #
    def checkpoint(self, seq, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of this boundary, filed under hash `h`.

        Two mechanisms, chosen by the backend's `StateTransfer`:

        `fork` — the slot cannot be shared while its owner still writes it, so
        the owner moves to a fresh slot and the old one, never written again,
        becomes the checkpoint. The next forward reads it and fills the
        replacement, which is the whole reason `min_fork_tokens` gates the
        position. Only the committed slot moves: the owner keeps its rollback
        scratch, and the checkpoint is one slot wide because a resuming prefix
        has no speculation to roll back.

        `copy` — the owner keeps writing where it is and a duplicate of its
        state goes to the index instead. Only the intent is recorded here; the
        destination slot and the copy pair come from `take_copies` when the next
        batch is built. Deferred because the bytes have to be moved by a forward, and
        a checkpoint indexed before its bytes exist would hand a resuming
        request whatever the destination happened to hold.

        `boundary_blocks` names the position, which is all this needs it for: a
        slot is a single entry, not a span of them, but whether the position
        is this prompt's own end decides where the checkpoint sits in the LRU
        order — see `mark_speculative`.

        Best-effort under both: with no free slot the seq simply keeps writing
        its own and no checkpoint is taken.
        """
        if not self.applies(seq):
            return
        old = seq.state_slot
        if old < 0:
            return
        anchor = getattr(seq, "checkpoint_end_pos", 0)
        guess = bool(anchor) and boundary_blocks * self.hash_block_size != anchor
        if self.transfer.copies:
            if seq.pending_checkpoint == -1:
                self._checkpoint_pending.append(seq)
            # A later boundary supersedes an earlier one: the slot holds the
            # state as of the last forward, so only the last position is true.
            seq.pending_checkpoint = h
            seq.pending_checkpoint_is_guess = guess
            return
        if not self.has_free():
            self.checkpoints_dropped += 1
            return
        seq.state_slot = self.pop()
        seq.state_fork_src = old
        # Not released: `state_fork_src` names `old` as what this request's
        # NEXT forward reads, and that forward is two passes off — one to build
        # the batch carrying the fork, one to issue it. Handing it back now put
        # it on the free list during the pass that admits the requests which
        # could pop it, and then one kernel would read and write it at once.
        # `_index` files it either way; `_set_hash` leaves a non-free slot be.
        self._index(h, old)
        if guess:
            self.mark_speculative(old)
        self.pin(old, reader_is_next_batch=True)
        self.checkpoints_kept += 1

    def _commit_pending(self) -> None:
        """Turn the last step's checkpoint intents into copy pairs.

        Each pending seq gets a destination slot, which goes straight back on
        the free list and into the index — the same capacity-neutral move
        `checkpoint` makes under `fork`, covered by the same lazy eviction:
        whoever pops the slot next invalidates the hash on the way out.

        A seq preempted or finished in between carries no slot any more and is
        skipped, so nothing is ever indexed over state that is gone. That check
        is only sound because this runs with the batch already decided — see
        `take_copies`.
        """
        if not self._checkpoint_pending:
            return
        for seq in self._checkpoint_pending:
            h, seq.pending_checkpoint = seq.pending_checkpoint, -1
            guess, seq.pending_checkpoint_is_guess = (
                seq.pending_checkpoint_is_guess,
                False,
            )
            src = seq.state_slot
            if h == -1 or src < 0:
                continue
            if not self.has_free():
                self.checkpoints_dropped += 1
                continue
            dst = self.pop()
            self.release(dst)
            # Before `_index`, which re-files the slot through `_set_hash` and
            # is therefore where the LRU position is actually decided.
            if guess:
                self.mark_speculative(dst)
            self._index(h, dst)
            self._copies.append((src, dst))
            self.checkpoints_kept += 1
        self._checkpoint_pending.clear()

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of the checkpoints the ladder asked this pool to keep.

        `checkpoints_dropped` reads 0 at every pool size, including one far too
        small, and is not the capacity signal it looks like: `_commit_pending`
        only counts a drop when `has_free()` is false, but a finished request
        hands its slot back, and `pop` never refuses — with nothing vacant it
        spends the LRU checkpoint and counts an eviction instead. This pool
        overwrites checkpoints, it does not turn them away.

        The capacity reading is `checkpoints_evicted` against
        `checkpoints_kept - num_slots`. When they match, every checkpoint but
        the resident set was destroyed to make room and more slots will buy
        hit rate directly. When eviction runs below that line, the pool is
        holding its working set and the misses are somewhere else — most
        likely `checkpoints_orphaned`, whose KV prefix died first and which
        more slots cannot prevent.
        """
        return {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": self.checkpoints_evicted,
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }

    def occupancy(self) -> dict[str, int]:
        """How much of this pool is live, held by checkpoints, or spare.

        Separate from `checkpoint_fates` because those are cumulative events
        and these are an instantaneous reading.

        `slots_vacant` at 0 is not a shortage. A slot with no live owner sits
        on the free list either way; vacant just means it also has no content
        left to lose. A steady state of all-held-none-vacant is this pool
        working, and admission never consults the split — `has_free` counts the
        whole free list.

        Nor is any field here the reading that says "too small" — see
        `checkpoint_fates`, which owns that question. This docstring used to
        point at `checkpoints_dropped`, which is wrong: that counter is
        structurally near-unreachable, because `pop` evicts rather than
        refusing. `kept - num_slots == evicted` is the thrash signal.
        """
        held = sum(1 for g in self._free if self.slot_hash[g] != -1)
        return {
            "slots_total": self.num_slots,
            "slots_used": self.num_slots - len(self._free),
            "slots_held": held,
            "slots_vacant": len(self._free) - held,
        }

    def record_copy(self, src: int, dst: int) -> None:
        """Schedule a state copy for the next batch's forward to issue."""
        self._copies.append((src, dst))

    def take_copies(self) -> list[tuple[int, int]]:
        """Every copy the batch now being built must issue before its forward.

        Called at the moment the batch is constructed, which is the whole point:
        a checkpoint's source is the owner's live slot, and it has to still be
        that owner's when the copy runs. Committing earlier in the pass would
        leave a window — an admission preempting that owner would return the
        slot to the free list, and the copy would then duplicate whatever the
        next request wrote there into a slot already indexed as a checkpoint.
        Nothing runs between here and the batch, so the window is empty.

        A checkpoint therefore becomes visible one pass later than the step that
        formed it, and this pass's own admissions get first claim on the free
        list. Both are the right way round: admission is throughput, a
        checkpoint is speculative reuse.

        The two kinds of pair cannot collide, so their order does not matter. A
        resume source is a claimed or pinned checkpoint and a keeper source is a
        live slot; neither is on the free list, so `_commit_pending`'s `pop`
        can return neither.
        """
        self._commit_pending()
        copies, self._copies = self._copies, []
        return copies

    def forget_pending(self, seq) -> None:
        """Drop `seq`'s uncommitted checkpoint — its slot is being released.

        The seq stays in `_checkpoint_pending` until the next commit, which
        skips it on the cleared hash. Cheaper than removing it, and the list is
        emptied every pass either way.
        """
        seq.pending_checkpoint = -1

    def _index(self, h: int, slot: int) -> None:
        """File `slot` as the checkpoint for hash `h`.

        Caller guarantees `slot` holds the state as of that boundary and has
        been handed back to the free list — a checkpointed slot is never
        written again, so the request that produced it must have moved on.
        """
        if not self.enabled:
            return
        self.invalidate(slot)
        prev = self.hash_to_slot.get(h, -1)
        self.hash_to_slot[h] = slot
        self._set_hash(slot, h)
        # Re-filing a hash onto a different slot orphans the old one. Drop its
        # back-pointer directly rather than through `invalidate`, which would
        # take the entry just written with it.
        if prev != -1 and prev != slot:
            self._set_hash(prev, -1)

    def unindex(self, h: int) -> int:
        """Drop the checkpoint filed under `h`. The dual of `_index`.

        Called when the KV block of that hash leaves the block index. The two
        pools are addressed by the same chained content hash and a prefix hit
        is a joint claim on both, so a checkpoint whose block is gone can never
        be reached again: `_gated_hit` caps at the last block still indexed.
        Until it is dropped it holds a slot and sits in the LRU queue ahead of
        checkpoints that are still worth something, so the pool spends a live
        one to make room for a dead one.

        Necessary but not sufficient for reachability — an *earlier* block in
        the chain may be the one that went — so this reclaims a subset. Making
        it exact would have the state pool watch every block of every prefix,
        which costs more than the tail it would catch.

        Returns the slot freed, or -1.
        """
        slot = self.hash_to_slot.get(h, -1)
        if slot < 0:
            return -1
        self.invalidate(slot)
        self.checkpoints_orphaned += 1
        return slot

    def invalidate(self, slot: int) -> None:
        """Drop `slot`'s checkpoint. Called when the slot is handed out."""
        if not self.enabled:
            return
        h = self.slot_hash[slot]
        if h == -1:
            return
        self._set_hash(slot, -1)
        if self.hash_to_slot.get(h) == slot:
            del self.hash_to_slot[h]

    # -------------------------------- pins --------------------------------- #
    def pin(self, slot: int, *, reader_is_next_batch: bool = False) -> None:
        """Hold `slot` off the free list until the forward that reads it ran.

        The caller states which batch that is, not when to let go: a resume
        pins while its own batch is being built, `checkpoint` pins after its
        batch went out and the forward that reads what it handed over is the
        one the next pass builds. `release_pins` turns the two into passes.
        """
        self._pinned[slot] = self._pinned.get(slot, 0) + 1
        if reader_is_next_batch:
            self._deferred.add(slot)

    def is_pinned(self, slot: int) -> bool:
        return slot in self._pinned

    def pin_count(self, slot: int) -> int:
        """Requests forking off `slot` in the in-flight step."""
        return self._pinned.get(slot, 0)

    def unpin(self, slot: int) -> None:
        """Drop one reference without freeing — the caller took the slot over.

        Only legal on the last reference; a slot another request still reads
        cannot be taken over. `BlockManager.cancel_state_fork` enforces that.
        """
        count = self._pinned.get(slot, 0)
        if count <= 1:
            self._pinned.pop(slot, None)
            self._deferred.discard(slot)
        else:
            self._pinned[slot] = count - 1

    def drop_reader(self, slot: int) -> None:
        """A reader of `slot` is gone before it read. Free it if it was last.

        A pin says some forward still has to read this slot. When the request
        that owed that forward is deallocated the obligation goes with it, and
        holding the slot to the clock would cost admission a slot for
        nothing — which is what keeps `checkpoint` capacity-neutral for a
        request that finishes or is preempted on the boundary it published.
        `release_pins` stays the backstop, for readers still to come.
        """
        if slot < 0 or not self.is_pinned(slot):
            return
        self.unpin(slot)
        if not self.is_pinned(slot):
            self.release(slot)

    def release_pins(self) -> None:
        """Return the sources whose reading forward has been issued.

        They go back to the free list, once each however many requests read
        them. Safe once that forward is out: a request handed one of these
        slots next pass runs its own forward after it, on the same stream.

        A pin taken while a batch was being built is read by that batch, so it
        clears here, one pass later. A pin marked `reader_is_next_batch` is
        read by the batch this pass is about to build, so it survives one more
        — and `checkpoint`'s source has to, because it is on nobody's free
        list to protect it during the very pass that admits the requests which
        could otherwise be handed it.

        Every pin clears within two passes whatever else happens. Nothing here
        waits on the obligation being consumed, so a request preempted between
        taking a checkpoint and its next batch cannot strand a slot — which
        is why this stays a clock rather than moving to `_consume_state_forks`,
        where it would be tighter and leak.
        """
        if not self._pinned:
            return
        held, self._deferred = self._deferred, set()
        for slot in sorted(self._pinned):
            if slot in held:
                continue
            self.release(slot)
            del self._pinned[slot]
