# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from math import inf

from atom.model_engine.page_unit_checkpoint import (
    CheckpointRestoreOp,
    CheckpointStoreOp,
    PageUnitCheckpointStore,
)

# `StateTransfer.kind` values. Plain strings keep the explicit runner RPC wire
# representation language-neutral and easy to validate.
FORK = "fork"
COPY = "copy"
NONE = "none"


@dataclass(frozen=True)
class StateTransfer:
    """How a backend hands one request's state over to another group.

    Three answers, and every checkpoint decision downstream follows from which:

      `none()`    no per-request state, or none that can be handed over at all.
                  Nothing is ever checkpointed and prefix hits shrink to 0.
      `fork(n)`   the state rolls. The owner gives its group to the index and
                  takes a fresh one, reading the old and writing the new for
                  exactly one forward — which has to leave the new group
                  self-contained, and that takes `n` committed tokens.
      `copy(id)`  one request's state has the versioned canonical byte layout
                  `id`, which can be scattered into PAGE units and gathered
                  into another Active Slot. Nothing is given away and no
                  successor forward is bound.

    The two mechanisms are not interchangeable, and which one a backend can
    offer decides where it may checkpoint. A copy's layout id is part of that
    capability rather than a second backend hook: the control plane, runner and
    copy kernel must all agree on the same canonical byte stream. A fork's
    contract binds the *next* forward, so it can only be taken where that
    forward is known to be long enough — true on a prompt, false during
    generation, where a step commits `1 + accepted_drafts` and acceptance is
    not knowable in advance. That is why DeepSeek-V4 copies: it is the only way
    to checkpoint at a decode boundary.

    GDN recurrent state uses `fork(1)`: its kernels rewrite the destination
    state whole, so one committed token leaves the new group self-contained.
    DeepSeek-V4 uses `copy(layout_id)`: its arena and slot planes form one
    canonical raw-byte stream, and a copy does not bind an unknowable successor
    decode length. Those backend declarations are capability summaries; this
    object is the complete scheduling and wire contract.

    These used to be one integer, `min_fork_tokens`, with 0 spelling `none()` —
    which is exactly the value `copy(id)` has to report, so the two were
    indistinguishable. Splitting the kind out is what lets a backend say "no
    successor needed" without saying "no state".
    """

    kind: str
    fork_tokens: int = 0
    paged_layout_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fork_tokens, int) or isinstance(self.fork_tokens, bool):
            raise TypeError("fork_tokens must be an integer")
        if self.kind == COPY:
            if self.fork_tokens != 0:
                raise ValueError("copy transfer cannot bind successor tokens")
            if not isinstance(self.paged_layout_id, str) or not self.paged_layout_id:
                raise ValueError("copy transfer requires a non-empty PAGE layout id")
            return
        if self.kind == FORK:
            if self.fork_tokens <= 0:
                raise ValueError("fork transfer requires positive fork_tokens")
            if self.paged_layout_id is not None:
                raise ValueError("fork transfer cannot declare a PAGE layout")
            return
        if self.kind == NONE:
            if self.fork_tokens != 0 or self.paged_layout_id is not None:
                raise ValueError("none transfer cannot carry tokens or a PAGE layout")
            return
        raise ValueError(f"unknown state transfer kind {self.kind!r}")

    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def fork(cls, tokens: int) -> "StateTransfer":
        return cls(FORK, tokens)

    @classmethod
    def copy(cls, layout_id: str) -> "StateTransfer":
        return cls(COPY, paged_layout_id=layout_id)

    def to_wire(self) -> dict[str, str | int | None]:
        """Return the validated capability payload used by runner RPC."""
        return {
            "kind": self.kind,
            "fork_tokens": self.fork_tokens,
            "paged_layout_id": self.paged_layout_id,
        }

    @classmethod
    def from_wire(cls, wire: object) -> "StateTransfer":
        """Rebuild and validate a capability received from another process."""
        if not isinstance(wire, Mapping):
            raise TypeError("state transfer capability must be a mapping")
        expected = {"kind", "fork_tokens", "paged_layout_id"}
        if set(wire) != expected:
            raise ValueError(
                "invalid state transfer capability fields: "
                f"expected={sorted(expected)}, got={sorted(wire)}"
            )
        return cls(
            kind=wire["kind"],  # type: ignore[arg-type]
            fork_tokens=wire["fork_tokens"],  # type: ignore[arg-type]
            paged_layout_id=wire["paged_layout_id"],  # type: ignore[arg-type]
        )

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
class StateMaintenanceOps:
    """All state movement that must run before one model batch.

    ``relocations`` move one contiguous Active Slot to another.  Checkpoint
    stores scatter an Active Slot into arbitrary PAGE units, while checkpoint
    restores gather those units into an Active Slot.  Keeping the three
    directions in one immutable bundle gives scheduling one drain point: an
    operation can neither ride a different batch nor be accidentally drained
    twice by independent consumers.
    """

    relocations: tuple[tuple[int, int], ...] = ()
    checkpoint_stores: tuple[CheckpointStoreOp, ...] = ()
    checkpoint_restores: tuple[CheckpointRestoreOp, ...] = ()

    @property
    def empty(self) -> bool:
        return not (
            self.relocations or self.checkpoint_stores or self.checkpoint_restores
        )


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
    """Per-request state groups, plus a content index over the free ones.

    A *group* is what one request occupies in the pre-allocated state tensor:
    `entries // entries_per_req` contiguous indices (GDN conv+ssm, the
    DeepSeek-V4 compressor ring and sliding window). This pool owns the free
    list, so it is the single answer to "can one more request be admitted".

    A per-request state cannot be rebuilt from cached KV blocks: the cache holds
    the compressor's *output*, the state is its rolling *input* window. So a
    prefix-cache hit is only recoverable up to a boundary where somebody
    checkpointed the state — which is what the index over the free list
    answers.

    A fork-transfer checkpoint is a group sitting on the free list with valid
    content. A copy-transfer checkpoint never uses that representation: groups
    are Active Slots only, immutable images live in ordered sets of arbitrary
    PAGE units, and this class schedules typed scatter/gather ops between them.

    *How* a group reaches the index is the backend's `StateTransfer`, and it is
    the only thing that differs between the two mechanisms this class runs:

      `fork`  the owner gives its group away and takes a fresh one, so the
              checkpoint costs no bytes but binds the very next forward, which
              has to leave the replacement self-contained (`min_fork_tokens`).
      `copy`  the state has a canonical byte stream, so it is scattered into
              PAGE units without disturbing the owner and gathered into the
              resumer's own Active Slot. No successor forward is bound.

    The count of groups is not fixed for life: `extend` and `retire_top` move it
    when the state pool's share of the byte budget changes. Retiring is
    index-forced but its cost is not — see `retire_top`.

    Vocabulary: *checkpoint* is the state sense throughout — a boundary this
    class kept resumable. *Publish* is reserved for a block entering the
    content-addressed KV index (`BlockManager.hash_blocks`, the KV events).

    `enabled` covers the *index* only. The free list stays live either way:
    admission needs a group whether or not anything is ever checkpointed.
    """

    def __init__(
        self,
        num_groups: int,
        transfer: StateTransfer | None = None,
        hash_block_size: int = 1,
        enabled: bool = True,
        page_checkpoints: PageUnitCheckpointStore | None = None,
    ):
        self.enabled: bool = enabled and num_groups > 0
        self.num_groups: int = num_groups
        self.transfer: StateTransfer = transfer or StateTransfer.none()
        # Committed tokens the forward after a fork must cover for the new group
        # to come out self-contained. 0 under `copy`, where the destination is
        # complete the moment the copy lands and no forward is involved.
        self.min_fork_tokens: int = self.transfer.fork_tokens
        self.successor_room: float = self.transfer.successor_room
        self.hash_block_size: int = hash_block_size
        self.page_checkpoints = page_checkpoints
        if self.transfer.copies and self.page_checkpoints is None:
            raise ValueError("copy transfer requires PAGE-unit state checkpoints")
        if not self.transfer.copies and self.page_checkpoints is not None:
            raise ValueError("PAGE-unit state checkpoints require copy transfer")
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
        # (src, dst) Active Slot relocation pairs. Checkpoints never feed this:
        # copy-transfer checkpoints use the typed PAGE ops below, while fork
        # checkpoints move ownership without copying.
        self._copies: list[tuple[int, int]] = []
        # Copy-transfer state separates the Active Slot free list above from
        # the immutable checkpoint image. The pending dict coalesces later
        # boundaries from the same request while retaining boundary metadata.
        self._paged_pending: dict[int, tuple[object, int, int]] = {}
        self._paged_store_ops: list[CheckpointStoreOp] = []
        self._paged_restore_ops: list[CheckpointRestoreOp] = []
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
        it and the scan keeps walking back. Under `copy` the resumer is handed
        the bytes instead of reading across two groups, so `min_fork_tokens` is
        0 and the test is vacuous — one expression covers both.

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
            checkpointed = (
                self.page_checkpoints.contains(block_hashes[i])
                if self.page_checkpoints is not None
                else block_hashes[i] in self.hash_to_group
            )
            if not assume_checkpointed and not checkpointed:
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
        if self.page_checkpoints is not None:
            return self.page_checkpoints.lookup(h)
        return self.hash_to_group.get(h, -1)

    # ---------------------------- checkpointing ---------------------------- #
    def checkpoint(self, seq, boundary_blocks: int, h: int) -> None:
        """Keep `seq`'s state as of this boundary, filed under hash `h`.

        Two mechanisms, chosen by the backend's `StateTransfer`:

        `fork` — the group cannot be shared while its owner still writes it, so
        the owner moves to a fresh group and the old one, never written again,
        becomes the checkpoint. The next forward reads it and fills the
        replacement, which is the whole reason `min_fork_tokens` gates the
        position.

        `copy` — the owner keeps writing where it is and an immutable image is
        scattered into PAGE units. Only the intent is recorded here; units are
        atomically reserved when the next batch is built. The checkpoint stays
        invisible until that batch has issued the scatter.

        `boundary_blocks` is carried by PAGE checkpoint records. Fork-backed
        groups are single entries and do not otherwise need it.
        """
        if not self.applies(seq):
            return
        old = seq.per_req_cache_group
        if old < 0:
            return
        if self.transfer.copies:
            self._paged_pending[id(seq)] = (seq, boundary_blocks, h)
            seq.pending_checkpoint = h
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

    def _commit_paged_pending(self) -> None:
        """Reserve arbitrary PAGE units and schedule Active Slot scatter ops."""
        if not self._paged_pending:
            return
        pending, self._paged_pending = self._paged_pending, {}
        store = self.page_checkpoints
        assert store is not None
        for seq, boundary_blocks, h in pending.values():
            # A later call may have cleared/superseded this intent.
            if seq.pending_checkpoint != h:
                continue
            seq.pending_checkpoint = -1
            src = seq.per_req_cache_group
            if src < 0:
                continue
            if store.contains_or_pending(h):
                continue
            op = store.begin_store(h, boundary_blocks, src)
            if op is None:
                self.checkpoints_dropped += 1
                continue
            self._paged_store_ops.append(op)
            self.checkpoints_kept += 1

    def checkpoint_fates(self) -> dict[str, int]:
        """What became of the checkpoints the ladder asked this pool to keep."""
        return {
            "checkpoints_kept": self.checkpoints_kept,
            "checkpoints_dropped": self.checkpoints_dropped,
            "checkpoints_evicted": (
                self.page_checkpoints.evictions
                if self.page_checkpoints is not None
                else self.checkpoints_evicted
            ),
            "checkpoints_orphaned": self.checkpoints_orphaned,
        }

    def record_copy(self, src: int, dst: int) -> None:
        """Schedule an Active Slot relocation for the next batch."""
        self._copies.append((src, dst))

    def record_restore(self, h: int, dst: int) -> bool:
        """Pin a PAGE-backed checkpoint and queue its gather into `dst`."""
        assert self.page_checkpoints is not None
        op = self.page_checkpoints.begin_restore(h, dst)
        if op is None:
            return False
        self._paged_restore_ops.append(op)
        return True

    def take_state_maintenance_ops(self) -> StateMaintenanceOps:
        """Drain every state move for the one real batch now being built."""
        if self.page_checkpoints is not None:
            self._commit_paged_pending()
        relocations, self._copies = self._copies, []
        stores, self._paged_store_ops = self._paged_store_ops, []
        restores, self._paged_restore_ops = self._paged_restore_ops, []
        return StateMaintenanceOps(
            relocations=tuple(relocations),
            checkpoint_stores=tuple(stores),
            checkpoint_restores=tuple(restores),
        )

    def forget_pending(self, seq) -> None:
        """Drop `seq`'s uncommitted checkpoint before releasing its slot."""
        seq.pending_checkpoint = -1
        self._paged_pending.pop(id(seq), None)

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

    def unindex(self, h: int) -> int:
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

        Returns the group freed, or -1.
        """
        if self.page_checkpoints is not None:
            checkpoint_id = self.page_checkpoints.unindex(h)
            if checkpoint_id >= 0:
                self.checkpoints_orphaned += 1
            return checkpoint_id
        group = self.hash_to_group.get(h, -1)
        if group < 0:
            return -1
        self.invalidate(group)
        self.checkpoints_orphaned += 1
        return group

    def clear_index(self) -> None:
        """Drop all checkpoint hashes, preserving only in-flight readers."""
        for seq, _, _ in self._paged_pending.values():
            seq.pending_checkpoint = -1
        self._paged_pending.clear()
        if self.page_checkpoints is not None:
            self.page_checkpoints.clear()
            return
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
        if self.page_checkpoints is not None:
            self.page_checkpoints.complete_inflight()
        if not self._pinned:
            return
        held, self._deferred = self._deferred, set()
        for group in sorted(self._pinned):
            if group in held:
                continue
            self.release(group)
            del self._pinned[group]
