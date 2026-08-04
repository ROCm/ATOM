# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from dataclasses import dataclass
from math import inf

# `StateTransfer.kind` values. Plain strings rather than an enum because the
# choice crosses a process boundary inside a dict (ModelRunner's `block_info`),
# where a scalar survives and a class does not.
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
      `copy()`    one request's state is a contiguous byte range another group
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

    @classmethod
    def none(cls) -> "StateTransfer":
        return cls(NONE)

    @classmethod
    def fork(cls, tokens: int) -> "StateTransfer":
        assert tokens > 0, "a fork binds its successor forward; use none()"
        return cls(FORK, tokens)

    @classmethod
    def copy(cls) -> "StateTransfer":
        return cls(COPY)

    @classmethod
    def from_config(cls, kind: str, fork_tokens: int) -> "StateTransfer":
        """Rebuild from the two scalars that crossed the process boundary."""
        if kind == FORK:
            return cls.fork(fork_tokens)
        assert kind in (COPY, NONE), f"unknown state transfer kind {kind!r}"
        return cls(kind)

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

    Capacity model: a checkpoint is a group sitting on the free list with its
    content still valid, filed under the content hash of the last block it
    covers. This is the KV block pool's lazy-eviction model (`pop` drops the
    hash at hand-out time, not at free time) applied to state groups. The index
    therefore holds nothing back — `pop` invalidates whatever it hands out, so a
    checkpoint can never shrink admission, and under full concurrency the
    checkpoint set drains to empty on its own.

    *How* a group reaches the index is the backend's `StateTransfer`, and it is
    the only thing that differs between the two mechanisms this class runs:

      `fork`  the owner gives its group away and takes a fresh one, so the
              checkpoint costs no bytes but binds the very next forward, which
              has to leave the replacement self-contained (`min_fork_tokens`).
      `copy`  the state is a byte range, so a duplicate goes to the index and
              the owner is not disturbed at all. Nothing is bound: no successor
              forward, and the resuming side copies rather than forking too.

    Both meet the same index and the same free list. Under `copy` the bytes are
    moved by a forward, so this class only schedules the pairs (`take_copies`)
    and the next batch issues them.

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
        # FIFO, matching the KV pool's `free_block_ids`: a group handed back
        # keeps its content and may still be indexed as a checkpoint, so
        # hand-out order must be least-recently-freed first or every checkpoint
        # is evicted by the next admission.
        self.free_groups: deque[int] = deque(range(num_groups))
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
        # `copy` only. Seqs whose last forward left their state on a boundary
        # worth keeping. `take_copies` turns each into a copy pair when the next
        # batch is built, which is the latest moment the owner is still known to
        # hold the group being duplicated.
        self._checkpoint_pending: list = []
        # (src, dst) group pairs the next batch must copy before its forward.
        # Both halves of the protocol feed this under `copy`: keeping a
        # checkpoint copies the owner's state out, resuming from one copies it
        # back in.
        self._copies: list[tuple[int, int]] = []

    # ------------------------------ free list ------------------------------ #
    def has_free(self) -> bool:
        return bool(self.free_groups)

    def pop(self) -> int:
        """Hand out a group, evicting its checkpoint if it carried one.

        The state twin of `BlockManager._pop_free_block`: groups sit in the FIFO
        carrying whatever the last owner left in them, and re-allocation — not
        the free — is the eviction event.
        """
        group = self.free_groups.popleft()
        self.invalidate(group)
        return group

    def claim(self, group: int) -> None:
        """Claim one specific free group — a checkpoint the caller looked up.

        Linear in the free list, unlike `pop`. That is deliberate: the queue
        stays the single source of truth for "how many groups are free", which
        admission and every caller of `has_free` rely on. The scan is bounded by
        max_num_seqs and runs once per resuming request, against a
        `can_allocate` that already hashed every block of the prompt.
        """
        self.free_groups.remove(group)

    def release(self, group: int) -> None:
        self.free_groups.append(group)

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
            if not assume_checkpointed and block_hashes[i] not in self.hash_to_group:
                continue
            if seq.num_tokens - (i + 1) * hbs >= self.min_fork_tokens:
                return i + 1
        return 0

    def lookup(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
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

        `copy` — the owner keeps writing where it is and a duplicate of its
        state goes to the index instead. Only the intent is recorded here; the
        destination group and the copy pair come from `take_copies` when the next
        batch is built. Deferred because the bytes have to be moved by a forward, and
        a checkpoint indexed before its bytes exist would hand a resuming
        request whatever the destination happened to hold.

        `boundary_blocks` is unused here: a group is a single entry, not a span
        of them. It is in the protocol for classes whose checkpoint is a run of
        entries ending at the boundary.

        Best-effort under both: with no free group the seq simply keeps writing
        its own and no checkpoint is taken.
        """
        if not self.applies(seq):
            return
        old = seq.per_req_cache_group
        if old < 0:
            return
        if self.transfer.copies:
            if seq.pending_checkpoint == -1:
                self._checkpoint_pending.append(seq)
            # A later boundary supersedes an earlier one: the group holds the
            # state as of the last forward, so only the last position is true.
            seq.pending_checkpoint = h
            return
        if not self.free_groups:
            return
        seq.per_req_cache_group = self.pop()
        seq.state_fork_src = old
        self.release(old)
        self._index(h, old)

    def _commit_pending(self) -> None:
        """Turn the last step's checkpoint intents into copy pairs.

        Each pending seq gets a destination group, which goes straight back on
        the free list and into the index — the same capacity-neutral move
        `checkpoint` makes under `fork`, covered by the same lazy eviction:
        whoever pops the group next invalidates the hash on the way out.

        A seq preempted or finished in between carries no group any more and is
        skipped, so nothing is ever indexed over state that is gone. That check
        is only sound because this runs with the batch already decided — see
        `take_copies`.
        """
        if not self._checkpoint_pending:
            return
        for seq in self._checkpoint_pending:
            h, seq.pending_checkpoint = seq.pending_checkpoint, -1
            src = seq.per_req_cache_group
            if h == -1 or src < 0 or not self.free_groups:
                continue
            dst = self.pop()
            self.release(dst)
            self._index(h, dst)
            self._copies.append((src, dst))
        self._checkpoint_pending.clear()

    def record_copy(self, src: int, dst: int) -> None:
        """Schedule a state copy for the next batch's forward to issue."""
        self._copies.append((src, dst))

    def take_copies(self) -> list[tuple[int, int]]:
        """Every copy the batch now being built must issue before its forward.

        Called at the moment the batch is constructed, which is the whole point:
        a checkpoint's source is the owner's live group, and it has to still be
        that owner's when the copy runs. Committing earlier in the pass would
        leave a window — an admission preempting that owner would return the
        group to the free list, and the copy would then duplicate whatever the
        next request wrote there into a group already indexed as a checkpoint.
        Nothing runs between here and the batch, so the window is empty.

        A checkpoint therefore becomes visible one pass later than the step that
        formed it, and this pass's own admissions get first claim on the free
        list. Both are the right way round: admission is throughput, a
        checkpoint is speculative reuse.

        The two kinds of pair cannot collide, so their order does not matter. A
        resume source is a claimed or pinned checkpoint and a keeper source is a
        live group; neither is on the free list, so `_commit_pending`'s `pop`
        can return neither.
        """
        self._commit_pending()
        copies, self._copies = self._copies, []
        return copies

    def forget_pending(self, seq) -> None:
        """Drop `seq`'s uncommitted checkpoint — its group is being released.

        The seq stays in `_checkpoint_pending` until the next commit, which
        skips it on the cleared hash. Cheaper than removing it, and the list is
        emptied every pass either way.
        """
        seq.pending_checkpoint = -1

    def _index(self, h: int, group: int) -> None:
        """File `group` as the checkpoint for hash `h`.

        Caller guarantees `group` holds the state as of that boundary and has
        been handed back to the free list — a checkpointed group is never
        written again, so the request that produced it must have moved on.
        """
        if not self.enabled:
            return
        stale = self.group_hash[group]
        if stale != -1 and self.hash_to_group.get(stale) == group:
            del self.hash_to_group[stale]
        # Re-filing a hash onto a different group orphans the old one; drop
        # its back-pointer so a later invalidate doesn't delete the new entry.
        prev = self.hash_to_group.get(h, -1)
        if prev != -1 and prev != group:
            self.group_hash[prev] = -1
        self.hash_to_group[h] = group
        self.group_hash[group] = h

    def invalidate(self, group: int) -> None:
        """Drop `group`'s checkpoint. Called when the group is handed out."""
        if not self.enabled:
            return
        h = self.group_hash[group]
        if h == -1:
            return
        self.group_hash[group] = -1
        if self.hash_to_group.get(h) == group:
            del self.hash_to_group[h]

    # -------------------------------- pins --------------------------------- #
    def pin(self, group: int) -> None:
        self._pinned[group] = self._pinned.get(group, 0) + 1

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
        else:
            self._pinned[group] = count - 1

    def release_pins(self) -> None:
        """Return the fork sources pinned for the step that just went out.

        They go back to the free list, once each however many requests read
        them. Safe once the reading forward has been issued: a request handed
        one of these groups next step runs its forward after the reader's, on
        the same stream.
        """
        if not self._pinned:
            return
        for group in sorted(self._pinned):
            self.release(group)
        self._pinned.clear()
