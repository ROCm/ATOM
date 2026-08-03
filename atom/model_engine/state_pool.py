# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from collections import deque
from math import inf


class StateGroupPool:
    """Per-request state groups, plus a content index over the free ones.

    A *group* is what one request occupies in the pre-allocated state tensor:
    `entries // entries_per_req` contiguous indices (GDN conv+ssm, the
    DeepSeek-V4 compressor ring). This pool owns the free list, so it is the
    single answer to "can one more request be admitted" — the twin of
    `SlidingWindowPool` owning its blocks.

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

    Vocabulary: *checkpoint* is the state sense throughout — a boundary this
    class kept resumable. *Publish* is reserved for a block entering the
    content-addressed KV index (`BlockManager.hash_blocks`, the KV events).

    `enabled` covers the *index* only. The free list stays live either way:
    admission needs a group whether or not anything is ever checkpointed.
    """

    def __init__(
        self,
        num_groups: int,
        min_fork_tokens: int = 0,
        hash_block_size: int = 1,
        enabled: bool = True,
    ):
        self.enabled: bool = enabled and num_groups > 0
        self.num_groups: int = num_groups
        # Tokens the forward after a fork must cover for the new group to come
        # out self-contained (see AttentionBackend.min_fork_tokens).
        self.min_fork_tokens: int = min_fork_tokens
        # Same number on the protocol's scale. The backend API spells "no
        # forkable state" as 0, which on that scale means the opposite — no
        # successor needed at all — so it is decoded here, where it enters, and
        # nowhere else.
        self.successor_room: float = min_fork_tokens if min_fork_tokens > 0 else inf
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
    def resumable_hit(self, seq, hit: int, block_hashes: list[int]) -> int:
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
        """
        if not self.applies(seq):
            return hit
        hbs = self.hash_block_size
        for i in range(hit - 1, -1, -1):
            if block_hashes[i] not in self.hash_to_group:
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
        """Hand `seq`'s state group to the index under hash `h`.

        The rolling half of the checkpoint protocol: the group cannot be shared
        while its owner still writes it, so the owner moves to a fresh group and
        the old one — never written again — becomes the checkpoint. The next
        forward reads it and fills the replacement, which is the whole reason
        `min_fork_tokens` gates the position.

        `boundary_blocks` is unused here: a group is a single entry, not a span
        of them. It is in the protocol for classes whose checkpoint is a run of
        entries ending at the boundary.

        Best-effort: with no free group the seq simply keeps writing its own and
        no checkpoint is taken.
        """
        if not self.applies(seq):
            return
        old = seq.per_req_cache_group
        if old < 0 or not self.free_groups:
            return
        seq.per_req_cache_group = self.pop()
        seq.state_fork_src = old
        self.release(old)
        self._index(h, old)

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
