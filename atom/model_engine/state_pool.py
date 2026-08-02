# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


class StateCheckpointPool:
    """Content index over per-request state groups that no request owns.

    A per-request state (GDN conv+ssm, V4 compressor ring) cannot be rebuilt
    from cached KV blocks: the cache holds the compressor's *output*, the state
    is its rolling *input* window. So a prefix-cache hit is only recoverable up
    to a boundary where somebody saved the state — which is what this pool
    indexes.

    Capacity model: a checkpoint is a group sitting on the free list with its
    content intact, indexed by the content hash of the last block it covers.
    This is the KV block pool's lazy-eviction model (`_allocate_block` drops the
    hash at hand-out time, not at free time) applied to state groups. The pool
    therefore holds nothing back — `invalidate` fires whenever a group is handed
    out, so a checkpoint can never shrink admission, and under full concurrency
    the checkpoint set drains to empty on its own.

    Ownership stays with `BlockManager`: it owns the free list and calls
    `invalidate` on hand-out. This class only maps hash <-> group and answers
    "how far back is the nearest recoverable boundary".

    Disabled (`num_groups == 0`, i.e. a model with no per-request state) makes
    every method a no-op and `bounded_hit` an identity, so the caller needs no
    branch.
    """

    def __init__(self, num_groups: int, enabled: bool = True):
        self.enabled: bool = enabled and num_groups > 0
        self.num_groups: int = num_groups
        # hash -> group holding the state as of that block boundary.
        self.hash_to_group: dict[int, int] = {}
        # Reverse map, for lazy eviction when a group is handed out. -1 = the
        # group carries no published checkpoint.
        self.group_hash: list[int] = [-1] * num_groups
        # Groups serving as a fork source in the in-flight step, counted by how
        # many requests fork off each. They stay off the free list until
        # `release_pins`, so the step that reads them cannot race a request that
        # was handed the same group. The count matters because a checkpoint is
        # read-only and several requests may share one: the group goes back only
        # once, and no reader may take it over while another still reads it.
        self._pinned: dict[int, int] = {}

    # ------------------------------- lookup -------------------------------- #
    def bounded_hit(
        self, compressed_hit: int, block_hashes: list[int], is_forkable=None
    ) -> int:
        """Shrink a compressed-prefix hit to the nearest recoverable boundary.

        Returns the largest `L <= compressed_hit` such that a state checkpoint
        exists for the prefix of `L` blocks AND `is_forkable(L)` holds, scanning
        right-to-left so the hit is cut as little as possible. 0 is always valid
        — a request starting from scratch needs no prior state.

        `is_forkable` is the caller's `min_fork_tokens` test: resuming reads the
        checkpoint and writes a fresh group for one forward, and that forward
        has to leave the fresh group self-contained. A boundary too close to the
        end of the prompt fails it and the scan keeps walking back.

        This is the state-side twin of `SlidingWindowPool.bounded_hit`; the
        caller chains the two. Without it, a hit hands the resumed forward a
        group freshly popped off the free list and it reads the previous
        occupant's state.
        """
        if not self.enabled:
            return compressed_hit
        for i in range(compressed_hit - 1, -1, -1):
            if block_hashes[i] not in self.hash_to_group:
                continue
            if is_forkable is None or is_forkable(i + 1):
                return i + 1
        return 0

    def lookup(self, h: int) -> int:
        """Group holding the checkpoint for hash `h`, or -1."""
        if not self.enabled:
            return -1
        return self.hash_to_group.get(h, -1)

    # ----------------------------- publication ----------------------------- #
    def publish(self, h: int, group: int) -> None:
        """Index `group` as the checkpoint for hash `h`.

        Caller guarantees `group` holds the state as of that boundary and has
        been handed back to the free list — a published group is never written
        again, so the request that produced it must have moved on to a new one.
        """
        if not self.enabled:
            return
        stale = self.group_hash[group]
        if stale != -1 and self.hash_to_group.get(stale) == group:
            del self.hash_to_group[stale]
        # Re-publishing a hash onto a different group orphans the old one; drop
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

    def take_pins(self) -> list[int]:
        """Drain the fork sources pinned for the step that just went out.

        The caller returns them to the free list, once each however many
        requests read them. Safe once the reading forward has been issued: a
        request handed one of these groups next step runs its forward after the
        reader's, on the same stream.
        """
        if not self._pinned:
            return []
        drained = sorted(self._pinned)
        self._pinned.clear()
        return drained
