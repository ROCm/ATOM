"""Logical-ID free-list shared by primary and windowed KV pools."""

from __future__ import annotations

from collections import deque


class PooledFreeList:
    """Track backed-free, unbacked-free, and used logical IDs.

    Queue entries may become stale; membership sets are authoritative.  Allocation
    prefers backed IDs because they can reuse physical storage without borrowing
    from an arena.
    """

    def __init__(self, capacity: int, *, initially_backed: bool = True):
        if capacity < 0:
            raise ValueError(f"capacity must be non-negative, got {capacity}")
        self.capacity = int(capacity)
        ids = range(self.capacity)
        self.backed_ids: deque[int] = deque(ids if initially_backed else ())
        self.backed_set: set[int] = (
            set(range(self.capacity)) if initially_backed else set()
        )
        self.unbacked_ids: deque[int] = deque(
            () if initially_backed else range(self.capacity)
        )
        self.unbacked_set: set[int] = (
            set() if initially_backed else set(range(self.capacity))
        )
        self.used_ids: set[int] = set()

    def _validate(self, logical_id: int) -> None:
        if not 0 <= logical_id < self.capacity:
            raise ValueError(f"logical id {logical_id} outside [0, {self.capacity})")

    @staticmethod
    def _pop_member(queue: deque[int], members: set[int]) -> int | None:
        while queue:
            logical_id = queue.popleft()
            if logical_id in members:
                members.discard(logical_id)
                return logical_id
        return None

    def pop_backed(self) -> int | None:
        """Pop only a backed-free ID, skipping stale queue entries."""
        return self._pop_member(self.backed_ids, self.backed_set)

    def pop(self, *, error_message: str = "No free blocks available") -> int:
        """Pop backed-first, falling back to an unbacked logical ID."""
        logical_id = self.pop_backed()
        if logical_id is None:
            logical_id = self._pop_member(self.unbacked_ids, self.unbacked_set)
        if logical_id is None:
            raise AssertionError(error_message)
        return logical_id

    def mark_used(self, logical_id: int) -> None:
        """Move an allocated or cache-hit ID into the used set."""
        self._validate(logical_id)
        self.backed_set.discard(logical_id)
        self.unbacked_set.discard(logical_id)
        self.used_ids.add(logical_id)

    def deallocate(self, logical_id: int) -> bool:
        """Return a used ID to backed-free; repeated release is a no-op."""
        self._validate(logical_id)
        if logical_id not in self.used_ids:
            return False
        self.used_ids.remove(logical_id)
        if logical_id not in self.backed_set:
            self.backed_ids.append(logical_id)
            self.backed_set.add(logical_id)
        self.unbacked_set.discard(logical_id)
        return True

    def move_to_unbacked(self, logical_id: int) -> None:
        """Record that true eviction returned this ID's physical backing."""
        self._validate(logical_id)
        self.used_ids.discard(logical_id)
        self.backed_set.discard(logical_id)
        if logical_id not in self.unbacked_set:
            self.unbacked_ids.append(logical_id)
            self.unbacked_set.add(logical_id)

    def move_all_to_unbacked(self) -> None:
        """Convert an initially fixed pool to lazy arena backing in place."""
        if self.used_ids:
            raise AssertionError("cannot detach backing while IDs are in use")
        for logical_id in sorted(self.backed_set):
            if logical_id not in self.unbacked_set:
                self.unbacked_ids.append(logical_id)
                self.unbacked_set.add(logical_id)
        self.backed_set.clear()
        self.backed_ids.clear()

    def ids_conserved(self) -> bool:
        """Whether every ID belongs to exactly one stable state."""
        if not self.used_ids.isdisjoint(self.backed_set):
            return False
        if not self.used_ids.isdisjoint(self.unbacked_set):
            return False
        if not self.backed_set.isdisjoint(self.unbacked_set):
            return False
        return (
            len(self.used_ids) + len(self.backed_set) + len(self.unbacked_set)
            == self.capacity
        )
