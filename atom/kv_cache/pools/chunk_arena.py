"""Byte-addressed physical chunks shared by heterogeneous KV pools.

``ChunkArena`` owns equal-sized byte chunks.  Each
``ChunkBackedFreeList`` borrows chunks on demand, subdivides them at its own
page size, and returns a chunk only after every page is truly free.  Lazy
ref-zero cache retention remains the responsibility of the logical KV pools.
"""

from __future__ import annotations

from collections import deque


class ArenaEmpty(Exception):
    """Raised when neither an owner page nor an arena chunk is available."""


class ChunkArena:
    """FIFO free-list of equal-size physical byte chunks."""

    def __init__(self, num_chunks: int, bytes_per_chunk: int):
        self.num_chunks = int(num_chunks)
        self.bytes_per_chunk = int(bytes_per_chunk)
        if self.num_chunks < 0 or self.bytes_per_chunk <= 0:
            raise ValueError(
                "num_chunks must be non-negative and bytes_per_chunk positive"
            )
        self.enabled = self.num_chunks > 0
        self._free: deque[int] = deque(range(self.num_chunks))
        self._free_set: set[int] = set(range(self.num_chunks))

    def num_free(self) -> int:
        return len(self._free_set)

    def acquire(self) -> int:
        while self._free:
            chunk_id = self._free.popleft()
            if chunk_id in self._free_set:
                self._free_set.discard(chunk_id)
                return chunk_id
        raise AssertionError("ChunkArena exhausted: no free chunks")

    def release(self, chunk_id: int) -> None:
        if not 0 <= chunk_id < self.num_chunks:
            raise ValueError(f"invalid chunk id {chunk_id}")
        if chunk_id in self._free_set:
            return
        self._free.append(chunk_id)
        self._free_set.add(chunk_id)


class ChunkBackedFreeList:
    """Per-owner page allocator backed by a shared :class:`ChunkArena`."""

    def __init__(self, arena: ChunkArena, page_bytes: int):
        self.arena = arena
        self.page_bytes = int(page_bytes)
        if self.page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        if arena.bytes_per_chunk % self.page_bytes:
            raise ValueError(
                f"page_bytes {self.page_bytes} must divide chunk bytes "
                f"{arena.bytes_per_chunk}"
            )
        self.pages_per_chunk = arena.bytes_per_chunk // self.page_bytes
        self._free_pages: deque[int] = deque()
        self._free_pages_set: set[int] = set()
        # chunk_id -> number of pages currently free while owned by this pool.
        self._chunk_free_count: dict[int, int] = {}

    def available(self) -> int:
        """Pages available without sibling eviction."""
        return len(self._free_pages_set) + self.arena.num_free() * self.pages_per_chunk

    def free_now(self) -> int:
        return len(self._free_pages_set)

    def owned_chunks(self) -> int:
        return len(self._chunk_free_count)

    def _grow(self) -> None:
        if self.arena.num_free() == 0:
            raise ArenaEmpty
        chunk_id = self.arena.acquire()
        base = chunk_id * self.pages_per_chunk
        self._chunk_free_count[chunk_id] = self.pages_per_chunk
        for local in range(self.pages_per_chunk):
            page_id = base + local
            self._free_pages.append(page_id)
            self._free_pages_set.add(page_id)

    def pop(self) -> int:
        """Allocate one page, borrowing a chunk when needed."""
        if not self._free_pages_set:
            self._grow()
        while self._free_pages:
            page_id = self._free_pages.popleft()
            if page_id in self._free_pages_set:
                self._free_pages_set.discard(page_id)
                self._chunk_free_count[page_id // self.pages_per_chunk] -= 1
                return page_id
        raise AssertionError("ChunkBackedFreeList: no free page after grow")

    def free(self, page_id: int) -> None:
        """Return a page and release its chunk when every page is free."""
        if page_id in self._free_pages_set:
            return
        chunk_id = page_id // self.pages_per_chunk
        if not 0 <= chunk_id < self.arena.num_chunks:
            raise ValueError(f"invalid page id {page_id}")
        self._free_pages.append(page_id)
        self._free_pages_set.add(page_id)
        self._chunk_free_count[chunk_id] = self._chunk_free_count.get(chunk_id, 0) + 1
        if self._chunk_free_count[chunk_id] == self.pages_per_chunk:
            base = chunk_id * self.pages_per_chunk
            for local in range(self.pages_per_chunk):
                self._free_pages_set.discard(base + local)
            del self._chunk_free_count[chunk_id]
            self.arena.release(chunk_id)
