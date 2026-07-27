# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared physical-chunk arena for the DeepSeek-V4 unified KV pool (byte arena).

The SWA pool (full-resolution window KV), the compressed pool (CSA/HCA), and the
CSA boundary-state snapshot pool have DIFFERENT page sizes AND different element
dtypes (fp8/bf16 KV vs fp32 boundary state). ATOM used to ship them as separate
fixed-size pools; whichever was under-used wasted its reserved capacity.

This arena removes the fixed split. One :class:`ChunkArena` owns a free-list of
equal-size physical *chunks*, measured in **bytes** so heterogeneous owners can
share the same physical memory. Each pool wraps the arena in a
:class:`ChunkBackedFreeList`: it borrows chunks on demand, sub-divides each chunk
into ``pages_per_chunk`` pages at its own byte-size (``page_bytes``), and returns
a chunk to the arena once ALL its pages are free.

Why bytes, not rows: SWA/compress pages are ``head_dim`` wide in fp8/bf16, while a
CSA boundary page is ``2*head_dim`` wide in fp32. A byte chunk lets an owner
reinterpret the same bytes as its own ``(dtype, width)`` view (see
``deepseek_v4_attn`` per-layer uint8 buffer + dtype views). Chunks are
single-owner at a time, so the byte ranges of different owners never alias.

Design note — the arena tracks only USED vs TRULY-FREE pages; it does NOT model
the pools' lazy ref-0-cached state. A page is ``free``d (and its chunk possibly
returned to the arena) ONLY at TRUE eviction (the owning pool has dropped the
block's hash). Lazy ref-0-cached blocks keep their pages held in the pool (its
own ``free_block_ids`` LRU); cross-pool lending is POOL-DRIVEN — when a pool
can't ``pop``, BlockManager evicts the coldest ref-0 sibling blocks (drop hash +
``free``) until a chunk returns to the arena, then retries. This keeps a single
source of truth for cache state (the pool) — the arena never holds a page whose
KV a cache hit might still want.

Addressing (keeps the Triton index kernels' ``page * stride`` form): a page id
*is* its arena physical page index within the owner. A chunk owns bytes
``[cid*bytes_per_chunk, (cid+1)*...)``; an owner with ``pages_per_chunk`` pages
numbers them ``cid*pages_per_chunk + local`` so ``page_id * page_bytes`` lands
inside that chunk. A chunk is single-owner at a time, so different owners'
(dtype, page_bytes) byte ranges never alias. Consumers turn a page id into a row
of their own dtype view via ``page_id * (page_bytes // (width * elem))``.

Self-guarding: ``num_chunks == 0`` disables the arena.
"""

from __future__ import annotations

from collections import deque


class ArenaEmpty(Exception):
    """Raised by ``ChunkBackedFreeList.pop`` when neither a free page nor a free
    arena chunk is available. The caller (BlockManager) then evicts a cold
    sibling block to return a chunk to the arena before retrying."""


class ChunkArena:
    """Free-list of equal-size physical chunks (in bytes) shared by the pools.

    A chunk is ``bytes_per_chunk`` bytes of the per-layer arena byte buffer.
    ``acquire`` hands a free chunk to a pool; ``release`` takes it back once the
    pool has freed every page in it. FIFO reuse mirrors the block pools.
    """

    def __init__(self, num_chunks: int, bytes_per_chunk: int):
        self.num_chunks: int = int(num_chunks)
        self.bytes_per_chunk: int = int(bytes_per_chunk)
        self.enabled: bool = self.num_chunks > 0
        self._free: deque[int] = deque(range(self.num_chunks))
        self._free_set: set[int] = set(range(self.num_chunks))

    def num_free(self) -> int:
        return len(self._free_set)

    def acquire(self) -> int:
        while self._free:
            cid = self._free.popleft()
            if cid in self._free_set:
                self._free_set.discard(cid)
                return cid
        raise AssertionError("ChunkArena exhausted: no free chunks")

    def release(self, chunk_id: int) -> None:
        if chunk_id in self._free_set:
            return
        self._free.append(chunk_id)
        self._free_set.add(chunk_id)


class ChunkBackedFreeList:
    """Per-owner page allocator backed by a shared :class:`ChunkArena`.

    Replaces a pool's flat ``deque(range(num_pages))`` free-list. A page id is
    the arena-physical page index ``chunk_id * pages_per_chunk + local``; the
    owning pool's hash/refcount logic operates on these ids unchanged. Capacity
    is elastic: a chunk is borrowed from the arena on demand and returned once
    all its pages are freed (at true eviction). ``page_bytes`` is this owner's
    per-page byte size, so ``page_bytes`` may differ across owners sharing the
    arena (SWA vs compress vs CSA boundary state).
    """

    def __init__(self, arena: ChunkArena, page_bytes: int):
        self.arena = arena
        self.page_bytes = int(page_bytes)
        assert self.page_bytes > 0
        assert arena.bytes_per_chunk % self.page_bytes == 0, (
            f"page_bytes {self.page_bytes} must divide chunk bytes "
            f"{arena.bytes_per_chunk}"
        )
        self.pages_per_chunk: int = arena.bytes_per_chunk // self.page_bytes
        self._free_pages: deque[int] = deque()
        self._free_pages_set: set[int] = set()
        # chunk_id -> number of its pages currently free (held by this pool).
        self._chunk_free_count: dict[int, int] = {}

    # ----------------------------- capacity ------------------------------- #
    def available(self) -> int:
        """Pages this pool can hand out WITHOUT sibling eviction: its own free
        pages plus pages from every currently-free arena chunk."""
        return len(self._free_pages_set) + self.arena.num_free() * self.pages_per_chunk

    def free_now(self) -> int:
        """Pages immediately poppable without touching the arena."""
        return len(self._free_pages_set)

    def owned_chunks(self) -> int:
        return len(self._chunk_free_count)

    # ----------------------------- grow / pop ----------------------------- #
    def _grow(self) -> None:
        """Borrow one chunk from the arena and register its pages as free.
        Raises :class:`ArenaEmpty` if the arena has no free chunk."""
        if self.arena.num_free() == 0:
            raise ArenaEmpty
        cid = self.arena.acquire()
        base = cid * self.pages_per_chunk
        self._chunk_free_count[cid] = self.pages_per_chunk
        for local in range(self.pages_per_chunk):
            pid = base + local
            self._free_pages.append(pid)
            self._free_pages_set.add(pid)

    def pop(self) -> int:
        """Allocate one page id, borrowing a fresh arena chunk if none are free.
        Raises :class:`ArenaEmpty` if neither a free page nor an arena chunk is
        available (caller evicts a cold sibling block, then retries)."""
        if not self._free_pages_set:
            self._grow()
        while self._free_pages:
            pid = self._free_pages.popleft()
            if pid in self._free_pages_set:
                self._free_pages_set.discard(pid)
                self._chunk_free_count[pid // self.pages_per_chunk] -= 1
                return pid
        raise AssertionError("ChunkBackedFreeList: no free page after grow")

    def free(self, page_id: int) -> None:
        """Return a page id at TRUE eviction. When its chunk becomes fully free,
        the chunk is returned to the arena so any pool can borrow it."""
        if page_id in self._free_pages_set:
            return
        cid = page_id // self.pages_per_chunk
        self._free_pages.append(page_id)
        self._free_pages_set.add(page_id)
        self._chunk_free_count[cid] = self._chunk_free_count.get(cid, 0) + 1
        if self._chunk_free_count[cid] == self.pages_per_chunk:
            # Whole chunk free: drop its pages from the set (stale ids left in the
            # deque are skipped by pop's membership check) and return it.
            base = cid * self.pages_per_chunk
            for local in range(self.pages_per_chunk):
                self._free_pages_set.discard(base + local)
            del self._chunk_free_count[cid]
            self.arena.release(cid)
