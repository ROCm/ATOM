# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the unified-KV chunk arena (v1-simple control-plane logic).

v1-simple: the arena tracks USED vs TRULY-FREE pages only. ``free`` is a true
eviction (chunk returns to the arena when fully free); lazy ref-0 caching and
cross-pool lending live in the pools / BlockManager, not here.
"""

import pytest

from atom.model_engine.chunk_arena import ArenaEmpty, ChunkArena, ChunkBackedFreeList

ROWS_PER_CHUNK = 128  # one SWA block


def test_arena_acquire_release_roundtrip():
    a = ChunkArena(num_chunks=4, bytes_per_chunk=ROWS_PER_CHUNK)
    assert a.num_free() == 4
    c0 = a.acquire()
    a.acquire()
    assert a.num_free() == 2
    a.release(c0)
    assert a.num_free() == 3
    a.release(c0)  # double release no-op
    assert a.num_free() == 3


def test_pop_raises_arena_empty_when_exhausted():
    a = ChunkArena(num_chunks=1, bytes_per_chunk=ROWS_PER_CHUNK)
    fl = ChunkBackedFreeList(a, page_bytes=ROWS_PER_CHUNK)  # 1 page/chunk
    fl.pop()
    with pytest.raises(ArenaEmpty):
        fl.pop()


def test_page_id_is_physical_page_index():
    a = ChunkArena(num_chunks=8, bytes_per_chunk=ROWS_PER_CHUNK)
    swa = ChunkBackedFreeList(a, page_bytes=128)  # 1 page/chunk
    cmp = ChunkBackedFreeList(a, page_bytes=32)  # 4 pages/chunk
    assert swa.pages_per_chunk == 1
    assert cmp.pages_per_chunk == 4
    swa_pid = swa.pop()
    assert swa_pid * 128 == swa_pid * ROWS_PER_CHUNK
    cmp_pid = cmp.pop()
    cmp_chunk, local = cmp_pid // 4, cmp_pid % 4
    assert cmp_pid * 32 == cmp_chunk * ROWS_PER_CHUNK + local * 32
    assert swa_pid != cmp_chunk  # different chunks, disjoint rows


def test_compressed_packs_four_pages_per_chunk():
    a = ChunkArena(num_chunks=2, bytes_per_chunk=ROWS_PER_CHUNK)
    cmp = ChunkBackedFreeList(a, page_bytes=32)
    pids = [cmp.pop() for _ in range(4)]
    assert len({p // 4 for p in pids}) == 1  # one chunk
    assert a.num_free() == 1
    cmp.pop()  # 5th forces a second chunk
    assert a.num_free() == 0


def test_free_returns_chunk_to_arena_only_when_fully_free():
    a = ChunkArena(num_chunks=1, bytes_per_chunk=ROWS_PER_CHUNK)
    cmp = ChunkBackedFreeList(a, page_bytes=32)
    p = [cmp.pop() for _ in range(4)]
    assert a.num_free() == 0
    cmp.free(p[0])
    cmp.free(p[1])
    cmp.free(p[2])
    assert a.num_free() == 0  # chunk NOT returned until ALL pages free
    assert cmp.free_now() == 3  # freed pages reusable within the pool
    reused = cmp.pop()
    assert reused in (p[0], p[1], p[2])
    cmp.free(reused)
    cmp.free(p[3])  # last one -> chunk returns to arena
    assert a.num_free() == 1
    assert cmp.free_now() == 0


def test_freed_chunk_reusable_by_sibling_pool():
    """After a compressed chunk fully frees, a DIFFERENT-page-size pool can
    borrow the same physical chunk from the arena (cross-pool elasticity)."""
    a = ChunkArena(num_chunks=1, bytes_per_chunk=ROWS_PER_CHUNK)
    cmp = ChunkBackedFreeList(a, page_bytes=32)  # 4 pages/chunk
    swa = ChunkBackedFreeList(a, page_bytes=128)  # 1 page/chunk
    p = [cmp.pop() for _ in range(4)]
    assert swa.available() == 0  # arena empty, nothing for SWA
    for pid in p:
        cmp.free(pid)  # chunk returns to arena
    assert swa.available() == 1
    sp = swa.pop()
    assert sp * 128 == (p[0] // 4) * ROWS_PER_CHUNK  # reused the freed rows


def test_free_idempotent():
    a = ChunkArena(num_chunks=1, bytes_per_chunk=ROWS_PER_CHUNK)
    cmp = ChunkBackedFreeList(a, page_bytes=32)
    pid = cmp.pop()
    cmp.free(pid)
    cmp.free(pid)  # no double-count corruption
    assert cmp.free_now() == 1


def test_available_counts_borrowable_arena_chunks():
    a = ChunkArena(num_chunks=3, bytes_per_chunk=ROWS_PER_CHUNK)
    cmp = ChunkBackedFreeList(a, page_bytes=32)
    assert cmp.available() == 3 * 4
    cmp.pop()
    assert cmp.available() == 3 * 4 - 1
    assert cmp.free_now() == 3
