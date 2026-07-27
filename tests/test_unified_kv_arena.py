# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the unified-KV control plane (v1-simple per-group BYTE arena).

v1-simple: per-group page allocator; free = true eviction. Cross-pool lending
is pool-driven (BlockManager evicts cold sibling blocks then retries), so the
arena raises ArenaEmpty when its own capacity is exhausted. Chunks are sized in
BYTES so SWA / compress owners (different page sizes) share them.
"""

import pytest

from atom.model_engine.chunk_arena import ArenaEmpty
from atom.model_engine.unified_kv_arena import UnifiedKvArena

BLOCK_SIZE = 128
BYTES_PER_CHUNK = 128  # toy chunk byte size (one SWA block)
C4_STRIDE = 32  # c4 compress page bytes -> 4 pages/chunk
C128_STRIDE = 1  # c128 compress page bytes -> 128 pages/chunk
SWA_PB = BYTES_PER_CHUNK  # swa page = whole chunk -> 1 page/chunk


def _grp(name, owners, num_chunks):
    return {
        "name": name,
        "num_chunks": num_chunks,
        "bytes_per_chunk": BYTES_PER_CHUNK,
        "owners": owners,
    }


def _arena(c4_chunks=8, c128_chunks=8):
    return UnifiedKvArena(
        block_size=BLOCK_SIZE,
        group_specs=[
            _grp("c4", {"swa": SWA_PB, "compress": C4_STRIDE}, c4_chunks),
            _grp("c128", {"swa": SWA_PB, "compress": C128_STRIDE}, c128_chunks),
        ],
    )


def test_disabled_is_noop():
    a = UnifiedKvArena(block_size=BLOCK_SIZE, group_specs=[])
    assert not a.enabled
    assert a.can_alloc_compressed(1000)
    a.alloc_compressed(0)
    a.alloc_swa(0)
    assert a.is_compressed_backed(0)  # disabled -> treated as backed


def test_compressed_physical_page_per_group():
    a = _arena()
    a.alloc_compressed(0)
    a.alloc_compressed(1)
    assert a.compress_page("c4", 0) == 0
    assert a.compress_page("c4", 1) == 1
    assert a.compress_page("c128", 0) == 0
    assert a.compress_page("c128", 1) == 1
    assert a.compress_page("c4", 1) * C4_STRIDE == 32
    assert a.compress_page("c128", 1) * C128_STRIDE == 1
    assert a.is_compressed_backed(0)
    assert not a.is_compressed_backed(99)


def test_swa_backed_in_every_group():
    a = _arena()
    a.alloc_swa(5)
    assert a.swa_page("c4", 5) >= 0
    assert a.swa_page("c128", 5) >= 0
    a.free_swa(5)
    with pytest.raises(KeyError):
        a.swa_page("c4", 5)


def test_compressed_packs_four_per_chunk_c4():
    a = _arena(c4_chunks=1, c128_chunks=64)
    for b in range(4):
        a.alloc_compressed(b)
    assert not a.can_alloc_compressed(1)  # c4 arena exhausted
    with pytest.raises(ArenaEmpty):
        a.alloc_compressed(4)
    assert not a.is_compressed_backed(4)  # rollback: not half-backed
    # free one block -> a page frees, but the chunk only returns when all 4 free
    for b in range(4):
        a.free_compressed(b)
    assert a.can_alloc_compressed(4)


def test_rollback_on_group_exhaustion():
    a = _arena(c4_chunks=64, c128_chunks=1)
    for b in range(128):
        a.alloc_compressed(b)  # fills the single c128 chunk (128 pages)
    assert not a.can_alloc_compressed(1)
    with pytest.raises(ArenaEmpty):
        a.alloc_compressed(128)
    assert not a.is_compressed_backed(128)
    assert 128 not in a.groups["c4"].phys["compress"]  # c4 grant rolled back


def test_free_then_realloc_reuses_capacity():
    a = _arena(c4_chunks=1, c128_chunks=64)
    for b in range(4):
        a.alloc_compressed(b)
    a.free_compressed(0)
    a.free_compressed(1)
    a.free_compressed(2)
    a.free_compressed(3)  # chunk fully returns
    a.alloc_compressed(10)  # reuse
    assert a.is_compressed_backed(10)
    assert a.compress_page("c4", 10) * C4_STRIDE >= 0


def test_swa_capacity_bounded_by_tightest_group():
    a = _arena(c4_chunks=2, c128_chunks=5)
    assert a.can_alloc_swa(2)
    assert not a.can_alloc_swa(3)  # c4 (2 chunks -> 2 SWA pages) caps it
    a.alloc_swa(0)
    a.alloc_swa(1)
    with pytest.raises(ArenaEmpty):
        a.alloc_swa(2)
    assert 2 not in a.groups["c128"].phys["swa"]  # rollback clean


def test_free_compressed_idempotent():
    a = _arena()
    a.alloc_compressed(3)
    a.free_compressed(3)
    a.free_compressed(3)
    assert a.can_alloc_compressed(1)


def test_cross_pool_free_enables_sibling_alloc():
    # C4 group: 1 chunk. SWA holds it; freeing SWA lets compressed alloc use it.
    a = _arena(c4_chunks=1, c128_chunks=64)
    a.alloc_swa(0)  # SWA takes the single c4 chunk
    assert not a.can_alloc_compressed(1)  # c4 arena empty
    a.free_swa(0)  # true eviction -> chunk returns to arena
    assert a.can_alloc_compressed(1)
    a.alloc_compressed(0)
    assert a.compress_page("c4", 0) * C4_STRIDE == 0  # reused freed bytes


# ----------------------------- CSA boundary owner ------------------------- #
# NOTE (feat/csa-swa-fusion): the CSA boundary snapshot is no longer an arena
# owner — it is fused into the c4 SWA chunk's tail bytes and rides the SWA block.
# The former csa_main/csa_idx owner tests (alloc_csa / csa_*_page / max_csa_pages
# / three-way borrow) were removed with that machinery.
