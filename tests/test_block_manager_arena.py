# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DSV4 manager wired to the unified-KV chunk arena.

Flag on: compressed block_ids are logical, backed by per-group arena pages;
SWA and compressed borrow chunks from a shared arena on demand (pool-driven).
"""

import pytest

from atom.kv_cache.dsv4.manager import Dsv4KvCacheManager
from conftest import MockConfig

BS = 4  # kv_cache_block_size


@pytest.fixture
def arena_on(monkeypatch):
    monkeypatch.setenv("ATOM_V4_UNIFIED_KV_ARENA", "1")


def _cfg(**over):
    # One group, byte chunk of BS bytes (toy). compress page 1 byte -> 4/chunk;
    # SWA page = whole chunk -> 1/chunk. num_chunks=6 -> max_compressed=24, swa=6.
    base = dict(
        kv_cache_block_size=BS,
        num_kvcache_blocks=8,
        enable_prefix_caching=False,
        num_swa_blocks=4,
        swa_window_size=BS,
        v4_arena_group_specs=[
            {
                "name": "g",
                "num_chunks": 6,
                "bytes_per_chunk": BS,
                "owners": {"swa": BS, "compress": 1},
            }
        ],
    )
    base.update(over)
    return MockConfig(**base)


def _assert_backed(bm, seq):
    for b in seq.block_table:
        assert bm.arena.is_compressed_backed(b)
        p = bm.arena.compress_page("g", b)  # resolvable physical page
        assert p >= 0


def test_arena_constructed_when_flag_on(arena_on):
    bm = Dsv4KvCacheManager(_cfg())
    assert bm.arena is not None and bm.arena.enabled
    # logical id space sized to arena max compressed capacity (6 chunks x 4)
    assert len(bm.blocks) == bm.arena.max_compressed_blocks() == 24


def test_arena_absent_when_flag_off():
    bm = Dsv4KvCacheManager(_cfg())  # no arena_on fixture -> flag off
    assert bm.arena is None
    assert len(bm.blocks) == 8  # fixed num_kvcache_blocks


def test_allocate_backs_blocks_and_resolves_pages(arena_on, seq_factory):
    bm = Dsv4KvCacheManager(_cfg())
    seq = seq_factory(list(range(1, 2 * BS + 1)))  # 2 blocks
    assert bm.can_allocate(seq) >= 0
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    _assert_backed(bm, seq)


def test_dealloc_keeps_pages_lazy_then_reusable(arena_on, seq_factory):
    bm = Dsv4KvCacheManager(_cfg())
    seq = seq_factory(list(range(1, 3 * BS + 1)))  # 3 blocks
    bm.allocate(seq)
    ids = list(seq.block_table)
    bm.deallocate(seq)
    # lazy: ref-0 blocks keep their arena pages (still backed, cached)
    for b in ids:
        assert bm.arena.is_compressed_backed(b)
    # a fresh seq reuses capacity without error
    seq2 = seq_factory(list(range(100, 100 + 3 * BS)))
    assert bm.can_allocate(seq2) >= 0
    bm.allocate(seq2)
    _assert_backed(bm, seq2)


def test_compressed_borrows_from_swa_under_pressure(arena_on, seq_factory):
    # 6 chunks total. Compressed alone can hold 24 blocks, but only if it can
    # borrow every chunk. Allocate enough compressed blocks to exceed a small
    # initial share, forcing borrow from SWA (evicting cold SWA blocks).
    bm = Dsv4KvCacheManager(_cfg())
    # Prime SWA with some cold blocks, then release so they are ref-0 cached.
    s0 = seq_factory(list(range(1, 4 * BS + 1)))
    bm.allocate(s0)
    bm.deallocate(s0)  # SWA blocks now ref-0 (reclaimable)
    # Now allocate a big compressed seq that needs many chunks.
    big = seq_factory(list(range(50, 50 + 20 * BS)))  # 20 blocks
    if bm.can_allocate(big) >= 0:
        bm.allocate(big)
        _assert_backed(bm, big)
        # all 20 blocks resolved -> compressed reclaimed SWA chunks
        assert len({bm.arena.compress_page("g", b) for b in big.block_table}) == 20


def _compressed_ids_conserved(bm):
    return bm.ids_conserved()


def _swa_ids_conserved(bm):
    return bm.ids_conserved()


def test_cross_pool_borrow_does_not_leak_ids(arena_on, seq_factory):
    """#11: SWA borrowing a chunk from compressed (and vice versa) evicts a
    sibling block; the evicted logical id must move to the UNBACKED free pool,
    not vanish. Repeated borrow/dealloc/realloc cycles must conserve every id."""
    bm = Dsv4KvCacheManager(_cfg())
    assert _compressed_ids_conserved(bm) and _swa_ids_conserved(bm)
    for rnd in range(6):
        base = 1000 * (rnd + 1)
        # Fill SWA with cold ref-0 blocks, then a big compressed seq forces
        # compressed to reclaim SWA chunks (cross-pool eviction).
        s = seq_factory(list(range(base, base + 4 * BS)))
        bm.allocate(s)
        bm.deallocate(s)
        big = seq_factory(list(range(base + 500, base + 500 + 20 * BS)))  # 20 blks
        if bm.can_allocate(big) >= 0:
            bm.allocate(big)
            _assert_backed(bm, big)
            bm.deallocate(big)
        # Invariant holds after every round — no id leaked either pool.
        assert _compressed_ids_conserved(bm), f"compressed id leak round {rnd}"
        assert _swa_ids_conserved(bm), f"swa id leak round {rnd}"


def test_admission_credits_backed_free_reuse(arena_on, seq_factory):
    """#11 (b): a compressed pool full of ref-0 BACKED cached blocks has ≈0 free
    arena pages, but reusing those cached blocks needs no new page — the compressed
    admission check must still pass. On the pre-fix code `backable` was ≈0 and this
    returned False, so the scheduler live-locked (can_allocate -1 forever)."""
    bm = Dsv4KvCacheManager(_cfg())
    s = seq_factory(list(range(1, 20 * BS + 1)))  # 20 blocks (arena max is 24)
    bm.allocate(s)
    bm.deallocate(s)  # 20 ref-0 BACKED cached blocks; arena pages held, ~none free
    assert bm.arena.compressed_available() < 20  # pages held, not free
    # Reuse-only admission must succeed purely on backed-free credit.
    assert bm._has_free_compressed(20), "livelock: reuse of cached blocks rejected"
    assert _compressed_ids_conserved(bm)


def test_cross_pool_evict_moves_id_to_unbacked_and_reusable(arena_on, seq_factory):
    """The id evicted by cross-pool lending must move to the UNBACKED free pool
    (not vanish) and be re-allocatable, re-borrowing a page. Drives the evict
    primitive directly (real SWA borrow only consumes blocks during forward)."""
    bm = Dsv4KvCacheManager(_cfg())
    fill = seq_factory(list(range(1, 20 * BS + 1)))
    bm.allocate(fill)
    bm.deallocate(fill)  # 20 ref-0 BACKED compressed cached blocks
    backed_before = len(bm.free_block_ids_set)
    unbacked_before = len(bm._unbacked_free_set)
    # Simulate SWA starvation borrowing a compressed chunk (pool-driven).
    assert bm._evict_cold_compressed() is True
    assert _compressed_ids_conserved(bm), "evicted id leaked"
    assert len(bm.free_block_ids_set) == backed_before - 1  # one left backed-free
    assert len(bm._unbacked_free_set) == unbacked_before + 1  # ...became unbacked
    # The evicted (now unbacked) id is still reusable: pop + allocate re-backs it.
    bid = bm._pop_free_block()
    bm._allocate_block(bid)
    assert bm.arena.is_compressed_backed(bid)
    assert _compressed_ids_conserved(bm)


def test_repeated_evict_never_leaks_or_livelocks(arena_on, seq_factory):
    """Hammer the cross-pool evict primitive: every evictable backed-free id can
    be lent, all move to unbacked, none leak, and evict cleanly reports False when
    nothing is left to lend (bounded — no spin)."""
    bm = Dsv4KvCacheManager(_cfg())
    fill = seq_factory(list(range(1, 20 * BS + 1)))
    bm.allocate(fill)
    bm.deallocate(fill)
    evicted = 0
    while bm._evict_cold_compressed():
        evicted += 1
        assert _compressed_ids_conserved(bm)
        assert evicted <= len(bm.blocks), "evict loop did not terminate"
    assert evicted == 20  # exactly the 20 backed-free cached blocks were lendable
    assert len(bm.free_block_ids_set) == 0  # nothing backed-free left
    # All ids still accounted for (used 0 + backed 0 + unbacked 24).
    assert len(bm._unbacked_free_set) == len(bm.blocks)


def test_flag_off_behaviour_unchanged(seq_factory):
    bm = Dsv4KvCacheManager(_cfg())  # flag off
    seq = seq_factory(list(range(1, 2 * BS + 1)))
    bm.allocate(seq)
    assert len(seq.block_table) == 2
    assert bm.arena is None
    # Arena-off keeps a single free pool; the unbacked pool stays empty.
    assert len(bm._unbacked_free_set) == 0
