# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU unit tests for the DeepSeek-V4 unified KV pool (ATOM_UNIFIED_KV_SHARE,
plan_atom_unified_kv_pool.md §10/§11): the per-layer-type slot allocator
(``UnifiedTypePool``) and the SWA↔compress coordinator (``UnifiedKvCoordinator``),
including cross-role cache eviction and admission accuracy. No GPU required."""

from atom.model_engine.unified_chunk_pool import (
    UnifiedKvCoordinator,
    UnifiedTypePool,
    unified_slot_counts,
)


# --------------------------- UnifiedTypePool ---------------------------- #
def test_swa_only():
    # k=0 (SWA-only / flag-off): phys = slot, byte-identical to old free-list.
    p = UnifiedTypePool(num_slots=4, k=0, block_size=128)
    assert p.enabled and p.has_free_swa(4) and not p.has_free_swa(5)
    a = p.alloc_swa()
    b = p.alloc_swa()
    assert a == 0 and b == 1  # FIFO from 0
    assert not p.is_free_swa(a) and p.num_free_slots() == 2
    p.free_swa(a)
    assert p.is_free_swa(a) and p.num_free_slots() == 3
    # cached-hit claim path
    c = p.alloc_swa()
    p.free_swa(c)
    assert p.is_free_swa(c)
    p.claim_swa(c)
    assert not p.is_free_swa(c)
    d = UnifiedTypePool(0, k=0)
    assert not d.enabled and d.has_free_swa(9)


def test_compress_csa():
    # CSA: k=32 -> blocks_per_slot=4. 2 slots -> 8 compress blocks.
    p = UnifiedTypePool(num_slots=2, k=32, block_size=128)
    assert (
        p.blocks_per_slot == 4
        and p.has_free_compress(8)
        and not p.has_free_compress(9)
    )
    got = [p.alloc_compress() for _ in range(8)]
    # slot0 -> phys 0..3, slot1 -> phys 4..7 (row = phys*32 covers [0,256))
    assert sorted(got) == [0, 1, 2, 3, 4, 5, 6, 7], got
    assert p.num_free_slots() == 0
    # free one block -> slot not fully free -> no slot returned
    p.free_compress(2)
    assert p.num_free_slots() == 0
    # realloc reuses the freed sub-slot
    assert p.alloc_compress() == 2
    # free a whole slot's worth (0,1,2,3) -> slot0 returns to pool
    for phys in (0, 1, 3):
        p.free_compress(phys)
    p.free_compress(2)
    assert p.num_free_slots() == 1, p.num_free_slots()


def test_mutual_reuse():
    # SWA and compress share one pool: a slot freed by SWA is reusable by compress.
    p = UnifiedTypePool(num_slots=3, k=32, block_size=128)  # 3 slots
    s0 = p.alloc_swa()
    p.alloc_swa()  # SWA takes 2 slots
    assert p.num_free_slots() == 1
    # compress grabs the last free slot -> 4 CSA blocks
    [p.alloc_compress() for _ in range(4)]
    assert (
        p.num_free_slots() == 0
        and not p.has_free_compress(1)
        and not p.has_free_swa(1)
    )
    # SWA frees a slot -> compress can now borrow it
    p.free_swa(s0)
    assert p.has_free_compress(1) and p.has_free_swa(1)
    more = [p.alloc_compress() for _ in range(4)]  # compress borrows SWA's slot
    assert p.num_free_slots() == 0
    # compress frees its borrowed slot -> SWA can take it back
    for phys in more:
        p.free_compress(phys)
    assert p.has_free_swa(1)
    assert p.alloc_swa() is not None


# ------------------------- UnifiedKvCoordinator ------------------------- #
def test_compress_borrows_swa_freed():
    c = UnifiedKvCoordinator(num_swa_base=4, n_csa_slots=8, n_hca_slots=5)
    swa = [c.alloc_swa() for _ in range(4)]
    assert sorted(swa) == [0, 1, 2, 3]
    assert c.has_free_csa(16) and not c.has_free_csa(17)  # own region [4,8)
    c.free_swa(1)  # SWA frees slot 1 -> CSA can borrow it
    assert c.has_free_csa(17)
    borrowed = [c.alloc_csa() for _ in range(20)]  # 16 own + 4 borrowed
    assert set([4, 5, 6, 7]).issubset(set(borrowed))  # slot 1 -> phys 4..7


def test_swa_cannot_reclaim_taken_slot():
    c = UnifiedKvCoordinator(num_swa_base=3, n_csa_slots=4, n_hca_slots=4)
    c.alloc_swa()
    c.alloc_swa()
    c.alloc_swa()
    c.free_swa(1)
    got = [c.alloc_csa() for _ in range(8)]  # CSA grabs borrowed slot 1
    assert set([4, 5, 6, 7]).issubset(set(got))
    try:
        c.alloc_swa()
        raise AssertionError("SWA should have no reclaimable slot")
    except AssertionError as e:
        assert "No free SWA" in str(e), e


def test_compress_frees_swa_reclaims():
    c = UnifiedKvCoordinator(num_swa_base=3, n_csa_slots=4, n_hca_slots=4)
    a = c.alloc_swa()
    b = c.alloc_swa()
    c.free_swa(a)
    assert c.has_free_csa(8) and not c.has_free_csa(9)
    csa = [c.alloc_csa() for _ in range(8)]
    assert len(set(csa)) == 8
    assert set(range(a * 4, a * 4 + 4)).issubset(set(csa))
    for phys in csa:
        c.free_csa(phys)
    got = c.alloc_swa()  # SWA reclaims after compress frees
    assert got in (a, [s for s in range(3) if s not in (a, b)][0]), got


def test_compress_borrow_evicts_swa_cache():
    """Compress borrowing an SWA-cached slot must fire the SWA evict callback
    (plan §11.1) — else a later SWA hash-hit double-claims a compress-held slot."""
    c = UnifiedKvCoordinator(num_swa_base=3, n_csa_slots=4, n_hca_slots=4)
    evicted = []
    c.set_swa_evict_cb(lambda slot: evicted.append(slot))
    c.alloc_swa()
    s1 = c.alloc_swa()
    c.alloc_swa()
    c.free_swa(s1)  # slot 1 freed but SWA-cached
    assert c.is_free_swa(s1)
    got = [c.alloc_csa() for _ in range(8)]  # slot3 own + slot1 borrowed
    assert set([4, 5, 6, 7]).issubset(set(got))
    assert evicted == [1], evicted  # exactly slot 1 evicted, once
    assert not c.is_free_swa(s1)


def test_swa_hit_reclaims_before_borrow():
    c = UnifiedKvCoordinator(num_swa_base=3, n_csa_slots=4, n_hca_slots=4)
    evicted = []
    c.set_swa_evict_cb(lambda slot: evicted.append(slot))
    c.alloc_swa()
    s1 = c.alloc_swa()
    c.alloc_swa()
    c.free_swa(s1)
    c.claim_swa(s1)  # SWA cache-hit reclaim before compress borrows
    assert not c.is_free_swa(s1)
    assert c.has_free_csa(4) and not c.has_free_csa(5)  # slot 1 re-reserved
    got = [c.alloc_csa() for _ in range(4)]
    assert set(range(12, 16)) == set(got)  # only own region
    assert evicted == []


def test_has_free_swa_accurate_after_borrow():
    """has_free_swa must not count SWA slots compress has borrowed (admission)."""
    c = UnifiedKvCoordinator(num_swa_base=3, n_csa_slots=4, n_hca_slots=4)
    a = c.alloc_swa()
    c.alloc_swa()
    c.alloc_swa()
    assert not c.has_free_swa(1)
    c.free_swa(a)
    assert c.has_free_swa(1)
    [c.alloc_csa() for _ in range(8)]  # compress borrows a
    assert not c.has_free_swa(1), "borrowed slot must not count as SWA-free"
    for phys in range(a * 4, a * 4 + 4):
        c.free_csa(phys)
    assert c.has_free_swa(1), "reclaimable after compress frees it"
    assert c.alloc_swa() == a


# ----------------------------- sizing ----------------------------------- #
def test_unified_slot_counts_matches_flagoff_size():
    # n_csa*128 == swa_pages + num_blocks*32 when num_blocks % 4 == 0 (flag-on
    # tensor size == flag-off), and covers rounding otherwise.
    n_csa, n_hca = unified_slot_counts(num_swa_blocks=16, num_compress_blocks=64)
    assert n_csa == 16 + 64 // 4  # 32 slots -> 4096 rows == 16*128 + 64*32
    assert n_hca == 16 + 1  # 64 HCA blocks pack into 1 slot (128/slot)
    n_csa2, _ = unified_slot_counts(16, 65)  # non-multiple rounds up
    assert n_csa2 == 16 + 17
