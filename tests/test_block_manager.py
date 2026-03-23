# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/block_manager.py — public API only

from atom.model_engine.block_manager import BlockManager
from conftest import MockConfig

# ── compute_hash ───────────────────────────────────────────────────────────


class TestComputeHash:
    def test_deterministic(self):
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([1, 2, 3, 4])
        assert h1 == h2

    def test_different_tokens_different_hash(self):
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([5, 6, 7, 8])
        assert h1 != h2

    def test_prefix_changes_hash(self):
        h1 = BlockManager.compute_hash([1, 2, 3, 4])
        h2 = BlockManager.compute_hash([1, 2, 3, 4], prefix=42)
        assert h1 != h2

    def test_hash_is_int(self):
        h = BlockManager.compute_hash([1, 2, 3, 4])
        assert isinstance(h, int)


# ── can_allocate ───────────────────────────────────────────────────────────


class TestCanAllocate:
    def test_can_allocate_when_free(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        assert block_manager.can_allocate(seq)

    def test_cannot_allocate_when_full(self, seq_factory):
        cfg = MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4])
        bm.allocate(s1)
        s2 = seq_factory([5, 6, 7, 8])
        assert not bm.can_allocate(s2)

    def test_can_allocate_multi_block(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5])
        assert block_manager.can_allocate(seq)


# ── allocate / deallocate ──────────────────────────────────────────────────


class TestAllocateDeallocate:
    def test_allocate_populates_block_table(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        block_manager.allocate(seq)
        assert len(seq.block_table) == 1

    def test_allocate_multi_block(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
        block_manager.allocate(seq)
        assert len(seq.block_table) == 3

    def test_deallocate_clears_seq(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        block_manager.allocate(seq)
        block_manager.deallocate(seq)
        assert seq.block_table == []
        assert seq.num_cached_tokens == 0

    def test_deallocate_restores_capacity(self, block_manager, seq_factory):
        s1 = seq_factory([1, 2, 3, 4])
        block_manager.allocate(s1)
        # Fill remaining capacity
        others = []
        for i in range(9):
            s = seq_factory([10 + i * 4, 11 + i * 4, 12 + i * 4, 13 + i * 4])
            block_manager.allocate(s)
            others.append(s)
        # Full — can't allocate more
        probe = seq_factory([100, 101, 102, 103])
        assert not block_manager.can_allocate(probe)
        # Deallocate one → can allocate again
        block_manager.deallocate(s1)
        assert block_manager.can_allocate(probe)


# ── Prefix caching ────────────────────────────────────────────────────────


class TestPrefixCaching:
    def test_prefix_cache_hit(self, block_manager_prefix, seq_factory):
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        block_manager_prefix.allocate(s1)
        block_manager_prefix.deallocate(s1)

        s2 = seq_factory([1, 2, 3, 4, 9, 10, 11, 12])
        block_manager_prefix.allocate(s2)
        assert s2.num_cached_tokens == 4

    def test_prefix_cache_miss_different_tokens(
        self, block_manager_prefix, seq_factory
    ):
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        block_manager_prefix.allocate(s1)
        block_manager_prefix.deallocate(s1)

        s2 = seq_factory([9, 10, 11, 12, 13, 14, 15, 16])
        block_manager_prefix.allocate(s2)
        assert s2.num_cached_tokens == 0

    def test_shared_prefix_doesnt_double_free(self, block_manager_prefix, seq_factory):
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        block_manager_prefix.allocate(s1)
        s2 = seq_factory([1, 2, 3, 4, 20, 21, 22, 23])
        block_manager_prefix.allocate(s2)

        # Deallocate s1 — s2 should still work fine
        block_manager_prefix.deallocate(s1)
        # s2 block_table still valid
        assert len(s2.block_table) == 2
        # Deallocate s2 — no crash
        block_manager_prefix.deallocate(s2)


# ── can_append / may_append ────────────────────────────────────────────────


class TestCanAppend:
    def test_can_append_within_block(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3])
        block_manager.allocate(seq)
        seq.append_token(4)
        assert block_manager.can_append(seq)

    def test_can_append_needs_new_block(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        block_manager.allocate(seq)
        seq.append_token(5)
        assert block_manager.can_append(seq)

    def test_cannot_append_no_free(self, seq_factory):
        cfg = MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        seq.append_token(5)
        assert not bm.can_append(seq)


class TestMayAppend:
    def test_no_new_block_within_boundary(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3])
        block_manager.allocate(seq)
        seq.append_token(4)
        block_manager.may_append(seq)
        assert len(seq.block_table) == 1

    def test_new_block_on_boundary_crossing(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        block_manager.allocate(seq)
        seq.append_token(5)
        block_manager.may_append(seq)
        assert len(seq.block_table) == 2

    def test_block_size_1(self, seq_factory):
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=1)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2], block_size=1)
        bm.allocate(seq)
        seq.append_token(3)
        bm.may_append(seq)
        assert len(seq.block_table) == 3


# ── Prefix caching: can_allocate with cache hits ─────────────────────────


class TestCanAllocateWithPrefixCaching:
    def test_can_allocate_accounts_for_cache_hits(self, seq_factory):
        """With 3 blocks total, allocate 2-block seq, deallocate, then a new
        2-block seq sharing block 1 should need only 1 free block."""
        cfg = MockConfig(
            num_kvcache_blocks=3, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        bm.deallocate(s1)  # blocks freed, hashes retained

        # Use up 2 of the 3 free blocks
        filler = seq_factory([50, 51, 52, 53, 60, 61, 62, 63])
        bm.allocate(filler)
        # Only 1 free block left; s2 needs 2 blocks but first is cached
        s2 = seq_factory([1, 2, 3, 4, 9, 10, 11, 12])
        assert bm.can_allocate(s2)

    def test_can_allocate_no_false_positive(self, seq_factory):
        """can_allocate should return False when even with cache hits
        there aren't enough free blocks."""
        cfg = MockConfig(
            num_kvcache_blocks=2, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        # 0 free blocks; new seq shares prefix but needs 1 new block
        s2 = seq_factory([1, 2, 3, 4, 9, 10, 11, 12])
        assert not bm.can_allocate(s2)


# ── Hash table cleanup ───────────────────────────────────────────────────


class TestHashTableCleanup:
    def test_stale_hash_entries_evicted_on_reuse(self, seq_factory):
        """When a cached block is reused for a different hash, the old
        hash_to_block_id entry should be cleaned up."""
        cfg = MockConfig(
            num_kvcache_blocks=2, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        h1 = bm.blocks[s1.block_table[0]].hash
        bm.deallocate(s1)

        # Allocate with completely different tokens — should overwrite blocks
        s2 = seq_factory([90, 91, 92, 93, 94, 95, 96, 97])
        bm.allocate(s2)
        # Old hash should no longer point to a valid block
        assert bm.hash_to_block_id.get(h1) != s2.block_table[0]

    def test_hash_table_bounded_growth(self, seq_factory):
        """hash_to_block_id should not grow beyond num_kvcache_blocks."""
        cfg = MockConfig(
            num_kvcache_blocks=4, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        for i in range(20):
            tokens = list(range(i * 4, i * 4 + 4))
            seq = seq_factory(tokens)
            if bm.can_allocate(seq):
                bm.allocate(seq)
                bm.deallocate(seq)
        assert len(bm.hash_to_block_id) <= cfg.num_kvcache_blocks


# ── can_append with multi-token decode (speculative decoding) ────────────


class TestCanAppendMultiToken:
    def test_can_append_multi_token_within_block(self, block_manager, seq_factory):
        """Appending 3 tokens that stay within the current block."""
        seq = seq_factory([1])
        block_manager.allocate(seq)
        seq.append_token(2)
        seq.append_token(3)
        assert block_manager.can_append(seq, num_new_tokens=3)

    def test_can_append_multi_token_crossing_boundary(self, seq_factory):
        """block_size=4, seq_len=14 (3.5 blocks=4 blocks allocated),
        appending 5 tokens crosses into block 5 — needs 1 new block."""
        cfg = MockConfig(num_kvcache_blocks=6, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory(list(range(14)))
        bm.allocate(seq)
        # seq_len=14, 4 blocks. Appending 5 tokens: positions 14..18 → need block 5
        for t in range(14, 19):
            seq.append_token(t)
        assert bm.can_append(seq, num_new_tokens=5)

    def test_cannot_append_multi_token_no_free(self, seq_factory):
        """block_size=4, 4 blocks total, seq fills 4 blocks (16 tokens),
        appending 5 tokens needs 2 new blocks but only 0 free."""
        cfg = MockConfig(num_kvcache_blocks=4, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory(list(range(14)))
        bm.allocate(seq)
        for t in range(14, 19):
            seq.append_token(t)
        assert not bm.can_append(seq, num_new_tokens=5)


# ── Prefix caching + preemption ──────────────────────────────────────────


class TestPrefixCachingPreemption:
    def test_preempt_and_reschedule_reuses_cache(self, seq_factory):
        """Preempted sequence re-discovers cache hits on re-allocation."""
        cfg = MockConfig(
            num_kvcache_blocks=10, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        # Simulate preemption
        bm.deallocate(s1)
        assert s1.num_cached_tokens == 0
        assert s1.block_table == []

        # Re-allocate — should get cache hits on both blocks
        s1_retry = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1_retry)
        assert s1_retry.num_cached_tokens == 8  # both blocks cached


# ── Edge cases ───────────────────────────────────────────────────────────


class TestPrefixCachingEdgeCases:
    def test_single_token_no_cache(self, seq_factory):
        """Single token seq (shorter than block_size) — hash is -1, no caching."""
        cfg = MockConfig(
            num_kvcache_blocks=4, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([42])
        bm.allocate(s1)
        bm.deallocate(s1)
        s2 = seq_factory([42])
        bm.allocate(s2)
        # Partial block → hash is -1 → no caching
        assert s2.num_cached_tokens == 0

    def test_exact_block_size_fully_cached(self, seq_factory):
        """Sequence with exactly block_size tokens — fully cached on reuse."""
        cfg = MockConfig(
            num_kvcache_blocks=4, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4])
        bm.allocate(s1)
        bm.deallocate(s1)
        s2 = seq_factory([1, 2, 3, 4])
        bm.allocate(s2)
        assert s2.num_cached_tokens == 4

    def test_free_block_ids_set_consistent(self, block_manager, seq_factory):
        """free_block_ids_set stays consistent through allocate/deallocate."""
        s1 = seq_factory([1, 2, 3, 4])
        block_manager.allocate(s1)
        initial_free = len(block_manager.free_block_ids_set)
        block_manager.deallocate(s1)
        assert len(block_manager.free_block_ids_set) == initial_free + 1
