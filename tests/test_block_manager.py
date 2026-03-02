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

    def test_at_block_boundary_needs_block(self, seq_factory):
        """seq_len=4, block_size=4 → at boundary, 1 new token needs 1 new block."""
        cfg = MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert bm.can_append(seq, num_new_tokens=1)

    def test_at_block_boundary_no_free(self, seq_factory):
        """seq_len=4, block_size=4, 0 free blocks → cannot append."""
        cfg = MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert not bm.can_append(seq, num_new_tokens=1)

    def test_multi_token_needs_two_blocks(self, seq_factory):
        """seq_len=7, block_size=4, num_new_tokens=4 → total 11, needs 3 blocks,
        2 allocated, need 1 more. With only 1 free block, should succeed."""
        cfg = MockConfig(num_kvcache_blocks=3, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7])
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert bm.can_append(seq, num_new_tokens=4)

    def test_multi_token_not_enough_free(self, seq_factory):
        """seq_len=5, block_size=4, num_new_tokens=4 → total 9, needs 3 blocks,
        2 allocated, need 1 more. With 0 free blocks, should fail."""
        cfg = MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4, 5])
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert not bm.can_append(seq, num_new_tokens=4)

    def test_multi_token_enough_free(self, seq_factory):
        """seq_len=7, block_size=4, num_new_tokens=4 → needs 1 more block.
        With enough free blocks, should succeed."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7])
        bm.allocate(seq)
        assert bm.can_append(seq, num_new_tokens=4)

    def test_multi_token_crosses_two_boundaries(self, seq_factory):
        """seq_len=5, block_size=4, num_new_tokens=4 → total 9, needs 3 blocks,
        but only 2 allocated. Need 1 more free block."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4, 5])
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        assert bm.can_append(seq, num_new_tokens=4)

    def test_multi_token_exact_fit(self, seq_factory):
        """seq_len=4, block_size=4, num_new_tokens=4 → total 8, needs 2 blocks.
        With exactly 1 free block, should succeed."""
        cfg = MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert bm.can_append(seq, num_new_tokens=4)

    def test_multi_token_one_short(self, seq_factory):
        """seq_len=4, block_size=4, num_new_tokens=5 → total 9, needs 3 blocks.
        With only 1 free block, should fail."""
        cfg = MockConfig(num_kvcache_blocks=2, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert not bm.can_append(seq, num_new_tokens=5)


class TestMayAppend:
    def test_no_new_block_within_boundary(self, block_manager, seq_factory):
        seq = seq_factory([1, 2])
        block_manager.allocate(seq)
        seq.append_token(3)
        block_manager.may_append(seq)
        assert len(seq.block_table) == 1

    def test_new_block_on_boundary_crossing(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4])
        block_manager.allocate(seq)
        seq.append_token(5)
        block_manager.may_append(seq)
        assert len(seq.block_table) == 2

    def test_block_size_1(self, seq_factory):
        """block_size=1: seq=[1,2] → 2 blocks. append(3) → seq_len=3.
        may_append(num_new_tokens=1) → needs ceil((3+1)/1) = 4 blocks."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=1)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2], block_size=1)
        bm.allocate(seq)
        seq.append_token(3)
        bm.may_append(seq)
        assert len(seq.block_table) == 4

    def test_multi_token_allocates_enough_blocks(self, seq_factory):
        """seq_len=5, block_size=4, num_new_tokens=4 → total 9, needs 3 blocks."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4, 5])
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        bm.may_append(seq, num_new_tokens=4)
        assert len(seq.block_table) == 3

    def test_multi_token_at_boundary(self, seq_factory):
        """seq_len=4, block_size=4, num_new_tokens=4 → total 8, needs 2 blocks."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert len(seq.block_table) == 1
        bm.may_append(seq, num_new_tokens=4)
        assert len(seq.block_table) == 2

    def test_multi_token_crosses_two_boundaries(self, seq_factory):
        """seq_len=4, block_size=4, num_new_tokens=5 → total 9, needs 3 blocks."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)
        assert len(seq.block_table) == 1
        bm.may_append(seq, num_new_tokens=5)
        assert len(seq.block_table) == 3

    def test_hash_registered_at_boundary(self, seq_factory):
        """When seq fills a block exactly, may_append should register its hash."""
        cfg = MockConfig(
            num_kvcache_blocks=10, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3])
        bm.allocate(seq)
        seq.append_token(4)
        bm.may_append(seq, num_new_tokens=1)
        last_block = bm.blocks[seq.block_table[0]]
        assert last_block.hash != -1
        assert last_block.hash in bm.hash_to_block_id

    def test_block_size_1_multi_token(self, seq_factory):
        """block_size=1: seq=[1,2] → 2 blocks. append(3) → seq_len=3.
        may_append(num_new_tokens=3) → needs ceil((3+3)/1) = 6 blocks."""
        cfg = MockConfig(num_kvcache_blocks=10, kv_cache_block_size=1)
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2], block_size=1)
        bm.allocate(seq)
        assert len(seq.block_table) == 2
        seq.append_token(3)
        bm.may_append(seq, num_new_tokens=3)
        assert len(seq.block_table) == 6


# ── Prefix caching during decode ──────────────────────────────────────────


class TestPrefixCachingDecode:
    def test_hash_registered_during_decode(self, seq_factory):
        """Block completed during decode should register its hash for reuse."""
        cfg = MockConfig(
            num_kvcache_blocks=10, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        seq = seq_factory([1, 2, 3])
        bm.allocate(seq)
        seq.append_token(4)
        bm.may_append(seq, num_new_tokens=1)

        block = bm.blocks[seq.block_table[0]]
        expected_hash = BlockManager.compute_hash([1, 2, 3, 4])
        assert block.hash == expected_hash
        assert bm.hash_to_block_id[expected_hash] == block.block_id

    def test_decode_block_reused_by_new_sequence(self, seq_factory):
        """A block completed and hashed during decode should be a cache hit
        for a new sequence with the same prefix."""
        cfg = MockConfig(
            num_kvcache_blocks=10, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)

        s1 = seq_factory([1, 2, 3])
        bm.allocate(s1)
        s1.append_token(4)
        bm.may_append(s1, num_new_tokens=1)
        bm.deallocate(s1)

        s2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s2)
        assert s2.num_cached_tokens == 4

    def test_multi_step_decode_builds_prefix(self, seq_factory):
        """Simulate multiple decode steps filling blocks, then verify
        a new sequence gets cache hits on the completed blocks."""
        cfg = MockConfig(
            num_kvcache_blocks=10, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)

        seq = seq_factory([1, 2, 3, 4])
        bm.allocate(seq)

        for tok in [5, 6, 7, 8]:
            seq.append_token(tok)
            bm.may_append(seq, num_new_tokens=1)

        bm.deallocate(seq)

        s2 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
        bm.allocate(s2)
        assert s2.num_cached_tokens == 8
