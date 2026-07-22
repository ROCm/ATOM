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
        assert block_manager.can_allocate(seq) >= 0

    def test_cannot_allocate_when_full(self, seq_factory):
        cfg = MockConfig(num_kvcache_blocks=1, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4])
        bm.allocate(s1)
        s2 = seq_factory([5, 6, 7, 8])
        assert bm.can_allocate(s2) < 0

    def test_can_allocate_multi_block(self, block_manager, seq_factory):
        seq = seq_factory([1, 2, 3, 4, 5])
        assert block_manager.can_allocate(seq) >= 0


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
        assert block_manager.can_allocate(probe) < 0
        # Deallocate one → can allocate again
        block_manager.deallocate(s1)
        assert block_manager.can_allocate(probe) >= 0


# ── Prefix caching ────────────────────────────────────────────────────────


class TestPrefixCaching:
    def test_prefix_cache_hit(self, block_manager_prefix, seq_factory):
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        block_manager_prefix.allocate(s1)
        block_manager_prefix.hash_blocks(s1, s1.num_tokens - s1.num_cached_tokens)
        block_manager_prefix.deallocate(s1)

        s2 = seq_factory([1, 2, 3, 4, 9, 10, 11, 12])
        n = block_manager_prefix.can_allocate(s2)
        block_manager_prefix.allocate(s2, n)
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
        """can_allocate must charge BOTH the cache-miss block AND the
        cache-hit-on-free-pool block to the free-block budget, because the
        cached block still has to be claimed out of free_block_ids_set."""
        cfg = MockConfig(
            num_kvcache_blocks=4, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        bm.hash_blocks(s1, s1.num_tokens - s1.num_cached_tokens)
        bm.deallocate(s1)  # blocks freed, hashes retained

        # Use up 2 of the 4 free blocks with non-overlapping tokens
        filler = seq_factory([50, 51, 52, 53, 60, 61, 62, 63])
        bm.allocate(filler)
        # 2 free blocks left. s2 needs 2 blocks (1 cached + 1 fresh): exactly fits.
        s2 = seq_factory([1, 2, 3, 4, 9, 10, 11, 12])
        n = bm.can_allocate(s2)
        assert n == 1
        bm.allocate(s2, n)
        assert s2.num_cached_tokens == 4

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
        assert bm.can_allocate(s2) < 0


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
        bm.hash_blocks(s1, s1.num_tokens - s1.num_cached_tokens)
        h1 = bm.blocks[s1.block_table[0]].hash
        bm.deallocate(s1)

        # Allocate with completely different tokens — should overwrite blocks
        s2 = seq_factory([90, 91, 92, 93, 94, 95, 96, 97])
        bm.allocate(s2)
        bm.hash_blocks(s2, s2.num_tokens - s2.num_cached_tokens)
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
            n = bm.can_allocate(seq)
            if n >= 0:
                bm.allocate(seq, n)
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
        bm.hash_blocks(s1, s1.num_tokens - s1.num_cached_tokens)
        # Simulate preemption
        bm.deallocate(s1)
        assert s1.num_cached_tokens == 0
        assert s1.block_table == []

        # Re-allocate — first block is a cache hit; the last full block is
        # force-recomputed so prefill has at least one token to forward.
        s1_retry = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        n = bm.can_allocate(s1_retry)
        bm.allocate(s1_retry, n)
        assert s1_retry.num_cached_tokens == 4


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

    def test_exact_block_size_last_block_recomputed(self, seq_factory):
        """Single-block prompt: last full block is force-recomputed on reuse so
        prefill has at least one token to forward and produce logits."""
        cfg = MockConfig(
            num_kvcache_blocks=4, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4])
        bm.allocate(s1)
        bm.deallocate(s1)
        s2 = seq_factory([1, 2, 3, 4])
        bm.allocate(s2)
        assert s2.num_cached_tokens == 0

    def test_free_block_ids_set_consistent(self, block_manager, seq_factory):
        """free_block_ids_set stays consistent through allocate/deallocate."""
        s1 = seq_factory([1, 2, 3, 4])
        block_manager.allocate(s1)
        initial_free = len(block_manager.free_block_ids_set)
        block_manager.deallocate(s1)
        assert len(block_manager.free_block_ids_set) == initial_free + 1


# ── M2 paged-SWA dual pool ───────────────────────────────────────────────────

_MC = MockConfig


def _swa_bm(num_blocks=10, num_swa=10, bs=4, window=8, prefix=True):
    return BlockManager(
        _MC(
            num_kvcache_blocks=num_blocks,
            num_swa_blocks=num_swa,
            kv_cache_block_size=bs,
            swa_window_size=window,
            enable_prefix_caching=prefix,
        )
    )


class TestM2DualSwaPool:
    def test_disabled_by_default(self, block_manager, seq_factory):
        # No num_swa_blocks → swa disabled, swa_block_table stays empty.
        assert block_manager.swa_enabled is False
        seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        n = block_manager.can_allocate(seq)
        block_manager.allocate(seq, n)
        assert seq.block_table and seq.swa_block_table == []

    def test_allocate_parallel(self, seq_factory):
        bm = _swa_bm()
        toks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 3 blocks (bs=4)
        seq = seq_factory(toks)
        n = bm.can_allocate(seq)
        bm.allocate(seq, n)
        # allocate() is lazy: swa_block_table starts as -1 placeholders (same
        # length as block_table); the scheduler fills the chunk's window before
        # forward via ensure_for_tokens (here: single-shot whole prompt).
        assert len(seq.swa_block_table) == len(seq.block_table) == seq.num_blocks
        assert all(s < 0 for s in seq.swa_block_table)
        bm.swa.ensure_for_tokens(seq, 0, len(toks))
        # After ensure, the touched logical blocks hold real disjoint phys ids.
        assert all(s in bm.swa.used_block_ids for s in seq.swa_block_table)

    def test_windowonly_prefill_no_cross_request_swa_reuse(self, seq_factory):
        # Window-only prefill writes/publishes SWA only for the trailing
        # `window` tokens relative to the FULL prompt. Cross-request prefix
        # reuse needs the trailing window *at the hit boundary* (mid-prompt)
        # to be SWA-present — which window-only never writes — so bounded_hit
        # finds no complete in-window run and gates the whole hit to 0. This
        # trades away cross-request SWA prefix reuse for prefill scatter-write
        # cost (see paged-SWA perf notes); correctness is preserved (a denied
        # hit just recomputes, never reads stale SWA).
        bm = _swa_bm()
        toks = list(range(1, 13))  # 3 blocks
        s1 = seq_factory(toks)
        bm.allocate(s1, bm.can_allocate(s1))
        bm.swa.ensure_for_tokens(
            s1, 0, len(toks)
        )  # fill SWA (scheduler does this pre-forward)
        bm.hash_blocks(s1, len(toks))  # publish hashes (compressed + swa)
        s2 = seq_factory(toks)
        n2 = bm.can_allocate(s2)
        assert n2 == 0  # window-only: no cross-request SWA prefix reuse
        bm.allocate(s2, n2)
        # reused cached blocks share the SAME swa phys ids as s1.
        for i in range(n2):
            assert s2.swa_block_table[i] == s1.swa_block_table[i]
            assert bm.swa.blocks[s2.swa_block_table[i]].ref_count >= 2

    def test_intersection_stops_hit_when_swa_evicted(self, seq_factory):
        bm = _swa_bm()
        toks = list(range(1, 13))
        s1 = seq_factory(toks)
        bm.allocate(s1, bm.can_allocate(s1))
        bm.hash_blocks(s1, len(toks))
        bm.deallocate(s1)
        # Manually evict the SWA hash for block 0 (simulate window-free/evict)
        # while leaving the compressed hash intact → hit must stop at block 0.
        h0 = BlockManager.compute_hash(toks[0:4])
        bm.swa.hash_to_block_id.pop(h0, None)
        s2 = seq_factory(toks)
        n2 = bm.can_allocate(s2)
        assert n2 == 0  # compressed block 0 cached but SWA gone → no reuse

    def test_deallocate_frees_both_pools(self, seq_factory):
        bm = _swa_bm()
        toks = list(range(1, 13))
        seq = seq_factory(toks)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.swa.ensure_for_tokens(seq, 0, len(toks))  # materialize real SWA blocks
        used_swa = {s for s in seq.swa_block_table if s >= 0}
        bm.deallocate(seq)
        assert seq.swa_block_table == []
        assert used_swa and used_swa.issubset(bm.swa.free_block_ids_set)

    def test_swa_pool_exhaustion(self, seq_factory):
        # SWA pool smaller than compressed → admission bounded by SWA pool.
        bm = _swa_bm(num_blocks=10, num_swa=2, bs=4, prefix=False)
        seq = seq_factory(list(range(1, 13)))  # needs 3 blocks > 2 swa
        assert bm.can_allocate(seq) == -1


class TestM2WindowFreeing:
    def test_out_of_window_swa_freed_trailing_kept(self, seq_factory):
        bs, window = 4, 8
        bm = _swa_bm(num_blocks=40, num_swa=40, bs=bs, window=window, prefix=False)
        seq = seq_factory(list(range(1, 9)))  # 2 full blocks (len 8)
        bm.allocate(seq, bm.can_allocate(seq))
        # Decode forward to len 24 (window=8 → blocks covering <16 fall out).
        for t in range(8, 24):
            seq.append_token(1000 + t)
            bm.may_append(seq)
        # blocks i with (i+1)*bs <= 24-8=16 → i in {0,1,2} freed (cover 0..11)
        # wait: (i+1)*4<=16 → i+1<=4 → i<=3 → i in {0,1,2,3} (cover 0..15)
        freed = [i for i, s in enumerate(seq.swa_block_table) if s < 0]
        assert set(freed) == {0, 1, 2, 3}, f"freed={freed}"
        # trailing blocks (in window) still held
        kept = [s for s in seq.swa_block_table if s >= 0]
        assert all(s in bm.swa.used_block_ids for s in kept)
        # live SWA footprint bounded ~ window/bs (+ slack), not full seq length.
        assert len(kept) <= window // bs + 2

    def test_freed_swa_slot_returned_to_pool(self, seq_factory):
        bs, window = 4, 8
        bm = _swa_bm(num_blocks=40, num_swa=40, bs=bs, window=window, prefix=False)
        seq = seq_factory(list(range(1, 9)))
        bm.allocate(seq, bm.can_allocate(seq))
        free0 = len(bm.swa.free_block_ids_set)
        for t in range(8, 30):
            seq.append_token(1000 + t)
            bm.may_append(seq)
        # Many SWA blocks were allocated then window-freed → free pool recovered.
        assert len(bm.swa.free_block_ids_set) > free0 - (window // bs + 3)

    def test_compressed_untouched_by_window_freeing(self, seq_factory):
        bs, window = 4, 8
        bm = _swa_bm(num_blocks=40, num_swa=40, bs=bs, window=window, prefix=False)
        seq = seq_factory(list(range(1, 9)))
        bm.allocate(seq, bm.can_allocate(seq))
        for t in range(8, 24):
            seq.append_token(1000 + t)
            bm.may_append(seq)
        # Every compressed block stays held (no -1 sentinels, all in used set).
        assert all(b >= 0 for b in seq.block_table)
        assert all(b in bm.used_block_ids for b in seq.block_table)
