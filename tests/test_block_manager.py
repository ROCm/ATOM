# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/block_manager.py — public API only


from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager

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
        assert len(seq.block_table) == 0
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


class TestPublishLoadedPrefix:
    def test_dcp_uses_hash_block_granularity(self, seq_factory, monkeypatch):
        cfg = MockConfig(
            num_kvcache_blocks=6,
            kv_cache_block_size=4,
            decode_context_parallel_size=2,
            enable_prefix_caching=True,
        )
        bm = BlockManager(cfg)
        # This test targets loaded-prefix publication, so isolate allocation
        # from the GPU-only dcp_ops module used to calculate local block counts.
        monkeypatch.setattr(
            bm,
            "num_pool_blocks",
            lambda seq_len: (seq_len + bm.hash_block_size - 1) // bm.hash_block_size,
        )
        loaded = seq_factory(list(range(16)))
        bm.allocate(loaded)

        assert bm.publish_loaded_prefix(loaded, start_token=0, end_token=8) == 8
        loaded_block = bm.kv.block(loaded.block_table[0])
        assert list(loaded_block.token_ids) == list(range(8))

        probe = seq_factory(list(range(8)) + list(range(100, 108)))
        num_cached_blocks = bm.can_allocate(probe)
        assert num_cached_blocks == 1
        bm.allocate(probe, num_cached_blocks)
        assert probe.num_cached_tokens == 8

    def test_skips_when_predecessor_unhashed(self, seq_factory):
        """A gap before the loaded range logs an error and skips."""
        cfg = MockConfig(num_kvcache_blocks=16, enable_prefix_caching=True)
        bm = BlockManager(cfg)
        tokens = list(range(24))

        loaded = seq_factory(tokens)
        bm.allocate(loaded)
        # Blocks 0-1 unhashed — publish_loaded_prefix should skip, not crash.
        assert bm.publish_loaded_prefix(loaded, start_token=8, end_token=16) == 0


class TestHashChainGapSkip:
    def test_hash_blocks_skips_on_gap(self, block_manager_prefix, seq_factory):
        """A gap in the hash chain must skip, not mint false-root hashes."""
        bm = block_manager_prefix
        tokens = list(range(16))

        seq = seq_factory(tokens)
        bm.allocate(seq)
        seq.num_cached_tokens = 8
        bm.hash_blocks(seq, 8)

        # Blocks 0-1 never hashed → hash_blocks should skip → blocks stay -1.
        for b in seq.block_table:
            assert bm.kv.block(b).hash == -1

    def test_no_skip_when_chain_intact(self, block_manager_prefix, seq_factory):
        bm = block_manager_prefix
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)
        bm.hash_blocks(seq, 16)
        for b in seq.block_table:
            assert bm.kv.block(b).hash != -1

    def test_hash_blocks_clamps_to_block_table(self, block_manager_prefix, seq_factory):
        bm = block_manager_prefix
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)
        seq.block_table.pop()
        bm.hash_blocks(seq, 16)  # must not IndexError


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
        cached block still has to be claimed off the free list."""
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
        content-hash entry should be cleaned up."""
        cfg = MockConfig(
            num_kvcache_blocks=2, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        s1 = seq_factory([1, 2, 3, 4, 5, 6, 7, 8])
        bm.allocate(s1)
        bm.hash_blocks(s1, s1.num_tokens - s1.num_cached_tokens)
        h1 = bm.kv.block(s1.block_table[0]).hash
        bm.deallocate(s1)

        # Allocate with completely different tokens — should overwrite blocks
        s2 = seq_factory([90, 91, 92, 93, 94, 95, 96, 97])
        bm.allocate(s2)
        bm.hash_blocks(s2, s2.num_tokens - s2.num_cached_tokens)
        # Old hash should no longer point to a valid block
        assert bm.kv.lookup(h1) != s2.block_table[0]

    def test_hash_table_bounded_growth(self, seq_factory):
        """The content index should not grow beyond num_kvcache_blocks."""
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
        assert bm.kv.num_indexed <= cfg.num_kvcache_blocks


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
        assert len(s1.block_table) == 0

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

    def test_free_count_consistent(self, block_manager, seq_factory):
        """The free count stays consistent through allocate/deallocate."""
        s1 = seq_factory([1, 2, 3, 4])
        block_manager.allocate(s1)
        initial_free = block_manager.kv.num_free
        block_manager.deallocate(s1)
        assert block_manager.kv.num_free == initial_free + 1


# ── decode-side block hashing ──────────────────────────────────────────────


class TestDecodeBlockHashing:
    """Generated blocks must enter the prefix cache, not just prompt blocks.

    The multi-turn case: turn 2's prompt is turn 1's prompt plus turn 1's
    answer. Hashing only the prompt caps every follow-up hit at the original
    prompt length, no matter how much of the conversation is still resident.
    """

    BS = 4

    def _bm(self, **overrides):
        cfg = {
            "num_kvcache_blocks": 100,
            "kv_cache_block_size": self.BS,
            "enable_prefix_caching": True,
            "max_model_len": 256,
        }
        cfg.update(overrides)
        return BlockManager(MockConfig(**cfg))

    def _run_turn(self, bm, seq, generated):
        """Prefill `seq`, then append `generated` and hash what filled up."""
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)
        for token in generated:
            seq.append_token(token)
            bm.may_append(seq)
        bm.hash_decode_blocks(seq, seq.num_tokens)

    def test_followup_turn_reuses_the_generated_blocks(self, seq_factory):
        bm = self._bm()
        prompt = list(range(8))  # 2 blocks
        generated = list(range(100, 112))  # 3 more blocks
        self._run_turn(bm, seq_factory(prompt), generated)

        # Turn 2 replays the whole conversation as its prompt: 20 tokens, 5
        # blocks. can_allocate never hands back the last block (the seq has to
        # forward something), so a full hit is 4.
        followup = seq_factory(prompt + generated)
        assert bm.can_allocate(followup) == 4

    def test_prompt_only_hashing_would_stop_at_the_prompt(self, seq_factory):
        """Pins what the fix buys: without it the hit stops at 2 blocks."""
        bm = self._bm()
        prompt = list(range(8))
        generated = list(range(100, 112))
        seq = seq_factory(prompt)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        for token in generated:
            seq.append_token(token)
            bm.may_append(seq)
        # Deliberately skip hash_decode_blocks — the pre-fix behaviour.
        followup = seq_factory(prompt + generated)
        assert bm.can_allocate(followup) == 2

    def test_uncommitted_tail_is_not_hashed(self, seq_factory):
        """Only whole blocks below the committed watermark may be published.

        The speculative-decoding hazard: tokens above the committed length can
        still be rewritten next step, and their KV with them.
        """
        bm = self._bm()
        prompt = list(range(8))
        seq = seq_factory(prompt)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        for token in range(100, 112):
            seq.append_token(token)
            bm.may_append(seq)
        # Commit only the first generated block; the rest is still in flight.
        bm.hash_decode_blocks(seq, 12)
        assert seq.num_hashed_tokens == 12

        followup = seq_factory(prompt + list(range(100, 112)))
        assert bm.can_allocate(followup) == 3  # 2 prompt + 1 committed

    def test_watermark_advances_past_the_prompt_boundary_block(self, seq_factory):
        """The block straddling prompt-end is hashed once generation fills it."""
        bm = self._bm()
        prompt = list(range(10))  # 2 whole blocks + 2 tokens
        seq = seq_factory(prompt)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        assert seq.num_hashed_tokens == 8  # block 2 is half full

        for token in range(100, 106):
            seq.append_token(token)
            bm.may_append(seq)
        bm.hash_decode_blocks(seq, seq.num_tokens)
        assert seq.num_hashed_tokens == 16

    def test_deallocate_clears_the_watermark(self, seq_factory):
        bm = self._bm()
        seq = seq_factory(list(range(8)))
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        assert seq.num_hashed_tokens == 8
        bm.deallocate(seq)
        # Preemption frees through here and re-prefills from scratch; a stale
        # watermark would make hash_decode_blocks index a block table that no
        # longer exists.
        assert seq.num_hashed_tokens == 0

    def test_no_op_without_prefix_caching(self, seq_factory):
        bm = self._bm(enable_prefix_caching=False)
        seq = seq_factory(list(range(8)))
        bm.allocate(seq, bm.can_allocate(seq))
        for token in range(100, 108):
            seq.append_token(token)
            bm.may_append(seq)
        bm.hash_decode_blocks(seq, seq.num_tokens)
        assert seq.num_hashed_tokens == 0
        assert not bm.kv.num_indexed


# ── register_received_prefix (PD consumer) ─────────────────────────────────


class TestRegisterReceivedPrefix:
    @staticmethod
    def _dcp_block_manager(monkeypatch):
        cfg = MockConfig(
            num_kvcache_blocks=12,
            kv_cache_block_size=4,
            decode_context_parallel_size=2,
            enable_prefix_caching=True,
        )
        bm = BlockManager(cfg)
        # Allocation normally asks dcp_ops for each rank's local token count.
        # These scheduler tests only need the equivalent virtual-block count.
        monkeypatch.setattr(
            bm,
            "num_pool_blocks",
            lambda seq_len: (seq_len + bm.hash_block_size - 1) // bm.hash_block_size,
        )
        return bm

    def test_registers_full_prompt_blocks_enabling_next_turn_hit(
        self, block_manager_prefix, seq_factory
    ):
        bm = block_manager_prefix
        # PD consumer: a remote-prefill request whose full prompt KV arrived via
        # RDMA. It never ran a prefill forward, so its blocks are unhashed until
        # register_received_prefix publishes them.
        a = seq_factory(list(range(1, 13)))  # 12 tokens, bs=4 -> 3 full blocks
        bm.allocate(a)
        registered = bm.register_received_prefix(a)
        assert registered == 3
        # Next turn with the same prefix now hits locally (last block excluded).
        b = seq_factory(list(range(1, 13)))
        assert bm.can_allocate(b) == 2

    def test_noop_without_prefix_caching(self, block_manager, seq_factory):
        a = seq_factory(list(range(1, 13)))
        block_manager.allocate(a)
        assert block_manager.register_received_prefix(a) == 0

    def test_excludes_trailing_partial_block(self, block_manager_prefix, seq_factory):
        bm = block_manager_prefix
        a = seq_factory(list(range(1, 11)))  # 10 tokens, bs=4 -> 2 full + partial
        bm.allocate(a)
        assert bm.register_received_prefix(a) == 2

    def test_dcp_registers_at_hash_block_granularity(self, seq_factory, monkeypatch):
        bm = self._dcp_block_manager(monkeypatch)
        assert bm.block_size == 4
        assert bm.hash_block_size == 8

        seq = seq_factory(list(range(20)))
        bm.allocate(seq)

        assert bm.register_received_prefix(seq) == 2
        assert seq.num_hashed_tokens == 16
        assert seq.prefix_hashes_published is True
        assert list(bm.kv.block(seq.block_table[0]).token_ids) == list(range(8))
        assert list(bm.kv.block(seq.block_table[1]).token_ids) == list(range(8, 16))
        assert bm.kv.block(seq.block_table[2]).hash == -1

    def test_dcp_registers_only_suffix_after_local_cache_hit(
        self, seq_factory, monkeypatch
    ):
        bm = self._dcp_block_manager(monkeypatch)

        cached = seq_factory(list(range(16)))
        bm.allocate(cached)
        bm.hash_blocks(cached, cached.num_prompt_tokens)
        bm.deallocate(cached)

        received = seq_factory(list(range(32)))
        num_cached_blocks = bm.can_allocate(received)
        assert num_cached_blocks == 2
        bm.allocate(received, num_cached_blocks)
        cached_hashes = [
            bm.kv.block(block_id).hash for block_id in received.block_table[:2]
        ]
        computed_token_groups = []
        compute_hash = bm.compute_hash

        def tracked_compute_hash(token_ids, prefix=-1):
            computed_token_groups.append(list(token_ids))
            return compute_hash(token_ids, prefix)

        monkeypatch.setattr(bm, "compute_hash", tracked_compute_hash)

        assert bm.register_received_prefix(received) == 2
        assert received.num_cached_tokens == 16
        assert received.num_hashed_tokens == 32
        assert computed_token_groups == [list(range(16, 24)), list(range(24, 32))]
        assert [
            bm.kv.block(block_id).hash for block_id in received.block_table[:2]
        ] == cached_hashes
        assert all(
            bm.kv.block(block_id).hash != -1 for block_id in received.block_table[2:]
        )
