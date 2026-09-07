# SPDX-License-Identifier: MIT
# Tests for atom/model_engine/block_manager.py — public API only

import logging

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


# ── the widened claim behind a joint boundary ──────────────────────────────


class TestJointClaimReusesResidentBlocks:
    """`can_allocate` returns one number, and the request needs two.

    What it may call *cached* is gated on a resumable state behind it. What it
    may point its block table at is every block the prefix walk matched -- the
    state gate cut resumability, not residency. Without the second number the
    KV leg pays LMCache to resend blocks the pool is already holding, and the
    reply lands in *fresh* blocks, so HBM ends up with two copies of the same
    prefix.
    """

    def _resident_prefix(self, seq_factory):
        """A 4-block prefix published in the index, then released."""
        cfg = MockConfig(
            num_kvcache_blocks=32, kv_cache_block_size=4, enable_prefix_caching=True
        )
        bm = BlockManager(cfg)
        tokens = list(range(20))
        first = seq_factory(tokens)
        bm.allocate(first, 0)
        bm.hash_blocks(first, len(tokens))
        resident = list(first.block_table[:4])
        bm.deallocate(first)
        return bm, tokens, resident

    def test_without_a_joint_boundary_the_claim_stops_at_the_gated_hit(
        self, seq_factory
    ):
        bm, tokens, resident = self._resident_prefix(seq_factory)
        seq = seq_factory(tokens)
        bm.allocate(seq, 2)
        assert list(seq.block_table[:2]) == resident[:2]
        # Blocks 2 and 3 are resident and match, but nothing above the hit will
        # ever be treated as computed, so claiming them would pin blocks the
        # forward is about to overwrite.
        assert seq.block_table[2] not in resident[2:]
        assert seq.num_cached_tokens == 8

    def test_a_joint_boundary_widens_the_claim_without_widening_the_hit(
        self, seq_factory
    ):
        bm, tokens, resident = self._resident_prefix(seq_factory)
        seq = seq_factory(tokens)
        # The state gate cut the hit to 2 blocks; the walk had matched 4.
        seq.offload_joint.claim_tokens = 16
        bm.allocate(seq, 2)

        # All four resident blocks are reused -- that is the transfer the KV
        # leg no longer has to make, and the second copy that no longer lands.
        assert list(seq.block_table[:4]) == resident
        # ...and the request still only calls two of them cached, so a failed
        # leg leaves the forward recomputing rather than skipping.
        assert seq.num_cached_tokens == 8

    def test_the_widened_claim_still_gives_every_position_a_block(self, seq_factory):
        bm, tokens, _ = self._resident_prefix(seq_factory)
        seq = seq_factory(tokens)
        seq.offload_joint.claim_tokens = 16
        bm.allocate(seq, 2)
        assert len(seq.block_table) == 5
        assert len(set(seq.block_table)) == 5


class TestDisownClaimedPrefix:
    """A disown recomputes from token 0 while the block table still points at
    hash-indexed blocks a *live* peer is decoding out of. Recomputing writes KV
    in place into those shared blocks and tears the peer's values (review
    finding #2 -- the precision-corruption main suspect). `disown_claimed_prefix`
    hands each shared claimed block back and drops a fresh private block into the
    same slot, so the forward can only overwrite blocks this seq owns alone.
    """

    def _shared_prefix(self, seq_factory, num_kvcache_blocks=32):
        """Two live seqs sharing a 4-block claimed prefix (ref_count == 2)."""
        cfg = MockConfig(
            num_kvcache_blocks=num_kvcache_blocks,
            kv_cache_block_size=4,
            enable_prefix_caching=True,
        )
        bm = BlockManager(cfg)
        tokens = list(range(20))
        holder = seq_factory(tokens)
        bm.allocate(holder, 0)
        bm.hash_blocks(holder, len(tokens))
        # A second live seq claims the same 4 resident blocks via the joint
        # boundary -- now every prefix block is held at ref_count == 2.
        seq = seq_factory(tokens)
        seq.offload_joint.claim_tokens = 16
        bm.allocate(seq, 2)
        shared = list(seq.block_table[:4])
        assert all(bm.kv.block(b).ref_count == 2 for b in shared)
        return bm, holder, seq, shared

    def test_disown_privatizes_shared_blocks_in_place(self, seq_factory):
        bm, holder, seq, shared = self._shared_prefix(seq_factory)
        table_before = list(seq.block_table)

        assert bm.disown_claimed_prefix(seq) is True

        # Same length, same tail slot -- only the shared prefix slots changed.
        assert len(seq.block_table) == len(table_before)
        assert seq.block_table[4] == table_before[4]
        # The four claimed slots now hold fresh private blocks: none is one of
        # the shared canonical blocks, and each is hash == -1 (private).
        for i in range(4):
            assert seq.block_table[i] not in shared
            assert bm.kv.block(seq.block_table[i]).hash == -1
        # The peer still holds every shared block, now back at ref_count == 1.
        assert list(holder.block_table[:4]) == shared
        for b in shared:
            assert bm.kv.block(b).ref_count == 1

    def test_disown_leaves_no_leak_after_deallocate(self, seq_factory):
        bm, holder, seq, _ = self._shared_prefix(seq_factory)
        bm.disown_claimed_prefix(seq)
        bm.deallocate(seq)
        bm.deallocate(holder)
        assert bm.kv.num_used == 0

    def test_disown_fails_when_pool_cannot_back_private_copies(self, seq_factory):
        # 6 blocks: holder takes 4, seq's fresh tail takes 1, leaving 1 free --
        # fewer than the 4 private copies the disown needs, so it must refuse
        # rather than silently reuse the shared blocks.
        bm, _holder, seq, shared = self._shared_prefix(
            seq_factory, num_kvcache_blocks=6
        )
        assert bm.kv.has_free(4) is False

        assert bm.disown_claimed_prefix(seq) is False

        # Refusal is total: the table is untouched and the peer's blocks are
        # still shared, so the caller can safely deallocate + requeue.
        assert list(seq.block_table[:4]) == shared
        for b in shared:
            assert bm.kv.block(b).ref_count == 2

    def test_disown_is_a_noop_without_prefix_caching(self, seq_factory):
        cfg = MockConfig(num_kvcache_blocks=8, kv_cache_block_size=4)
        bm = BlockManager(cfg)
        seq = seq_factory(list(range(16)))
        bm.allocate(seq)
        table_before = list(seq.block_table)
        assert bm.disown_claimed_prefix(seq) is True
        assert list(seq.block_table) == table_before


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


# ── state_offload disabled by default ─────────────────────────────────────


def test_state_offload_is_none_when_tier_is_off(block_manager):
    """No `lmcache_offload` connector means no tier, whatever the ring size:
    the connector is the feature's only on/off switch."""
    assert block_manager.state_offload is None


# ── the index is built only where something can report to it ─────────────

# The layout is part of the capability, not an incidental detail: only K3
# offloads per-request state, so an offload connector on a dense model hosts no
# tier however it is configured.
_OFFLOAD_KVC = {
    "kv_connector": "lmcache_offload",
    "kv_role": "offload",
    "offload_layout": "kimi_k3",
}


def _bm_with_state_tier(monkeypatch, kv_transfer_config):
    cfg = MockConfig(
        enable_prefix_caching=True,
        kv_transfer_config=kv_transfer_config,
        pool_entries={"state": 4},
        pool_entries_per_req={"state": 1},
    )
    return BlockManager(cfg)


def test_no_ring_is_installed_without_a_connector_that_hosts_the_tier(
    monkeypatch, caplog
):
    """The silent-permanent failure. A load is resolved only by a worker
    report, and without a hosting connector no report ever comes -- so an index
    built here would offer loads that park their request forever. Refuse to
    build it."""
    with caplog.at_level(logging.WARNING, logger="atom"):
        bm = _bm_with_state_tier(monkeypatch, None)

    assert bm.state_offload is None


def test_a_connector_that_cannot_host_the_tier_gets_no_index_either(
    monkeypatch, caplog
):
    """Same failure, one step subtler: moriio is a KV connector, so a plain
    truthiness test on kv_transfer_config would build the index, but only
    lmcache_offload's worker half ever builds a `_state_tier`."""
    with caplog.at_level(logging.WARNING, logger="atom"):
        bm = _bm_with_state_tier(monkeypatch, {"kv_connector": "moriio"})

    assert bm.state_offload is None


def test_the_index_is_built_for_the_offload_connector(monkeypatch):
    bm = _bm_with_state_tier(monkeypatch, _OFFLOAD_KVC)
    assert bm.state_offload is not None
    # #2045 moved the checkpoint into the KV pool, so the tier no longer
    # reaches into the slot pool at all -- that coupling was the staging ring.
    assert not any(hasattr(cache, "offload") for cache in bm.state_caches)


def test_a_dense_layout_hosts_no_state_tier_however_it_is_configured(monkeypatch):
    """The name check said yes to every `lmcache_offload`. Only K3's worker
    half builds a `StateOffloadTier`: dense has no per-request state and DSV4
    keeps its own in the SLOT sidecar."""
    for layout in ("dense", "hybrid"):
        bm = _bm_with_state_tier(
            monkeypatch, {**_OFFLOAD_KVC, "offload_layout": layout}
        )
        assert bm.state_offload is None, layout
        assert layout in bm.state_tier_capability.reason


def test_pipeline_parallelism_hosts_no_state_tier(monkeypatch):
    """The worker refuses PP outright -- `CacheEngineKey` has no PP component,
    so two stages at one TP rank would overwrite each other. The engine used to
    build an index anyway and emit stores with nowhere to go."""
    cfg = MockConfig(
        enable_prefix_caching=True,
        kv_transfer_config=_OFFLOAD_KVC,
        pool_entries={"state": 4},
        pool_entries_per_req={"state": 1},
    )
    cfg.pipeline_parallel_size = 2
    bm = BlockManager(cfg)
    assert bm.state_offload is None
    assert "pipeline_parallel_size=2" in bm.state_tier_capability.reason


def test_a_producer_role_stores_but_never_votes_for_a_load(monkeypatch):
    bm = _bm_with_state_tier(monkeypatch, {**_OFFLOAD_KVC, "kv_role": "kv_producer"})
    assert bm.state_offload is not None
    assert bm.state_offload.can_store and not bm.state_offload.can_load
    bm.state_offload.note_stored(11)
    assert bm.state_offload.request_load("r1", 11) is False


def test_a_consumer_role_votes_but_never_hands_over_a_store(monkeypatch):
    bm = _bm_with_state_tier(monkeypatch, {**_OFFLOAD_KVC, "kv_role": "kv_consumer"})
    assert bm.state_offload is not None
    assert bm.state_offload.can_load and not bm.state_offload.can_store
    assert bm.take_state_stores(4) == []


def test_a_multi_listing_two_offload_connectors_is_refused(monkeypatch):
    """The tier's bytes ride one connector's worker half and its completions
    ride that connector's `get_finished`, so two providers would each hold half
    an answer. Refused *loudly* at startup, not degraded to a no-tier fallback:
    the silent fallback left the KV load path live over a block table the other
    sub was writing (review round 5, finding 0)."""
    try:
        _bm_with_state_tier(
            monkeypatch,
            {
                "kv_connector": "multi",
                "connectors": [_OFFLOAD_KVC, dict(_OFFLOAD_KVC)],
            },
        )
    except ValueError as exc:
        assert "offload connectors" in str(exc)
        assert "at most one" in str(exc)
    else:
        raise AssertionError("two offload sub-connectors must raise at startup")


def test_the_index_is_built_for_a_multi_that_lists_the_offload_backend(
    monkeypatch,
):
    bm = _bm_with_state_tier(
        monkeypatch,
        {
            "kv_connector": "multi",
            "connectors": [{"kv_connector": "moriio"}, _OFFLOAD_KVC],
        },
    )
    assert bm.state_offload is not None


def test_a_multi_without_the_offload_backend_gets_no_index(monkeypatch):
    bm = _bm_with_state_tier(
        monkeypatch,
        {"kv_connector": "multi", "connectors": [{"kv_connector": "moriio"}]},
    )
    assert bm.state_offload is None


# ── a store whose source was reclaimed must not be indexed ─────────────────


class TestReclaimedStoresAreNotIndexed:
    """`reclaim_stale_offload_pins` cannot prove the worker stopped reading:
    a K3 state store bypasses `CacheEngine.store()` and gathers ATOM PAGE units
    directly, so LMCache's GPU-source pin monitor does not cover them. If the
    reader had not stopped, the pool may have handed those units to another
    request whose writes the gather picked up -- a CPU image that is a mix of
    two prefixes under the first one's hash."""

    class _Coordinator:
        def __init__(self, reclaimed):
            self._reclaimed = reclaimed
            self.settled = []
            self.released = []

        def was_reclaimed(self, op):
            return op in self._reclaimed

        def settle_offload_store(self, op):
            self.settled.append(op)

        def release_offload_store_source(self, op):
            self.released.append(op)

    def _bm(self, reclaimed=()):
        from atom.model_engine.state_offload import StateOffloadIndex

        bm = object.__new__(BlockManager)
        bm.paged_state_checkpoints = self._Coordinator(set(reclaimed))
        bm.state_offload = StateOffloadIndex()
        return bm

    def test_a_normal_store_is_indexed(self):
        from atom.kv_transfer.disaggregation.types import StateStoreOperationId

        bm = self._bm()
        op = StateStoreOperationId(11, 1)
        bm.settle_state_store(op, ok=True)
        assert 11 in bm.state_offload.hashes
        assert bm.state_offload.stores_completed == 1
        assert bm.state_offload.stores_untrusted == 0

    def test_a_reclaimed_store_reporting_success_is_forfeited(self):
        from atom.kv_transfer.disaggregation.types import StateStoreOperationId

        op = StateStoreOperationId(11, 1)
        bm = self._bm(reclaimed=[op])
        bm.settle_state_store(op, ok=True)
        assert 11 not in bm.state_offload.hashes, "voting for it is wrong output"
        assert bm.state_offload.stores_completed == 0
        assert bm.state_offload.stores_untrusted == 1

    def test_the_units_go_back_either_way(self):
        from atom.kv_transfer.disaggregation.types import StateStoreOperationId

        op = StateStoreOperationId(11, 1)
        bm = self._bm(reclaimed=[op])
        bm.settle_state_store(op, ok=True)
        assert bm.paged_state_checkpoints.settled == [op]

    def test_the_source_release_is_what_normally_returns_them(self):
        from atom.kv_transfer.disaggregation.types import StateStoreOperationId

        bm = self._bm()
        op = StateStoreOperationId(11, 1)
        bm.release_state_store_source(op)
        # Phase one hands the units back but leaves the pin in place, so the
        # store still reads as dispatched-but-unreported until its report lands.
        assert bm.paged_state_checkpoints.released == [op]
        assert bm.paged_state_checkpoints.settled == [], "the pin is not retired"
        # ...and it does not touch the index, which the store report owns.
        assert bm.state_offload.hashes == set()


# ── the orphan load slot lives in one dict of (slot, stamp) ────────────────


class TestOrphanLoadSlotSingleDict:
    """A load slot whose request was torn down before its bytes landed is
    parked in one dict of `(slot, stamp)`. It used to live across two parallel
    dicts keyed the same way, mutated in pairs at four sites -- a shape where an
    overwrite or a half-applied pop could desync them, so the reconciler
    (iterating the stamp dict) and the release (reading the slot dict) could
    disagree and strand a slot off the pool free list. One dict makes that
    unrepresentable: an entry is present with its slot and stamp together, or it
    is gone."""

    class _Pool:
        def __init__(self):
            self.released = []

        def release(self, slot):
            self.released.append(slot)

    class _Index:
        def __init__(self):
            self.abandoned, self.completed, self.failed = [], [], []

        def abandon_load(self, req_id):
            self.abandoned.append(req_id)

        def complete_load(self, req_id):
            self.completed.append(req_id)

        def fail_load(self, req_id):
            self.failed.append(req_id)

    def _bm(self):
        bm = object.__new__(BlockManager)
        bm.state = self._Pool()
        bm.state_offload = self._Index()
        bm._orphan_load_slots = {}
        bm._orphan_load_slots_reclaimed = 0
        return bm

    def test_settle_frees_the_parked_slot_and_clears_the_entry(self):
        from time import monotonic

        bm = self._bm()
        bm._orphan_load_slots["r"] = [(7, monotonic())]
        bm.settle_state_load("r", ok=True)
        assert bm.state.released == [7]
        assert "r" not in bm._orphan_load_slots
        assert bm.state_offload.completed == ["r"]

    def test_abandon_frees_the_parked_slot(self):
        from time import monotonic

        bm = self._bm()
        bm._orphan_load_slots["r"] = [(7, monotonic())]
        bm.abandon_state_load("r")
        assert bm.state.released == [7]
        assert "r" not in bm._orphan_load_slots

    def test_reconcile_frees_a_slot_whose_report_never_came(self):
        from time import monotonic

        bm = self._bm()
        bm._orphan_load_slots["r"] = [(7, monotonic())]
        assert bm.reconcile_orphan_load_slots(timeout_s=1e-9) == 1
        assert bm.state.released == [7]
        assert bm._orphan_load_slots_reclaimed == 1
        assert "r" not in bm._orphan_load_slots

    def test_reconcile_spares_a_slot_still_inside_its_window(self):
        from time import monotonic

        bm = self._bm()
        bm._orphan_load_slots["r"] = [(7, monotonic())]
        assert bm.reconcile_orphan_load_slots(timeout_s=3600) == 0
        assert bm.state.released == []
        assert "r" in bm._orphan_load_slots

    def test_a_late_report_after_reconcile_releases_nothing(self):
        from time import monotonic

        bm = self._bm()
        bm._orphan_load_slots["r"] = [(7, monotonic())]
        bm.reconcile_orphan_load_slots(timeout_s=1e-9)
        bm.settle_state_load("r", ok=True)  # the report finally arrives
        assert bm.state.released == [7], "freed once by reconcile, not twice"

    def test_a_preempt_readmit_parks_the_same_id_twice_without_dropping_a_slot(
        self,
    ):
        # `seq.id` is per-request, not per-admission: a preempt/re-admit can park
        # the same id twice while the first load is still in flight. A single
        # tuple would overwrite -- leaking slot 7 forever and, worse, releasing
        # slot 8 (still being written) when the first report lands. Appended and
        # released oldest-first, each report frees the slot its load actually
        # settled.
        from time import monotonic

        bm = self._bm()
        first = (7, monotonic())
        second = (8, monotonic())
        bm._orphan_load_slots["r"] = [first]  # first admission
        bm._orphan_load_slots["r"].append(second)  # re-admission, load in flight

        bm.settle_state_load("r", ok=True)  # first (oldest) load reports
        assert bm.state.released == [7]
        assert bm._orphan_load_slots["r"] == [second]  # slot 8 still parked

        bm.settle_state_load("r", ok=True)  # second load reports
        assert bm.state.released == [7, 8]
        assert "r" not in bm._orphan_load_slots  # entry cleared when empty

    def test_reconcile_expires_one_admission_and_keeps_the_fresh_one(self):
        # Each parked admission ages on its own stamp: a stale one is reclaimed
        # even while a newer park for the same id is still inside the window.
        from time import monotonic

        bm = self._bm()
        now = monotonic()
        bm._orphan_load_slots["r"] = [(7, now - 3600), (8, now)]  # old, fresh

        assert bm.reconcile_orphan_load_slots(timeout_s=1.0) == 1
        assert bm.state.released == [7]
        assert bm._orphan_load_slots["r"] == [(8, now)]
        assert bm._orphan_load_slots_reclaimed == 1


# ── the LMCache chunk probe is gated on the capability ─────────────────────


class TestJointChunkProbeIsGated:
    """Reading the chunk size imports LMCache, which is an optional dependency.
    Doing it unconditionally meant every engine without an offload connector
    logged a full `ModuleNotFoundError` traceback at WARNING on startup, for a
    number it has no use for -- and any caplog assertion downstream inherited
    that noise."""

    def _bm(self, kv_transfer_config):
        cfg = MockConfig(
            enable_prefix_caching=True,
            kv_transfer_config=kv_transfer_config,
            pool_entries={"state": 4},
            pool_entries_per_req={"state": 1},
        )
        return BlockManager(cfg)

    def test_an_engine_with_no_tier_does_not_probe_or_warn(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="atom"):
            bm = self._bm({"kv_connector": "moriio"})
        assert bm._joint_chunk_tokens == 0
        assert "LMCache chunk size" not in caplog.text
        assert "lmcache" not in caplog.text.lower()

    def test_a_dense_offload_engine_does_not_probe_either(self, caplog):
        """It has an offload connector but no state to offload, so a joint KV
        load is impossible and the chunk grid is not its business."""
        with caplog.at_level(logging.DEBUG, logger="atom"):
            bm = self._bm(
                {
                    "kv_connector": "lmcache_offload",
                    "kv_role": "offload",
                    "offload_layout": "dense",
                }
            )
        assert bm._joint_chunk_tokens == 0
        assert "LMCache chunk size" not in caplog.text

    def test_a_hosted_tier_still_probes_and_says_so_when_it_cannot_read(self, caplog):
        """The warning is worth printing exactly here: the tier is on, so a
        missing chunk size really does disable the joint KV load."""
        with caplog.at_level(logging.WARNING, logger="atom"):
            bm = self._bm(
                {
                    "kv_connector": "lmcache_offload",
                    "kv_role": "offload",
                    "offload_layout": "kimi_k3",
                }
            )
        # lmcache is not installed in the unit-test environment, so the probe
        # runs and fails -- which is the point: it ran.
        assert bm._joint_chunk_tokens == 0
        assert "LMCache chunk size" in caplog.text
