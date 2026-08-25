# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the recurrent-state checkpoint pool.

Covers `atom/model_engine/state_cache_pool.py` and its integration into
`BlockManager`: a prefix-cache hit for a linear-attention hybrid must be
truncated to a token count where the conv/ssm state can actually be resumed,
and the pool must not leak slots on any path (publish, short chunk, preempt,
abort).

Everything here drives the real classes through their public API rather than
asserting on source text — these are behaviours, and behaviours are testable.
"""

import pytest

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.state_cache_pool import StateCachePool
from conftest import MockConfig

BS = 4  # paged block size used throughout (matches seq_factory's default)


def make_pool(
    num_blocks=8,
    m=16,
    block_size=BS,
    enable_prefix_caching=True,
    enable_chunked_prefill=True,
):
    return StateCachePool(
        num_blocks=num_blocks,
        state_cache_block_size=m,
        block_size=block_size,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
    )


def tokens(n, base=0):
    return list(range(base, base + n))


def hashes_for(seq, n):
    """The chained content hashes BlockManager would compute for blocks [0, n)."""
    out, h = [], -1
    for i in range(n):
        h = BlockManager.compute_hash(seq.block(i), h)
        out.append(h)
    return out


def step_prefill(pool, seq, hs, done, chunk):
    """One scheduler step: align the chunk, reserve, forward, publish, end.

    Mirrors the real call order — the scheduler trims the chunk back to an M
    boundary and reserves before the forward, `BlockManager.hash_blocks`
    publishes after it, and the step ends. Returns the tokens actually
    forwarded, which is `chunk` or less.
    """
    pool.reserve_write(seq, done, chunk)
    start, end = done // BS, (done + chunk) // BS
    for i in range(start, end):
        pool.publish(seq, i, hs[i], seq.block(i))
    pool.end_step(seq)
    pool.release_restore(seq)
    return chunk


def run_prefill(pool, seq, chunks):
    """Drive one sequence's (chunked) prefill through the pool's lifecycle.

    Each entry in `chunks` is what the scheduler WANTS to forward in one step;
    each chunk is forwarded at its requested width. The recurrent kernels write
    every M-boundary crossed inside that forward.
    """
    hs = hashes_for(seq, seq.num_blocks)
    done = 0
    for chunk in chunks:
        target = done + chunk
        while done < target:
            forwarded = step_prefill(pool, seq, hs, done, target - done)
            done += forwarded
    return hs


def boundary_chunks(num_tokens, m):
    """Chunk sizes that make a prefill checkpoint at every M boundary.

    One chunk per span, so each step ends on a boundary and snapshots it, plus
    the unaligned tail. This is what a deployment that wants maximum prefix-cache
    granularity would get by setting `max_num_batched_tokens` to M; the default
    is far larger, so real chunks span many spans and checkpoint only their last.
    """
    out = [m] * (num_tokens // m)
    if num_tokens % m:
        out.append(num_tokens % m)
    return out


# ── disabled pool is inert ─────────────────────────────────────────────────


class TestDisabled:
    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(num_blocks=0),
            dict(m=0),
            dict(enable_prefix_caching=False),
            dict(enable_chunked_prefill=False),
        ],
        ids=[
            "no-blocks",
            "no-granularity",
            "no-prefix-caching",
            "no-chunked-prefill",
        ],
    )
    def test_every_method_is_a_no_op(self, kwargs, seq_factory):
        pool = make_pool(**kwargs)
        assert not pool.enabled
        seq = seq_factory(tokens(16))
        # bounded_hit is an identity so it can never shorten a hit for a model
        # that has no recurrent state to resume.
        assert pool.bounded_hit(seq, 4, [1, 2, 3, 4]) == 4
        # trim_chunk_to_boundary is an identity too, so a disabled pool cannot
        # perturb the scheduler's chunking by a single token.
        assert pool.trim_chunk_to_boundary(0, 16) == 16
        assert pool.trim_chunk_to_boundary(5, 3) == 3
        pool.reserve_write(seq, 0, 16)
        assert seq.state_ckpt_writes == {}
        pool.publish(seq, 3, 99, seq.block(3))
        pool.claim_restore(seq, 99)
        assert seq.state_restore_slot == -1
        pool.release_restore(seq)
        pool.end_step(seq)
        pool.release(seq)
        pool.clear_cache()

    def test_disabled_pool_reports_zero_span(self):
        assert make_pool(num_blocks=0).blocks_per_ckpt == 0


# ── granularity ────────────────────────────────────────────────────────────


class TestGranularity:
    @pytest.mark.parametrize("m,expected", [(4, 1), (16, 4), (64, 16)])
    def test_span_is_m_over_block_size(self, m, expected):
        assert make_pool(m=m).blocks_per_ckpt == expected

    def test_checkpoint_written_only_at_m_boundaries(self, seq_factory):
        pool = make_pool(m=16)  # = 4 paged blocks
        seq = seq_factory(tokens(48))  # 12 blocks = 3 spans
        run_prefill(pool, seq, boundary_chunks(48, 16))
        # 3 spans published; keyed by the LAST block of each span (3, 7, 11).
        assert len(pool.hash_to_block_id) == 3
        hs = hashes_for(seq, 12)
        for last in (3, 7, 11):
            assert hs[last] in pool.hash_to_block_id

    def test_partial_trailing_span_is_not_checkpointed(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(40))  # 10 blocks: 2 full spans + 2 leftover
        run_prefill(pool, seq, boundary_chunks(40, 16))
        assert len(pool.hash_to_block_id) == 2


# ── bounded_hit ────────────────────────────────────────────────────────────


class TestBoundedHit:
    def test_hit_truncated_down_to_last_checkpoint(self, seq_factory):
        pool = make_pool(m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        later = seq_factory(tokens(48))
        # Paged pool would happily hand back 11 blocks; only 8 (= 2 spans) are
        # resumable, since block 11 is the last block of span 3 but the paged
        # hit stops at 11 exclusive.
        assert pool.bounded_hit(later, 11, hs) == 8

    @pytest.mark.parametrize(
        "paged_hit,expected", [(0, 0), (3, 0), (4, 4), (7, 4), (8, 8), (12, 12)]
    )
    def test_hit_rounds_down_to_a_span_multiple(self, paged_hit, expected, seq_factory):
        pool = make_pool(m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        later = seq_factory(tokens(48))
        assert pool.bounded_hit(later, paged_hit, hs) == expected

    def test_no_checkpoints_means_no_hit(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        hs = hashes_for(seq, 12)
        assert pool.bounded_hit(seq, 12, hs) == 0

    def test_evicted_checkpoint_falls_back_to_an_earlier_boundary(self, seq_factory):
        pool = make_pool(num_blocks=3, m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        assert pool.bounded_hit(warm, 12, hs) == 12
        # Pin the first two spans (as two resuming requests would), leaving only
        # the LAST span's slot free. A third request's prefill then evicts it,
        # and the hit must fall back to span 1's boundary rather than keep
        # claiming a checkpoint whose slot now holds someone else's state.
        r0, r1 = seq_factory(tokens(48)), seq_factory(tokens(48))
        pool.claim_restore(r0, hs[3])
        pool.claim_restore(r1, hs[7])
        greedy = seq_factory(tokens(16, base=1000))
        run_prefill(pool, greedy, [16])
        assert hs[11] not in pool.hash_to_block_id
        assert pool.bounded_hit(warm, 12, hs) == 8

    def test_hash_collision_is_rejected_by_token_comparison(self, seq_factory):
        pool = make_pool(m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        # A different sequence whose (forged) hashes collide must not resume:
        # the stored token_ids don't match, so the boundary is rejected.
        other = seq_factory(tokens(48, base=500))
        assert pool.bounded_hit(other, 12, hs) == 0


# ── chunked prefill ────────────────────────────────────────────────────────


class TestChunkedPrefill:
    def test_every_chunking_checkpoints_a_subset_of_the_boundaries(self, seq_factory):
        """A step snapshots the one boundary its chunk ends on, so coarser
        chunking means fewer checkpoints — never wrong ones. Whatever the
        chunking, every published checkpoint is a real M boundary of this
        prompt, and per-span chunking gets all of them."""
        m = 16
        every = None
        for chunks in ([48], [16, 16, 16], [8, 8, 8, 8, 8, 8], [40, 8], [20, 28]):
            pool = make_pool(m=m)
            seq = seq_factory(tokens(48))
            hs = run_prefill(pool, seq, chunks)
            published = set(pool.hash_to_block_id)
            boundaries = {hs[i] for i in range(m // BS - 1, seq.num_blocks, m // BS)}
            assert published <= boundaries, f"{chunks} checkpointed a non-boundary"
            if chunks == [16, 16, 16]:
                every = published
        assert len(every) == 3, "per-span chunking must checkpoint every boundary"

    def test_a_wide_chunk_is_trimmed_not_capped(self, seq_factory):
        """The throughput property: one big chunk stays one big forward.

        Cutting at the NEXT boundary instead would cap every chunk at M,
        turning a long prefill into hundreds of tiny passes.
        """
        pool = make_pool(m=16)
        assert pool.trim_chunk_to_boundary(0, 1024) == 1024
        assert pool.trim_chunk_to_boundary(0, 1000) == 992  # trimmed, not 16

    def test_aligned_chunk_reserves_the_boundary_it_ends_on(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        pool.reserve_write(seq, 0, 16)  # ends exactly on span 0's boundary
        assert sorted(seq.state_ckpt_writes) == [3]

    def test_chunk_inside_one_span_reserves_nothing(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        pool.reserve_write(seq, 4, 8)  # tokens [4, 12): lands mid-span
        assert seq.state_ckpt_writes == {}

    def test_unaligned_chunk_end_reserves_nothing(self, seq_factory):
        """Only the END matters: a chunk that crosses a boundary but overshoots
        it leaves the working slot holding a mid-span state, which is not a
        checkpoint of anything the paged hashes name."""
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        pool.reserve_write(seq, 0, 20)  # crosses span 0, ends at token 20
        assert seq.state_ckpt_writes == {}

    def test_resumed_chunk_reserves_its_own_boundary(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        pool.reserve_write(seq, 32, 16)  # spans 0/1 already done
        assert sorted(seq.state_ckpt_writes) == [11]


# ── chunk alignment ────────────────────────────────────────────────────────


class TestChunkToBoundary:
    @pytest.mark.parametrize(
        "done,chunk,expected",
        [
            (0, 16, 16),  # already lands on a boundary: unchanged
            (0, 48, 48),  # exactly three spans: still unchanged
            (0, 20, 16),  # overshoots span 0 by 4: trimmed back to it
            (4, 32, 28),  # mid-span start: trimmed to end on token 32
            (32, 16, 16),  # resumed, aligned start, exact span
            (16, 100, 96),  # wide chunk keeps its width, minus the overshoot
        ],
    )
    def test_trims_back_to_the_last_boundary_spanned(self, done, chunk, expected):
        assert make_pool(m=16).trim_chunk_to_boundary(done, chunk) == expected

    @pytest.mark.parametrize("done,chunk", [(0, 4), (0, 15), (4, 8), (33, 2)])
    def test_chunk_short_of_a_boundary_is_left_alone(self, done, chunk):
        """Nothing to align to, so the scheduler's chunk stands and the step
        simply checkpoints nothing."""
        assert make_pool(m=16).trim_chunk_to_boundary(done, chunk) == chunk

    @pytest.mark.parametrize("chunk", [1, 7, 16, 64, 1000])
    def test_never_grows_and_never_zeroes_a_chunk(self, chunk):
        pool = make_pool(m=16)
        for done in range(0, 40):
            got = pool.trim_chunk_to_boundary(done, chunk)
            assert 0 < got <= chunk

    def test_repeated_alignment_walks_a_prefill_to_completion(self):
        """The property the scheduler relies on: applying it step after step
        terminates, and every step but a trailing partial one ends on M.

        With a budget wide enough for the whole prompt this is two steps (the
        aligned bulk, then the tail) rather than one per span — the point of
        trimming down instead of cutting at the next boundary.
        """
        pool = make_pool(m=16)
        done, ends = 0, []
        while done < 40:
            done += pool.trim_chunk_to_boundary(done, 40 - done)
            ends.append(done)
        assert ends == [32, 40]

    @pytest.mark.parametrize("budget", [16, 24, 64, 1024, 16384])
    @pytest.mark.parametrize("prompt", [40, 100, 512, 8192])
    def test_every_pass_consumes_nearly_the_whole_budget(self, budget, prompt):
        """Each forward advances to within M of everything it was offered.
        Trimming can only ever give back the overshoot past the last boundary,
        which is under M by construction.

        This is the guard on the throughput regression: cutting at the NEXT
        boundary instead makes every pass advance exactly M no matter how large
        the budget, so an 8k prompt at M=16 takes 512 passes instead of 1.
        """
        m = 16
        pool = make_pool(m=m)
        done = 0
        while done < prompt:
            offered = min(budget, prompt - done)
            got = pool.trim_chunk_to_boundary(done, offered)
            assert 0 < got <= offered
            assert got > offered - m
            done += got

    def test_non_positive_chunk_is_returned_unchanged(self):
        pool = make_pool(m=16)
        assert pool.trim_chunk_to_boundary(0, 0) == 0
        assert pool.trim_chunk_to_boundary(0, -1) == -1


# ── slot accounting ────────────────────────────────────────────────────────


class TestSlotAccounting:
    def _free(self, pool):
        return len(pool.free_block_ids_set)

    def test_published_checkpoints_do_not_pin_slots(self, seq_factory):
        pool = make_pool(num_blocks=8, m=16)
        seq = seq_factory(tokens(48))
        run_prefill(pool, seq, boundary_chunks(48, 16))
        # All 3 published checkpoints are back in the free queue with their
        # hashes intact — hittable without holding memory hostage.
        assert self._free(pool) == 8
        assert len(pool.hash_to_block_id) == 3

    def test_unpublished_reservation_is_reclaimed_by_end_step(self, seq_factory):
        pool = make_pool(num_blocks=8, m=16)
        seq = seq_factory(tokens(48))
        pool.reserve_write(seq, 0, 16)
        assert self._free(pool) == 7
        pool.end_step(seq)
        assert self._free(pool) == 8
        assert seq.state_ckpt_writes == {}

    def test_release_reclaims_restore_and_reservations(self, seq_factory):
        pool = make_pool(num_blocks=8, m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        seq = seq_factory(tokens(48))
        pool.claim_restore(seq, hs[7])
        pool.reserve_write(seq, 32, 16)
        assert self._free(pool) < 8
        pool.release(seq)
        assert self._free(pool) == 8
        assert seq.state_restore_slot == -1
        assert seq.state_ckpt_writes == {}

    def test_release_restore_is_idempotent(self, seq_factory):
        pool = make_pool(num_blocks=8, m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        seq = seq_factory(tokens(48))
        pool.claim_restore(seq, hs[7])
        pool.release_restore(seq)
        pool.release_restore(seq)
        assert self._free(pool) == 8

    def test_exhausted_pool_skips_checkpoints_instead_of_raising(self, seq_factory):
        """A checkpoint is a cache entry, never a correctness requirement, so a
        step that cannot get a slot forwards normally and just misses later."""
        pool = make_pool(num_blocks=1, m=16)
        a, b = seq_factory(tokens(48)), seq_factory(tokens(48, base=500))
        pool.reserve_write(a, 0, 16)  # takes the pool's only slot
        pool.reserve_write(b, 0, 16)  # nothing left
        assert len(a.state_ckpt_writes) == 1
        assert b.state_ckpt_writes == {}

    def test_restore_reference_survives_a_competing_prefill(self, seq_factory):
        """A claimed checkpoint must not be evicted out from under its reader."""
        pool = make_pool(num_blocks=2, m=16)
        warm = seq_factory(tokens(32))
        hs = run_prefill(pool, warm, boundary_chunks(32, 16))
        reader = seq_factory(tokens(32))
        pool.claim_restore(reader, hs[7])
        held = reader.state_restore_slot
        # A long competing prefill cycles through every free slot, span after
        # span; the claimed one must stay out of its reach the whole way.
        greedy = seq_factory(tokens(64, base=1000))
        ghs, done = hashes_for(greedy, greedy.num_blocks), 0
        while done < 64:
            chunk = pool.trim_chunk_to_boundary(done, 64 - done)
            pool.reserve_write(greedy, done, chunk)
            assert held not in greedy.state_ckpt_writes.values()
            for i in range(done // BS, (done + chunk) // BS):
                pool.publish(greedy, i, ghs[i], greedy.block(i))
            pool.end_step(greedy)
            done += chunk

    def test_claiming_a_free_checkpoint_keeps_it_hittable(self, seq_factory):
        """Claiming must not run reset(), which would evict the matched hash."""
        pool = make_pool(num_blocks=8, m=16)
        warm = seq_factory(tokens(48))
        hs = run_prefill(pool, warm, boundary_chunks(48, 16))
        seq = seq_factory(tokens(48))
        pool.claim_restore(seq, hs[7])
        assert hs[7] in pool.hash_to_block_id
        assert pool.blocks[seq.state_restore_slot].token_ids == warm.block(7)


# ── clear_cache ────────────────────────────────────────────────────────────


class TestClearCache:
    def test_clear_drops_every_checkpoint(self, seq_factory):
        pool = make_pool(m=16)
        seq = seq_factory(tokens(48))
        hs = run_prefill(pool, seq, boundary_chunks(48, 16))
        pool.clear_cache()
        assert pool.bounded_hit(seq, 12, hs) == 0


# ── BlockManager integration ───────────────────────────────────────────────


def bm_with_state_cache(m=16, num_state_cache_blocks=8, **overrides):
    kwargs = dict(
        kv_cache_block_size=BS,
        num_kvcache_blocks=64,
        enable_prefix_caching=True,
        state_cache_block_size=m,
        num_state_cache_blocks=num_state_cache_blocks,
    )
    kwargs.update(overrides)
    return BlockManager(MockConfig(**kwargs))


def bm_prefill(bm, seq, num_cached_blocks=0, budget=None):
    """Admit + prefill one sequence, chunking as the scheduler would.

    `budget` is the per-step token budget (`max_num_batched_tokens`), defaulting
    to one span so the prompt checkpoints at every boundary — the granularity
    these hit tests are about. A step snapshots only the boundary its chunk ends
    on, so a wider budget legitimately publishes fewer checkpoints; see
    `test_a_wide_budget_checkpoints_only_where_chunks_end`.
    """
    bm.allocate(seq, num_cached_blocks)
    step = budget or (bm.state_cache.state_cache_block_size or len(seq))
    while seq.num_cached_tokens < len(seq):
        chunk = min(step, len(seq) - seq.num_cached_tokens)
        chunk = bm.state_cache.trim_chunk_to_boundary(seq.num_cached_tokens, chunk)
        bm.state_cache.reserve_write(seq, seq.num_cached_tokens, chunk)
        bm.hash_blocks(seq, chunk)
        seq.num_cached_tokens += chunk
        bm.state_cache.end_step(seq)
        bm.state_cache.release_restore(seq)


class TestBlockManagerIntegration:
    def test_absent_for_models_without_recurrent_state(self, seq_factory):
        bm = BlockManager(
            MockConfig(
                kv_cache_block_size=BS,
                num_kvcache_blocks=64,
                enable_prefix_caching=True,
            )
        )
        assert not bm.state_cache_enabled
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        later = seq_factory(tokens(48))
        # Untouched paged behaviour: hit is the full prefix minus the last block.
        assert bm.can_allocate(later) == 11

    def test_hit_is_truncated_to_the_last_checkpoint(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        assert bm.state_cache_enabled
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        later = seq_factory(tokens(48))
        # Paged pool offers 11; only 8 tokens' worth of state is resumable.
        assert bm.can_allocate(later) == 8

    def test_hit_never_exceeds_the_paged_hit(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        # Diverges at block 5, so the paged hit is 5 and the state hit is 4.
        later = seq_factory(tokens(20) + tokens(28, base=900))
        assert bm.can_allocate(later) == 4

    def test_a_wide_budget_checkpoints_only_where_chunks_end(self, seq_factory):
        """The granularity trade the trim-down design makes explicit.

        A budget spanning the whole prompt runs ONE forward, which leaves one
        state behind, so only the last boundary is checkpointed — a later
        request resumes from token 32, not 48's worth of intermediate points.
        Passes stay at 1 either way; only hit granularity moves.
        """
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm, budget=1024)
        assert len(bm.state_cache.hash_to_block_id) == 1
        later = seq_factory(tokens(48))
        assert bm.can_allocate(later) == 8

    def test_allocate_claims_the_boundary_checkpoint(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        later = seq_factory(tokens(48))
        hit = bm.can_allocate(later)
        bm.allocate(later, hit)
        assert later.state_restore_slot >= 0
        # The claimed slot holds the state for the boundary block (hit - 1).
        block = bm.state_cache.blocks[later.state_restore_slot]
        assert block.token_ids == warm.block(hit - 1)

    def test_no_hit_means_no_restore_slot(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        seq = seq_factory(tokens(48))
        bm.allocate(seq, bm.can_allocate(seq))
        assert seq.state_restore_slot == -1

    def test_deallocate_returns_every_slot(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        free_before = len(bm.state_cache.free_block_ids_set)
        later = seq_factory(tokens(48))
        hit = bm.can_allocate(later)
        bm.allocate(later, hit)
        bm.state_cache.reserve_write(later, later.num_cached_tokens, 16)
        bm.deallocate(later)
        assert len(bm.state_cache.free_block_ids_set) == free_before
        assert later.state_restore_slot == -1
        assert later.state_ckpt_writes == {}

    def test_clear_cache_clears_both_pools(self, seq_factory):
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(48))
        bm_prefill(bm, warm)
        bm.clear_cache()
        assert bm.state_cache.hash_to_block_id == {}
        later = seq_factory(tokens(48))
        assert bm.can_allocate(later) == 0

    def test_pool_disabled_when_prefix_caching_is_off(self):
        bm = bm_with_state_cache(m=16, enable_prefix_caching=False)
        assert not bm.state_cache_enabled

    def test_pool_disabled_when_chunked_prefill_is_off(self):
        """Checkpointing writes by cutting a prefill short at a boundary, which
        a deployment that turned chunking off has explicitly opted out of."""
        bm = bm_with_state_cache(m=16, enable_chunked_prefill=False)
        assert not bm.state_cache_enabled

    def test_pool_disabled_when_no_blocks_were_budgeted(self):
        assert not bm_with_state_cache(
            m=16, num_state_cache_blocks=0
        ).state_cache_enabled

    def test_pool_disabled_when_granularity_is_zero(self):
        assert not bm_with_state_cache(m=0).state_cache_enabled

    def test_checkpoint_and_paged_hit_agree_on_the_token_count(self, seq_factory):
        """The property the whole design rests on: one hit length is valid in
        both pools, so a resumed request never reads paged KV past the state it
        restored, nor restores state past the KV it kept."""
        bm = bm_with_state_cache(m=16)
        warm = seq_factory(tokens(64))
        bm_prefill(bm, warm)
        later = seq_factory(tokens(64))
        hit = bm.can_allocate(later)
        assert hit % (16 // BS) == 0
        bm.allocate(later, hit)
        assert later.num_cached_tokens == hit * BS
        # Every reused paged block is present...
        assert len(later.block_table) == later.num_blocks
        # ...and the restored checkpoint is exactly the one at that boundary.
        slot = later.state_restore_slot
        assert bm.state_cache.blocks[slot].token_ids == warm.block(hit - 1)
