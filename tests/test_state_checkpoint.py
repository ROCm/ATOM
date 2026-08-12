# SPDX-License-Identifier: MIT
# Tests for per-request state checkpoints: the third prefix-cache gate.
#
# Neither the GDN recurrent state nor the V4 compressor ring can be rebuilt
# from cached KV blocks, so a prefix hit is only resumable at a boundary where
# some earlier request published its state. `StateGroupPool` indexes those
# boundaries and `BlockManager` shrinks the hit to the rightmost one — without
# it, a hit hands the resumed forward a group straight off the free list and it
# reads the previous occupant's state.
#
# Capacity model under test: a checkpoint is a FREE group whose content is
# still valid (the KV block pool's lazy eviction, applied to state groups). So
# checkpoints must never reduce the number of admissible requests, and the
# eviction event is hand-out, not free.

from math import inf, isinf
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import CacheStats, ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.model_engine.state_cache import StateCache
from atom.model_engine.state_pool import StateGroupPool, StateTransfer

BLOCK = 4
MIN_FORK = 8


def ckpt_config(**overrides):
    defaults = {
        "kv_cache_block_size": BLOCK,
        "num_kvcache_blocks": 200,
        "enable_prefix_caching": True,
        "max_num_seqs": 4,
        "max_num_batched_tokens": 256,
        "max_model_len": 256,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "stop_token_ids": [],
        "scheduler_delay_factor": 0.0,
        "speculative_config": None,
        "pool_entries": {"state": 4},
        "state_transfer_kind": "fork",
        "state_fork_tokens": MIN_FORK,
        "state_checkpoint_interval_tokens": BLOCK,
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def stateful_seq(token_ids):
    return Sequence(token_ids, BLOCK, has_per_req_cache=True)


def run_prompt(bm: BlockManager, seq: Sequence) -> None:
    """Admit `seq` and finalize its whole prompt as one forward."""
    hit = bm.can_allocate(seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    bm.hash_blocks(seq, seq.num_prompt_tokens - seq.num_cached_tokens)


def publish_at_boundary(bm: BlockManager, seq: Sequence) -> int:
    """Admit `seq`, forward exactly up to its checkpoint boundary, return its hash."""
    hit = bm.can_allocate(seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    boundary = bm.checkpoint_limit(seq)
    assert boundary > 0
    bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
    return boundary_hash(bm, seq)


def publisher_has_read_its_source(bm: BlockManager) -> None:
    """Step past the two passes `checkpoint` holds its fork source for.

    `checkpoint` runs in postprocess, after its own batch went out, so the
    forward that reads the source it handed over is the one the *next* pass
    builds and the pin clears the pass after that. Until then the group is off
    the free list — handing it to somebody else in between is one kernel
    reading and writing it at once.

    Tests about a resumer, not about the publisher, step over that here rather
    than each spelling out two `release_state_pins` calls.
    """
    bm.release_state_pins()
    bm.release_state_pins()


def run_prompt_on_the_ladder(bm: BlockManager, seq: Sequence) -> list[int]:
    """Admit `seq`, then forward its prompt on the ladder."""
    bm.allocate(seq, bm.can_allocate(seq))
    return forward_on_the_ladder(bm, seq)


def forward_on_the_ladder(bm: BlockManager, seq: Sequence) -> list[int]:
    """Forward an admitted seq's remaining prompt, cutting where the ladder says.

    What the scheduler does minus the token budget: each chunk runs to the end
    of the prompt unless `checkpoint_cut` pulls it back. Returns the positions
    it was cut at, which is the cost side of every checkpoint kept.
    """
    cuts = []
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        start = seq.num_cached_tokens
        chunk = seq.num_prompt_tokens - start
        target = bm.checkpoint_cut(seq, start, start + chunk)
        if target:
            chunk = target - start
            cuts.append(target)
        bm.hash_blocks(seq, chunk, start_tokens=start)
        seq.num_cached_tokens = start + chunk
    return cuts


def boundary_hash(bm: BlockManager, seq: Sequence) -> int:
    """Content hash of the last block before this seq's checkpoint boundary."""
    last = bm.checkpoint_limit(seq) // bm.hash_block_size - 1
    return bm.kv.block(seq.block_table[last]).hash


# ── StateGroupPool in isolation ────────────────────────────────────────────


def idx_seq(num_tokens: int = 1000):
    """The two Sequence fields `resumable_hit` reads, and nothing else."""
    return SimpleNamespace(num_tokens=num_tokens, has_per_req_cache=True)


class TestPoolIndex:

    def test_disabled_is_identity(self):
        pool = StateGroupPool(0)
        assert pool.resumable_hit(idx_seq(), 5, [1, 2, 3, 4, 5]) == 5
        assert pool.lookup(1) == -1

    def test_resumable_hit_picks_rightmost_checkpoint(self):
        pool = StateGroupPool(4, StateTransfer.fork(1), hash_block_size=1)
        pool._index(10, 0)
        pool._index(30, 1)
        # hashes for blocks 0..4; checkpoints exist after block 0 and block 2
        assert pool.resumable_hit(idx_seq(), 5, [10, 20, 30, 40, 50]) == 3

    def test_resumable_hit_zero_when_nothing_published(self):
        pool = StateGroupPool(4, StateTransfer.fork(1), hash_block_size=1)
        assert pool.resumable_hit(idx_seq(), 5, [10, 20, 30, 40, 50]) == 0

    def test_resumable_hit_walks_back_when_the_fork_has_no_room(self):
        pool = StateGroupPool(4, StateTransfer.fork(4), hash_block_size=1)
        pool._index(10, 0)
        pool._index(30, 1)
        # One token per block, five in the seq: the rightmost checkpoint
        # (boundary 3) leaves only 2 tokens to forward, short of the 4 a fork
        # needs, so the scan walks back to boundary 1, which leaves 4.
        assert pool.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 1

    def test_invalidate_drops_both_directions(self):
        pool = StateGroupPool(4)
        pool._index(10, 2)
        pool.invalidate(2)
        assert pool.lookup(10) == -1
        # A later invalidate of the same group must not delete a new tenant.
        pool._index(10, 3)
        pool.invalidate(2)
        assert pool.lookup(10) == 3

    def test_republishing_a_hash_orphans_the_old_group(self):
        pool = StateGroupPool(4)
        pool._index(10, 1)
        pool._index(10, 2)
        assert pool.lookup(10) == 2
        # Group 1 no longer backs hash 10; invalidating it leaves 2 indexed.
        pool.invalidate(1)
        assert pool.lookup(10) == 2

    def test_pins_drain_once(self):
        pool = StateGroupPool(4)
        while pool.has_free():  # every group out with a request
            pool.pop()
        pool.pin(1)
        pool.pin(3)
        assert pool.is_pinned(1)
        pool.release_pins()
        assert pool.num_free() == 2
        assert pool.is_free(1) and pool.is_free(3)
        pool.release_pins()  # idempotent: a drained pin is not freed twice
        assert pool.num_free() == 2
        assert not pool.is_pinned(1)


# ── The free list is two halves: vacant, and checkpoints in LRU order ──────
#
# Splitting them is what lets the pool shrink from the top without spending
# whatever happens to sit there. Vacant is drawn from first and packs towards
# index 0; checkpoints are spent oldest-first, wherever they are.


def drain(pool):
    """Hand out every group, as if that many requests were running."""
    while pool.has_free():
        pool.pop()


class TestFreeListHalves:
    def test_a_vacant_group_is_spent_before_any_checkpoint(self):
        """The single release-ordered queue this replaced got this wrong.

        Group 0 is checkpointed and handed back first, group 1 is handed back
        after it carrying nothing. In release order 0 comes out first and the
        checkpoint dies while a group with nothing to lose waits behind it.
        """
        pool = StateGroupPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool.release(1)

        assert pool.pop() == 1
        assert pool.lookup(10) == 0

    def test_admission_packs_towards_index_zero(self):
        pool = StateGroupPool(4)
        drain(pool)
        for group in (3, 1, 2):
            pool.release(group)
        assert [pool.pop() for _ in range(3)] == [1, 2, 3]

    def test_checkpoints_are_spent_least_recently_used_first(self):
        pool = StateGroupPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11), (2, 12)):
            pool.release(group)
            pool._index(h, group)

        assert pool.pop() == 0
        assert pool.pop() == 1

    def test_resuming_from_a_checkpoint_refreshes_it(self):
        """Reuse has to count as use or the hottest checkpoint dies first.

        `claim` deliberately leaves the hash in place, so the group comes back
        through `release` still checkpointed — and lands at the LRU tail.
        """
        pool = StateGroupPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.claim(0)  # a resumer reads the oldest checkpoint
        pool.pin(0)
        pool.release_pins()

        assert pool.pop() == 1  # 11 is now the older of the two
        assert pool.lookup(10) == 0

    def test_republishing_a_hash_returns_the_orphan_to_the_vacant_half(self):
        pool = StateGroupPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool.release(1)
        pool._index(10, 1)  # group 0 no longer backs anything

        assert pool.pop() == 0  # vacant again, so it goes before the checkpoint
        assert pool.lookup(10) == 1


class TestShrinking:
    def test_a_vacant_top_costs_nothing(self):
        pool = StateGroupPool(4)
        out = pool.retire_top()
        assert (out.retired, out.relocated_to) == (3, -1)
        assert pool.num_groups == 3
        assert not pool.is_free(3)

    def test_a_live_top_moves_into_the_lowest_vacant_group(self):
        pool = StateGroupPool(4)
        drain(pool)
        pool.release(2)  # only group 2 is free; 3 is held by a request

        out = pool.retire_top()
        assert (out.retired, out.relocated_to, out.held_checkpoint) == (3, 2, False)
        assert pool.num_groups == 3

    def test_shrinking_spends_the_oldest_checkpoint_not_the_top_one(self):
        """The whole reason `retire_top` relocates instead of just dropping.

        A group's index records the concurrency high-water mark when it was
        handed out and is never refreshed by use, so the hottest checkpoint can
        sit at the top. Retiring by index alone would spend it and leave one
        nothing has touched in minutes.
        """
        pool = StateGroupPool(4)
        drain(pool)
        for group, h in ((0, 10), (3, 13)):
            pool.release(group)
            pool._index(h, group)
        pool.claim(3)  # 13 is hot: someone just resumed from it
        pool.pin(3)
        pool.release_pins()

        out = pool.retire_top()

        assert out.retired == 3 and out.held_checkpoint
        assert out.relocated_to == 0
        assert pool.lookup(13) == 0  # the hot one survived, at a new address
        assert pool.lookup(10) == -1  # the cold one is what we spent
        assert pool.num_groups == 3

    def test_the_top_is_spent_when_it_is_itself_the_oldest(self):
        pool = StateGroupPool(2)
        drain(pool)
        pool.release(1)
        pool._index(13, 1)

        out = pool.retire_top()
        assert (out.retired, out.relocated_to, out.held_checkpoint) == (1, -1, True)
        assert pool.lookup(13) == -1

    def test_a_pinned_top_is_refused_rather_than_moved(self):
        """It is being read by the in-flight step; the pin drains next pass."""
        pool = StateGroupPool(4)
        drain(pool)
        pool.pin(3)
        assert pool.retire_top() is None
        assert pool.num_groups == 4

    def test_a_live_top_with_nowhere_to_go_is_refused(self):
        pool = StateGroupPool(4)
        drain(pool)
        assert pool.retire_top() is None
        assert pool.num_groups == 4

    def test_growing_adds_groups_at_the_top(self):
        pool = StateGroupPool(2)
        drain(pool)
        pool.extend(2)
        assert pool.num_groups == 4
        assert [pool.pop() for _ in range(2)] == [2, 3]

    def test_the_vacant_heap_does_not_grow_without_bound(self):
        """Taking a hash while vacant leaves an entry behind; churn compacts.

        Nothing observable depends on this, which is why it is asserted
        directly: on a long-lived server the stale entries otherwise outnumber
        the live ones by the number of checkpoints ever taken.
        """
        pool = StateGroupPool(4)
        for round_ in range(200):
            group = pool.pop()
            pool.release(group)
            pool._index(round_, group)  # promotes it, stranding a heap entry
            pool.claim(group)
            pool.group_hash[group] = -1
            pool.release(group)
        assert len(pool._vacant) <= 2 * pool.num_groups + 2

    def test_regrowing_a_retired_index_reuses_its_hash_slot(self):
        """Not appending a second one, which would shift every index above it."""
        pool = StateGroupPool(3)
        assert pool.retire_top().retired == 2
        pool.extend(1)

        assert pool.num_groups == 3
        assert len(pool.group_hash) == 3
        drain(pool)
        pool.release(2)
        pool._index(12, 2)
        assert pool.lookup(12) == 2


# ── BlockManager: the hit is shrunk to a resumable boundary ────────────────


class TestHitShrink:

    def test_hit_is_zero_without_a_checkpoint(self):
        """The correctness fix: a stateful model cannot resume a bare KV hit."""
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        run_prompt(bm, first)
        # Same prompt again: compressed blocks are all cached, but the first
        # request published nothing (its forward never ended on the boundary).
        second = stateful_seq(list(range(40)))
        assert bm.can_allocate(second) == 0
        assert second.num_compressed_hit_blocks > 0

    def test_stateless_model_keeps_the_full_hit(self):
        bm = BlockManager(
            ckpt_config(
                pool_entries={}, state_transfer_kind="none", state_fork_tokens=0
            )
        )
        first = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        run_prompt(bm, first)
        second = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        # 10 blocks of prompt, the last never reused → full 9-block hit.
        assert bm.can_allocate(second) == 9

    def test_hit_lands_on_the_published_boundary(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        publish_at_boundary(bm, first)
        boundary = bm.checkpoint_limit(first)

        second = stateful_seq(list(range(40)))
        assert bm.can_allocate(second) * bm.hash_block_size == boundary

    def test_resume_reads_the_checkpoint_and_writes_a_fresh_group(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        src = bm.state.lookup(h)
        assert src >= 0

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert second.state_fork_src == src
        assert second.per_req_cache_group != src
        # The checkpoint survives the resume, so a third request still finds it.
        assert bm.state.lookup(h) == src


# ── Capacity: checkpoints live on the free list, never hold it back ────────


class TestCapacity:

    def test_checkpoints_do_not_reduce_admission(self):
        """A published checkpoint is a free group; concurrency is unchanged."""
        bm = BlockManager(ckpt_config())
        for i in range(4):
            seq = stateful_seq(list(range(100 * i, 100 * i + 20 + 4 * i)))
            publish_at_boundary(bm, seq)
            bm.deallocate(seq)
        # Some checkpoints survive, older ones were recycled by the FIFO — the
        # point is that neither outcome costs a group.
        assert bm.state.hash_to_group
        # Every group is back, so the pool admits its full concurrency.
        assert bm.state.num_free() == 4
        for i in range(4):
            seq = stateful_seq(list(range(900 + 20 * i, 920 + 20 * i)))
            assert bm.can_allocate(seq) >= 0
            bm.allocate(seq, 0)
        assert bm.state.num_free() == 0

    def test_handout_evicts_the_checkpoint_it_lands_on(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        group = bm.state.lookup(h)
        bm.deallocate(first)
        # Drain the queue until the checkpoint's group comes back around.
        while bm.state.has_free():
            seq = stateful_seq(list(range(900, 920)))
            bm.allocate(seq, 0)
            if seq.per_req_cache_group == group:
                break
        assert bm.state.lookup(h) == -1

    def test_resume_without_a_spare_group_adopts_the_checkpoint(self):
        # Two groups: the publisher keeps one, so the only free group when the
        # resume arrives is the checkpoint itself.
        bm = BlockManager(ckpt_config(pool_entries={"state": 2}))
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        group = bm.state.lookup(h)
        assert bm.state.num_free() == 1

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        # No second group to fork into, so the resume spends the checkpoint —
        # still exactly the state it wanted, just no longer shareable.
        assert second.per_req_cache_group == group
        assert second.state_fork_src == -1
        assert bm.state.lookup(h) == -1


# ── Fork lifecycle ─────────────────────────────────────────────────────────


class TestForkLifecycle:

    def test_publish_moves_the_writer_to_a_new_group(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hit = bm.can_allocate(seq)
        bm.allocate(seq, hit)
        before = seq.per_req_cache_group
        boundary = bm.checkpoint_limit(seq)
        bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
        assert seq.per_req_cache_group != before
        assert seq.state_fork_src == before
        assert bm.state.lookup(boundary_hash(bm, seq)) == before

    def test_no_publish_when_the_forward_misses_the_boundary(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        group = seq.per_req_cache_group
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) + BLOCK)
        assert seq.per_req_cache_group == group
        assert not bm.state.hash_to_group

    def test_boundary_leaves_room_for_the_fork_forward(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        boundary = bm.checkpoint_limit(seq)
        assert boundary % bm.hash_block_size == 0
        assert seq.num_prompt_tokens - boundary >= MIN_FORK

    def test_every_block_boundary_up_to_the_limit_qualifies(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        limit = bm.checkpoint_limit(seq)
        assert bm.checkpointers_at(seq, BLOCK)
        assert bm.checkpointers_at(seq, limit)
        assert not bm.checkpointers_at(seq, limit + BLOCK)  # no room to fork
        assert not bm.checkpointers_at(seq, BLOCK + 2)  # not block aligned
        assert not bm.checkpointers_at(seq, 0)

    def test_chunked_prefill_leaves_a_ladder_of_checkpoints(self):
        """Intermediate boundaries publish too — the CPU-offload resume points."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        for _ in range(4):
            # One scheduling pass per chunk: each publish hands its source to
            # the next forward, and that forward is what lets the group go.
            # Without the boundary four publishes would hold four sources at
            # once and the pool would run out mid-ladder.
            bm.release_state_pins()
            bm.hash_blocks(seq, 2 * BLOCK, start_tokens=seq.num_cached_tokens)
            seq.num_cached_tokens += 2 * BLOCK
        # Four publishes into four groups: the oldest was recycled to serve the
        # last one, the rest stand as distinct resume points.
        assert len(bm.state.hash_to_group) == 3
        assert bm.state.lookup(boundary_hash(bm, seq)) >= 0  # the rightmost one

    def test_interval_thins_the_ladder(self):
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=3 * BLOCK))
        seq = stateful_seq(list(range(40)))
        limit = bm.checkpoint_limit(seq)
        published = [
            pos
            for pos in range(BLOCK, limit + BLOCK, BLOCK)
            if bm.checkpointers_at(seq, pos)
        ]
        # 40 tokens, 8 reserved for the fork forward: rungs at 12 and 24, and
        # the limit is the last rung rather than the last block boundary (32).
        assert limit == 6 * BLOCK
        assert published == [3 * BLOCK, 6 * BLOCK]

    def test_interval_zero_publishes_nothing(self):
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=0))
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 0
        assert not any(bm.checkpointers_at(seq, pos) for pos in range(BLOCK, 40, BLOCK))

    def test_prompt_shorter_than_the_interval_publishes_nothing(self):
        """The zero-cost case: no reuse to be had, so no forward is spent.

        A prompt that cannot even reach one rung must not be cut, or every
        request on a short-prompt workload pays an extra forward for a
        checkpoint nothing will ever hit.
        """
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=8 * BLOCK))
        seq = stateful_seq(list(range(30)))  # 30 < 8 * BLOCK
        assert bm.checkpoint_limit(seq) == 0
        run_prompt(bm, seq)
        assert not bm.state.hash_to_group
        assert seq.state_fork_src == -1

    def test_interval_snaps_onto_the_hash_block_grid(self):
        """A rung off the block grid has no content hash to be filed under.

        The interval defaults to 8192 while the grid follows `--block-size` and
        `--decode-context-parallel-size`, so an off-grid interval is something
        ordinary flag combinations produce rather than something the user asked
        for. Snapping down keeps the ladder on positions a lookup can reach; the
        alternative the pool used to take — refusing to construct — turned a
        block-size choice into a startup failure naming a flag nobody set.
        """
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=BLOCK + 1))
        assert bm.state_checkpoint_interval_tokens == BLOCK
        # Below one block there is no reachable rung at all, so the ladder is
        # off rather than snapped to something unusable.
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=BLOCK - 1))
        assert bm.state_checkpoint_interval_tokens == 0

    def test_hit_never_lands_where_swa_cannot_follow(self):
        """The two gates settle jointly; neither is applied to the other's answer.

        `swa.resumable_hit` promises the rightmost boundary whose trailing window
        is present. Shrinking that answer to a checkpoint boundary can land
        somewhere SWA never approved, and `allocate` would then claim an SWA
        hash the pool never promised.
        """
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        published = [2, 5]  # checkpoint boundaries, in blocks

        bm.state.hash_to_group = {}
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate(published):
            bm.state._index(hashes[boundary - 1], group)
        # A second class that accepts at most 5 — exactly the rightmost
        # checkpoint, so the fixpoint should settle there.
        bm.state_caches = (*bm.state_caches, StubStateCache(cap=5))
        assert bm._gated_hit(seq, 9, hashes) == 5

        # Now it accepts only 4: the rightmost checkpoint (5) is out of reach,
        # so the answer must fall back to 2 rather than stay at 5 or become 4.
        bm.state_caches = (bm.state_caches[0], StubStateCache(cap=4))
        assert bm._gated_hit(seq, 9, hashes) == 2

    def test_no_boundary_when_the_backend_cannot_fork(self):
        bm = BlockManager(ckpt_config(state_transfer_kind="none", state_fork_tokens=0))
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 0
        assert not bm.checkpointers_at(seq, 16)

    def test_cancel_adopts_the_source_and_returns_the_new_group(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        source = seq.per_req_cache_group
        free_before_publish = bm.state.num_free()
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        # Publishing costs a group until the forward that reads the source has
        # run: the seq now owns a fresh group and the source is pinned for it.
        assert bm.state.num_free() == free_before_publish - 1

        bm.cancel_state_fork(seq)
        assert seq.per_req_cache_group == source
        assert seq.state_fork_src == -1
        assert not bm.state.hash_to_group
        # Cancelling gives back exactly what publishing took.
        assert bm.state.num_free() == free_before_publish

    def test_two_resumers_in_one_step_share_the_checkpoint(self):
        # A checkpoint is read-only, so a second request hitting the same prefix
        # before the pins are released must fork off it too — not try to claim a
        # group the first one already took off the free list.
        bm = BlockManager(ckpt_config(pool_entries={"state": 8}))
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)

        resumers = [stateful_seq(list(range(40))) for _ in range(3)]
        for seq in resumers:
            bm.allocate(seq, bm.can_allocate(seq))

        assert bm.state.pin_count(src) == len(resumers)
        assert all(s.state_fork_src == src for s in resumers)
        # Distinct write groups, none of them the shared source.
        groups = {s.per_req_cache_group for s in resumers}
        assert len(groups) == len(resumers)
        assert src not in groups
        # However many read it, the group goes back exactly once.
        before = bm.state.num_free()
        bm.release_state_pins()
        assert bm.state.num_free() == before + 1

    def test_cancel_refuses_to_adopt_a_shared_source(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)

        sharers = [stateful_seq(list(range(40))) for _ in range(2)]
        for seq in sharers:
            bm.allocate(seq, bm.can_allocate(seq))

        # Taking the source over would write into a group the other request's
        # forward still has to read, so the fork has to stay.
        assert bm.cancel_state_fork(sharers[0]) is False
        assert sharers[0].state_fork_src == src
        # Once only one reader is left, adopting is legal again.
        bm.state.unpin(src)
        assert bm.cancel_state_fork(sharers[1]) is True
        assert sharers[1].per_req_cache_group == src

    def test_cancel_of_a_resume_releases_the_pin(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert bm.state.is_pinned(src)
        bm.cancel_state_fork(second)
        assert second.per_req_cache_group == src
        assert not bm.state.is_pinned(src)
        # The pin must not also hand the group back — it has an owner now.
        bm.release_state_pins()
        assert not bm.state.is_free(src)

    def test_pinned_source_returns_to_the_free_list_next_step(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)
        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert not bm.state.is_free(src)
        bm.release_state_pins()
        assert bm.state.is_free(src)

    def test_a_published_source_is_not_handed_out_before_its_reader_runs(self):
        """The source is what the publisher's NEXT forward reads.

        `checkpoint` runs in postprocess, so that forward belongs to the batch
        the next pass builds — one pass further off than a resume's reader.
        Handing the group back straight away, as this used to, put it on the
        free list during the very pass that admits the requests which could pop
        it, and then one kernel reads and writes it at once.
        """
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        assert first.state_fork_src == src

        assert not bm.state.is_free(src)  # the pass that admits cannot get it
        bm.release_state_pins()  # the batch carrying the fork is built
        assert not bm.state.is_free(src)  # its forward has not been issued yet
        bm.release_state_pins()  # it has now
        assert bm.state.is_free(src)
        # And it comes back as a checkpoint, at the LRU tail — publishing is
        # not what spends it.
        assert bm.state.lookup(bm.state.group_hash[src]) == src

    def test_a_finished_publisher_gives_its_source_back_at_once(self):
        """Nobody is left to read it, so the clock should not hold it.

        This is what keeps publishing capacity-neutral for the common shape —
        a request that crosses a rung and then finishes or is preempted.
        """
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        whole = bm.state.num_free()  # nothing handed out yet
        h = publish_at_boundary(bm, first)
        src = bm.state.lookup(h)
        assert not bm.state.is_free(src)

        bm.deallocate(first)
        assert bm.state.is_free(src)
        # Source and write group both back: the pool is whole again, without
        # waiting out the two passes the clock would have taken.
        assert bm.state.num_free() == whole
        assert bm.state.lookup(h) == src  # the checkpoint itself survives


class TestCheckpointsDieWithTheirPrefix:
    """A checkpoint whose KV block left the index can never be reached again.

    The two pools are addressed by one chained content hash and a prefix hit
    claims both, so `_gated_hit` caps at the last block still indexed. Until
    the state pool is told, the dead checkpoint holds a group and sits in the
    LRU queue ahead of live ones — the pool spends something usable to make
    room for something that is not.
    """

    def test_evicting_the_block_frees_the_checkpoint_group(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        src = bm.state.lookup(h)
        assert bm.state.holds_checkpoint(src)

        bm._record_evicted(h)
        assert bm.state.lookup(h) == -1
        assert bm.state.is_free(src)
        assert not bm.state.holds_checkpoint(src)  # vacant, spent before live ones
        assert bm.state.checkpoint_fates()["checkpoints_orphaned"] == 1

    def test_an_orphan_is_spent_before_a_live_checkpoint(self):
        pool = StateGroupPool(4)
        while pool.has_free():
            pool.pop()
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.unindex(10)  # group 0's prefix is gone
        assert pool.pop() == 0
        assert pool.lookup(11) == 1

    def test_unindex_of_an_unknown_hash_is_a_no_op(self):
        pool = StateGroupPool(4)
        pool._index(10, 0)
        assert pool.unindex(999) == -1
        assert pool.lookup(10) == 0
        assert pool.checkpoint_fates()["checkpoints_orphaned"] == 0


# ── The scheduler side: what a checkpoint costs the publisher ──────────────


class TestPrefillChunkAlignment:
    """`_finalize_prefill_chunk` cuts a prompt only where a rung is reachable.

    Every cut is an extra forward for the publisher, so the interval's whole
    job is to keep that off prompts too short to have anything to publish.
    """

    def test_prompt_shorter_than_the_interval_is_not_cut(self):
        sched = Scheduler(ckpt_config(state_checkpoint_interval_tokens=8 * BLOCK))
        seq = stateful_seq(list(range(30)))  # 30 < 8 * BLOCK
        assert sched._finalize_prefill_chunk(seq, 0, 30) == 30

    def test_chunk_stops_at_the_rung(self):
        sched = Scheduler(ckpt_config(state_checkpoint_interval_tokens=3 * BLOCK))
        seq = stateful_seq(list(range(40)))
        limit = sched.block_manager.checkpoint_limit(seq)
        assert limit == 24
        # A whole-prompt chunk is cut at the last rung...
        assert sched._finalize_prefill_chunk(seq, 0, 40) == limit
        # ...one that ends between rungs is pulled back to the one below...
        assert sched._finalize_prefill_chunk(seq, 0, 20) == 3 * BLOCK
        # ...and one starting past the limit is left whole, since nothing more
        # will be published there.
        assert sched._finalize_prefill_chunk(seq, limit, 16) == 16


# ── Copy lifecycle ─────────────────────────────────────────────────────────


def copy_config(**overrides):
    """A backend whose state is one byte range: it checkpoints by copying."""
    overrides.setdefault("state_transfer_kind", "copy")
    overrides.setdefault("state_fork_tokens", 0)
    return ckpt_config(**overrides)


class TestCopyLifecycle:
    """The other half of the protocol: a duplicate goes to the index.

    Everything the fork binds — a successor forward long enough to refill the
    replacement, and therefore a boundary with room behind it — is gone. What
    replaces it is a deferral: the bytes need a forward to move them, so the
    index entry cannot appear until the copy has been scheduled.
    """

    def _admitted(self, bm, tokens=None):
        seq = stateful_seq(tokens or list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        return seq

    def test_the_owner_is_not_disturbed(self):
        bm = BlockManager(copy_config())
        seq = self._admitted(bm)
        group = seq.per_req_cache_group
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        # No hand-over: the group and the read slot are exactly as they were.
        assert seq.per_req_cache_group == group
        assert seq.state_fork_src == -1
        assert seq.pending_checkpoint != -1
        # And nothing is claimable yet — the bytes do not exist.
        assert not bm.state.hash_to_group

    def test_the_next_batch_turns_it_into_a_pair(self):
        bm = BlockManager(copy_config())
        seq = self._admitted(bm)
        src = seq.per_req_cache_group
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        h = boundary_hash(bm, seq)

        copies = bm.state_copies_for_batch()
        assert seq.pending_checkpoint == -1
        assert len(copies) == 1
        got_src, dst = copies[0]
        assert got_src == src and dst != src
        assert bm.state.lookup(h) == dst
        # Capacity-neutral: the destination went straight back on the free list.
        assert bm.state.is_free(dst)
        assert not bm.state_copies_for_batch()  # drained once, not twice

    def test_a_request_freed_before_the_commit_indexes_nothing(self):
        """Its group is back on the free list, so there is nothing to copy."""
        bm = BlockManager(copy_config())
        seq = self._admitted(bm)
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        bm.deallocate(seq)

        # committed by state_copies_for_batch()
        assert not bm.state.hash_to_group
        assert not bm.state_copies_for_batch()

    def test_a_full_pool_keeps_no_checkpoint(self):
        """Best-effort, exactly as under a fork: no group, no checkpoint."""
        bm = BlockManager(copy_config())
        seq = self._admitted(bm)
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        while bm.state.has_free():
            bm.state.pop()

        # committed by state_copies_for_batch()
        assert not bm.state.hash_to_group
        assert not bm.state_copies_for_batch()

    @pytest.mark.parametrize(("extra_groups", "kept"), [(0, False), (1, True)])
    def test_checkpoint_capacity_starts_above_the_live_floor(self, extra_groups, kept):
        live_floor = 4
        config = copy_config(
            max_num_seqs=live_floor,
            pool_entries={"state": live_floor + extra_groups},
        )
        bm = BlockManager(config)
        owners = [
            self._admitted(bm, list(range(100 * i, 100 * i + 40)))
            for i in range(config.max_num_seqs)
        ]
        owner_groups = {seq.per_req_cache_group for seq in owners}
        assert len(owner_groups) == live_floor
        assert bm.state.num_free() == extra_groups

        publisher = owners[0]
        bm.hash_blocks(
            publisher,
            bm.checkpoint_limit(publisher) - publisher.num_cached_tokens,
        )
        h = boundary_hash(bm, publisher)
        assert publisher.pending_checkpoint != -1

        copies = bm.state_copies_for_batch()
        assert publisher.pending_checkpoint == -1
        assert bool(copies) is kept
        assert (bm.state.lookup(h) >= 0) is kept
        assert bm.state.checkpoint_fates() == {
            "checkpoints_kept": int(kept),
            "checkpoints_dropped": int(not kept),
            "checkpoints_evicted": 0,
            "checkpoints_orphaned": 0,
        }
        if kept:
            src, dst = copies[0]
            assert src == publisher.per_req_cache_group
            assert dst == bm.state.lookup(h)
            assert dst not in owner_groups

    @pytest.mark.parametrize("free_groups", [1, 2])
    def test_pending_copies_deduplicate_hashes_and_reserve_destinations(
        self, free_groups
    ):
        config = copy_config(
            max_num_seqs=3,
            pool_entries={"state": 3 + free_groups},
        )
        bm = BlockManager(config)
        owners = [
            self._admitted(bm, list(range(40))),
            self._admitted(bm, list(range(40))),
            self._admitted(bm, list(range(100, 140))),
        ]
        hashes = []
        for seq in owners:
            bm.hash_blocks(
                seq,
                bm.checkpoint_limit(seq) - seq.num_cached_tokens,
            )
            hashes.append(boundary_hash(bm, seq))

        assert hashes[0] == hashes[1]
        assert hashes[0] != hashes[2]
        copies = bm.state_copies_for_batch()

        # The duplicate H intent needs no second copy. With one free group only
        # one distinct boundary fits; with two, both do. In neither case may a
        # destination be released and reused while its copy is still pending.
        kept = min(free_groups, 2)
        assert len(copies) == kept
        destinations = [dst for _, dst in copies]
        assert len(destinations) == len(set(destinations))
        assert len(bm.state.hash_to_group) == kept
        assert set(bm.state.hash_to_group).issubset(set(hashes))
        assert set(destinations) == set(bm.state.hash_to_group.values())
        assert bm.state.checkpoint_fates() == {
            "checkpoints_kept": kept,
            "checkpoints_dropped": 2 - kept,
            "checkpoints_evicted": 0,
            "checkpoints_orphaned": 0,
        }

    def test_duplicate_copy_destinations_fail_before_the_batch(self):
        pool = StateGroupPool(4)
        pool.record_copy(0, 3)
        pool.record_copy(1, 3)

        with pytest.raises(RuntimeError, match="destinations must be unique"):
            pool.take_copies()

    def test_a_resume_is_handed_a_duplicate_not_a_fork(self):
        bm = BlockManager(copy_config())
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        # committed by state_copies_for_batch()
        src = bm.state_copies_for_batch()[0][1]

        # A follow-up turn, not a repeat: with no room reserved behind it the
        # checkpoint sits on the prompt's last block, and a request of the same
        # length can never reach it (its own hit stops one block short).
        second = stateful_seq(list(range(48)))
        hit = bm.can_allocate(second)
        assert hit > 0
        bm.allocate(second, hit)
        # The read side stays untouched; the bytes arrive by copy instead.
        assert second.state_fork_src == -1
        assert bm.state_copies_for_batch() == [(src, second.per_req_cache_group)]
        # And the source is held until the forward that reads it has been issued.
        assert bm.state.is_pinned(src)

    def test_the_checkpoint_is_only_claimable_once_its_batch_is_decided(self):
        """Why the commit waits for the batch instead of opening the pass.

        The source of a keeper copy is the owner's *live* group. Anything that
        can preempt that owner between the commit and the batch — an admission,
        in the same pass — would put the group back on the free list, and the
        copy would then duplicate the next request's state into a group already
        indexed as a checkpoint. Waiting until the batch is decided leaves no
        such window, at the price of the checkpoint landing one pass later.
        """
        bm = BlockManager(copy_config())
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)

        # An admission in the same pass cannot see it yet.
        second = stateful_seq(list(range(48)))
        assert bm.can_allocate(second) == 0

        bm.state_copies_for_batch()  # the batch is decided; now it exists
        assert bm.can_allocate(second) > 0

    def test_admissions_get_the_free_list_before_checkpoints_do(self):
        """Committing after admissions is also the right priority order."""
        bm = BlockManager(copy_config())
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        # Leave exactly one group: the admission takes it, the checkpoint yields.
        while bm.state.num_free() > 1:
            bm.state.pop()

        newcomer = stateful_seq(list(range(40)))
        bm.allocate(newcomer, bm.can_allocate(newcomer))
        assert newcomer.per_req_cache_group >= 0
        assert bm.state_copies_for_batch() == []
        assert not bm.state.hash_to_group

    def test_the_batch_carries_what_was_drained(self):
        """The copies have to reach the forward, which means riding a batch."""
        sched = Scheduler(copy_config())
        sched.add(stateful_seq(list(range(BLOCK))))
        sched.block_manager.state.record_copy(2, 3)
        batch, _ = sched.schedule()
        assert batch.state_copy_pairs == [(2, 3)]
        # Carried once: the next batch is not asked to repeat them.
        batch, _ = sched.schedule()
        assert batch.state_copy_pairs == []

    def test_a_copy_checkpoints_where_a_fork_cannot(self):
        """Speculation and a one-token step both stop a fork, neither a copy."""
        spec = SimpleNamespace(num_speculative_tokens=3, use_dspark=lambda: False)
        seq = stateful_seq(list(range(40)))
        seq.type = SequenceType.DECODE
        forking = Scheduler(ckpt_config(state_fork_tokens=1, speculative_config=spec))
        copying = Scheduler(copy_config(speculative_config=spec))
        assert forking._checkpoint_room(seq, False) == 0
        assert copying._checkpoint_room(seq, False) == 1
        # A finishing request still keeps nothing: no next batch to copy on.
        assert copying._checkpoint_room(seq, True) == 0


# ── Checkpoints past the prompt ────────────────────────────────────────────


class TestDecodePointPublishing:
    """The same ladder, walked by generation instead of by prompt.

    A long answer crosses rungs the prompt never reached, and a follow-up turn
    replaying the conversation wants to resume from them. What decides whether a
    rung is usable there is the same number as in prefill — how many tokens the
    next forward carries — except that number is now 1, which is why the
    backends split: GDN fills a fresh group from one token, V4's ring needs 131.
    """

    def _generate_to(self, bm, seq, end, room=1):
        """Append tokens one at a time, hashing at each committed KV length."""
        while seq.num_tokens < end:
            seq.append_token(500 + seq.num_tokens)
            bm.may_append(seq)
            bm.hash_decode_blocks(seq, seq.num_tokens, next_forward_tokens=room)

    def _prompt_of_10(self, bm):
        """A prompt that ends between rungs, so prefill publishes nothing."""
        seq = stateful_seq(list(range(10)))
        run_prompt(bm, seq)
        assert not bm.state.hash_to_group
        return seq

    def test_a_rung_past_the_prompt_publishes(self):
        bm = BlockManager(ckpt_config(state_fork_tokens=1))
        seq = self._prompt_of_10(bm)
        group = seq.per_req_cache_group

        self._generate_to(bm, seq, 3 * BLOCK)
        assert seq.per_req_cache_group != group
        assert seq.state_fork_src == group
        assert bm.state.lookup(bm.kv.block(seq.block_table[2]).hash) == group

    def test_a_backend_needing_a_long_fork_never_publishes_mid_generation(self):
        """Self-gating: no `min_fork` special case, the number decides.

        One decode token cannot fill a group that needs `MIN_FORK` of them, so
        the rung is simply not a publish position for this backend.
        """
        bm = BlockManager(ckpt_config())  # state_fork_tokens=MIN_FORK
        seq = self._prompt_of_10(bm)
        group = seq.per_req_cache_group

        self._generate_to(bm, seq, 4 * BLOCK)
        assert seq.per_req_cache_group == group
        assert not bm.state.hash_to_group

    def test_no_publish_on_the_step_that_finishes_the_request(self):
        """Nothing will fork from it, and the fresh group would go straight back."""
        bm = BlockManager(ckpt_config(state_fork_tokens=1))
        seq = self._prompt_of_10(bm)
        group = seq.per_req_cache_group

        self._generate_to(bm, seq, 3 * BLOCK, room=0)
        assert seq.per_req_cache_group == group
        assert not bm.state.hash_to_group

    def test_blocks_are_still_hashed_where_no_checkpoint_is_taken(self):
        """Prefix caching and state checkpoints are separate gates."""
        bm = BlockManager(ckpt_config())
        seq = self._prompt_of_10(bm)
        self._generate_to(bm, seq, 3 * BLOCK)
        assert seq.num_hashed_tokens == 3 * BLOCK

    def test_followup_turn_resumes_from_a_generated_rung(self):
        """The payoff: turn 2 reuses KV *and* the state that goes with it."""
        bm = BlockManager(ckpt_config(state_fork_tokens=1))
        seq = self._prompt_of_10(bm)
        self._generate_to(bm, seq, 4 * BLOCK)

        followup = stateful_seq(seq.token_ids[: 4 * BLOCK])
        # can_allocate never hands back the last block — the seq has to forward
        # something — so the hit caps at 3, which is exactly where generation
        # left a checkpoint.
        assert bm.can_allocate(followup) == 3
        bm.allocate(followup, 3)
        assert followup.state_fork_src == bm.state.lookup(
            bm.kv.block(seq.block_table[2]).hash
        )


class TestDecodePublishGate:
    """`Scheduler._state_publish_room`: who is allowed to checkpoint at decode."""

    def _sched(self, **overrides):
        return Scheduler(ckpt_config(state_fork_tokens=1, **overrides))

    def _decoding_seq(self):
        seq = stateful_seq(list(range(40)))
        seq.type = SequenceType.DECODE
        return seq

    def test_plain_decode_offers_its_one_token(self):
        assert self._sched()._checkpoint_room(self._decoding_seq(), False) == 1

    def test_finishing_request_offers_nothing(self):
        assert self._sched()._checkpoint_room(self._decoding_seq(), True) == 0

    def test_a_seq_still_on_its_prompt_offers_nothing(self):
        """Prefill decides with the prompt's own remainder, not with this."""
        seq = stateful_seq(list(range(40)))
        seq.type = SequenceType.PREFILL
        assert self._sched()._checkpoint_room(seq, False) == 0

    def test_speculative_decode_offers_nothing(self):
        """A fork must never reach the spec path — it has no read-side index.

        Prefill publishing stays live on the same models: `min_fork_tokens`
        keeps prompt behind every rung, and prompt forwards down the non-spec
        path.
        """
        sched = self._sched(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=3, use_dspark=lambda: False
            )
        )
        assert sched._checkpoint_room(self._decoding_seq(), False) == 0
        assert sched.block_manager.checkpoint_limit(stateful_seq(list(range(40)))) > 0

    def test_postprocess_carries_the_room_to_a_real_checkpoint(self):
        """End to end: generation alone leaves a resume point behind.

        A four-token prompt is too short for a rung of its own, so anything in
        the index at the end got there from a decode step, and the fork it
        raised has to be seen by the batch that follows.
        """
        sched = self._sched()
        bm = sched.block_manager
        seq = stateful_seq(list(range(BLOCK)))
        assert bm.checkpoint_limit(seq) == 0
        sched.add(seq)
        batch, _ = sched.schedule()

        forks = []
        for token in range(500, 505):
            sched.postprocess(
                list(sched.running),
                ScheduledBatchOutput(
                    req_ids=[seq.id],
                    token_ids=[(token,)],
                    num_rejected=None,
                    num_bonus=None,
                    draft_token_ids=None,
                ),
                batch=batch,
            )
            batch, _ = sched.schedule()
            forks.extend(s for s in batch.state_fork_srcs if s >= 0)

        published = bm.state.lookup(bm.kv.block(seq.block_table[1]).hash)
        assert published >= 0
        # The seq moved off the group it gave away, and the forward right after
        # the publish was told to read it.
        assert seq.per_req_cache_group != published
        assert forks == [published]


# ── One ladder, N state classes ────────────────────────────────────────────
#
# The ladder treats `Pool.STATE` classes as a set: each scales with in-flight
# requests, each can keep a boundary resumable, each can veto a hit. They differ
# only in mutability, and `successor_room` is that difference quantified — which
# is all the ladder knows about any of them.
#
# There is one real class today (the compressor ring; the sliding window became
# a per-request ring carried by the checkpoint and left the protocol). These
# tests use a stub for the second member on purpose: the multi-class behaviour
# is a property of the ladder, not of whichever classes happen to exist, and it
# has to keep working for the next one to arrive (GDN, once it stops forking).
# Testing it through a real second class would make these tests hostage to that
# class's own lifecycle — which is exactly what happened when it was SWA.


class StubStateCache:
    """Minimal `StateCache`: a fixed room and a hit it can be told to cap."""

    def __init__(self, successor_room=inf, cap=None, enabled=True):
        self.successor_room = successor_room
        self.enabled = enabled
        self._cap = cap

    def applies(self, seq):
        return self.enabled

    def resumable_hit(self, seq, P, block_hashes, assume_checkpointed=False):
        return P if self._cap is None else min(P, self._cap)

    def checkpoint(self, seq, boundary_blocks, h):
        pass


def second_class(**overrides):
    """A second state class for the protocol tests.

    A stub rather than a real one: multi-class behaviour is a property of the
    ladder, not of whichever class happens to exist beside `StateGroupPool`,
    and testing it through a real one made these tests hostage to that class's
    lifetime — which is how they broke when the sliding window stopped being a
    pool of its own.
    """
    return StubStateCache(**overrides)


class TestStateCacheProtocol:

    def test_both_classes_satisfy_the_protocol(self):
        assert isinstance(second_class(), StateCache)
        assert isinstance(StateGroupPool(4), StateCache)

    def test_a_class_that_keeps_nothing_reports_inf(self):
        """`inf` is what stops the ladder cutting chunks for a class in vain.

        The window pool only ever materializes the trailing window, so no older
        boundary has anything left to hold on to; reporting 0 would have the
        scheduler cut prefill chunks at every rung for a class that stores
        nothing there — cost with no reuse.
        """
        assert isinf(second_class().successor_room)
        assert isinf(StateGroupPool(4, StateTransfer.none()).successor_room)

    def test_the_limit_follows_the_class_that_reaches_furthest(self):
        """The smallest room reaches furthest right; a larger one must not cap it."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 32  # the ring alone: 40 - MIN_FORK
        bm.state_caches = (*bm.state_caches, StubStateCache(successor_room=0))
        assert bm.checkpoint_limit(seq) == 40

    def test_the_three_transfers_land_on_three_different_rooms(self):
        """The reason a backend declares a kind and not a token count.

        `none` and `copy` both have nothing to hand over, so a single integer
        could not separate "no state at all" from "no successor needed" — which
        are opposite ends of the room scale.
        """
        assert isinf(StateGroupPool(4, StateTransfer.none()).successor_room)
        assert StateGroupPool(4, StateTransfer.copy()).successor_room == 0
        assert StateGroupPool(4, StateTransfer.fork(7)).successor_room == 7

    def test_a_copy_never_asks_the_resumer_for_room(self):
        """`resumable_hit`'s fork test is vacuous under `copy`, not skipped."""
        forking = StateGroupPool(4, StateTransfer.fork(4), hash_block_size=1)
        copying = StateGroupPool(4, StateTransfer.copy(), hash_block_size=1)
        for pool in (forking, copying):
            pool._index(10, 0)
            pool._index(50, 1)
        # Five one-token blocks; the rightmost checkpoint leaves no room to
        # forward, so a fork walks back to the first and a copy does not.
        assert forking.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 1
        assert copying.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 5

    def test_the_immutable_class_qualifies_where_the_rolling_one_cannot(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        # A rung one token from the end: the ring has no room to hand over, an
        # immutable class needs none.
        pos = seq.num_prompt_tokens - BLOCK
        assert bm.state not in bm.checkpointers_at(seq, pos)
        bm.state_caches = (*bm.state_caches, StubStateCache(successor_room=0))
        assert bm.checkpointers_at(seq, pos) == [bm.state_caches[-1]]

    def test_cut_and_ladder_agree_position_for_position(self):
        """The chunk is cut where — and only where — something gets kept."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        cuts = {
            bm.checkpoint_cut(seq, pos - 1, pos)
            for pos in range(1, seq.num_prompt_tokens + 1)
        }
        rungs = {
            pos
            for pos in range(1, seq.num_prompt_tokens + 1)
            if bm.checkpointers_at(seq, pos)
        }
        assert cuts - {0} == rungs


class TestGatedHitFixpoint:

    def test_the_answer_is_accepted_by_every_class(self):
        """What a fixpoint means, asserted directly rather than by construction."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate([2, 5]):
            bm.state._index(hashes[boundary - 1], group)
        bm.state_caches = (*bm.state_caches, StubStateCache(cap=4))

        answer = bm._gated_hit(seq, 9, hashes)
        for cache in bm.state_caches:
            assert cache.resumable_hit(seq, answer, hashes) == answer

    def test_order_between_classes_does_not_change_the_answer(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate([2, 5]):
            bm.state._index(hashes[boundary - 1], group)
        bm.state_caches = (*bm.state_caches, StubStateCache(cap=4))

        forward = bm._gated_hit(seq, 9, hashes)
        bm.state_caches = tuple(reversed(bm.state_caches))
        assert bm._gated_hit(seq, 9, hashes) == forward


# ── Demand-driven checkpoints ──────────────────────────────────────────────


INTERVAL = 4 * BLOCK
PROMPT = list(range(44))  # 11 blocks; last never reused, so 10 are hittable


def demand_config(**overrides):
    """A grid too coarse to cover the prompt, so demand has room to show.

    `INTERVAL` of 16 over a 4-token hash block puts rungs at 16 and 32, while
    the fork test allows a checkpoint as far right as 36 — the gap between
    those two is what a demand rung fills.
    """
    overrides.setdefault("state_checkpoint_interval_tokens", INTERVAL)
    overrides.setdefault("pool_entries", {"state": 8})
    overrides.setdefault("max_num_seqs", 8)
    return ckpt_config(**overrides)


class TestDemandDrivenCheckpoints:
    """A rung placed where a request was seen to want one.

    The interval is a guess about where reuse will resume; the requests know.
    Whenever the state gates cut a hit short, `can_allocate` asks the same
    question again with every ladder assumed dense, and the gap between the two
    answers is reuse being declined only for want of a checkpoint. The request
    that finds the gap is the one that pays for it — it collects none of that
    reuse and has to compute the prefix anyway.
    """

    def test_the_gap_becomes_a_rung_off_the_grid(self):
        bm = BlockManager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 8  # the grid's last rung, 32 tokens
        assert second.num_wanted_hit_blocks == 9  # what a checkpoint would give
        assert second.checkpoint_demand_pos == 36
        # Off the grid, and to the right of the last rung the grid offers: the
        # demand carries its own fork room, so `limit` does not cap it.
        assert 36 % INTERVAL
        assert bm.checkpoint_limit(second) == 32

    def test_the_third_request_finds_what_the_second_was_missing(self):
        """Self-limiting: nothing to want, want it once, want nothing again."""
        bm = BlockManager(demand_config())

        first = stateful_seq(PROMPT)
        assert run_prompt_on_the_ladder(bm, first) == [32]  # the grid alone
        assert first.checkpoint_demand_pos == 0  # nothing was cached to fall short

        second = stateful_seq(PROMPT)
        bm.allocate(second, bm.can_allocate(second))
        assert second.num_cached_tokens == 32  # the grid got it this far...
        assert second.checkpoint_demand_pos == 36  # ...one block short of the rest
        assert forward_on_the_ladder(bm, second) == [36]  # one cut, for the gap

        third = stateful_seq(PROMPT)
        bm.allocate(third, bm.can_allocate(third))
        assert third.num_cached_tokens == 36
        assert third.checkpoint_demand_pos == 0  # nothing left to want
        assert forward_on_the_ladder(bm, third) == []

    def test_reuse_another_class_declines_is_not_charged_to_the_ladder(self):
        """The counterfactual keeps every other gate applied.

        A boundary whose sliding window is gone stays out of reach however
        densely the ring is checkpointed, so it must not buy a cut. Attributing
        the whole gap to the ladder would have every request pay for a
        checkpoint the next one still cannot use.
        """
        bm = BlockManager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        bm.state_caches = (*bm.state_caches, StubStateCache(cap=8))

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 8
        assert second.num_compressed_hit_blocks == 10  # 2 blocks declined...
        assert second.num_wanted_hit_blocks == 8  # ...none of it recoverable
        assert second.checkpoint_demand_pos == 0

    def test_a_demand_the_grid_cannot_express_is_kept_anyway(self):
        """The grid's granularity does not gate the evidence.

        A prompt with no room for a rung — shorter than an interval, or with
        its whole tail inside the last one — used to decline every reusable
        block it had: the demand was measured, compared against the interval,
        and dropped. But the interval is a guess about where reuse might
        resume, while a demand is reuse that was asked for and refused, and one
        is no reason to discard the other. This is the workload that motivates
        it: prompts under the interval, sharing a real prefix.
        """
        bm = BlockManager(demand_config())
        short = list(range(16))
        run_prompt_on_the_ladder(bm, stateful_seq(short))

        second = stateful_seq(short)
        assert bm.can_allocate(second) == 0
        assert bm.checkpoint_limit(second) == 0  # the grid places no rung here
        assert second.checkpoint_demand_pos == 8  # the demand is its own rung
        assert run_prompt_on_the_ladder(bm, second) == [8]

        third = stateful_seq(short)
        assert bm.can_allocate(third) == 2  # ...and the next one collects it
        assert third.checkpoint_demand_pos == 0  # nothing left to want
        assert run_prompt_on_the_ladder(bm, third) == []

    def test_the_demand_is_cut_and_kept_at_the_same_position(self):
        """The cut and the keep read the same call, so they cannot drift."""
        bm = BlockManager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert seq.checkpoint_demand_pos == 36

        n = len(PROMPT)
        cuts = {bm.checkpoint_cut(seq, pos - 1, pos) for pos in range(1, n + 1)}
        rungs = {pos for pos in range(1, n + 1) if bm.checkpointers_at(seq, pos)}
        assert cuts - {0} == rungs == {16, 32, 36}

    def test_a_recorded_demand_is_always_a_position_something_keeps(self):
        """Otherwise the cut is an extra forward that stores nothing.

        The demand comes out of the same fork test the ladder applies, on the
        same request, so it satisfies `successor_room` by construction. Swept
        rather than argued, because the two derivations sit in different files.
        """
        for n in range(20, 60, 3):
            bm = BlockManager(demand_config())
            tokens = list(range(1000 * n, 1000 * n + n))
            run_prompt_on_the_ladder(bm, stateful_seq(tokens))
            seq = stateful_seq(tokens)
            bm.allocate(seq, bm.can_allocate(seq))
            demand = seq.checkpoint_demand_pos
            assert not demand or bm.checkpointers_at(seq, demand), n

    def test_a_stateless_model_records_no_demand(self):
        bm = BlockManager(
            demand_config(
                pool_entries={}, state_transfer_kind="none", state_fork_tokens=0
            )
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        run_prompt_on_the_ladder(bm, cold)
        warm = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        assert bm.can_allocate(warm) == 10  # nothing was gating it
        assert warm.checkpoint_demand_pos == 0


class TestCacheStatsAttribution:
    """Splitting declined reuse into the part a checkpoint reaches and the rest.

    One number for both makes "does demand-driven checkpointing apply to this
    workload" unfalsifiable, which is the whole reason the counterfactual is
    computed outside the tests.
    """

    def test_the_split_accounts_for_every_declined_token(self):
        stats = CacheStats(log_interval=10**6)
        stats.update(32, 44, 40, 36)
        lost_to_checkpoint = stats.total_wanted_tokens - stats.total_cached_tokens
        lost_hard = stats.total_compressed_tokens - stats.total_wanted_tokens
        assert lost_to_checkpoint == 4
        assert lost_hard == 4
        assert lost_to_checkpoint + lost_hard == 40 - 32

    def test_hit_tokens_are_counted_in_hash_blocks(self):
        """Under DCP one block_table entry spans `dcp` blocks of tokens."""
        sched = Scheduler(demand_config(decode_context_parallel_size=2))
        assert sched.block_manager.hash_block_size == 2 * BLOCK
        seq = stateful_seq(PROMPT)
        seq.num_compressed_hit_blocks = 3
        seq.num_wanted_hit_blocks = 2
        sched._schedule_prefill_seq(seq, 44, {}, [], 0, 0)
        assert sched.cache_stats.total_compressed_tokens == 3 * 2 * BLOCK
        assert sched.cache_stats.total_wanted_tokens == 2 * 2 * BLOCK


class TestGenerationIsHeldToSpacingNotTheGrid:
    """A step that cannot choose where it ends is judged by distance instead.

    Prefill lands where `checkpoint_cut` puts it, so it meets the grid exactly.
    A speculative decode step commits `1 + accepted` and steps over most rungs;
    held to the grid it would keep a checkpoint only when the arithmetic
    happened to divide out. The grid is there to space checkpoints, and any
    hash-block boundary far enough past the last one spaces them just as well —
    a resumer finds a checkpoint by hash, never by arithmetic.

    `demand_config`, whose grid is several hash blocks wide: where the two
    coincide there is no rule to tell apart.
    """

    def keepers(self, bm, seq, pos, aimed):
        # Room to spare: what is under test is which positions qualify, not
        # whether a class has enough forward left to take one there.
        return bm.checkpointers_at(seq, pos, MIN_FORK, aimed=aimed)

    def test_an_aimed_step_is_held_to_the_grid(self):
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        assert self.keepers(bm, seq, INTERVAL, aimed=True)
        assert not self.keepers(bm, seq, INTERVAL + BLOCK, aimed=True)

    def test_an_unaimed_step_keeps_off_the_grid(self):
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        assert self.keepers(bm, seq, INTERVAL + BLOCK, aimed=False)

    def test_an_unaimed_step_still_has_to_land_on_a_block(self):
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        # The checkpoint is filed under the hash of a whole block, so a landing
        # between two of them has nothing to file it under.
        assert not self.keepers(bm, seq, INTERVAL + 1, aimed=False)

    def test_spacing_is_measured_from_the_last_one_kept(self):
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        seq.last_checkpoint_pos = INTERVAL + BLOCK
        assert not self.keepers(bm, seq, 2 * INTERVAL, aimed=False)
        assert self.keepers(bm, seq, 2 * INTERVAL + BLOCK, aimed=False)

    def test_the_grid_ignores_the_watermark(self):
        # An aimed caller answers to `checkpoint_cut`, which knows nothing of
        # the watermark; letting it in here would put the two out of step.
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        seq.last_checkpoint_pos = INTERVAL
        assert self.keepers(bm, seq, 2 * INTERVAL, aimed=True)

    def test_a_demand_is_out_of_generation_s_reach(self):
        # Not a rule, an arithmetic fact: a demand is bounded by the prompt's
        # own hit ceiling, and generation only ever asks about positions at or
        # past the end of the prompt. The unaimed branch omits the demand
        # because of this, so the day it stops holding, this fails first.
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        second = stateful_seq(PROMPT)
        bm.allocate(second, bm.can_allocate(second))
        assert second.checkpoint_demand_pos < second.num_prompt_tokens
