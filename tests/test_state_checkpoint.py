# SPDX-License-Identifier: MIT
# Tests for per-request state checkpoints: the third prefix-cache gate.
#
# Neither the GDN recurrent state nor the V4 compressor ring can be rebuilt
# from cached KV blocks, so a prefix hit is only resumable at a boundary where
# some earlier request published its state. `StateCheckpointPool` indexes those
# boundaries and `BlockManager` shrinks the hit to the rightmost one — without
# it, a hit hands the resumed forward a group straight off the free list and it
# reads the previous occupant's state.
#
# Capacity model under test: a checkpoint is a FREE group whose content is
# still valid (the KV block pool's lazy eviction, applied to state groups). So
# checkpoints must never reduce the number of admissible requests, and the
# eviction event is hand-out, not free.

from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_pool import StateCheckpointPool

BLOCK = 4
MIN_FORK = 8


def ckpt_config(**overrides):
    defaults = dict(
        kv_cache_block_size=BLOCK,
        num_kvcache_blocks=200,
        enable_prefix_caching=True,
        max_num_seqs=4,
        max_num_batched_tokens=256,
        max_model_len=256,
        bos_token_id=1,
        eos_token_id=2,
        stop_token_ids=[],
        scheduler_delay_factor=0.0,
        speculative_config=None,
        pool_entries={"state": 4},
        state_min_fork_tokens=MIN_FORK,
        state_checkpoint_interval=1,
    )
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
    boundary = bm.state_publish_limit(seq)
    assert boundary > 0
    bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
    return boundary_hash(bm, seq)


def boundary_hash(bm: BlockManager, seq: Sequence) -> int:
    """Content hash of the last block before this seq's checkpoint boundary."""
    last = bm.state_publish_limit(seq) // bm.hash_block_size - 1
    return bm.blocks[seq.block_table[last]].hash


# ── StateCheckpointPool in isolation ───────────────────────────────────────


class TestPoolIndex:

    def test_disabled_is_identity(self):
        pool = StateCheckpointPool(0)
        assert pool.bounded_hit(5, [1, 2, 3, 4, 5]) == 5
        assert pool.lookup(1) == -1

    def test_bounded_hit_picks_rightmost_checkpoint(self):
        pool = StateCheckpointPool(4)
        pool.publish(10, 0)
        pool.publish(30, 1)
        # hashes for blocks 0..4; checkpoints exist after block 0 and block 2
        assert pool.bounded_hit(5, [10, 20, 30, 40, 50]) == 3

    def test_bounded_hit_zero_when_nothing_published(self):
        pool = StateCheckpointPool(4)
        assert pool.bounded_hit(5, [10, 20, 30, 40, 50]) == 0

    def test_bounded_hit_respects_forkable_predicate(self):
        pool = StateCheckpointPool(4)
        pool.publish(10, 0)
        pool.publish(30, 1)
        # The rightmost boundary (3) is rejected, so the scan walks back to 1.
        assert pool.bounded_hit(5, [10, 20, 30, 40, 50], lambda L: L < 3) == 1

    def test_invalidate_drops_both_directions(self):
        pool = StateCheckpointPool(4)
        pool.publish(10, 2)
        pool.invalidate(2)
        assert pool.lookup(10) == -1
        # A later invalidate of the same group must not delete a new tenant.
        pool.publish(10, 3)
        pool.invalidate(2)
        assert pool.lookup(10) == 3

    def test_republishing_a_hash_orphans_the_old_group(self):
        pool = StateCheckpointPool(4)
        pool.publish(10, 1)
        pool.publish(10, 2)
        assert pool.lookup(10) == 2
        # Group 1 no longer backs hash 10; invalidating it leaves 2 indexed.
        pool.invalidate(1)
        assert pool.lookup(10) == 2

    def test_pins_drain_once(self):
        pool = StateCheckpointPool(4)
        pool.pin(1)
        pool.pin(3)
        assert pool.is_pinned(1)
        assert pool.take_pins() == [1, 3]
        assert pool.take_pins() == []
        assert not pool.is_pinned(1)


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
        bm = BlockManager(ckpt_config(pool_entries={}, state_min_fork_tokens=0))
        first = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        run_prompt(bm, first)
        second = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        # 10 blocks of prompt, the last never reused → full 9-block hit.
        assert bm.can_allocate(second) == 9

    def test_hit_lands_on_the_published_boundary(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        publish_at_boundary(bm, first)
        boundary = bm.state_publish_limit(first)

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
        assert len(bm.free_per_req_cache_groups) == 4
        for i in range(4):
            seq = stateful_seq(list(range(900 + 20 * i, 920 + 20 * i)))
            assert bm.can_allocate(seq) >= 0
            bm.allocate(seq, 0)
        assert len(bm.free_per_req_cache_groups) == 0

    def test_handout_evicts_the_checkpoint_it_lands_on(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        group = bm.state.lookup(h)
        bm.deallocate(first)
        # Drain the queue until the checkpoint's group comes back around.
        while bm.free_per_req_cache_groups:
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
        group = bm.state.lookup(h)
        assert len(bm.free_per_req_cache_groups) == 1

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
        boundary = bm.state_publish_limit(seq)
        bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
        assert seq.per_req_cache_group != before
        assert seq.state_fork_src == before
        assert bm.state.lookup(boundary_hash(bm, seq)) == before

    def test_no_publish_when_the_forward_misses_the_boundary(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        group = seq.per_req_cache_group
        bm.hash_blocks(seq, bm.state_publish_limit(seq) + BLOCK)
        assert seq.per_req_cache_group == group
        assert not bm.state.hash_to_group

    def test_boundary_leaves_room_for_the_fork_forward(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        boundary = bm.state_publish_limit(seq)
        assert boundary % bm.hash_block_size == 0
        assert seq.num_prompt_tokens - boundary >= MIN_FORK

    def test_every_block_boundary_up_to_the_limit_qualifies(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        limit = bm.state_publish_limit(seq)
        assert bm.is_state_publish_pos(seq, BLOCK)
        assert bm.is_state_publish_pos(seq, limit)
        assert not bm.is_state_publish_pos(seq, limit + BLOCK)  # no room to fork
        assert not bm.is_state_publish_pos(seq, BLOCK + 2)  # not block aligned
        assert not bm.is_state_publish_pos(seq, 0)

    def test_chunked_prefill_leaves_a_ladder_of_checkpoints(self):
        """Intermediate boundaries publish too — the CPU-offload resume points."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        for _ in range(4):
            bm.hash_blocks(seq, 2 * BLOCK, start_tokens=seq.num_cached_tokens)
            seq.num_cached_tokens += 2 * BLOCK
        # Four publishes into four groups: the oldest was recycled to serve the
        # last one, the rest stand as distinct resume points.
        assert len(bm.state.hash_to_group) == 3
        assert bm.state.lookup(boundary_hash(bm, seq)) >= 0  # the rightmost one

    def test_interval_thins_the_ladder_but_keeps_the_limit(self):
        bm = BlockManager(ckpt_config(state_checkpoint_interval=3))
        seq = stateful_seq(list(range(40)))
        limit = bm.state_publish_limit(seq)
        published = [
            pos
            for pos in range(BLOCK, limit + BLOCK, BLOCK)
            if bm.is_state_publish_pos(seq, pos)
        ]
        # Every third block, and the limit whatever the interval says.
        assert published == [3 * BLOCK, 6 * BLOCK, limit]

    def test_interval_zero_keeps_only_the_limit(self):
        bm = BlockManager(ckpt_config(state_checkpoint_interval=0))
        seq = stateful_seq(list(range(40)))
        limit = bm.state_publish_limit(seq)
        assert limit > 0
        assert bm.is_state_publish_pos(seq, limit)
        assert not any(
            bm.is_state_publish_pos(seq, pos) for pos in range(BLOCK, limit, BLOCK)
        )

    def test_hit_never_lands_where_swa_cannot_follow(self):
        """The two gates settle jointly; neither is applied to the other's answer.

        `swa.bounded_hit` promises the rightmost boundary whose trailing window
        is present. Shrinking that answer to a checkpoint boundary can land
        somewhere SWA never approved, and `allocate` would then claim an SWA
        hash the pool never promised.
        """
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        published = [2, 5]  # checkpoint boundaries, in blocks
        approved = 5  # the only boundary SWA accepts

        bm.state.hash_to_group = {}
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate(published):
            bm.state.publish(hashes[boundary - 1], group)
        bm.swa.bounded_hit = lambda s, p, h, _a=approved: min(p, _a)

        assert bm._gated_hit(seq, 9, hashes) == approved

        # Now SWA only accepts 4: the rightmost checkpoint (5) is out of reach,
        # so the answer must fall back to 2 rather than stay at 5 or become 4.
        approved = 4
        bm.swa.bounded_hit = lambda s, p, h, _a=approved: min(p, _a)
        assert bm._gated_hit(seq, 9, hashes) == 2

    def test_no_boundary_when_the_backend_cannot_fork(self):
        bm = BlockManager(ckpt_config(state_min_fork_tokens=0))
        seq = stateful_seq(list(range(40)))
        assert bm.state_publish_limit(seq) == 0
        assert not bm.is_state_publish_pos(seq, 16)

    def test_cancel_adopts_the_source_and_returns_the_new_group(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        source = seq.per_req_cache_group
        bm.hash_blocks(seq, bm.state_publish_limit(seq) - seq.num_cached_tokens)
        free_after_publish = len(bm.free_per_req_cache_groups)

        bm.cancel_state_fork(seq)
        assert seq.per_req_cache_group == source
        assert seq.state_fork_src == -1
        assert not bm.state.hash_to_group
        assert len(bm.free_per_req_cache_groups) == free_after_publish

    def test_two_resumers_in_one_step_share_the_checkpoint(self):
        # A checkpoint is read-only, so a second request hitting the same prefix
        # before the pins are released must fork off it too — not try to claim a
        # group the first one already took off the free list.
        bm = BlockManager(ckpt_config(pool_entries={"state": 8}))
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))

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
        before = len(bm.free_per_req_cache_groups)
        bm.release_state_pins()
        assert len(bm.free_per_req_cache_groups) == before + 1

    def test_cancel_refuses_to_adopt_a_shared_source(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))

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

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert bm.state.is_pinned(src)
        bm.cancel_state_fork(second)
        assert second.per_req_cache_group == src
        assert not bm.state.is_pinned(src)
        # The pin must not also hand the group back — it has an owner now.
        bm.release_state_pins()
        assert src not in bm.free_per_req_cache_groups

    def test_pinned_source_returns_to_the_free_list_next_step(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert src not in bm.free_per_req_cache_groups
        bm.release_state_pins()
        assert src in bm.free_per_req_cache_groups
