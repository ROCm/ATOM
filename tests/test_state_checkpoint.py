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

from math import isinf
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import CacheStats, ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.model_engine.state_cache import StateCache
from atom.model_engine.state_pool import StateGroupPool, StateTransfer
from atom.model_engine.swa_pool import SlidingWindowPool

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
        pool.free_groups.clear()  # every group out with a request
        pool.pin(1)
        pool.pin(3)
        assert pool.is_pinned(1)
        pool.release_pins()
        assert list(pool.free_groups) == [1, 3]
        pool.release_pins()  # idempotent: a drained pin is not freed twice
        assert list(pool.free_groups) == [1, 3]
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
        assert len(bm.state.free_groups) == 4
        for i in range(4):
            seq = stateful_seq(list(range(900 + 20 * i, 920 + 20 * i)))
            assert bm.can_allocate(seq) >= 0
            bm.allocate(seq, 0)
        assert len(bm.state.free_groups) == 0

    def test_handout_evicts_the_checkpoint_it_lands_on(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        group = bm.state.lookup(h)
        bm.deallocate(first)
        # Drain the queue until the checkpoint's group comes back around.
        while bm.state.free_groups:
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
        assert len(bm.state.free_groups) == 1

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

    def test_interval_must_divide_the_hash_block_size(self):
        """A rung off the block grid has no content hash to be filed under."""
        with pytest.raises(AssertionError, match="must be a multiple"):
            BlockManager(ckpt_config(state_checkpoint_interval_tokens=BLOCK + 1))

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
        approved = 5  # the only boundary SWA accepts

        bm.state.hash_to_group = {}
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate(published):
            bm.state._index(hashes[boundary - 1], group)
        bm.swa.resumable_hit = (
            lambda s, p, h, assume_checkpointed=False, _a=approved: min(p, _a)
        )

        assert bm._gated_hit(seq, 9, hashes) == approved

        # Now SWA only accepts 4: the rightmost checkpoint (5) is out of reach,
        # so the answer must fall back to 2 rather than stay at 5 or become 4.
        approved = 4
        bm.swa.resumable_hit = (
            lambda s, p, h, assume_checkpointed=False, _a=approved: min(p, _a)
        )
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
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        free_after_publish = len(bm.state.free_groups)

        bm.cancel_state_fork(seq)
        assert seq.per_req_cache_group == source
        assert seq.state_fork_src == -1
        assert not bm.state.hash_to_group
        assert len(bm.state.free_groups) == free_after_publish

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
        before = len(bm.state.free_groups)
        bm.release_state_pins()
        assert len(bm.state.free_groups) == before + 1

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
        assert src not in bm.state.free_groups

    def test_pinned_source_returns_to_the_free_list_next_step(self):
        bm = BlockManager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert src not in bm.state.free_groups
        bm.release_state_pins()
        assert src in bm.state.free_groups


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
        assert dst in bm.state.free_groups
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
        bm.state.free_groups.clear()

        # committed by state_copies_for_batch()
        assert not bm.state.hash_to_group
        assert not bm.state_copies_for_batch()

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
        while len(bm.state.free_groups) > 1:
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
        spec = SimpleNamespace(num_speculative_tokens=3)
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
            speculative_config=SimpleNamespace(num_speculative_tokens=3)
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


# ── One ladder, two state classes ──────────────────────────────────────────
#
# SWA and the compressor ring are both `Pool.STATE` (sub_pool_spec.py): both
# scale with in-flight requests, both can keep a boundary resumable, both can
# veto a hit. They differ only in mutability, and `successor_room` is that
# difference quantified — which is all the ladder knows about either.


def swa_pool(**overrides):
    kwargs = {"num_blocks": 64, "window": 16, "block_size": BLOCK, "mtp_k": 0}
    kwargs.update(overrides)
    return SlidingWindowPool(**kwargs)


class TestStateCacheProtocol:

    def test_both_classes_satisfy_the_protocol(self):
        assert isinstance(swa_pool(), StateCache)
        assert isinstance(StateGroupPool(4), StateCache)

    def test_a_class_that_keeps_nothing_reports_inf(self):
        """`inf` is what stops the ladder cutting chunks for a class in vain.

        The window pool only ever materializes the trailing window, so no older
        boundary has anything left to hold on to; reporting 0 would have the
        scheduler cut prefill chunks at every rung for a class that stores
        nothing there — cost with no reuse.
        """
        assert isinf(swa_pool().successor_room)
        assert isinf(StateGroupPool(4, StateTransfer.none()).successor_room)

    def test_the_limit_follows_the_class_that_reaches_furthest(self):
        """The smallest room reaches furthest right; a larger one must not cap it."""
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 32  # the ring alone: 40 - MIN_FORK
        bm.swa.enabled = True
        bm.swa.successor_room = 0
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
        bm.swa.enabled = True
        bm.swa.successor_room = 0
        assert bm.checkpointers_at(seq, pos) == [bm.swa]

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
        bm.swa.resumable_hit = lambda s, p, h, assume_checkpointed=False: min(p, 4)

        answer = bm._gated_hit(seq, 9, hashes)
        for cache in bm.state_caches:
            assert cache.resumable_hit(seq, answer, hashes) == answer

    def test_order_between_classes_does_not_change_the_answer(self):
        bm = BlockManager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate([2, 5]):
            bm.state._index(hashes[boundary - 1], group)
        bm.swa.resumable_hit = lambda s, p, h, assume_checkpointed=False: min(p, 4)

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
        bm.swa.resumable_hit = lambda s, p, h, assume_checkpointed=False: min(p, 8)

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 8
        assert second.num_compressed_hit_blocks == 10  # 2 blocks declined...
        assert second.num_wanted_hit_blocks == 8  # ...none of it recoverable
        assert second.checkpoint_demand_pos == 0

    def test_demand_under_one_interval_costs_nothing(self):
        """`checkpoint_limit`'s promise, from the other side.

        A demand of at least one interval implies a limit above zero, because
        the demand carries `successor_room` behind it by construction. So a
        workload keeping no checkpoints today cannot start paying for cuts —
        the threshold is what makes that statable rather than probable.
        """
        bm = BlockManager(demand_config())
        short = list(range(16))  # under one interval
        run_prompt_on_the_ladder(bm, stateful_seq(short))

        second = stateful_seq(short)
        assert bm.can_allocate(second) == 0
        assert second.num_wanted_hit_blocks == 2  # a checkpoint at 8 would land
        assert bm.checkpoint_limit(second) == 0  # but this prompt keeps none
        assert second.checkpoint_demand_pos == 0
        assert run_prompt_on_the_ladder(bm, second) == []

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
