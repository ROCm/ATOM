# SPDX-License-Identifier: MIT
# Tests for per-request state checkpoints: the third prefix-cache gate.
#
# Neither the GDN recurrent state nor the V4 compressor ring can be rebuilt
# from cached KV blocks, so a prefix hit is only resumable at a boundary where
# some earlier request published its state. `StateSlotPool` indexes those
# boundaries and `BlockManager` shrinks the hit to the rightmost one — without
# it, a hit hands the resumed forward a group straight off the free list and it
# reads the previous occupant's state.
#
# Fork-transfer checkpoints are FREE groups whose content is still valid.
# Copy-transfer checkpoints are immutable PAGE-unit images; Active Slots are
# reserved only for resident requests and never serve as checkpoint backing.

import logging
from math import inf, isinf
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.engine_stats import EngineStats
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.model_engine.state_cache import StateCache
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import (
    StateRuntime,
    StateTransfer,
)

BLOCK = 4
MIN_FORK = 8
PAGED_COPY_SPEC = PagedStateCheckpointSpec(10, 25, "test-layout-v1", image_bytes=25)
DEFAULT_STATE_TRANSFER = StateTransfer.fork(MIN_FORK)
PAGED_COPY_TRANSFER = StateTransfer.copy(PAGED_COPY_SPEC.layout_id)
DEFAULT_STATE_RUNTIME = StateRuntime(transfer=DEFAULT_STATE_TRANSFER)
PAGED_COPY_RUNTIME = StateRuntime(
    transfer=PAGED_COPY_TRANSFER,
    checkpoint_spec=PAGED_COPY_SPEC,
)


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
        "state_checkpoint_interval_tokens": BLOCK,
    }
    defaults.update(overrides)
    return MockConfig(**defaults)


def make_block_manager(
    config,
    *,
    state_runtime=DEFAULT_STATE_RUNTIME,
):
    return BlockManager(
        config,
        state_runtime=state_runtime,
    )


def make_scheduler(
    config,
    *,
    state_runtime=DEFAULT_STATE_RUNTIME,
):
    return Scheduler(
        config,
        state_runtime=state_runtime,
    )


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
    than each spelling out two lifecycle calls.
    """
    bm.complete_previous_state_batch()
    bm.complete_previous_state_batch()


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


# ── StateSlotPool in isolation ────────────────────────────────────────────


def idx_seq(num_tokens: int = 1000):
    """The two Sequence fields `resumable_hit` reads, and nothing else."""
    return SimpleNamespace(num_tokens=num_tokens, has_per_req_cache=True)


class TestPoolIndex:

    def test_disabled_is_identity(self):
        pool = StateSlotPool(0)
        assert pool.resumable_hit(idx_seq(), 5, [1, 2, 3, 4, 5]) == 5
        assert pool.lookup(1) == -1

    def test_resumable_hit_picks_rightmost_checkpoint(self):
        pool = StateSlotPool(4, StateTransfer.fork(1), hash_block_size=1)
        pool._index(10, 0)
        pool._index(30, 1)
        # hashes for blocks 0..4; checkpoints exist after block 0 and block 2
        assert pool.resumable_hit(idx_seq(), 5, [10, 20, 30, 40, 50]) == 3

    def test_resumable_hit_zero_when_nothing_published(self):
        pool = StateSlotPool(4, StateTransfer.fork(1), hash_block_size=1)
        assert pool.resumable_hit(idx_seq(), 5, [10, 20, 30, 40, 50]) == 0

    def test_resumable_hit_walks_back_when_the_fork_has_no_room(self):
        pool = StateSlotPool(4, StateTransfer.fork(4), hash_block_size=1)
        pool._index(10, 0)
        pool._index(30, 1)
        # One token per block, five in the seq: the rightmost checkpoint
        # (boundary 3) leaves only 2 tokens to forward, short of the 4 a fork
        # needs, so the scan walks back to boundary 1, which leaves 4.
        assert pool.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 1

    def test_invalidate_drops_both_directions(self):
        pool = StateSlotPool(4)
        pool._index(10, 2)
        pool.invalidate(2)
        assert pool.lookup(10) == -1
        # A later invalidate of the same group must not delete a new tenant.
        pool._index(10, 3)
        pool.invalidate(2)
        assert pool.lookup(10) == 3

    def test_republishing_a_hash_orphans_the_old_group(self):
        pool = StateSlotPool(4)
        pool._index(10, 1)
        pool._index(10, 2)
        assert pool.lookup(10) == 2
        # Group 1 no longer backs hash 10; invalidating it leaves 2 indexed.
        pool.invalidate(1)
        assert pool.lookup(10) == 2

    def test_pins_drain_once(self):
        pool = StateSlotPool(4)
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
        pool = StateSlotPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool.release(1)

        assert pool.pop() == 1
        assert pool.lookup(10) == 0

    def test_admission_packs_towards_index_zero(self):
        pool = StateSlotPool(4)
        drain(pool)
        for group in (3, 1, 2):
            pool.release(group)
        assert [pool.pop() for _ in range(3)] == [1, 2, 3]

    def test_checkpoints_are_spent_least_recently_used_first(self):
        pool = StateSlotPool(4)
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
        pool = StateSlotPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.claim(0)  # a resumer reads the oldest checkpoint
        pool.pin(0)
        pool.release_pins()

        assert pool.pop() == 1  # 11 is now the older of the two
        assert pool.lookup(10) == 0

    def test_a_speculative_checkpoint_is_spent_before_any_anchor(self):
        """A guess must never evict knowledge, however old the knowledge is.

        Group 0 holds an anchor released first, so plain LRU would spend it.
        Group 1 is marked speculative and lands at the head instead, which is
        what makes the demand rung cost the anchors nothing.

        Indexed before it is released, which is the order the fork path takes:
        the group is still its owner's when the hash is filed.
        """
        pool = StateSlotPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool._index(11, 1)
        pool.mark_speculative(1)
        pool.release(1)

        assert pool.pop() == 1
        assert pool.lookup(10) == 0

    def test_a_seq_with_no_anchor_files_only_guesses(self):
        """`checkpoint_end_pos == 0` means no known resume point, not "0 is it".

        `_record_checkpoint_end` leaves the anchor at 0 on four paths, one of
        which is every prompt too short to have a keepable end. Reading that as
        "nothing to demote" files those seqs' ladder and demand rungs at the
        LRU tail beside the real anchors of long prompts -- and then spends the
        anchors first, which is the ordering `mark_speculative` exists to
        invert.
        """
        pool = StateSlotPool(4)
        drain(pool)
        anchored = SimpleNamespace(checkpoint_end_pos=64)
        unanchored = SimpleNamespace(checkpoint_end_pos=0)

        pool.publish_midstep([(0, 64, 10)], anchored)
        pool.publish_midstep([(1, 32, 11)], unanchored)

        assert 0 not in pool._speculative, "the anchor's own position is known"
        assert 1 in pool._speculative, "a seq with no anchor has only guesses"
        assert pool.pop() == 1, "and the guess is spent first"

    def test_publishing_without_a_seq_demotes_nothing(self):
        """The caller cannot tell a guess from knowledge, so it keeps.

        Over-keeping costs one eviction later; over-demoting spends an anchor
        that would have been read back, and there is no way to get it back.
        """
        pool = StateSlotPool(4)
        drain(pool)
        pool.publish_midstep([(0, 32, 10), (1, 64, 11)], None)
        assert not pool._speculative

    def test_speculative_checkpoints_keep_lru_among_themselves(self):
        pool = StateSlotPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11)):
            pool._index(h, group)
            pool.mark_speculative(group)
            pool.release(group)

        # Filed at the head, so the *later* one is spent first: neither has
        # been read, and the older has had longer to prove it never will be.
        assert pool.pop() == 1
        assert pool.pop() == 0

    def test_a_read_speculative_checkpoint_is_promoted(self):
        """Being resumed from is the evidence the guess was right.

        `BlockManager._attach_state_slots` promotes the source it is about to
        fork off, so a demand rung that pays off stops being spent first.
        """
        pool = StateSlotPool(4)
        drain(pool)
        pool._index(10, 0)
        pool.mark_speculative(0)
        pool.release(0)
        pool.release(1)
        pool._index(11, 1)

        pool.promote(0)  # a resumer reads the speculative checkpoint

        assert pool.pop() == 1  # 10 is no longer the first thing spent
        assert pool.lookup(10) == 0

    def test_promoting_a_group_nobody_marked_leaves_the_order_alone(self):
        """`_attach_state_slots` promotes every source, most of them anchors."""
        pool = StateSlotPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.promote(1)

        assert pool.pop() == 0  # still the older of the two

    def test_republishing_a_hash_returns_the_orphan_to_the_vacant_half(self):
        pool = StateSlotPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool.release(1)
        pool._index(10, 1)  # group 0 no longer backs anything

        assert pool.pop() == 0  # vacant again, so it goes before the checkpoint
        assert pool.lookup(10) == 1


class TestShrinking:
    def test_a_vacant_top_costs_nothing(self):
        pool = StateSlotPool(4)
        out = pool.retire_top()
        assert (out.retired, out.relocated_to) == (3, -1)
        assert pool.num_slots == 3
        assert not pool.is_free(3)

    def test_a_live_top_moves_into_the_lowest_vacant_group(self):
        pool = StateSlotPool(4)
        drain(pool)
        pool.release(2)  # only group 2 is free; 3 is held by a request

        out = pool.retire_top()
        assert (out.retired, out.relocated_to, out.held_checkpoint) == (3, 2, False)
        assert pool.num_slots == 3

    def test_shrinking_spends_the_oldest_checkpoint_not_the_top_one(self):
        """The whole reason `retire_top` relocates instead of just dropping.

        A group's index records the concurrency high-water mark when it was
        handed out and is never refreshed by use, so the hottest checkpoint can
        sit at the top. Retiring by index alone would spend it and leave one
        nothing has touched in minutes.
        """
        pool = StateSlotPool(4)
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
        assert pool.num_slots == 3

    def test_the_top_is_spent_when_it_is_itself_the_oldest(self):
        pool = StateSlotPool(2)
        drain(pool)
        pool.release(1)
        pool._index(13, 1)

        out = pool.retire_top()
        assert (out.retired, out.relocated_to, out.held_checkpoint) == (1, -1, True)
        assert pool.lookup(13) == -1

    def test_a_pinned_top_is_refused_rather_than_moved(self):
        """It is being read by the in-flight step; the pin drains next pass."""
        pool = StateSlotPool(4)
        drain(pool)
        pool.pin(3)
        assert pool.retire_top() is None
        assert pool.num_slots == 4

    def test_a_live_top_with_nowhere_to_go_is_refused(self):
        pool = StateSlotPool(4)
        drain(pool)
        assert pool.retire_top() is None
        assert pool.num_slots == 4

    def test_growing_adds_groups_at_the_top(self):
        pool = StateSlotPool(2)
        drain(pool)
        pool.extend(2)
        assert pool.num_slots == 4
        assert [pool.pop() for _ in range(2)] == [2, 3]

    def test_the_vacant_heap_does_not_grow_without_bound(self):
        """Taking a hash while vacant leaves an entry behind; churn compacts.

        Nothing observable depends on this, which is why it is asserted
        directly: on a long-lived server the stale entries otherwise outnumber
        the live ones by the number of checkpoints ever taken.
        """
        pool = StateSlotPool(4)
        for round_ in range(200):
            group = pool.pop()
            pool.release(group)
            pool._index(round_, group)  # promotes it, stranding a heap entry
            pool.claim(group)
            pool.slot_hash[group] = -1
            pool.release(group)
        assert len(pool._vacant) <= 2 * pool.num_slots + 2

    def test_regrowing_a_retired_index_reuses_its_hash_slot(self):
        """Not appending a second one, which would shift every index above it."""
        pool = StateSlotPool(3)
        assert pool.retire_top().retired == 2
        pool.extend(1)

        assert pool.num_slots == 3
        assert len(pool.slot_hash) == 3
        drain(pool)
        pool.release(2)
        pool._index(12, 2)
        assert pool.lookup(12) == 2


# ── BlockManager: the hit is shrunk to a resumable boundary ────────────────


class TestHitShrink:

    def test_hit_is_zero_without_a_checkpoint(self):
        """The correctness fix: a stateful model cannot resume a bare KV hit."""
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        run_prompt(bm, first)
        # Same prompt again: compressed blocks are all cached, but the first
        # request published nothing (its forward never ended on the boundary).
        second = stateful_seq(list(range(40)))
        assert bm.can_allocate(second) == 0
        assert second.num_compressed_hit_blocks > 0

    def test_stateless_model_keeps_the_full_hit(self):
        bm = make_block_manager(
            ckpt_config(pool_entries={}),
            state_runtime=StateRuntime(),
        )
        first = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        run_prompt(bm, first)
        second = Sequence(list(range(40)), BLOCK, has_per_req_cache=False)
        # 10 blocks of prompt, the last never reused → full 9-block hit.
        assert bm.can_allocate(second) == 9

    def test_hit_lands_on_the_published_boundary(self):
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        publish_at_boundary(bm, first)
        boundary = bm.checkpoint_limit(first)

        second = stateful_seq(list(range(40)))
        assert bm.can_allocate(second) * bm.hash_block_size == boundary

    def test_resume_reads_the_checkpoint_and_writes_a_fresh_group(self):
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        src = bm.state.lookup(h)
        assert src >= 0

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert second.state_fork_src == src
        assert second.state_slot != src
        # The checkpoint survives the resume, so a third request still finds it.
        assert bm.state.lookup(h) == src


# ── Capacity: checkpoints live on the free list, never hold it back ────────


class TestCapacity:

    def test_checkpoints_do_not_reduce_admission(self):
        """A published checkpoint is a free group; concurrency is unchanged."""
        bm = make_block_manager(ckpt_config())
        for i in range(4):
            seq = stateful_seq(list(range(100 * i, 100 * i + 20 + 4 * i)))
            publish_at_boundary(bm, seq)
            bm.deallocate(seq)
        # Some checkpoints survive, older ones were recycled by the FIFO — the
        # point is that neither outcome costs a group.
        assert bm.state.hash_to_slot
        # Every group is back, so the pool admits its full concurrency.
        assert bm.state.num_free() == 4
        for i in range(4):
            seq = stateful_seq(list(range(900 + 20 * i, 920 + 20 * i)))
            assert bm.can_allocate(seq) >= 0
            bm.allocate(seq, 0)
        assert bm.state.num_free() == 0

    def test_handout_evicts_the_checkpoint_it_lands_on(self):
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        group = bm.state.lookup(h)
        bm.deallocate(first)
        # Drain the queue until the checkpoint's group comes back around.
        while bm.state.has_free():
            seq = stateful_seq(list(range(900, 920)))
            bm.allocate(seq, 0)
            if seq.state_slot == group:
                break
        assert bm.state.lookup(h) == -1

    def test_resume_without_a_spare_group_adopts_the_checkpoint(self):
        # Two groups: the publisher keeps one, so the only free group when the
        # resume arrives is the checkpoint itself.
        bm = make_block_manager(ckpt_config(pool_entries={"state": 2}))
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        group = bm.state.lookup(h)
        assert bm.state.num_free() == 1

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        # No second group to fork into, so the resume spends the checkpoint —
        # still exactly the state it wanted, just no longer shareable.
        assert second.state_slot == group
        assert second.state_fork_src == -1
        assert bm.state.lookup(h) == -1


# ── A request is wide, a checkpoint is one slot ────────────────────────────
#
# The asymmetry this whole pool exists to express. Every other test in this
# file runs at `state_slots_per_req == 1`, where a request and a checkpoint
# happen to be the same size and nothing can tell the two apart. These run at
# 3 — `--num-speculative-tokens 2` — which is the config the change was made
# for: there, a checkpoint that took a request's width would waste two thirds
# of its bytes, and those bytes come out of the same budget as the KV cache.


def spec_config(slots, **overrides):
    """`slots` raw slots, three of which one live request takes."""
    return ckpt_config(
        pool_entries={"state": slots},
        pool_entries_per_req={"state": 3},
        **overrides,
    )


class TestPerNeedWidth:

    def test_a_live_request_takes_its_whole_width(self):
        bm = make_block_manager(spec_config(9))
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        assert len(seq.state_slots) == 3
        assert len(set(seq.state_slots)) == 3  # no slot handed out twice
        assert bm.state.num_free() == 6

    def test_a_checkpoint_takes_one(self):
        """The point of the change: 3 slots per request, 1 per checkpoint.

        The publisher hands its committed slot to the index and takes a fresh
        one, so it still holds 3 afterwards — and the checkpoint beside it is
        one slot, not another 3. At `spr == 3` the old sizing spent 9 slots to
        end up here; this spends 4.
        """
        bm = make_block_manager(spec_config(9))
        seq = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, seq)
        assert bm.state.lookup(h) >= 0
        assert len(seq.state_slots) == 3
        # 9 - 3 held by the seq - 1 pinned checkpoint.
        assert bm.state.num_free() == 5

    def test_admission_gates_on_the_full_width(self):
        """Not on one slot: a request admitted on 1 would find no scratch."""
        bm = make_block_manager(spec_config(4))
        first = stateful_seq(list(range(40)))
        assert bm.can_allocate(first) >= 0
        bm.allocate(first, 0)
        assert bm.state.num_free() == 1  # non-zero, but short of a width

        second = stateful_seq(list(range(900, 940)))
        assert bm.can_allocate(second) == -1

    def test_a_resume_settles_at_what_a_cold_start_costs(self):
        """Three slots in steady state, four while the fork is in flight.

        The fourth is the checkpoint itself: the resumer's first forward reads
        it, so it is pinned off the free list until that forward is out, and
        then it goes back — still indexed, so the next resumer hits it too.
        That transient is the whole difference between resuming and starting
        cold, and it lasts two passes rather than the request's lifetime.
        """
        bm = make_block_manager(spec_config(12))
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        bm.deallocate(first)
        free_before = bm.state.num_free()

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert second.state_fork_src == bm.state.lookup(h)
        assert len(second.state_slots) == 3
        assert free_before - bm.state.num_free() == 4  # its own 3, plus the source

        publisher_has_read_its_source(bm)
        assert free_before - bm.state.num_free() == 3
        assert bm.state.lookup(h) == second.state_fork_src  # survives to be hit again

    def test_adopting_a_checkpoint_still_yields_a_full_width(self):
        """The narrow path: too few slots to fork into, so the checkpoint is
        taken over as the committed slot and the scratch comes from the rest.

        The seq must still end up 3 wide. Adopting changes where its committed
        state lives, not how much speculation room it has.
        """
        # 6 slots: the publisher holds 3 and its checkpoint is 1, leaving 2 —
        # one short of a width, which is exactly the adopt case.
        bm = make_block_manager(spec_config(6))
        first = stateful_seq(list(range(40)))
        h = publish_at_boundary(bm, first)
        publisher_has_read_its_source(bm)
        checkpoint = bm.state.lookup(h)
        assert bm.state.num_free() == 3  # 2 vacant + the checkpoint

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert second.state_slot == checkpoint
        assert second.state_fork_src == -1
        assert len(second.state_slots) == 3
        assert len(set(second.state_slots)) == 3
        assert bm.state.lookup(h) == -1

    def test_deallocate_returns_every_slot(self):
        """Including the scratch. Releasing only the committed one would leak
        `num_spec` slots per request and starve the pool within a few hundred.
        """
        bm = make_block_manager(spec_config(9))
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        bm.deallocate(seq)
        assert bm.state.num_free() == 9
        assert seq.state_slots == []

    def test_a_fork_moves_only_the_committed_slot(self):
        """The scratch persists across forwards — step N's accepted slot is
        step N+1's initial state — so it belongs to the request, not to
        whichever slot it currently commits into.
        """
        bm = make_block_manager(spec_config(9))
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        scratch = list(seq.state_slots[1:])
        committed = seq.state_slot

        # Forward exactly to the boundary: publishing is what forks.
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)

        assert bm.state.lookup(boundary_hash(bm, seq)) == committed
        assert seq.state_fork_src == committed
        assert seq.state_slot != committed  # took a fresh one
        assert seq.state_slots[1:] == scratch  # left its scratch alone
        assert len(seq.state_slots) == 3
        assert len(set(seq.state_slots)) == 3


# ── Fork lifecycle ─────────────────────────────────────────────────────────


class TestForkLifecycle:

    def test_publish_moves_the_writer_to_a_new_group(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hit = bm.can_allocate(seq)
        bm.allocate(seq, hit)
        before = seq.state_slot
        boundary = bm.checkpoint_limit(seq)
        bm.hash_blocks(seq, boundary - seq.num_cached_tokens)
        assert seq.state_slot != before
        assert seq.state_fork_src == before
        assert bm.state.lookup(boundary_hash(bm, seq)) == before

    def test_no_publish_when_the_forward_misses_the_boundary(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        group = seq.state_slot
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) + BLOCK)
        assert seq.state_slot == group
        assert not bm.state.hash_to_slot

    def test_boundary_leaves_room_for_the_fork_forward(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        boundary = bm.checkpoint_limit(seq)
        assert boundary % bm.hash_block_size == 0
        assert seq.num_prompt_tokens - boundary >= MIN_FORK

    def test_every_block_boundary_up_to_the_limit_qualifies(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        limit = bm.checkpoint_limit(seq)
        assert bm.checkpointers_at(seq, BLOCK)
        assert bm.checkpointers_at(seq, limit)
        assert not bm.checkpointers_at(seq, limit + BLOCK)  # no room to fork
        assert not bm.checkpointers_at(seq, BLOCK + 2)  # not block aligned
        assert not bm.checkpointers_at(seq, 0)

    def test_chunked_prefill_leaves_a_ladder_of_checkpoints(self):
        """Intermediate boundaries publish too — the CPU-offload resume points."""
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        for _ in range(4):
            # One scheduling pass per chunk: each publish hands its source to
            # the next forward, and that forward is what lets the group go.
            # Without the boundary four publishes would hold four sources at
            # once and the pool would run out mid-ladder.
            bm.complete_previous_state_batch()
            bm.hash_blocks(seq, 2 * BLOCK, start_tokens=seq.num_cached_tokens)
            seq.num_cached_tokens += 2 * BLOCK
        # Four publishes into four groups: the oldest was recycled to serve the
        # last one, the rest stand as distinct resume points.
        assert len(bm.state.hash_to_slot) == 3
        assert bm.state.lookup(boundary_hash(bm, seq)) >= 0  # the rightmost one

    def test_interval_thins_the_ladder(self):
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=3 * BLOCK))
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
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=0))
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 0
        assert not any(bm.checkpointers_at(seq, pos) for pos in range(BLOCK, 40, BLOCK))

    def test_prompt_shorter_than_the_interval_publishes_nothing(self):
        """The zero-cost case: no reuse to be had, so no forward is spent.

        A prompt that cannot even reach one rung must not be cut, or every
        request on a short-prompt workload pays an extra forward for a
        checkpoint nothing will ever hit.
        """
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=8 * BLOCK))
        seq = stateful_seq(list(range(30)))  # 30 < 8 * BLOCK
        assert bm.checkpoint_limit(seq) == 0
        run_prompt(bm, seq)
        assert not bm.state.hash_to_slot
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
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=BLOCK + 1))
        assert bm.state_checkpoint_interval_tokens == BLOCK
        # Below one block there is no reachable rung at all, so the ladder is
        # off rather than snapped to something unusable.
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=BLOCK - 1))
        assert bm.state_checkpoint_interval_tokens == 0

    def test_hit_never_lands_where_swa_cannot_follow(self):
        """The two gates settle jointly; neither is applied to the other's answer.

        `swa.resumable_hit` promises the rightmost boundary whose trailing window
        is present. Shrinking that answer to a checkpoint boundary can land
        somewhere SWA never approved, and `allocate` would then claim an SWA
        hash the pool never promised.
        """
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        published = [2, 5]  # checkpoint boundaries, in blocks

        bm.state.hash_to_slot = {}
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
        bm = make_block_manager(
            ckpt_config(),
            state_runtime=StateRuntime(),
        )
        seq = stateful_seq(list(range(40)))
        assert bm.checkpoint_limit(seq) == 0
        assert not bm.checkpointers_at(seq, 16)

    def test_cancel_adopts_the_source_and_returns_the_new_group(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        source = seq.state_slot
        free_before_publish = bm.state.num_free()
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        # Publishing costs a group until the forward that reads the source has
        # run: the seq now owns a fresh group and the source is pinned for it.
        assert bm.state.num_free() == free_before_publish - 1

        bm.cancel_state_fork(seq)
        assert seq.state_slot == source
        assert seq.state_fork_src == -1
        assert not bm.state.hash_to_slot
        # Cancelling gives back exactly what publishing took.
        assert bm.state.num_free() == free_before_publish

    def test_two_resumers_in_one_step_share_the_checkpoint(self):
        # A checkpoint is read-only, so a second request hitting the same prefix
        # before the pins are released must fork off it too — not try to claim a
        # group the first one already took off the free list.
        bm = make_block_manager(ckpt_config(pool_entries={"state": 8}))
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)

        resumers = [stateful_seq(list(range(40))) for _ in range(3)]
        for seq in resumers:
            bm.allocate(seq, bm.can_allocate(seq))

        assert bm.state.pin_count(src) == len(resumers)
        assert all(s.state_fork_src == src for s in resumers)
        # Distinct write groups, none of them the shared source.
        groups = {s.state_slot for s in resumers}
        assert len(groups) == len(resumers)
        assert src not in groups
        # However many read it, the group goes back exactly once.
        before = bm.state.num_free()
        bm.complete_previous_state_batch()
        assert bm.state.num_free() == before + 1

    def test_cancel_refuses_to_adopt_a_shared_source(self):
        bm = make_block_manager(ckpt_config())
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
        assert sharers[1].state_slot == src

    def test_cancel_of_a_resume_releases_the_pin(self):
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)

        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert bm.state.is_pinned(src)
        bm.cancel_state_fork(second)
        assert second.state_slot == src
        assert not bm.state.is_pinned(src)
        # The pin must not also hand the group back — it has an owner now.
        bm.complete_previous_state_batch()
        assert not bm.state.is_free(src)

    def test_pinned_source_returns_to_the_free_list_next_step(self):
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        publisher_has_read_its_source(bm)
        second = stateful_seq(list(range(40)))
        bm.allocate(second, bm.can_allocate(second))
        assert not bm.state.is_free(src)
        bm.complete_previous_state_batch()
        assert bm.state.is_free(src)

    def test_a_published_source_is_not_handed_out_before_its_reader_runs(self):
        """The source is what the publisher's NEXT forward reads.

        `checkpoint` runs in postprocess, so that forward belongs to the batch
        the next pass builds — one pass further off than a resume's reader.
        Handing the group back straight away, as this used to, put it on the
        free list during the very pass that admits the requests which could pop
        it, and then one kernel reads and writes it at once.
        """
        bm = make_block_manager(ckpt_config())
        first = stateful_seq(list(range(40)))
        src = bm.state.lookup(publish_at_boundary(bm, first))
        assert first.state_fork_src == src

        assert not bm.state.is_free(src)  # the pass that admits cannot get it
        bm.complete_previous_state_batch()  # the batch carrying the fork is built
        assert not bm.state.is_free(src)  # its forward has not been issued yet
        bm.complete_previous_state_batch()  # it has now
        assert bm.state.is_free(src)
        # And it comes back as a checkpoint, at the LRU tail — publishing is
        # not what spends it.
        assert bm.state.lookup(bm.state.slot_hash[src]) == src

    def test_a_finished_publisher_gives_its_source_back_at_once(self):
        """Nobody is left to read it, so the clock should not hold it.

        This is what keeps publishing capacity-neutral for the common shape —
        a request that crosses a rung and then finishes or is preempted.
        """
        bm = make_block_manager(ckpt_config())
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
        assert bm.state.lookup(h) == src


class TestCheckpointsDieWithTheirPrefix:
    """A checkpoint whose KV block left the index can never be reached again.

    The two pools are addressed by one chained content hash and a prefix hit
    claims both, so `_gated_hit` caps at the last block still indexed. Until
    the state pool is told, the dead checkpoint holds a group and sits in the
    LRU queue ahead of live ones — the pool spends something usable to make
    room for something that is not.
    """

    def test_evicting_the_block_frees_the_checkpoint_group(self):
        bm = make_block_manager(ckpt_config())
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
        pool = StateSlotPool(4)
        while pool.has_free():
            pool.pop()
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.unindex(10)  # group 0's prefix is gone
        assert pool.pop() == 0
        assert pool.lookup(11) == 1

    def test_unindex_of_an_unknown_hash_is_a_no_op(self):
        pool = StateSlotPool(4)
        pool._index(10, 0)
        pool.unindex(999)
        assert pool.lookup(10) == 0
        assert pool.checkpoint_fates()["checkpoints_orphaned"] == 0

    def test_a_thrashing_pool_reports_no_drops_at_all(self):
        """`checkpoints_dropped` is not the capacity signal it looks like.

        Take four times as many checkpoints as the pool can hold, each one
        returned to the free list the way a finished request returns its
        group. `pop` never refuses — it spends the LRU checkpoint — so this
        pool overwrites rather than turning anything away, and `dropped`
        stays 0 through the whole thrash.

        Read `checkpoints_evicted` against `kept - num_slots` instead. On
        hardware that identity held exactly (kept 198, evicted 166, 32 groups)
        at the moment the state hit rate fell 6 points, while the 0 in
        `dropped` was read as proof the pool had room to spare.
        """
        pool = StateSlotPool(4)
        for h in range(16):
            group = pool.pop()
            pool.release(group)
            pool._index(h, group)

        fates = pool.checkpoint_fates()
        assert fates["checkpoints_dropped"] == 0
        assert pool.num_free() == 4  # never once out of groups to hand out
        # 16 taken, 4 resident: every other one was destroyed for space.
        assert fates["checkpoints_evicted"] == 16 - 4
        assert sum(1 for h in range(16) if pool.lookup(h) != -1) == 4


# ── The scheduler side: what a checkpoint costs the publisher ──────────────


class TestPrefillChunkAlignment:
    """`_finalize_prefill_chunk` cuts a prompt only where a rung is reachable.

    Every cut is an extra forward for the publisher, so the interval's whole
    job is to keep that off prompts too short to have anything to publish.
    """

    def test_prompt_shorter_than_the_interval_is_not_cut(self):
        sched = make_scheduler(ckpt_config(state_checkpoint_interval_tokens=8 * BLOCK))
        seq = stateful_seq(list(range(30)))  # 30 < 8 * BLOCK
        assert sched._finalize_prefill_chunk(seq, 0, 30) == 30

    def test_chunk_stops_at_the_rung(self):
        sched = make_scheduler(ckpt_config(state_checkpoint_interval_tokens=3 * BLOCK))
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


# ── PAGE-backed copy lifecycle ─────────────────────────────────────────────


def paged_copy_config(**overrides):
    return ckpt_config(**overrides)


class TestPagedCopyCheckpoint:
    def _admitted(self, bm, tokens=None):
        seq = stateful_seq(tokens or list(range(40)))
        bm.allocate(seq, bm.can_allocate(seq))
        return seq

    def test_allocate_resets_live_slot_readiness(self):
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = stateful_seq(list(range(40)))
        seq._state_initialized_after_alloc = True

        bm.allocate(seq, bm.can_allocate(seq))

        assert seq._state_initialized_after_alloc is False

    def test_validated_runtime_is_explicit_from_wire_through_block_manager(self):
        config = paged_copy_config()
        engine_runtime = StateRuntime.from_wire(PAGED_COPY_RUNTIME.to_wire())

        scheduler = make_scheduler(
            config,
            state_runtime=engine_runtime,
        )

        checkpoints = scheduler.block_manager.paged_state_checkpoints
        assert scheduler.block_manager.state_caches == (checkpoints,)
        assert checkpoints.store.spec is engine_runtime.checkpoint_spec
        assert checkpoints.store.units_per_checkpoint == 3
        assert scheduler.block_manager.state.transfer == StateTransfer.none()
        assert not hasattr(scheduler.block_manager, "state_runtime")
        assert not hasattr(scheduler.block_manager, "page_checkpoints")
        assert not any(
            hasattr(config, field)
            for field in (
                "paged_state_page_unit_bytes",
                "paged_state_slot_bytes",
                "paged_state_units_per_checkpoint",
                "paged_state_layout_id",
                "state_transfer_kind",
                "state_fork_tokens",
            )
        )

    def test_empty_batch_does_not_drain_state_maintenance(self):
        scheduler = make_scheduler(
            paged_copy_config(state_checkpoint_interval_tokens=0),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        scheduler.block_manager.state.record_relocation(1, 2)

        scheduled = scheduler.schedule()

        assert scheduled is None
        pending = scheduler.block_manager.take_state_maintenance_ops()
        assert pending.relocations == ((1, 2),)
        assert scheduler.block_manager.take_state_maintenance_ops().empty

    def test_real_batch_drains_all_state_maintenance_once(self):
        scheduler = make_scheduler(
            paged_copy_config(state_checkpoint_interval_tokens=0),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        checkpoints = scheduler.block_manager.paged_state_checkpoints
        seed = checkpoints.store.begin_store(33, src_slot=0)
        assert seed is not None
        checkpoints.store.complete_inflight()
        assert checkpoints.begin_restore(33, dst_slot=2)
        publisher = stateful_seq(list(range(BLOCK)))
        publisher.state_slot = 1
        checkpoints.checkpoint(publisher, boundary_blocks=1, h=13)
        scheduler.block_manager.state.record_relocation(3, 4)
        scheduler.add(stateful_seq(list(range(BLOCK))))

        batch, scheduled = scheduler.schedule()

        assert scheduled
        ops = batch.state_maintenance_ops
        assert ops.relocations == ((3, 4),)
        assert len(ops.checkpoint_stores) == 1
        assert ops.checkpoint_stores[0].src_slot == 1
        assert len(ops.checkpoint_restores) == 1
        assert ops.checkpoint_restores[0].dst_slot == 2
        assert scheduler.block_manager.take_state_maintenance_ops().empty
        assert not hasattr(scheduler.block_manager, "state_copies_for_batch")
        assert not hasattr(scheduler.block_manager, "state_transfers_for_batch")

    def test_only_the_boundary_the_slot_holds_is_stored(self):
        """Two undrained boundaries of one seq store one image, not two.

        A store reads `seq.state_slot` at the drain, so two entries surviving
        into one drain would both be copied out of whatever the last forward
        left there -- the earlier hash filed over the later state. Storing both
        is exactly the bug: two findable images, one of which returns a state
        from further along the prompt than the hash it answers to.

        Ordinarily a drain follows each forward and the two boundaries are
        stored separately and correctly; this is the empty-pass case, where
        `state_maintenance_ops=None` carries `_pending` forward.
        """
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = self._admitted(bm)
        checkpoints = bm.paged_state_checkpoints

        checkpoints.checkpoint(seq, boundary_blocks=1, h=101)
        checkpoints.checkpoint(seq, boundary_blocks=2, h=202)
        ops = bm.take_state_maintenance_ops()

        assert len(ops.checkpoint_stores) == 1
        bm.complete_previous_state_batch()
        assert checkpoints.store.contains(202), "the state the slot holds"
        assert not checkpoints.store.contains(101), "would have been mis-stored"

    def test_a_drain_between_forwards_stores_both(self):
        """The ordinary case: one boundary per forward, each drained in turn.

        This is what the anchor rests on -- the anchor and the prompt-end
        checkpoint a chunk later are separate forwards, so both are stored,
        each from the slot as it was.
        """
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = self._admitted(bm)
        checkpoints = bm.paged_state_checkpoints

        checkpoints.checkpoint(seq, boundary_blocks=1, h=101)
        assert len(bm.take_state_maintenance_ops().checkpoint_stores) == 1
        bm.complete_previous_state_batch()

        checkpoints.checkpoint(seq, boundary_blocks=2, h=202)
        assert len(bm.take_state_maintenance_ops().checkpoint_stores) == 1
        bm.complete_previous_state_batch()

        assert checkpoints.store.contains(101)
        assert checkpoints.store.contains(202)

    def test_reaching_the_same_boundary_twice_is_still_one_checkpoint(self):
        """The hash in the key is what tells "two boundaries" from "one
        boundary, reached again"."""
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = self._admitted(bm)
        checkpoints = bm.paged_state_checkpoints

        checkpoints.checkpoint(seq, boundary_blocks=1, h=303)
        checkpoints.checkpoint(seq, boundary_blocks=1, h=303)
        ops = bm.take_state_maintenance_ops()

        assert len(ops.checkpoint_stores) == 1

    def test_prefix_eviction_drops_an_uncommitted_checkpoint(self):
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = self._admitted(bm)
        checkpoints = bm.paged_state_checkpoints

        checkpoints.checkpoint(seq, boundary_blocks=1, h=101)
        bm._record_evicted(101)

        assert bm.take_state_maintenance_ops().checkpoint_stores == ()
        assert checkpoints.checkpoint_fates()["checkpoints_orphaned"] == 1

    def test_checkpoint_uses_page_units_not_an_active_slot(self):
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        seq = self._admitted(bm)
        free_slots = bm.state.num_free()
        free_pages = bm.kv.num_free
        bm.hash_blocks(seq, bm.checkpoint_limit(seq) - seq.num_cached_tokens)
        h = boundary_hash(bm, seq)

        transfers = bm.take_state_maintenance_ops()
        assert transfers.relocations == ()
        assert len(transfers.checkpoint_stores) == 1
        assert bm.state.num_free() == free_slots
        assert bm.kv.num_free == free_pages - 3
        checkpoints = bm.paged_state_checkpoints
        assert checkpoints.store.lookup(h) == -1

        bm.complete_previous_state_batch()
        assert checkpoints.store.contains(h)

    def test_hit_gathers_into_a_distinct_contiguous_active_slot(self):
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        store = bm.take_state_maintenance_ops().checkpoint_stores[0]
        bm.complete_previous_state_batch()

        second = stateful_seq(list(range(48)))
        hit = bm.can_allocate(second)
        assert hit > 0
        bm.allocate(second, hit)
        transfers = bm.take_state_maintenance_ops()
        assert transfers.checkpoint_stores == ()
        assert len(transfers.checkpoint_restores) == 1
        restore = transfers.checkpoint_restores[0]
        assert restore.unit_ids == store.unit_ids
        assert restore.dst_slot == second.state_slot
        assert second.state_fork_src == -1
        # The checkpoint stays canonical and shareable. Its fragments were not
        # adopted as the request's kernel-visible slot.
        assert bm.paged_state_checkpoints.store.contains(h)

    def test_deallocate_cancels_a_queued_restore_before_reusing_its_slot(self):
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()

        second = stateful_seq(list(range(48)))
        bm.allocate(second, bm.can_allocate(second))
        dst = second.state_slot
        checkpoint_id = bm.paged_state_checkpoints.store.lookup(h)
        assert bm.paged_state_checkpoints.store.records[checkpoint_id].pin_count == 1

        bm.deallocate(second)

        assert bm.take_state_maintenance_ops().checkpoint_restores == ()
        assert bm.paged_state_checkpoints.store.records[checkpoint_id].pin_count == 0
        assert bm.state.is_free(dst)

        third = stateful_seq(list(range(100, 140)))
        bm.allocate(third, bm.can_allocate(third))
        assert third.state_slot == dst
        assert bm.take_state_maintenance_ops().checkpoint_restores == ()

    # ── the CPU tier's leg of a PAGE resume ────────────────────────────

    class _TierIndex:
        """The engine-side index, reduced to what `_attach_state_slots` reads."""

        def __init__(self, *hashes):
            self.hashes = set(hashes)
            self.pending_loads = {}
            self.requested = []

        def could_serve(self, h):
            return h in self.hashes

        def request_load(self, req_id, h):
            if h not in self.hashes:
                return False
            self.pending_loads[req_id] = h
            self.requested.append((req_id, h))
            return True

    def _with_tier(self, bm, *hashes):
        index = self._TierIndex(*hashes)
        bm.state_offload = index
        bm.paged_state_checkpoints.attach_offload(index)
        return index

    def test_a_boundary_only_the_tier_has_becomes_a_load(self):
        """The path the whole tier exists for. HBM misses, the tier votes, and
        the request parks on a load instead of disowning the boundary."""
        bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        # The image left HBM but the tier still advertises it.
        bm.paged_state_checkpoints.unindex(h)
        index = self._with_tier(bm, h)

        second = stateful_seq(list(range(48)))
        assert bm._attach_state_slots(second, h) is True

        assert second.offload_joint.load_hash == h
        assert second.state_slot >= 0, "the H2D writes this slot directly"
        assert index.requested == [(second.id, h)]
        # No restore queued: there is nothing in HBM to gather from, and the
        # bytes land in the slot rather than in PAGE units.
        assert bm.take_state_maintenance_ops().checkpoint_restores == ()
        assert bm.take_state_loads() == [(second.id, h, second.state_slot)]

    def test_hbm_is_preferred_over_the_tier(self):
        """Both tiers are keyed by the same hash, so the gate does not say which
        one answered -- `_attach_state_slots` tries HBM first and only falls to
        a load on a miss. A resident image must not pay a park."""
        bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        index = self._with_tier(bm, h)  # advertised in BOTH

        second = stateful_seq(list(range(48)))
        assert bm._attach_state_slots(second, h) is True

        assert second.offload_joint.load_hash == -1
        assert index.requested == [], "a resident image must not pay a park"
        assert len(bm.take_state_maintenance_ops().checkpoint_restores) == 1

    def test_a_tier_that_declines_disowns_rather_than_parking(self):
        """`request_load` refuses a hash it never stored, because a load is
        resolved only by a report -- offering one would park the request
        against bytes no `get` can produce."""
        bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        bm.paged_state_checkpoints.unindex(h)
        self._with_tier(bm)  # votes for nothing

        second = stateful_seq(list(range(48)))
        assert bm._attach_state_slots(second, h) is False
        assert second.offload_joint.load_hash == -1
        assert bm.checkpoint_funnel()["state_gate_lost_boundary"] == 1

    # ── the joint boundary, which #2045 makes the only source of value ──

    def _joint_bm(self, chunk=BLOCK, **overrides):
        bm = make_block_manager(
            paged_copy_config(**overrides), state_runtime=PAGED_COPY_RUNTIME
        )
        # Normally read off the LMCache config at construction; that import is
        # unavailable here, and the value is the KV leg's transfer grid.
        bm._joint_chunk_tokens = chunk
        return bm

    def _prompt_with_a_rung_at(self, bm, blocks: int):
        """Run PROMPT once and leave one READY checkpoint `blocks` blocks in.

        Placed explicitly rather than by the ladder because the position
        matters: `can_allocate` matches over `range(n_hash_blocks - 1)`, so a
        checkpoint filed under the LAST block is one no scan can ever look up.
        An interior rung is what a resume actually lands on.
        """
        seq = self._admitted(bm, PROMPT)
        # Publish the prefix, which is what makes its blocks lookup-able and so
        # what a second request's walk can match against.
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        # Whatever the ladder placed is not the subject here; keep exactly one
        # rung, at a position chosen for being interior.
        bm.paged_state_checkpoints.clear_index()

        h = bm._chain_to(seq, [], blocks)[blocks - 1]
        bm.paged_state_checkpoints.checkpoint(seq, blocks, h)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        assert bm.paged_state_checkpoints.contains(h), "precondition: it is READY"
        assert bm.kv.lookup(h) >= 0, "precondition: its KV block is published"
        return seq, h

    @staticmethod
    def _break_the_kv_chain_at(bm, block_id: int) -> None:
        """Evict one published KV block so the prefix walk stops before the rung.

        Without this there is nothing for a joint boundary to do: a fully
        resident prefix already gates to its rung, `_attach_state_slots` issues
        the state load on its own, and the KV leg has nothing to fetch. The
        joint path exists for exactly the case this creates -- HBM lost the KV,
        LMCache still has it.

        Call after `deallocate`: `allocate` is the eviction event and it asserts
        the block is unheld.
        """
        assert bm.kv.block(block_id).hash != -1, "precondition: it was published"
        bm.kv.allocate(block_id)  # takes it for fresh content, dropping the hash

    def _reset_joint_counters(self, bm):
        bm.joint_boundaries = bm.state_hbm_boundaries = bm.state_tier_boundaries = 0
        bm.joint_skips.clear()

    def test_a_page_class_now_gets_a_joint_boundary(self):
        """The inversion, and the single most important assertion in Phase 5.

        This gate used to refuse every PAGE seq (`not_hybrid`), which was right
        while a K3 checkpoint was an Active Slot the tier spilled out of the
        slot pool. #2045 moved the image into the KV pool, and HBM's
        `state ⊆ KV` means a checkpoint can no longer outlive its KV there --
        so when LMCache hands the KV back, nothing hands the state back unless
        the two are fetched together. Refusing here makes the whole tier dead
        weight, and no other counter in the system would say so.
        """
        bm = self._joint_bm()
        first, h = self._prompt_with_a_rung_at(bm, 8)  # 32 tokens in

        victim = first.block_table[1]
        bm.deallocate(first)
        self._break_the_kv_chain_at(bm, victim)
        bm.paged_state_checkpoints.unindex(h)  # the image left HBM with it
        self._with_tier(bm, h)  # ...but LMCache still has it
        self._reset_joint_counters(bm)

        second = stateful_seq(PROMPT)
        second.offload_joint.kv_prefix_tokens = len(PROMPT)
        hbm_hit = bm.can_allocate(second)

        assert bm.joint_boundaries == 1, bm.joint_skips
        assert second.offload_joint.boundary_hash == h
        assert second.offload_joint.boundary_tokens == 8 * BLOCK
        # The KV leg has real work: the walk stopped well below the boundary.
        assert hbm_hit * BLOCK < second.offload_joint.kv_tokens

    def test_the_split_says_which_tier_the_state_leg_came_from(self):
        """`state_tier` is the only counter here that cannot be non-zero with
        the CPU tier switched off, which makes it the honest test of "did this
        feature run" -- no passing unit test can produce it in production."""
        bm = self._joint_bm()
        first, h = self._prompt_with_a_rung_at(bm, 8)
        victim = first.block_table[1]
        bm.deallocate(first)
        self._break_the_kv_chain_at(bm, victim)
        self._with_tier(bm, h)
        self._reset_joint_counters(bm)

        # The checkpoint outlived the block that broke the chain, so the state
        # leg is free -- a gather out of resident units, no transfer.
        resident = stateful_seq(PROMPT)
        resident.offload_joint.kv_prefix_tokens = len(PROMPT)
        bm.can_allocate(resident)
        assert (bm.state_hbm_boundaries, bm.state_tier_boundaries) == (1, 0)

        # Drop it from HBM too: now the state leg costs an image-sized H2D.
        bm.paged_state_checkpoints.unindex(h)
        from_cpu = stateful_seq(PROMPT)
        from_cpu.offload_joint.kv_prefix_tokens = len(PROMPT)
        bm.can_allocate(from_cpu)
        assert (bm.state_hbm_boundaries, bm.state_tier_boundaries) == (1, 1)

    def test_no_tier_means_no_joint_boundary(self):
        """`state_offload is None` short-circuits before anything else: a
        boundary both legs must reach is meaningless with one leg missing."""
        bm = self._joint_bm()
        first, _h = self._prompt_with_a_rung_at(bm, 8)
        victim = first.block_table[1]
        bm.deallocate(first)
        self._break_the_kv_chain_at(bm, victim)
        self._reset_joint_counters(bm)

        seq = stateful_seq(PROMPT)
        seq.offload_joint.kv_prefix_tokens = len(PROMPT)
        bm.can_allocate(seq)
        assert bm.joint_boundaries == 0
        assert bm.joint_skips.get("off") == 1

    def test_a_gated_boundary_neither_tier_has_is_disowned_not_raised(self):
        """This used to raise, and could not stay that way.

        The raise asserted that the gate and the HBM store agree, which held
        while `can_allocate` only ever accepted boundaries the HBM index
        carried. The gate now consults the CPU tier as well, so a hash it
        accepted may live only there -- and with no tier able to produce it,
        the answer is to disown the boundary, not to take the engine down.

        The slots are KEPT, unlike the old abort: the request is about to
        recompute its whole prefix and it writes that state into these very
        slots. Releasing them here would hand the next request a buffer this
        one is still filling.
        """
        bm = make_block_manager(
            paged_copy_config(),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        first = self._admitted(bm)
        bm.hash_blocks(first, bm.checkpoint_limit(first) - first.num_cached_tokens)
        h = boundary_hash(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        free_slots = bm.state.num_free()
        bm.paged_state_checkpoints.unindex(h)

        second = stateful_seq(list(range(48)))
        assert bm._attach_state_slots(second, h) is False

        assert second.state_slot >= 0, "the seq keeps slots to recompute into"
        assert bm.state.num_free() == free_slots - bm.state_slots_per_req
        assert bm.checkpoint_funnel()["state_gate_lost_boundary"] == 1

    def test_copy_transfer_can_checkpoint_a_speculative_decode_boundary(self):
        spec = SimpleNamespace(num_speculative_tokens=3, use_dspark=lambda: False)
        seq = stateful_seq(list(range(40)))
        seq.type = SequenceType.DECODE
        forking = make_scheduler(
            ckpt_config(speculative_config=spec),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)),
        )
        copying = make_scheduler(
            paged_copy_config(speculative_config=spec),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        assert (
            copying.block_manager.paged_state_checkpoints.store.spec is PAGED_COPY_SPEC
        )
        assert forking._checkpoint_room(seq, False) == 0
        assert copying._checkpoint_room(seq, False) == 1
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
        assert not bm.state.hash_to_slot
        return seq

    def test_a_rung_past_the_prompt_publishes(self):
        bm = make_block_manager(
            ckpt_config(),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)),
        )
        seq = self._prompt_of_10(bm)
        group = seq.state_slot

        self._generate_to(bm, seq, 3 * BLOCK)
        assert seq.state_slot != group
        assert seq.state_fork_src == group
        assert bm.state.lookup(bm.kv.block(seq.block_table[2]).hash) == group

    def test_a_backend_needing_a_long_fork_never_publishes_mid_generation(self):
        """Self-gating: no `min_fork` special case, the number decides.

        One decode token cannot fill a group that needs `MIN_FORK` of them, so
        the rung is simply not a publish position for this backend.
        """
        bm = make_block_manager(ckpt_config())  # DEFAULT_STATE_TRANSFER needs MIN_FORK.
        seq = self._prompt_of_10(bm)
        group = seq.state_slot

        self._generate_to(bm, seq, 4 * BLOCK)
        assert seq.state_slot == group
        assert not bm.state.hash_to_slot

    def test_no_publish_on_the_step_that_finishes_the_request(self):
        """Nothing will fork from it, and the fresh group would go straight back."""
        bm = make_block_manager(
            ckpt_config(),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)),
        )
        seq = self._prompt_of_10(bm)
        group = seq.state_slot

        self._generate_to(bm, seq, 3 * BLOCK, room=0)
        assert seq.state_slot == group
        assert not bm.state.hash_to_slot

    def test_blocks_are_still_hashed_where_no_checkpoint_is_taken(self):
        """Prefix caching and state checkpoints are separate gates."""
        bm = make_block_manager(ckpt_config())
        seq = self._prompt_of_10(bm)
        self._generate_to(bm, seq, 3 * BLOCK)
        assert seq.num_hashed_tokens == 3 * BLOCK

    def test_followup_turn_resumes_from_a_generated_rung(self):
        """The payoff: turn 2 reuses KV *and* the state that goes with it."""
        bm = make_block_manager(
            ckpt_config(),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)),
        )
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
        return make_scheduler(
            ckpt_config(**overrides),
            state_runtime=StateRuntime(transfer=StateTransfer.fork(1)),
        )

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
        assert seq.state_slot != published
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

    def __init__(
        self, successor_room=inf, cap=None, enabled=True, readable_midstep=False
    ):
        self.successor_room = successor_room
        self.enabled = enabled
        self.readable_midstep = readable_midstep
        self._cap = cap

    def applies(self, seq):
        return self.enabled

    def resumable_hit(self, seq, P, block_hashes, assume_checkpointed=False):
        return P if self._cap is None else min(P, self._cap)

    def checkpoint(self, seq, boundary_blocks, h):
        pass

    def reserve_midstep(self, seq, positions):
        return []

    def publish_midstep(self, reservations, seq=None):
        pass

    def cancel_midstep(self, reservations):
        pass


def second_class(**overrides):
    """A second state class for the protocol tests.

    A stub rather than a real one: multi-class behaviour is a property of the
    ladder, not of whichever class happens to exist beside `StateSlotPool`,
    and testing it through a real one made these tests hostage to that class's
    lifetime — which is how they broke when the sliding window stopped being a
    pool of its own.
    """
    return StubStateCache(**overrides)


class TestStateCacheProtocol:

    def test_copy_transfer_has_no_slot_backed_fallback(self):
        with pytest.raises(ValueError, match="do not belong"):
            StateSlotPool(4, StateTransfer.copy("test-layout"))

    def test_both_classes_satisfy_the_protocol(self):
        assert isinstance(second_class(), StateCache)
        assert isinstance(StateSlotPool(4), StateCache)

    def test_a_class_that_keeps_nothing_reports_inf(self):
        """`inf` is what stops the ladder cutting chunks for a class in vain.

        The window pool only ever materializes the trailing window, so no older
        boundary has anything left to hold on to; reporting 0 would have the
        scheduler cut prefill chunks at every rung for a class that stores
        nothing there — cost with no reuse.
        """
        assert isinf(second_class().successor_room)
        assert isinf(StateSlotPool(4, StateTransfer.none()).successor_room)

    def test_the_limit_follows_the_class_that_reaches_furthest(self):
        """The smallest room reaches furthest right; a larger one must not cap it."""
        bm = make_block_manager(ckpt_config())
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
        assert isinf(StateSlotPool(4, StateTransfer.none()).successor_room)
        assert StateTransfer.copy("test-layout").successor_room == 0
        assert StateSlotPool(4, StateTransfer.fork(7)).successor_room == 7

    def test_a_copy_never_asks_the_resumer_for_room(self):
        """`resumable_hit`'s fork test is vacuous under `copy`, not skipped."""
        forking = StateSlotPool(4, StateTransfer.fork(4), hash_block_size=1)
        copying = PagedStateCheckpointCoordinator(
            BlockPool(4),
            PagedStateCheckpointSpec(1, 1, "test-layout", image_bytes=1),
            enabled=True,
        )
        assert isinstance(copying, StateCache)
        forking._index(10, 0)
        forking._index(50, 1)
        assert copying.store.begin_store(10, 0) is not None
        assert copying.store.begin_store(50, 1) is not None
        copying.store.complete_inflight()
        # Five one-token blocks; the rightmost checkpoint leaves no room to
        # forward, so a fork walks back to the first and a copy does not.
        assert forking.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 1
        assert copying.resumable_hit(idx_seq(5), 5, [10, 20, 30, 40, 50]) == 5

    def test_the_immutable_class_qualifies_where_the_rolling_one_cannot(self):
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        # A rung one token from the end: the ring has no room to hand over, an
        # immutable class needs none.
        pos = seq.num_prompt_tokens - BLOCK
        assert bm.state not in bm.checkpointers_at(seq, pos)
        bm.state_caches = (*bm.state_caches, StubStateCache(successor_room=0))
        assert bm.checkpointers_at(seq, pos) == [bm.state_caches[-1]]

    def test_cut_and_ladder_agree_position_for_position(self):
        """The chunk is cut where — and only where — something gets kept."""
        bm = make_block_manager(ckpt_config())
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
        bm = make_block_manager(ckpt_config())
        seq = stateful_seq(list(range(40)))
        hashes = [1000 + i for i in range(9)]
        for group, boundary in enumerate([2, 5]):
            bm.state._index(hashes[boundary - 1], group)
        bm.state_caches = (*bm.state_caches, StubStateCache(cap=4))

        answer = bm._gated_hit(seq, 9, hashes)
        for cache in bm.state_caches:
            assert cache.resumable_hit(seq, answer, hashes) == answer

    def test_order_between_classes_does_not_change_the_answer(self):
        bm = make_block_manager(ckpt_config())
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
# A prompt that diverges from `PROMPT` at token 28, mid-interval and nowhere
# near either prompt's end. This is the traffic the demand is for now that the
# prompt-end anchor exists: on a conversation that just grows, the position the
# next turn resumes at *is* the previous turn's end, and the anchor reserves it
# up front rather than one disappointed request late. What the anchor cannot
# reserve is a branch point, because no prompt ever ended there.
BRANCH = list(range(28)) + list(range(900, 916))
# `PROMPT` resent with a further turn appended. This is the shape a demand
# needs now that the anchor exists: the anchor reserves the previous prompt's
# end, so a continuation resumes there (10 blocks) and still wants the block
# past it — the "resumed from a checkpoint AND wants a further one" state the
# two gates have to be observed meeting in. An identical re-send no longer
# produces it: the anchor covers every hittable block, so the hit is complete
# and there is no gap left for a demand to name.
CONTINUATION = PROMPT + list(range(900, 916))

# An image that costs more units than a request's blocks do. That is the shape
# where the ladder's question and admission's question can disagree: the pool
# still has room for the request and not for the checkpoint. With an image the
# size of a couple of blocks the two run out together and there is nothing to
# test.
BIG_IMAGE_SPEC = PagedStateCheckpointSpec(10, 400, "test-layout-big", image_bytes=400)
BIG_IMAGE_RUNTIME = StateRuntime(
    transfer=StateTransfer.copy(BIG_IMAGE_SPEC.layout_id),
    checkpoint_spec=BIG_IMAGE_SPEC,
)


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


def an_image_fits_on_its_own(checkpoints) -> bool:
    """What the demand gate used to ask, kept as the contrast it is read against.

    The gate now asks whether an image fits *after* the admission has taken
    its own blocks, and the tests below turn on the pool state where the two
    answers differ. Written out here rather than left as a method on the
    store, which would be a production API nothing in production asks.
    """
    return checkpoints.has_available_units(checkpoints.store.units_per_checkpoint)


class TestDemandDrivenCheckpoints:
    """A rung placed where a request was seen to want one.

    The interval is a guess about where reuse will resume; the requests know.
    Whenever the state gates cut a hit short, `can_allocate` asks the same
    question again with every ladder assumed dense, and the gap between the two
    answers is reuse being declined only for want of a checkpoint. The request
    that finds the gap is the one that pays for it — it collects none of that
    reuse and has to compute the prefix anyway.

    Scoped to branch points since the prompt-end anchor landed. A conversation
    that only grows resumes at the previous turn's end, and the anchor reserves
    that proactively — see `TestPromptEndAnchor`, which inherited the cases
    these tests used to make. What no anchor can reserve is a position no prompt
    ever ended at, and that is what these now use.
    """

    def test_the_gap_becomes_a_rung_off_the_grid(self):
        bm = make_block_manager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))

        second = stateful_seq(BRANCH)
        assert bm.can_allocate(second) == 0  # nothing resumable at the branch
        assert second.num_wanted_hit_blocks == 7  # what a checkpoint would give
        assert second.checkpoint_demand_pos == 28
        # Off the grid: the demand carries its own fork room, so it sits where
        # the request asked rather than where the interval would have put it.
        assert 28 % INTERVAL
        assert bm.checkpoint_limit(second) == 32

    def test_the_rung_can_be_switched_off_without_the_grid(self):
        """`--no-state-checkpoint-demand` drops the rung, nothing else.

        The refusal is still measured — `num_wanted_hit_blocks` is what
        `EngineStats` splits declined reuse by, and turning the placement off
        must not blind that. What goes is only the placement, leaving the grid
        and this prompt's own anchor to carry the checkpoints.
        """
        bm = make_block_manager(demand_config(state_checkpoint_demand=False))
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))

        second = stateful_seq(BRANCH)
        bm.allocate(second, bm.can_allocate(second))
        assert second.num_cached_tokens == 0
        assert second.num_wanted_hit_blocks == 7  # still measured...
        assert second.checkpoint_demand_pos == 0  # ...but no longer placed
        assert bm.demands_recorded == 0
        # 28 is the demand's rung and it is gone; the grid and anchor remain.
        assert forward_on_the_ladder(bm, second) == [32, 36]

    def test_the_env_var_overrides_the_flag_in_both_directions(self, monkeypatch):
        """`ATOM_STATE_CHECKPOINT_DEMAND` beats the config field.

        Both directions are pinned because the override is asymmetric in
        practice: =0 turns the rung off for one run without editing a launch
        script, and =1 has to be able to turn it back on over a script that
        already passes --no-state-checkpoint-demand. An unset variable must
        change nothing, or merely having it exported on the box would pin the
        policy for every server running there.
        """
        # =0 beats a config that asks for the rung.
        monkeypatch.setenv("ATOM_STATE_CHECKPOINT_DEMAND", "0")
        bm = make_block_manager(demand_config(state_checkpoint_demand=True))
        assert bm.state_checkpoint_demand is False

        # =1 beats a config that refuses it.
        monkeypatch.setenv("ATOM_STATE_CHECKPOINT_DEMAND", "1")
        bm = make_block_manager(demand_config(state_checkpoint_demand=False))
        assert bm.state_checkpoint_demand is True

        # Exported-but-empty is not "set" — the flag still decides.
        monkeypatch.setenv("ATOM_STATE_CHECKPOINT_DEMAND", "")
        bm = make_block_manager(demand_config(state_checkpoint_demand=False))
        assert bm.state_checkpoint_demand is False

        monkeypatch.delenv("ATOM_STATE_CHECKPOINT_DEMAND")
        bm = make_block_manager(demand_config(state_checkpoint_demand=True))
        assert bm.state_checkpoint_demand is True

    def test_the_third_request_finds_what_the_second_was_missing(self):
        """Self-limiting: nothing to want, want it once, want nothing again."""
        bm = make_block_manager(demand_config())

        first = stateful_seq(PROMPT)
        # 32 is the grid's last rung; 36 is `first`'s own end, anchored.
        assert run_prompt_on_the_ladder(bm, first) == [32, 36]
        assert first.checkpoint_demand_pos == 0  # nothing was cached to fall short

        second = stateful_seq(BRANCH)
        bm.allocate(second, bm.can_allocate(second))
        assert second.num_cached_tokens == 0  # the branch point is unreachable...
        assert second.checkpoint_demand_pos == 28  # ...and this is where it is
        # The demand at 28, then the grid rung and this prompt's own anchor.
        assert forward_on_the_ladder(bm, second) == [28, 32, 36]

        third = stateful_seq(BRANCH)
        bm.allocate(third, bm.can_allocate(third))
        assert third.num_cached_tokens == 36
        assert third.checkpoint_demand_pos == 0  # nothing left to want
        assert forward_on_the_ladder(bm, third) == []

    def test_a_demand_the_floor_would_refuse_is_not_recorded(self):
        """A cut costs a forward; buying one for a refused store is pure loss.

        `begin_store` drops a checkpoint whose units are not reachable.
        Recording a demand for it anyway would still shorten the
        request's prefill chunk, so the ladder asks the same question the
        store will — and the reuse attribution is unaffected, because that
        reuse really was declined for want of a checkpoint.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        checkpoints = bm.paged_state_checkpoints
        assert an_image_fits_on_its_own(checkpoints)

        # Live KV takes the pool down to where an image no longer fits but the
        # resumer's own blocks still do -- the state under real pressure, where
        # admission goes through and only the store cannot.
        spare = -(-len(CONTINUATION) // BLOCK) + 1
        assert spare < checkpoints.store.units_per_checkpoint
        bm.kv.reserve_units(bm.kv.num_free - spare, ("live-kv", 0))
        assert not an_image_fits_on_its_own(checkpoints)

        second = stateful_seq(CONTINUATION)
        hit = bm.can_allocate(second)
        bm.allocate(second, hit)

        # The reuse is still attributed to a missing checkpoint...
        assert second.num_wanted_hit_blocks > hit, "the attribution was suppressed too"
        # ...but nothing is cut for a store that would be refused.
        assert second.checkpoint_demand_pos == 0, "a refused store still cut a chunk"
        funnel = bm.checkpoint_funnel()
        assert funnel["demands_declined_no_room"] == 1
        assert funnel["demands_recorded"] == 0
        assert funnel["chunks_cut_for_demand"] == 0
        # The grid rung and the prompt-end anchor. The anchor is not the thing
        # under test here -- it is reserved from `num_prompt_tokens` alone and
        # is unaffected by the pool being tight -- but it is a cut, so it
        # appears. What must be absent is a *demand* cut: that is the one the
        # refused store would have bought for nothing.
        cuts = forward_on_the_ladder(bm, second)
        assert 48 in cuts, "the grid rung went missing"
        assert second.checkpoint_demand_pos not in cuts, "a demand cut slipped in"

    def _tighten_past_an_image(self, bm):
        """Leave room for a resumer's blocks but not for a checkpoint image."""
        checkpoints = bm.paged_state_checkpoints
        spare = -(-len(PROMPT) // BLOCK) + 1
        assert spare < checkpoints.store.units_per_checkpoint
        bm.kv.reserve_units(bm.kv.num_free - spare, ("live-kv", 0))
        assert not an_image_fits_on_its_own(checkpoints)

    def test_a_demand_is_refused_when_the_admission_itself_drains_the_pool(self):
        """The blocks this request takes come first, so they count.

        A pool with room for an image but not for the request *and* the image
        answers yes to "does an image fit" -- and then the admission takes its
        block table, `begin_store` refuses many forwards later, and the cut
        this gate exists to withhold has already been bought. The funnel shows
        nothing, because the decline happened somewhere that does not count.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        checkpoints = bm.paged_state_checkpoints
        image = checkpoints.store.units_per_checkpoint
        blocks = -(-len(PROMPT) // BLOCK)

        # Enough for an image on its own, not for this request and an image.
        bm.kv.reserve_units(bm.kv.num_free - (image + blocks // 2), ("live-kv", 0))
        assert an_image_fits_on_its_own(checkpoints), "the old question still says yes"

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) >= 0, "admission itself must still go through"

        assert second.checkpoint_demand_pos == 0, "bought a cut the store cannot use"
        assert bm.checkpoint_funnel()["demands_declined_no_room"] == 1

    def test_both_gates_in_one_pass_protect_the_same_checkpoint(self):
        """`_checkpoint_has_room` and `_has_page_units` agree on what is spendable.

        The second excludes the checkpoint this admission is about to pin --
        it is about to be read, so eviction cannot have it. The first used to
        count it as reclaimable, so with the pool resting on exactly that one
        image the two gates in a single pass gave opposite answers.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        first = stateful_seq(PROMPT)
        published = publish_at_boundary(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        checkpoints = bm.paged_state_checkpoints
        assert checkpoints.store.contains(published)

        # Nothing spare: the only spendable units are that one checkpoint's.
        bm.kv.reserve_units(bm.kv.num_free, ("live-kv", 0))

        assert bm._checkpoint_has_room(0, protected_hash=None), "the setup is wrong"
        assert not bm._checkpoint_has_room(
            0, protected_hash=published
        ), "the checkpoint about to be pinned was counted as spendable"

    def test_the_checkpoint_being_resumed_from_is_not_counted_as_spendable(self):
        """Through `can_allocate`, where the two gates actually meet.

        The seq hits a checkpoint and wants a further one, so the pin and the
        demand happen in the same call. Rest the pool on exactly that one
        image and the answer turns on whether the gate knows it is spoken for:
        counting it leaves the ladder cutting a chunk for a store that has no
        units left to take.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        first = stateful_seq(PROMPT)
        run_prompt_on_the_ladder(bm, first)
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        store = bm.paged_state_checkpoints.store
        image = store.units_per_checkpoint
        # A prompt now stores its grid rung *and* its prompt-end anchor, and
        # this test needs the pool resting on exactly one image. Spend all but
        # the deepest, which is the one a continuation resumes from -- and the
        # one `_next_victim` would keep longest, so this is also the state the
        # gate would find on its own.
        for cid in list(store._lru)[:-1]:
            store._evict(cid)
        assert len(store.records) == 1, "rested on one image"

        second = stateful_seq(CONTINUATION)
        # Leave the request's own blocks plus half an image: reachable only by
        # spending the very checkpoint `second` is about to resume from.
        spare = -(-len(CONTINUATION) // BLOCK) + image // 2
        bm.kv.reserve_units(bm.kv.num_free - spare, ("live-kv", 0))

        hit = bm.can_allocate(second)

        assert hit > 0, "the seq is supposed to resume from that checkpoint"
        assert second.num_wanted_hit_blocks > hit, "and to want a further one"
        assert second.checkpoint_demand_pos == 0, "spent an image already spoken for"
        assert bm.checkpoint_funnel()["demands_declined_no_room"] == 1

    def test_a_demand_recorded_while_there_was_room_is_withdrawn_when_it_goes(self):
        """The gate is the store's question, so it has to be asked afresh.

        `can_allocate` re-runs for a sequence the queue keeps deferring. One
        that recorded a demand while the pool had room, and is then re-admitted
        against a pool that does not, is exactly the case the gate exists for:
        the cut it would buy is now pure loss. Reading the position the gate is
        about to overwrite made it a one-shot and let that cut through.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) >= 0
        recorded = second.checkpoint_demand_pos
        assert recorded, "the first attempt was supposed to record a demand"

        self._tighten_past_an_image(bm)
        bm.can_allocate(second)

        assert second.checkpoint_demand_pos == 0, "a stale answer bought the cut"
        assert bm.checkpoint_funnel()["demands_declined_no_room"] == 1

    def test_a_deferred_sequence_is_counted_once_however_often_it_asks(self):
        """One request under pressure, not one per admission attempt.

        `demands_declined_no_room` is read against `demands_recorded`, so a
        counter that fires per attempt makes the funnel unreadable under the
        only pressure anyone reads it in -- and a decline writes 0 into the
        position, so the position cannot be the marker that stops it.
        """
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        self._tighten_past_an_image(bm)

        second = stateful_seq(PROMPT)
        for _ in range(5):
            bm.can_allocate(second)

        funnel = bm.checkpoint_funnel()
        assert funnel["demands_declined_no_room"] == 1, "counted per attempt"
        assert funnel["demands_recorded"] == 0

    def test_a_demand_survives_being_asked_twice_without_being_counted_twice(self):
        """The mirror: room throughout, so the recorded counter must not move."""
        bm = make_block_manager(demand_config(), state_runtime=BIG_IMAGE_RUNTIME)
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))

        second = stateful_seq(PROMPT)
        for _ in range(5):
            assert bm.can_allocate(second) >= 0

        assert second.checkpoint_demand_pos, "the demand was lost"
        funnel = bm.checkpoint_funnel()
        assert funnel["demands_recorded"] == 1, "counted per attempt"
        assert funnel["demands_declined_no_room"] == 0

    def test_reuse_another_class_declines_is_not_charged_to_the_ladder(self):
        """The counterfactual keeps every other gate applied.

        A boundary whose sliding window is gone stays out of reach however
        densely the ring is checkpointed, so it must not buy a cut. Attributing
        the whole gap to the ladder would have every request pay for a
        checkpoint the next one still cannot use.
        """
        bm = make_block_manager(demand_config())
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
        bm = make_block_manager(demand_config())
        short = list(range(16))
        short_branch = list(range(4)) + list(range(900, 912))
        first = stateful_seq(short)
        run_prompt_on_the_ladder(bm, first)
        assert bm.checkpoint_limit(first) == 0  # the grid places no rung here

        second = stateful_seq(short_branch)
        assert bm.can_allocate(second) == 0
        assert bm.checkpoint_limit(second) == 0
        assert second.checkpoint_demand_pos == 4  # the demand is its own rung
        # 4 is the demand, 8 is this prompt's own end anchored — and the anchor
        # is reachable here only because it is the demand's neighbour, not its
        # substitute: no prompt has ever ended at 4.
        assert run_prompt_on_the_ladder(bm, second) == [4, 8]

        third = stateful_seq(short_branch)
        assert bm.can_allocate(third) == 2  # ...and the next one collects it
        assert third.checkpoint_demand_pos == 0  # nothing left to want
        assert run_prompt_on_the_ladder(bm, third) == []

    def test_the_demand_is_cut_and_kept_at_the_same_position(self):
        """The cut and the keep read the same call, so they cannot drift."""
        bm = make_block_manager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        seq = stateful_seq(BRANCH)
        bm.allocate(seq, bm.can_allocate(seq))
        assert seq.checkpoint_demand_pos == 28
        assert seq.checkpoint_end_pos == 36

        n = len(BRANCH)
        cuts = {bm.checkpoint_cut(seq, pos - 1, pos) for pos in range(1, n + 1)}
        rungs = {pos for pos in range(1, n + 1) if bm.checkpointers_at(seq, pos)}
        # 16 and 32 from the grid, 28 the demand, 36 the anchor. Swept one
        # token at a time, so every position is offered to both sides — which
        # is what would catch `checkpoint_cut` picking a target `checkpointers_at`
        # then refuses, the failure the two-candidate ladder made possible.
        assert cuts - {0} == rungs == {16, 28, 32, 36}

    def test_a_recorded_demand_is_always_a_position_something_keeps(self):
        """Otherwise the cut is an extra forward that stores nothing.

        The demand comes out of the same fork test the ladder applies, on the
        same request, so it satisfies `successor_room` by construction. Swept
        rather than argued, because the two derivations sit in different files.
        """
        for n in range(20, 60, 3):
            bm = make_block_manager(demand_config())
            tokens = list(range(1000 * n, 1000 * n + n))
            run_prompt_on_the_ladder(bm, stateful_seq(tokens))
            seq = stateful_seq(tokens)
            bm.allocate(seq, bm.can_allocate(seq))
            demand = seq.checkpoint_demand_pos
            assert not demand or bm.checkpointers_at(seq, demand), n

    def test_a_stateless_model_records_no_demand(self):
        bm = make_block_manager(
            demand_config(pool_entries={}),
            state_runtime=StateRuntime(),
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        run_prompt_on_the_ladder(bm, cold)
        warm = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        assert bm.can_allocate(warm) == 10  # nothing was gating it
        assert warm.checkpoint_demand_pos == 0


class TestPromptEndAnchor:
    """A rung reserved at this prompt's own end, before anyone asks for it.

    The demand is reactive: it exists only once a hit has already been refused
    for want of a checkpoint, which is one request too late for the position
    that serves the next turn of a conversation. On agentic traffic that
    position is where nearly all the reuse is — over the SemiAnalysis cc-traces
    93.5% of resumes land on a previous prompt's end and 0.0% on the interval
    ladder — so it is reserved up front instead of waited for.

    These cases are the ones `TestDemandDrivenCheckpoints` used to make, before
    the anchor started serving the growing-conversation traffic they replayed.
    """

    def test_the_second_request_resumes_where_the_first_ended(self):
        """No disappointed request in between — this is the whole point."""
        bm = make_block_manager(demand_config())
        first = stateful_seq(PROMPT)
        run_prompt_on_the_ladder(bm, first)
        assert first.checkpoint_end_pos == 36

        second = stateful_seq(PROMPT)
        # 9 blocks = 36 tokens, the anchor. The grid alone would have given 8,
        # and the demand would have taken until the third request to find it.
        assert bm.can_allocate(second) == 9
        assert second.num_wanted_hit_blocks == 9  # nothing left on the table
        assert second.checkpoint_demand_pos == 0  # so nothing to demand

    def test_the_anchor_steps_back_to_a_position_that_is_keepable(self):
        """The exact end is never keepable, so insisting on it anchors nothing.

        A checkpoint at P binds the forward after it to carry `successor_room`
        tokens, and a grid-floored prompt end leaves at most `hash_block_size`
        minus one. Wherever the room reaches a block or more — MIN_FORK 8
        against BLOCK 4 here, V4's 131 against 256 in production — the floored
        end fails that test for *every* prompt: `checkpoint_cut` would shorten
        a chunk and `checkpointers_at` would then refuse to keep anything, with
        no error to show for it. Stepping back to the rightmost keepable grid
        position costs at most one block of the next turn's reuse.
        """
        bm = make_block_manager(demand_config())
        for n in (12, 40, 44, 45, 50):
            seq = stateful_seq(list(range(1000 * n, 1000 * n + n)))
            bm.can_allocate(seq)
            anchor = seq.checkpoint_end_pos
            assert anchor % BLOCK == 0, n  # on the hash grid
            assert n - anchor >= MIN_FORK, n  # and it leaves the fork its room
            assert bm.checkpointers_at(seq, anchor), n  # so it is really kept

    def test_a_prompt_with_no_room_for_an_anchor_gets_none(self):
        bm = make_block_manager(demand_config())
        seq = stateful_seq(list(range(MIN_FORK)))
        bm.can_allocate(seq)
        assert seq.checkpoint_end_pos == 0

    def test_the_cut_and_the_keep_agree_at_every_anchor(self):
        """Swept, because a cut nothing keeps is a forward spent on nothing."""
        for n in range(BLOCK, 80):
            bm = make_block_manager(demand_config())
            tokens = list(range(1000 * n, 1000 * n + n))
            for _ in range(3):
                seq = stateful_seq(tokens)
                bm.allocate(seq, bm.can_allocate(seq))
                cuts = set(forward_on_the_ladder(bm, seq))
                keeps = {p for p in range(1, n + 1) if bm.checkpointers_at(seq, p)}
                assert not cuts - keeps, (n, sorted(cuts), sorted(keeps))

    def test_the_anchor_does_not_displace_the_grid_rung(self):
        """Both are cut for, because they serve different classes.

        `checkpoint_cut` takes the *earliest* candidate for exactly this: with
        the anchor at 36 and a rung at 32, returning the later one means the
        forward never ends at 32 and the rung is not deferred but lost. A class
        the anchor is out of reach for would then lose the rung it had been
        resuming from, on every request, permanently.
        """
        bm = make_block_manager(demand_config())
        first = stateful_seq(PROMPT)
        assert run_prompt_on_the_ladder(bm, first) == [32, 36]

        bm.state_caches = (*bm.state_caches, StubStateCache(cap=8))
        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 8  # the rung, still there
        assert second.num_wanted_hit_blocks == 8  # and no gap to demand
        assert second.checkpoint_demand_pos == 0

    def test_a_stateless_model_records_no_anchor(self):
        bm = make_block_manager(
            demand_config(pool_entries={}),
            state_runtime=StateRuntime(),
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        bm.can_allocate(cold)
        assert cold.checkpoint_end_pos == 0

    def test_deallocate_clears_the_anchor(self):
        """Sequences are recycled; a stale anchor would cut the next prompt."""
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        run_prompt_on_the_ladder(bm, seq)
        assert seq.checkpoint_end_pos == 36
        bm.deallocate(seq)
        assert seq.checkpoint_end_pos == 0

    def test_anchor_cuts_are_counted_apart_from_demand_cuts(self):
        """The demand counter is a convergence signal and must stay readable.

        The anchor fires on nearly every prompt while the demand is supposed to
        fall silent once the gap it found is filled. Folding the two together
        would leave `chunks_cut_for_demand` growing forever on healthy traffic,
        which is precisely the shape it exists to expose.
        """
        bm = make_block_manager(demand_config())
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        assert bm.checkpoint_funnel()["chunks_cut_for_demand"] == 0
        assert bm.checkpoint_funnel()["chunks_cut_for_end"] == 1

        second = stateful_seq(BRANCH)
        bm.allocate(second, bm.can_allocate(second))
        forward_on_the_ladder(bm, second)
        assert bm.checkpoint_funnel()["chunks_cut_for_demand"] == 1
        assert bm.checkpoint_funnel()["chunks_cut_for_end"] == 2


class TestLadderOffButCheckpointingOn:
    """`-1`: no interval rungs, demand and anchor still place checkpoints.

    Every rung costs the prompt that keeps it an extra prefill chunk, and the
    interval is a guess about where reuse will resume. The other two placements
    are not guesses — one is a position a request was refused at, the other is
    where the next turn of a conversation demonstrably starts. On the
    SemiAnalysis cc-traces the ladder placed ~30x the writes of the two of them
    together and caught reuse they already reach: 0.0% of resumes landed on an
    8192 rung.

    Spelled `-1` rather than folded into `0` because `0` is the documented off
    switch *and* reachable by accident — `test_interval_snaps_onto_the_hash_-
    block_grid` shows an off-grid interval snapping down to it. Giving `0` a
    second meaning would turn a `--block-size` typo from failing safe into
    silently enabling a policy.
    """

    def test_minus_one_survives_the_grid_snap(self):
        bm = make_block_manager(ckpt_config(state_checkpoint_interval_tokens=-1))
        assert bm.state_checkpoint_interval_tokens == -1

    def test_the_grid_places_no_rung(self):
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=-1))
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert bm.checkpoint_limit(seq) == 0
        # 32 is a rung under the default interval, and nothing under -1. The
        # anchor at 36 is the only aimed position left.
        assert not bm.checkpointers_at(seq, 32)
        assert bm.checkpointers_at(seq, 36)

    def test_the_anchor_still_reaches_the_same_hit(self):
        """The point of the mode: the ladder's reuse for one cut, not two."""
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=-1))
        first = stateful_seq(PROMPT)
        assert run_prompt_on_the_ladder(bm, first) == [36]  # the ladder cut 32 too

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 9  # what the full ladder also gave
        assert second.checkpoint_demand_pos == 0

    def test_the_demand_still_fires(self):
        """This is what -1 buys over 0, and why it is not spelled 0."""
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=-1))
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))

        second = stateful_seq(BRANCH)
        assert bm.can_allocate(second) == 0
        assert second.checkpoint_demand_pos == 28
        bm.allocate(second, 0)
        assert forward_on_the_ladder(bm, second) == [28, 36]  # no rung at 32

        third = stateful_seq(BRANCH)
        # 9, not the demand's 7: by now `second` has left its own anchor at 36,
        # which is further along than the branch point. The demand's rung is
        # what got `second` past 28 to reach the end and place that anchor —
        # under interval=0 the pair would still be stuck at 0.
        assert bm.can_allocate(third) == 9

    def test_zero_would_have_left_that_reuse_on_the_floor(self):
        """The same three requests under 0, as the contrast -1 exists for."""
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=0))
        run_prompt_on_the_ladder(bm, stateful_seq(PROMPT))
        for _ in range(3):
            seq = stateful_seq(BRANCH)
            assert bm.can_allocate(seq) == 0
            bm.allocate(seq, 0)
            assert forward_on_the_ladder(bm, seq) == []

    def test_generation_keeps_no_checkpoints(self):
        """Decode spacing is measured in intervals, and there is no interval.

        Both aimed placements are prompt positions, so an unaimed position past
        the prompt has nothing to match. Stated as a test because the arithmetic
        that would otherwise run — `pos - last < -1` — is true for every pos,
        which would checkpoint on every decode step.
        """
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=-1))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert not any(
            bm.checkpointers_at(seq, pos, aimed=False)
            for pos in range(BLOCK, 200, BLOCK)
        )

    def test_it_costs_fewer_cuts_than_the_ladder(self):
        """The whole justification, swept rather than asserted at one length."""
        totals = {}
        for interval in (INTERVAL, -1):
            cuts = 0
            for n in range(BLOCK, 80):
                bm = make_block_manager(
                    demand_config(state_checkpoint_interval_tokens=interval)
                )
                tokens = list(range(1000 * n, 1000 * n + n))
                for _ in range(3):
                    seq = stateful_seq(tokens)
                    bm.allocate(seq, bm.can_allocate(seq))
                    cuts += len(forward_on_the_ladder(bm, seq))
            totals[interval] = cuts
        assert totals[-1] < totals[INTERVAL]

    def test_every_cut_is_still_kept(self):
        for n in range(BLOCK, 80):
            bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=-1))
            tokens = list(range(1000 * n, 1000 * n + n))
            for _ in range(3):
                seq = stateful_seq(tokens)
                bm.allocate(seq, bm.can_allocate(seq))
                cuts = set(forward_on_the_ladder(bm, seq))
                keeps = {p for p in range(1, n + 1) if bm.checkpointers_at(seq, p)}
                assert not cuts - keeps, (n, sorted(cuts), sorted(keeps))

    def test_interval_zero_anchors_nothing(self):
        """0 is off for *all three* placements, not just the grid.

        The anchor is recorded outside the grid, so it does not inherit the
        grid's off switch — it has to check the interval itself. Without that
        check `checkpoint_cut` shortens a chunk on every prompt and
        `checkpointers_at` then refuses to keep anything, which is a per-request
        cost with nothing stored and no error raised.
        """
        bm = make_block_manager(demand_config(state_checkpoint_interval_tokens=0))
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert seq.checkpoint_end_pos == 0
        assert run_prompt_on_the_ladder(bm, stateful_seq(PROMPT)) == []

    def test_both_classes_are_anchored_for(self):
        """The anchor costs a prefill chunk, so it has to buy something.

        It does on both paths now. `PagedStateCheckpointCoordinator` keys its
        pending checkpoints by `(seq, hash)`, so the anchor and the prompt-end
        checkpoint a chunk later are two entries and both survive to be stored.

        This test previously pinned the opposite, and was right to: with one
        pending entry per seq the later write overwrote the anchor before
        either was stored, so the chunk was bought and thrown away. What
        changed is the key, and what makes keeping both affordable is the
        image's price -- 127 blocks against a whole Active Slot under `fork`.

        The anchor is also the placement that pays: of 4,808 cc-trace resumes
        with a nonzero KV hit, 93.5% land on a previous prompt end and 0.0% on
        the 8192 ladder.

        Asked through `checkpoint_end_pos` rather than a capability flag: what
        matters is that both paths actually anchor, and the flag that used to
        gate this answered `True` from every implementor there was.
        """
        fork = make_block_manager(ckpt_config())
        copy = make_block_manager(ckpt_config(), state_runtime=PAGED_COPY_RUNTIME)
        forked, copied = stateful_seq(PROMPT), stateful_seq(PROMPT)
        fork.can_allocate(forked)
        copy.can_allocate(copied)
        assert forked.checkpoint_end_pos > 0, "fork should anchor"
        assert copied.checkpoint_end_pos > 0, "copy should anchor too now"

    def test_a_later_boundary_supersedes_an_undrained_earlier_one(self):
        """One pending boundary per seq, because one slot holds one state.

        A pending entry names a hash and is stored from `seq.state_slot` at the
        drain. Two entries surviving into one drain would both be stored from
        whatever the last forward left in that slot, filing the earlier hash
        over the later state -- a resuming request would then continue from
        ahead of its own prefix, and `_validate_paged_state_op` would pass,
        because layout, size and unit count are all still right.

        This test previously asserted the opposite and was wrong to: it pinned
        the coexistence without pinning that each was stored from its own slot,
        which the drain cannot do. The newer boundary wins because it is the
        one the slot actually holds.
        """
        copy = make_block_manager(ckpt_config(), state_runtime=PAGED_COPY_RUNTIME)
        coord = copy.paged_state_checkpoints
        seq = stateful_seq(PROMPT)
        seq.state_slots = [0]

        coord.checkpoint(seq, 4, 111)
        coord.checkpoint(seq, 8, 222)
        assert [h for _sid, h in coord._pending] == [222], "the slot holds 222"
        assert coord.checkpoints_dropped == 1, "111 is lost reuse, and counted"

        coord.checkpoint(seq, 8, 222)
        assert len(coord._pending) == 1, "the same boundary twice is still one"
        assert coord.checkpoints_dropped == 2, "superseding itself still counts"

        coord.forget_pending(seq)
        assert not coord._pending, "a released slot takes it with it"

    def test_two_seqs_do_not_supersede_each_other(self):
        """Superseding is per sequence -- each has its own slot."""
        copy = make_block_manager(ckpt_config(), state_runtime=PAGED_COPY_RUNTIME)
        coord = copy.paged_state_checkpoints
        first, second = stateful_seq(PROMPT), stateful_seq(PROMPT)
        first.state_slots, second.state_slots = [0], [1]

        coord.checkpoint(first, 4, 111)
        coord.checkpoint(second, 4, 222)
        assert len(coord._pending) == 2
        assert coord.checkpoints_dropped == 0


class TestCacheAttribution:
    """Splitting declined reuse into the part a checkpoint reaches and the rest.

    One number for both makes "does demand-driven checkpointing apply to this
    workload" unfalsifiable, which is the whole reason the counterfactual is
    computed outside the tests.
    """

    def test_the_split_accounts_for_every_declined_token(self):
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        stats.update_cache(32, 48, 40, 36, 44)
        lost_to_checkpoint = stats.total_wanted_tokens - stats.total_cached_tokens
        lost_hard = stats.total_compressed_tokens - stats.total_wanted_tokens
        assert lost_to_checkpoint == 4
        assert lost_hard == 4
        assert lost_to_checkpoint + lost_hard == 40 - 32

    def test_a_perfect_run_is_reported_as_perfect(self):
        """The regression that motivated `reusable` as the denominator.

        Against `full`, every rate here read below 100% on a run where both
        caches did everything they possibly could: `can_allocate` never matches
        the trailing block, so `compressed < full` holds for every request that
        could exist and the shortfall is charged to a pool that was never
        offered the block.
        """
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        # 100 tokens, 90 reusable: every reusable token was served by cache.
        stats.update_cache(90, 100, 90, 90, 90)

        assert stats.cache_hit_rate == 1.0
        assert stats.paged_hit_rate == 1.0
        assert stats.state_hit_rate == 1.0

    def test_each_pool_is_scored_against_what_it_was_actually_asked_for(self):
        """The two rates must isolate their own pool, and compose exactly.

        The paged pool is asked for `reusable` and supplies `compressed`. The
        state cache never sees what the paged pool already lost, so it is
        scored against `compressed`, not `reusable` -- otherwise a KV eviction
        shows up as a state-cache failure and sends tuning at the wrong pool.
        """
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        # Of 100 reusable, the paged pool had 80 (80%); of those 80 the state
        # gates admitted 60 (75%). End to end: 60%.
        stats.update_cache(60, 128, 80, 70, 100)

        assert stats.paged_hit_rate == 0.80
        assert stats.state_hit_rate == 0.75
        assert stats.cache_hit_rate == 0.60
        # approx, not ==: the identity is exact over the integer counters, but
        # each rate is a float division first, so the product carries rounding.
        assert stats.paged_hit_rate * stats.state_hit_rate == pytest.approx(
            stats.cache_hit_rate
        )

    def test_a_kv_eviction_does_not_lower_the_state_cache_score(self):
        """Independence, stated as the property that makes the split useful.

        Two runs whose state cache behaves identically -- admitting every
        boundary the paged pool offered -- must score the same on
        `state_hit_rate` however much prefix the paged pool lost.
        """
        healthy = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        healthy.update_cache(100, 128, 100, 100, 100)
        evicted = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        evicted.update_cache(40, 128, 40, 40, 100)  # paged pool lost 60% of the prefix

        assert evicted.paged_hit_rate < healthy.paged_hit_rate
        assert evicted.state_hit_rate == healthy.state_hit_rate == 1.0

    def test_the_recoverable_share_bounds_what_checkpointing_can_buy(self):
        """`state_hit + recoverable` is the ceiling a dense ladder would reach.

        The distance from there to 1.0 is loss no checkpoint touches, and so
        the honest cap on what more groups are worth.
        """
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        # 80 offered, 50 admitted; a dense ladder would have reached 70.
        stats.update_cache(50, 128, 80, 70, 100)

        assert stats.state_hit_rate == 0.625
        assert stats.state_recoverable_loss_rate == 0.25
        assert stats.state_hit_rate + stats.state_recoverable_loss_rate == 0.875

    def test_a_violated_ordering_is_clamped_not_fatal(self, caplog):
        """Every rate is a difference of two totals, so an out-of-order update
        would report a negative percentage. Clamp it, and say so.

        Not an assert: `num_cached_tokens` has four independent writers and the
        CPU-offload wake can legitimately load more prefix than the GPU index
        held, so an `AssertionError` here would take the engine down to protect
        a log line -- and only in builds without `-O`, so the two would differ
        in behaviour.
        """
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        with caplog.at_level(logging.WARNING):
            stats.update_cache(50, 128, 40, 45, 100)  # cached > compressed
            stats.update_cache(50, 128, 90, 60, 80)  # compressed > reusable

        assert stats.total_requests == 2, "both were counted, neither raised"
        assert sum("clamping" in r.message for r in caplog.records) == 2
        # Clamped into order, so every rate stays inside [0, 1].
        assert 0.0 <= stats.cache_hit_rate <= 1.0
        assert 0.0 <= stats.paged_hit_rate <= 1.0
        assert 0.0 <= stats.state_hit_rate <= 1.0
        assert stats.total_cached_tokens <= stats.total_compressed_tokens
        assert stats.total_compressed_tokens <= stats.total_reusable_tokens

    def test_hit_tokens_are_counted_in_hash_blocks(self):
        """Under DCP one block_table entry spans `dcp` blocks of tokens."""
        sched = make_scheduler(demand_config(decode_context_parallel_size=2))
        assert sched.block_manager.hash_block_size == 2 * BLOCK
        seq = stateful_seq(PROMPT)
        seq.num_compressed_hit_blocks = 3
        seq.num_wanted_hit_blocks = 2
        sched._schedule_prefill_seq(seq, 44, {}, [], 0, 0)
        assert sched.engine_stats.total_compressed_tokens == 3 * 2 * BLOCK
        assert sched.engine_stats.total_wanted_tokens == 2 * 2 * BLOCK

    def test_the_reuse_ceiling_matches_the_matcher_that_sets_it(self):
        """The scheduler's ceiling and `can_allocate`'s match loop are the same
        rule written twice, so pin them to each other rather than to a literal.

        Drift here is silent and one-directional: a ceiling above what the
        matcher can reach makes a perfect run look imperfect forever.
        """
        sched = make_scheduler(demand_config())
        hbs = sched.block_manager.hash_block_size
        seq = stateful_seq(PROMPT)
        sched._schedule_prefill_seq(seq, 44, {}, [], 0, 0)

        matchable_blocks = sched.block_manager._n_hash_blocks(seq) - 1
        assert sched.engine_stats.total_reusable_tokens == matchable_blocks * hbs
        assert sched.engine_stats.total_reusable_tokens < seq.num_tokens

    def test_a_sequence_below_one_block_has_nothing_to_reuse(self):
        """The `n_hash_blocks - 1` ceiling goes negative for a short prompt.

        Its reuse ceiling is genuinely zero -- the only block it has is the one
        prefill must compute -- and a negative denominator would invert every
        rate on the line.
        """
        stats = EngineStats(enable_prefix_caching=True, cache_log_interval=10**6)
        stats.update_cache(0, 10, 0, 0, 0)
        assert stats.total_reusable_tokens == 0
        assert stats.cache_hit_rate == 0.0
        assert stats.paged_hit_rate == 0.0


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
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        assert self.keepers(bm, seq, INTERVAL, aimed=True)
        assert not self.keepers(bm, seq, INTERVAL + BLOCK, aimed=True)

    def test_an_unaimed_step_keeps_off_the_grid(self):
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        assert self.keepers(bm, seq, INTERVAL + BLOCK, aimed=False)

    def test_an_unaimed_step_still_has_to_land_on_a_block(self):
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        # The checkpoint is filed under the hash of a whole block, so a landing
        # between two of them has nothing to file it under.
        assert not self.keepers(bm, seq, INTERVAL + 1, aimed=False)

    def test_spacing_is_measured_from_the_last_one_kept(self):
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        seq.last_checkpoint_pos = INTERVAL + BLOCK
        assert not self.keepers(bm, seq, 2 * INTERVAL, aimed=False)
        assert self.keepers(bm, seq, 2 * INTERVAL + BLOCK, aimed=False)

    def test_the_grid_ignores_the_watermark(self):
        # An aimed caller answers to `checkpoint_cut`, which knows nothing of
        # the watermark; letting it in here would put the two out of step.
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        seq.last_checkpoint_pos = INTERVAL
        assert self.keepers(bm, seq, 2 * INTERVAL, aimed=True)

    def test_a_demand_is_out_of_generation_s_reach(self):
        # Not a rule, an arithmetic fact: a demand is bounded by the prompt's
        # own hit ceiling, and generation only ever asks about positions at or
        # past the end of the prompt. The unaimed branch omits the demand
        # because of this, so the day it stops holding, this fails first.
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        second = stateful_seq(PROMPT)
        bm.allocate(second, bm.can_allocate(second))
        assert second.checkpoint_demand_pos < second.num_prompt_tokens


class TestTheCacheCannotStarveLiveKv:
    """Why no floor is held back for live KV.

    `_fresh_block` raises when the pool is dry and nothing is evictable, and
    the checkpoint cache shares that pool. These pin the three facts that keep
    the cache from ever taking it there, so that a future reader looking for a
    reserve finds the argument instead of re-inventing one.
    """

    def _pool_of_three_with_one_checkpoint(self, pin: bool):
        """A pool holding exactly one image, optionally being read.

        Three units, one checkpoint, nothing spare -- the tightest state the
        cache can put the pool in.
        """
        bm = make_block_manager(
            paged_copy_config(num_kvcache_blocks=3, state_checkpoint_interval_tokens=0),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        checkpoints = bm.paged_state_checkpoints
        assert checkpoints.store.units_per_checkpoint == 3
        assert checkpoints.store.begin_store(33, src_slot=0) is not None
        checkpoints.store.complete_inflight()
        if pin:
            assert checkpoints.begin_restore(33, dst_slot=1)
        assert bm.kv.num_free == 0, "the pool is meant to have nothing spare"
        return bm, checkpoints.store

    def test_a_ready_unpinned_checkpoint_is_available_to_live_kv(self):
        """The cache's size is not the variable: a spendable image is free space.

        `has_available_units` counts it and `ensure_free_units` spends it, so
        holding checkpoints costs live KV nothing and there is nothing for a
        floor to ration.
        """
        bm, store = self._pool_of_three_with_one_checkpoint(pin=False)
        assert store.has_available_units(3), "the image was not counted as free space"
        assert not store.has_available_units(4), "more was counted than exists"

        seq = stateful_seq(list(range(BLOCK)))
        assert bm.can_allocate(seq) == 0, "a spendable image was not counted"

        bm.allocate(seq, 0)
        assert store.lookup(33) < 0, "it was counted but could not be spent"

    def test_a_pinned_cache_refuses_an_admission_rather_than_raising(self):
        """The reachable outcome under contention, and the one that is not.

        A restore pin is the one thing that makes an image unspendable while
        allocation is running. Even with the whole cache pinned and the free
        list empty, the gate answers no and the request waits for the pass that
        releases the pin -- `_fresh_block` is never reached.
        """
        bm, store = self._pool_of_three_with_one_checkpoint(pin=True)
        assert not store.has_available_units(1), "the cache is meant to be pinned"

        seq = stateful_seq(list(range(BLOCK)))

        assert bm.can_allocate(seq) < 0, "the gate admitted a seq it cannot serve"

    def test_bypassing_the_gate_reaches_the_raise(self):
        """The sibling that gives the test above its meaning.

        Without this one, `can_allocate` returning -1 would be indistinguishable
        from a scenario that was never tight enough to matter.
        """
        bm, _ = self._pool_of_three_with_one_checkpoint(pin=True)
        seq = stateful_seq(list(range(BLOCK)))

        with pytest.raises(AssertionError, match="No PAGE unit"):
            bm.allocate(seq, 0)

    def test_a_pass_releases_the_previous_pins_before_it_allocates(self):
        """Why the decode loop never sees a pin.

        Pins live one pass: `schedule` releases the previous batch's before it
        admits anything. Observable here because the admission below is only
        possible once the pinned image becomes spendable again.
        """
        scheduler = make_scheduler(
            paged_copy_config(num_kvcache_blocks=3, state_checkpoint_interval_tokens=0),
            state_runtime=PAGED_COPY_RUNTIME,
        )
        # The same pinned pool as the tests above, built inside a scheduler,
        # with the batch that reads the restore gone out -- which is what the
        # pin is waiting on.
        checkpoints = scheduler.block_manager.paged_state_checkpoints
        assert checkpoints.store.begin_store(33, src_slot=0) is not None
        checkpoints.store.complete_inflight()
        assert checkpoints.begin_restore(33, dst_slot=1)
        scheduler.block_manager.take_state_maintenance_ops()
        assert not checkpoints.store.has_available_units(1)
        assert (
            scheduler.block_manager.can_allocate(stateful_seq(list(range(BLOCK)))) < 0
        )

        scheduler.add(stateful_seq(list(range(BLOCK))))
        _, scheduled = scheduler.schedule()

        assert scheduled, "the pass allocated before releasing the previous pin"


# ── midstep checkpoints ────────────────────────────────────────────────────


#: A backend that reads its recurrent state at interior chunk boundaries.
MIDSTEP_STATE_RUNTIME = StateRuntime(
    transfer=StateTransfer.fork(MIN_FORK, readable_midstep=True)
)


def make_midstep_bm(**overrides):
    """A `BlockManager` on `demand_config` for a midstep-readable backend.

    Readability is a property of the backend, not of the config, so it arrives
    through `state_runtime` rather than a config field — the same route
    `ModelRunner` hands `GDNStateMixin.state_transfer` down.
    """
    return make_block_manager(
        demand_config(**overrides), state_runtime=MIDSTEP_STATE_RUNTIME
    )


def forward_midstep(bm: BlockManager, seq: Sequence) -> list[int]:
    """Run an admitted seq's prompt the way a readable backend does.

    The scheduler's loop with the cut still consulted — it should never fire —
    and `plan_midstep` where `Scheduler.schedule` puts it, once the chunk is
    settled. Returns the positions checkpointed, which under this backend is
    what the ladder yields *without* the forwards it used to cost.
    """
    kept = []
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        start = seq.num_cached_tokens
        chunk = seq.num_prompt_tokens - start
        assert not bm.checkpoint_cut(seq, start, start + chunk)
        bm.plan_midstep(seq, start, start + chunk)
        kept.extend(p for _g, p, _h in seq.midstep_reservations)
        bm.hash_blocks(seq, chunk, start_tokens=start)
        seq.num_cached_tokens = start + chunk
    return kept


class TestMidstepCheckpoints:
    """Every rung of the ladder, kept inside one full-length forward.

    A checkpoint is state as of position P, and the only reason the scheduler
    shortens a prefill chunk onto P is that most backends can hand back state
    only as of the forward's last token. A chunk kernel does not have that
    limitation: it materializes the recurrent state at every interior chunk
    boundary on its way through, so P is a copy rather than a forward.

    So the ladder's cost model changes and its reach does not. `checkpoint_cut`
    returns 0 for every seq and `checkpointers_at` defers to the midstep path;
    the two gates are one change, because suppressing the cut alone leaves
    `checkpointers_at` refusing off-grid positions it is then handed and keeping
    nothing at all, silently.
    """

    def test_the_ladder_costs_no_forwards(self):
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        # The unreadable backend cuts at 32 and again at 36 for this prompt.
        assert forward_midstep(bm, seq) == [32, 36]
        assert bm.checkpoint_funnel()["chunks_cut_for_end"] == 0
        assert bm.checkpoint_funnel()["chunks_cut_for_demand"] == 0

    def test_the_reuse_is_the_same_reuse(self):
        """The point: same hit as the cutting ladder, without the cuts."""
        for readable in (False, True):
            bm = make_block_manager(
                demand_config(),
                state_runtime=(
                    MIDSTEP_STATE_RUNTIME if readable else DEFAULT_STATE_RUNTIME
                ),
            )
            first = stateful_seq(PROMPT)
            bm.allocate(first, bm.can_allocate(first))
            (forward_midstep if readable else forward_on_the_ladder)(bm, first)

            second = stateful_seq(PROMPT)
            assert bm.can_allocate(second) == 9, readable

    def test_both_positions_are_separately_resumable(self):
        """Not one checkpoint at the rightmost — one per position, each keyed.

        A single group filed under the last position would look identical on a
        prompt that reuses the whole prefix, and fail the moment a request
        branches before it.
        """
        bm = make_midstep_bm()
        first = stateful_seq(PROMPT)
        bm.allocate(first, bm.can_allocate(first))
        forward_midstep(bm, first)

        assert len(set(bm.state.hash_to_slot.values())) == 2

        # A request sharing 32 tokens and then diverging cannot use the anchor
        # at 36, so its hit of 8 blocks is 32's checkpoint and could have come
        # from nowhere else. Filing both positions under one group would leave
        # this at 0.
        branch_at_32 = stateful_seq(list(range(32)) + list(range(900, 916)))
        assert bm.can_allocate(branch_at_32) == 8
        # And the whole-prefix case still reaches the further one.
        assert bm.can_allocate(stateful_seq(PROMPT)) == 9

    def test_the_boundary_is_not_kept_twice(self):
        """`checkpointers_at` has to defer, or both paths keep the same rung.

        The midstep path already filed 32, and a forward that also ends there
        is exactly what the ladder used to produce — so without the gate the
        rung is kept a second time. Two groups on one hash, the loser sitting
        free and unindexed; and under `fork` the seq gives its live group away
        and takes a fresh one, binding the next forward to refill a replacement
        it had no reason to need.
        """
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        group = seq.state_slot
        bm.plan_midstep(seq, 0, 32)
        bm.hash_blocks(seq, 32, start_tokens=0)

        assert bm.checkpoint_funnel()["checkpoints_kept"] == 1
        assert seq.state_slot == group  # not forked out from under it
        assert seq.state_fork_src == -1

    def test_a_position_the_hash_chain_cannot_name_is_skipped(self):
        """No hash, no way back — so reserving one would spend a group on air."""
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        seq.block_hashes = seq.block_hashes[:2]  # 8 tokens' worth
        assert bm.midstep_positions(seq, 0, 44) == []

    def test_the_chain_covers_the_whole_prompt_past_the_miss(self):
        """`block_hashes` stops at the first miss; the anchor is past it."""
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert len(seq.block_hashes) == len(PROMPT) // BLOCK
        # And it is the same chain `hash_blocks` publishes, or a resumer would
        # look the checkpoint up under a hash nothing files it under.
        bm.allocate(seq, 0)
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        published = [bm.kv.block(b).hash for b in seq.block_table]
        assert published == seq.block_hashes

    def test_an_unreadable_backend_keeps_its_chain_empty(self):
        """A hash pass over every prompt, for a field nothing would read."""
        bm = make_block_manager(demand_config())
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert seq.block_hashes == []

    def test_nothing_is_findable_until_the_forward_has_run(self):
        """Publishing at reservation time indexes bytes nobody wrote."""
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        assert seq.midstep_reservations
        assert bm.state.hash_to_slot == {}

        bm.hash_blocks(seq, 44)
        assert len(bm.state.hash_to_slot) == 2
        assert seq.midstep_reservations == []  # drained, not left to re-publish

    def test_a_cancelled_reservation_is_returned_vacant(self):
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        free_before = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        assert bm.state.num_free() == free_before - 2

        bm.cancel_midstep(seq)
        assert bm.state.num_free() == free_before
        assert bm.state.hash_to_slot == {}  # holding nothing findable

    def test_replanning_returns_the_previous_forward_s_groups(self):
        """A plan is good for one forward; a second means the first never ran."""
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        free_with_one_plan = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        assert bm.state.num_free() == free_with_one_plan

    def test_deallocate_returns_them_too(self):
        """Preemption frees through here, and the forward is not going to run."""
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        free_before = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        bm.deallocate(seq)
        # `free_before` counted the seq's own group as taken; deallocate hands
        # that back as well, so the reservations are the difference.
        assert bm.state.num_free() == free_before + 1
        assert seq.midstep_reservations == []

    def test_a_shortage_keeps_the_earliest_position(self):
        """Best-effort, in the order a later forward would reach them.

        The earliest is the one an earlier chunk arrives at, and the one a
        branching request is most likely to still be able to use.
        """
        bm = make_midstep_bm(pool_entries={"state": 2})
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        assert [p for _g, p, _h in seq.midstep_reservations] == [32]
        assert bm.checkpoint_funnel()["checkpoints_dropped"] == 1

    def test_reservations_never_starve_an_admission(self):
        """`has_free` is the gate, so the worst case is a deferred admission."""
        bm = make_midstep_bm(pool_entries={"state": 3})
        first = stateful_seq(PROMPT)
        bm.allocate(first, bm.can_allocate(first))
        bm.plan_midstep(first, 0, 44)
        # Two groups reserved, one held by `first` — the pool is empty, and a
        # second request is refused rather than handed a reserved group.
        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == -1
        assert bm.state.num_free() == 0

    def test_generation_still_checkpoints_the_ordinary_way(self):
        """Midstep is a prefill affair; a decode step ends where acceptance says.

        `checkpointers_at` defers only on the aimed path, so an unaimed caller
        gets the same answer a fork backend has always given.
        """
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.checkpointers_at(seq, INTERVAL + BLOCK, MIN_FORK, aimed=False)

    def test_the_prompt_s_checkpoints_space_the_decode_ones(self):
        """`last_checkpoint_pos` is the decode spacing rule's only input.

        A prompt that filed a midstep checkpoint at its end and left the
        watermark at 0 would let the first decode boundary keep another one
        immediately, which is what the interval exists to prevent.
        """
        bm = make_midstep_bm()
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        forward_midstep(bm, seq)
        assert seq.last_checkpoint_pos == 36

    def test_one_unreadable_class_keeps_the_cut(self):
        """The gate is `all`, not `any`: that class still needs the forward.

        A readable class loses nothing by being handed a position it would have
        taken anyway, and an unreadable one loses everything by being handed a
        forward that does not end there.
        """
        bm = make_midstep_bm()
        bm.state_caches = (*bm.state_caches, StubStateCache(successor_room=0))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.checkpoint_cut(seq, 0, 44) == 32
        assert bm.checkpointers_at(seq, 32)

    def test_interval_zero_reserves_nothing(self):
        """0 is off for the midstep path too, as it is for the other three."""
        bm = make_midstep_bm(state_checkpoint_interval_tokens=0)
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.midstep_positions(seq, 0, 44) == []

    def test_minus_one_reserves_the_anchor_alone(self):
        """The two changes compose: no grid, no cuts, and the reuse still there."""
        bm = make_midstep_bm(state_checkpoint_interval_tokens=-1)
        first = stateful_seq(PROMPT)
        bm.allocate(first, bm.can_allocate(first))
        assert forward_midstep(bm, first) == [36]

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 9

    def test_a_stateless_model_reserves_nothing(self):
        bm = make_block_manager(
            demand_config(pool_entries={}),
            state_runtime=StateRuntime(),
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        bm.can_allocate(cold)
        assert cold.block_hashes == []
        assert bm.midstep_positions(cold, 0, 44) == []


def test_fates_report_every_counter():
    """checkpoint_fates() must expose all four fate counters.

    NOTE: the actual dict keys carry the ``checkpoints_`` prefix
    (``checkpoints_kept``, ``checkpoints_evicted``, ``checkpoints_orphaned``,
    ``checkpoints_dropped``).  The task-0 brief assumed short keys
    (``kept`` / ``evicted`` / …); those differ — see the report for the
    discrepancy note and the rationale for leaving the public API unchanged.
    """
    pool = StateSlotPool(num_slots=2, transfer=StateTransfer.fork(1), hash_block_size=4)
    fates = pool.checkpoint_fates()
    assert set(fates) == {
        "checkpoints_kept",
        "checkpoints_evicted",
        "checkpoints_orphaned",
        "checkpoints_dropped",
    }


def test_state_checkpoint_fates_warns_on_missing_method(caplog):
    """A state cache that lacks checkpoint_fates() emits a warning.

    The aggregator must not silently under-count: when a class registered in
    ``state_caches`` does not implement ``checkpoint_fates()``, a WARNING is
    logged naming the skipped class so an operator knows the total is partial.
    """
    import logging

    bm = BlockManager(ckpt_config())
    # StubStateCache (defined above) satisfies StateCache but has no
    # checkpoint_fates — exactly the class of future lightweight implementations
    # the warning is meant to catch.
    bm.state_caches = (bm.state_caches[0], StubStateCache())

    with caplog.at_level(logging.WARNING, logger="atom"):
        bm.state_checkpoint_fates()

    assert any(
        "StubStateCache" in r.message and "checkpoint_fates" in r.message
        for r in caplog.records
    ), "expected a WARNING naming StubStateCache; got: " + str(
        [r.message for r in caplog.records]
    )


def test_state_checkpoint_fates_warns_once_per_class(caplog):
    """The caller is the scheduler's every-100-ticks stats line and
    `state_caches` never changes during a run, so an unlatched warning is the
    same line forever -- drowning the log it is trying to draw attention to.
    What it reports is a static property of the build: true on tick 1, no
    truer on tick 10000.
    """
    import logging

    bm = BlockManager(ckpt_config())
    bm.state_caches = (bm.state_caches[0], StubStateCache())

    with caplog.at_level(logging.WARNING, logger="atom"):
        for _ in range(5):
            bm.state_checkpoint_fates()

    hits = [r for r in caplog.records if "StubStateCache" in r.message]
    assert len(hits) == 1, [r.message for r in hits]


def test_state_checkpoint_fates_log_order_is_stable(caplog):
    """The periodic log line emitted by `Scheduler.schedule()` sorts its keys.

    Read by an operator diffing one stats line against the next, so the field
    order has to be a property of the line rather than of whichever order the
    counters happen to be declared in — `StateSlotPool.checkpoint_fates()`
    returns them kept/dropped/evicted/orphaned, which is not alphabetical, and
    a new counter appended to that dict would otherwise land in the middle of
    the line for every deployment at once.

    Asserted against the emitted message, not against a `sorted()` this test
    applies itself: dropping the `sorted()` from the scheduler's join has to be
    what fails here, and the non-alphabetical insertion order above is what
    makes the two distinguishable.
    """
    import logging

    sched = Scheduler(ckpt_config())
    sched.add(stateful_seq(list(range(40))))
    pool = sched.block_manager.state
    pool.checkpoints_kept += 3
    pool.checkpoints_dropped += 1
    # The premise: if the counters were declared alphabetically the emitted
    # line would be sorted whether or not the scheduler sorted it.
    assert list(pool.checkpoint_fates()) != sorted(pool.checkpoint_fates())

    # The line is periodic — one pass in a hundred — so the batch has to be
    # driven to the tick that emits it.
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(100):
            sched.schedule()

    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("state checkpoints: ")
    ]
    assert lines, "the periodic state-checkpoint line was never emitted"

    fields = lines[-1].removeprefix("state checkpoints: ").split()
    keys = [f.split("=")[0] for f in fields]
    assert keys == [
        "checkpoints_dropped",
        "checkpoints_evicted",
        "checkpoints_kept",
        "checkpoints_orphaned",
    ], f"fields are not in alphabetical order: {lines[-1]}"
    # The values ride along with their keys rather than being sorted apart.
    assert "checkpoints_kept=3" in fields and "checkpoints_dropped=1" in fields


def test_checkpoint_funnel_includes_second_state_class():
    """checkpoint_funnel() must aggregate fates across ALL state classes.

    Prior to the fix, ``checkpoint_funnel()`` called ``self.state.checkpoint_fates()``
    directly, which would miss any second state class added to ``state_caches``.
    After routing through ``state_checkpoint_fates()``, a second class that
    implements ``checkpoint_fates()`` is included in the funnel output.
    """

    class SecondPoolStub:
        """Minimal StateCache with checkpoint_fates — mimics a second real class."""

        successor_room = 0

        def applies(self, seq):
            return False

        def resumable_hit(self, seq, P, block_hashes, assume_checkpointed=False):
            return 0

        def checkpoint(self, seq, boundary_blocks, h):
            pass

        def checkpoint_fates(self) -> dict:
            return {"checkpoints_kept": 7, "checkpoints_dropped": 2}

    bm = BlockManager(ckpt_config())
    bm.state_caches = (bm.state_caches[0], SecondPoolStub())

    funnel = bm.checkpoint_funnel()

    # The two ladder-level counters must still be present.
    assert "demands_recorded" in funnel
    assert "chunks_cut_for_demand" in funnel

    # The second class's counters must appear in the funnel — they would have
    # been absent before the fix.
    assert (
        funnel.get("checkpoints_kept", 0) >= 7
    ), "checkpoints_kept from SecondPoolStub missing from checkpoint_funnel()"
    assert (
        funnel.get("checkpoints_dropped", 0) >= 2
    ), "checkpoints_dropped from SecondPoolStub missing from checkpoint_funnel()"


# ── a claimed joint boundary must have a state leg behind it ───────────────


class TestPagedAllocateAimsTheStateLegAtTheJointBoundary:
    """`can_allocate` picks the boundary and `_attach_state_slots` secures the
    state, and the PAGE branch of `allocate` used to hand it `hit_hash` -- the
    hash at the *HBM* hit -- while the fork branch already used the joint one.

    Both ways of getting that wrong end identically: the KV leg loads to the
    boundary, `_claim_after_load` raises `num_cached_tokens` to it, and the
    forward resumes over a prefix the recurrent state does not cover.
    """

    class _TierIndex:
        def __init__(self, *hashes):
            self.hashes = set(hashes)
            self.pending_loads = {}
            self.requested = []
            self.can_store = True
            self.can_load = True

        def could_serve(self, h):
            return self.can_load and h in self.hashes

        def request_load(self, req_id, h):
            if h not in self.hashes:
                return False
            self.pending_loads[req_id] = h
            self.requested.append((req_id, h))
            return True

    def _bm(self, *tier_hashes):
        bm = make_block_manager(paged_copy_config(), state_runtime=PAGED_COPY_RUNTIME)
        index = self._TierIndex(*tier_hashes)
        bm.state_offload = index
        bm.paged_state_checkpoints.attach_offload(index)
        return bm, index

    def _joint_seq(self, bm, tokens, *, boundary_hash_value, claim_tokens):
        seq = stateful_seq(list(tokens))
        seq.offload_joint.boundary_hash = boundary_hash_value
        seq.offload_joint.boundary_tokens = claim_tokens
        seq.offload_joint.claim_tokens = claim_tokens
        return seq

    def test_the_state_leg_is_aimed_at_the_boundary_not_the_hbm_hit(self):
        """`num_cached_blocks == 0`, so `hit_hash` is never assigned and the
        old code took the cold-start exit: no restore, no load, and a boundary
        still claimed."""
        bm, index = self._bm(4242)
        seq = self._joint_seq(bm, range(48), boundary_hash_value=4242, claim_tokens=0)
        bm.allocate(seq, 0)

        assert index.requested == [
            (seq.id, 4242)
        ], "the state leg must be aimed at the joint boundary"
        assert seq.offload_joint.boundary_hash == 4242, "the boundary still stands"

    def test_a_boundary_with_no_state_behind_it_is_disowned(self):
        """The backstop. Whatever the gate decided, a request may not resume
        over a prefix nothing put state behind."""
        bm, _index = self._bm()  # the tier has nothing
        seq = self._joint_seq(bm, range(48), boundary_hash_value=4242, claim_tokens=0)
        before = bm.state_gate_lost_boundary
        bm.allocate(seq, 0)

        assert seq.offload_joint.boundary_hash == -1
        assert seq.offload_joint.boundary_tokens == 0
        assert seq.num_cached_tokens == 0
        assert bm.state_gate_lost_boundary > before

    def test_a_cold_start_without_a_boundary_is_untouched(self):
        """The gate must not fire on the ordinary path: no boundary claimed,
        so a cold slot is exactly right."""
        bm, _index = self._bm()
        seq = stateful_seq(list(range(48)))
        before = bm.state_gate_lost_boundary
        bm.allocate(seq, 0)

        assert seq.state_slot >= 0
        assert bm.state_gate_lost_boundary == before
