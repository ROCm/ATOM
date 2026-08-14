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
from unittest.mock import MagicMock

import pytest
import torch
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

    def test_a_speculative_checkpoint_is_spent_before_any_anchor(self):
        """A guess must never evict knowledge, however old the knowledge is.

        Group 0 holds an anchor released first, so plain LRU would spend it.
        Group 1 is marked speculative and lands at the head instead, which is
        what makes the demand rung cost the anchors nothing.

        Indexed before it is released, which is the order the fork path takes:
        the group is still its owner's when the hash is filed.
        """
        pool = StateGroupPool(4)
        drain(pool)
        pool.release(0)
        pool._index(10, 0)
        pool._index(11, 1)
        pool.mark_speculative(1)
        pool.release(1)

        assert pool.pop() == 1
        assert pool.lookup(10) == 0

    def test_speculative_checkpoints_keep_lru_among_themselves(self):
        pool = StateGroupPool(4)
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

        `BlockManager._attach_state_group` promotes the source it is about to
        fork off, so a demand rung that pays off stops being spent first.
        """
        pool = StateGroupPool(4)
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
        """`_attach_state_group` promotes every source, most of them anchors."""
        pool = StateGroupPool(4)
        drain(pool)
        for group, h in ((0, 10), (1, 11)):
            pool.release(group)
            pool._index(h, group)

        pool.promote(1)

        assert pool.pop() == 0  # still the older of the two

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
# A prompt that diverges from `PROMPT` at token 28, mid-interval and nowhere
# near either prompt's end. This is the traffic the demand is for now that the
# prompt-end anchor exists: on a conversation that just grows, the position the
# next turn resumes at *is* the previous turn's end, and the anchor reserves it
# up front rather than one disappointed request late. What the anchor cannot
# reserve is a branch point, because no prompt ever ended there.
BRANCH = list(range(28)) + list(range(900, 916))


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

    Scoped to branch points since the prompt-end anchor landed. A conversation
    that only grows resumes at the previous turn's end, and the anchor reserves
    that proactively — see `TestPromptEndAnchor`, which inherited the cases
    these tests used to make. What no anchor can reserve is a position no prompt
    ever ended at, and that is what these now use.
    """

    def test_the_gap_becomes_a_rung_off_the_grid(self):
        bm = BlockManager(demand_config())
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
        `CacheStats` splits declined reuse by, and turning the placement off
        must not blind that. What goes is only the placement, leaving the grid
        and this prompt's own anchor to carry the checkpoints.
        """
        bm = BlockManager(demand_config(state_checkpoint_demand=False))
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
        bm = BlockManager(demand_config(state_checkpoint_demand=True))
        assert bm.state_checkpoint_demand is False

        # =1 beats a config that refuses it.
        monkeypatch.setenv("ATOM_STATE_CHECKPOINT_DEMAND", "1")
        bm = BlockManager(demand_config(state_checkpoint_demand=False))
        assert bm.state_checkpoint_demand is True

        # Exported-but-empty is not "set" — the flag still decides.
        monkeypatch.setenv("ATOM_STATE_CHECKPOINT_DEMAND", "")
        bm = BlockManager(demand_config(state_checkpoint_demand=False))
        assert bm.state_checkpoint_demand is False

        monkeypatch.delenv("ATOM_STATE_CHECKPOINT_DEMAND")
        bm = BlockManager(demand_config(state_checkpoint_demand=True))
        assert bm.state_checkpoint_demand is True

    def test_the_third_request_finds_what_the_second_was_missing(self):
        """Self-limiting: nothing to want, want it once, want nothing again."""
        bm = BlockManager(demand_config())

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
        bm = BlockManager(demand_config())
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
        bm = BlockManager(demand_config())
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
        bm = BlockManager(demand_config())
        for n in (12, 40, 44, 45, 50):
            seq = stateful_seq(list(range(1000 * n, 1000 * n + n)))
            bm.can_allocate(seq)
            anchor = seq.checkpoint_end_pos
            assert anchor % BLOCK == 0, n  # on the hash grid
            assert n - anchor >= MIN_FORK, n  # and it leaves the fork its room
            assert bm.checkpointers_at(seq, anchor), n  # so it is really kept

    def test_a_prompt_with_no_room_for_an_anchor_gets_none(self):
        bm = BlockManager(demand_config())
        seq = stateful_seq(list(range(MIN_FORK)))
        bm.can_allocate(seq)
        assert seq.checkpoint_end_pos == 0

    def test_the_cut_and_the_keep_agree_at_every_anchor(self):
        """Swept, because a cut nothing keeps is a forward spent on nothing."""
        for n in range(BLOCK, 80):
            bm = BlockManager(demand_config())
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
        bm = BlockManager(demand_config())
        first = stateful_seq(PROMPT)
        assert run_prompt_on_the_ladder(bm, first) == [32, 36]

        bm.state_caches = (*bm.state_caches, StubStateCache(cap=8))
        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 8  # the rung, still there
        assert second.num_wanted_hit_blocks == 8  # and no gap to demand
        assert second.checkpoint_demand_pos == 0

    def test_a_stateless_model_records_no_anchor(self):
        bm = BlockManager(
            demand_config(
                pool_entries={}, state_transfer_kind="none", state_fork_tokens=0
            )
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        bm.can_allocate(cold)
        assert cold.checkpoint_end_pos == 0

    def test_deallocate_clears_the_anchor(self):
        """Sequences are recycled; a stale anchor would cut the next prompt."""
        bm = BlockManager(demand_config())
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
        bm = BlockManager(demand_config())
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
        bm = BlockManager(ckpt_config(state_checkpoint_interval_tokens=-1))
        assert bm.state_checkpoint_interval_tokens == -1

    def test_the_grid_places_no_rung(self):
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=-1))
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert bm.checkpoint_limit(seq) == 0
        # 32 is a rung under the default interval, and nothing under -1. The
        # anchor at 36 is the only aimed position left.
        assert not bm.checkpointers_at(seq, 32)
        assert bm.checkpointers_at(seq, 36)

    def test_the_anchor_still_reaches_the_same_hit(self):
        """The point of the mode: the ladder's reuse for one cut, not two."""
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=-1))
        first = stateful_seq(PROMPT)
        assert run_prompt_on_the_ladder(bm, first) == [36]  # the ladder cut 32 too

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 9  # what the full ladder also gave
        assert second.checkpoint_demand_pos == 0

    def test_the_demand_still_fires(self):
        """This is what -1 buys over 0, and why it is not spelled 0."""
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=-1))
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
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=0))
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
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=-1))
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
                bm = BlockManager(
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
            bm = BlockManager(demand_config(state_checkpoint_interval_tokens=-1))
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
        bm = BlockManager(demand_config(state_checkpoint_interval_tokens=0))
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert seq.checkpoint_end_pos == 0
        assert run_prompt_on_the_ladder(bm, stateful_seq(PROMPT)) == []


class TestCacheStatsAttribution:
    """Splitting declined reuse into the part a checkpoint reaches and the rest.

    One number for both makes "does demand-driven checkpointing apply to this
    workload" unfalsifiable, which is the whole reason the counterfactual is
    computed outside the tests.
    """

    def test_the_split_accounts_for_every_declined_token(self):
        stats = CacheStats(log_interval=10**6)
        stats.update(32, 48, 40, 36, 44)
        lost_to_checkpoint = stats.total_wanted_tokens - stats.total_cached_tokens
        lost_hard = stats.total_compressed_tokens - stats.total_wanted_tokens
        assert lost_to_checkpoint == 4
        assert lost_hard == 4
        assert lost_to_checkpoint + lost_hard == 40 - 32

    def test_each_pool_is_counted_where_it_cost_the_request_reuse(self):
        """`cached <= wanted <= compressed <= reusable`, one request per case.

        Request-weighted, because the token totals are decided by the largest
        conversations and cannot say whether a loss was broad or concentrated.
        """
        stats = CacheStats(log_interval=10**6)
        # Everything reusable was reused.
        stats.update(44, 48, 44, 44, 44)
        # Paged pool had it; a checkpoint at that boundary would have unlocked
        # it. The state cache's own miss.
        stats.update(32, 48, 44, 40, 44)
        # Paged pool had it, but nothing a checkpoint reaches: wanted == cached.
        stats.update(32, 48, 40, 32, 44)
        # The prefix itself was absent -- state tuning is powerless.
        stats.update(20, 48, 20, 20, 44)

        assert stats.reqs_full_reuse == 1
        assert stats.reqs_state_miss_recoverable == 1
        # Superset: both middle requests had paged prefix the gates declined.
        assert stats.reqs_state_miss == 2
        assert stats.reqs_no_paged == 2  # the absent-prefix one, and case 3

    def test_a_request_losing_at_both_pools_is_counted_at_both(self):
        """The buckets overlap on purpose.

        A prompt whose paged prefix ran out early *and* whose remaining reuse
        needed a checkpoint has two independent problems. Charging it to one
        pool would undercount the other and point sizing at the wrong one, so
        the counters are not required to sum to the request count.
        """
        stats = CacheStats(log_interval=10**6)
        stats.update(20, 100, 60, 40, 90)  # cached < wanted < compressed < reusable
        assert stats.reqs_state_miss_recoverable == 1
        assert stats.reqs_no_paged == 1
        assert stats.reqs_full_reuse == 0
        assert (
            stats.reqs_state_miss + stats.reqs_no_paged > stats.total_requests
        ), "buckets must be free to overlap"

    def test_a_perfect_run_is_reported_as_perfect(self):
        """The regression that motivated `reusable`.

        Against `full`, `reqs_full_reuse` and `reqs_no_paged` were tautologies:
        `can_allocate` never matches the trailing block, so `compressed < full`
        held for every request that could exist and the pair read 0% and 100%
        on all workloads -- including this one, where both caches did
        everything they possibly could.
        """
        stats = CacheStats(log_interval=10**6)
        # 100 tokens, 90 reusable: every reusable token was served by cache.
        stats.update(90, 100, 90, 90, 90)

        assert stats.reqs_full_reuse == 1
        assert stats.reqs_no_paged == 0, "a perfect run must not report a paged miss"
        assert stats.hit_rate == 1.0
        assert stats.paged_hit_rate == 1.0
        assert stats.state_hit_rate == 1.0

    def test_each_pool_is_scored_against_what_it_was_actually_asked_for(self):
        """The two rates must isolate their own pool, and compose exactly.

        The paged pool is asked for `reusable` and supplies `compressed`. The
        state cache never sees what the paged pool already lost, so it is
        scored against `compressed`, not `reusable` -- otherwise a KV eviction
        shows up as a state-cache failure and sends tuning at the wrong pool.
        """
        stats = CacheStats(log_interval=10**6)
        # Of 100 reusable, the paged pool had 80 (80%); of those 80 the state
        # gates admitted 60 (75%). End to end: 60%.
        stats.update(60, 128, 80, 70, 100)

        assert stats.paged_hit_rate == 0.80
        assert stats.state_hit_rate == 0.75
        assert stats.hit_rate == 0.60
        # approx, not ==: the identity is exact over the integer counters, but
        # each rate is a float division first, so the product carries rounding.
        assert stats.paged_hit_rate * stats.state_hit_rate == pytest.approx(
            stats.hit_rate
        )

    def test_a_kv_eviction_does_not_lower_the_state_cache_score(self):
        """Independence, stated as the property that makes the split useful.

        Two runs whose state cache behaves identically -- admitting every
        boundary the paged pool offered -- must score the same on
        `state_hit_rate` however much prefix the paged pool lost.
        """
        healthy = CacheStats(log_interval=10**6)
        healthy.update(100, 128, 100, 100, 100)
        evicted = CacheStats(log_interval=10**6)
        evicted.update(40, 128, 40, 40, 100)  # paged pool lost 60% of the prefix

        assert evicted.paged_hit_rate < healthy.paged_hit_rate
        assert evicted.state_hit_rate == healthy.state_hit_rate == 1.0

    def test_the_recoverable_share_bounds_what_checkpointing_can_buy(self):
        """`state_hit + recoverable` is the ceiling a dense ladder would reach.

        The distance from there to 1.0 is loss no checkpoint touches, and so
        the honest cap on what more groups are worth.
        """
        stats = CacheStats(log_interval=10**6)
        # 80 offered, 50 admitted; a dense ladder would have reached 70.
        stats.update(50, 128, 80, 70, 100)

        assert stats.state_hit_rate == 0.625
        assert stats.state_recoverable_loss_rate == 0.25
        assert stats.state_hit_rate + stats.state_recoverable_loss_rate == 0.875

    def test_the_nesting_invariant_is_enforced_not_assumed(self):
        """Every rate is a difference of two totals, so a violation reports a
        negative percentage rather than failing. Catch it at the source."""
        stats = CacheStats(log_interval=10**6)
        with pytest.raises(AssertionError, match="nesting"):
            stats.update(50, 128, 40, 45, 100)  # cached > compressed
        with pytest.raises(AssertionError, match="ceiling"):
            stats.update(50, 128, 90, 60, 80)  # compressed > reusable

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

    def test_the_reuse_ceiling_matches_the_matcher_that_sets_it(self):
        """The scheduler's ceiling and `can_allocate`'s match loop are the same
        rule written twice, so pin them to each other rather than to a literal.

        Drift here is silent and one-directional: a ceiling above what the
        matcher can reach makes a perfect run look imperfect forever.
        """
        sched = Scheduler(demand_config())
        hbs = sched.block_manager.hash_block_size
        seq = stateful_seq(PROMPT)
        sched._schedule_prefill_seq(seq, 44, {}, [], 0, 0)

        matchable_blocks = sched.block_manager._n_hash_blocks(seq) - 1
        assert sched.cache_stats.total_reusable_tokens == matchable_blocks * hbs
        assert sched.cache_stats.total_reusable_tokens < seq.num_tokens

    def test_a_sequence_below_one_block_has_nothing_to_reuse(self):
        """The `n_hash_blocks - 1` ceiling goes negative for a short prompt.

        Its reuse ceiling is genuinely zero -- the only block it has is the one
        prefill must compute -- and a negative denominator would invert every
        rate on the line.
        """
        stats = CacheStats(log_interval=10**6)
        stats.update(0, 10, 0, 0, 0)
        assert stats.total_reusable_tokens == 0
        assert stats.hit_rate == 0.0
        assert stats.paged_hit_rate == 0.0
        assert stats.reqs_no_paged == 0, "nothing was reusable, so nothing was lost"


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


# ── midstep checkpoints ────────────────────────────────────────────────────


def midstep_config(**overrides):
    """`demand_config` for a backend that reads its state mid-forward."""
    overrides.setdefault("state_readable_midstep", True)
    return demand_config(**overrides)


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
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        # The unreadable backend cuts at 32 and again at 36 for this prompt.
        assert forward_midstep(bm, seq) == [32, 36]
        assert bm.checkpoint_funnel()["chunks_cut_for_end"] == 0
        assert bm.checkpoint_funnel()["chunks_cut_for_demand"] == 0

    def test_the_reuse_is_the_same_reuse(self):
        """The point: same hit as the cutting ladder, without the cuts."""
        for readable in (False, True):
            bm = BlockManager(demand_config(state_readable_midstep=readable))
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
        bm = BlockManager(midstep_config())
        first = stateful_seq(PROMPT)
        bm.allocate(first, bm.can_allocate(first))
        forward_midstep(bm, first)

        assert len(set(bm.state.hash_to_group.values())) == 2

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
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        group = seq.per_req_cache_group
        bm.plan_midstep(seq, 0, 32)
        bm.hash_blocks(seq, 32, start_tokens=0)

        assert bm.checkpoint_funnel()["checkpoints_kept"] == 1
        assert seq.per_req_cache_group == group  # not forked out from under it
        assert seq.state_fork_src == -1

    def test_a_position_the_hash_chain_cannot_name_is_skipped(self):
        """No hash, no way back — so reserving one would spend a group on air."""
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        seq.block_hashes = seq.block_hashes[:2]  # 8 tokens' worth
        assert bm.midstep_positions(seq, 0, 44) == []

    def test_the_chain_covers_the_whole_prompt_past_the_miss(self):
        """`block_hashes` stops at the first miss; the anchor is past it."""
        bm = BlockManager(midstep_config())
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
        bm = BlockManager(demand_config())
        seq = stateful_seq(PROMPT)
        bm.can_allocate(seq)
        assert seq.block_hashes == []

    def test_nothing_is_findable_until_the_forward_has_run(self):
        """Publishing at reservation time indexes bytes nobody wrote."""
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        assert seq.midstep_reservations
        assert bm.state.hash_to_group == {}

        bm.hash_blocks(seq, 44)
        assert len(bm.state.hash_to_group) == 2
        assert seq.midstep_reservations == []  # drained, not left to re-publish

    def test_a_cancelled_reservation_is_returned_vacant(self):
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        free_before = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        assert bm.state.num_free() == free_before - 2

        bm.cancel_midstep(seq)
        assert bm.state.num_free() == free_before
        assert bm.state.hash_to_group == {}  # holding nothing findable

    def test_replanning_returns_the_previous_forward_s_groups(self):
        """A plan is good for one forward; a second means the first never ran."""
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        free_with_one_plan = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        assert bm.state.num_free() == free_with_one_plan

    def test_deallocate_returns_them_too(self):
        """Preemption frees through here, and the forward is not going to run."""
        bm = BlockManager(midstep_config())
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
        bm = BlockManager(midstep_config(pool_entries={"state": 2}))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        bm.plan_midstep(seq, 0, 44)
        assert [p for _g, p, _h in seq.midstep_reservations] == [32]
        assert bm.checkpoint_funnel()["checkpoints_dropped"] == 1

    def test_reservations_never_starve_an_admission(self):
        """`has_free` is the gate, so the worst case is a deferred admission."""
        bm = BlockManager(midstep_config(pool_entries={"state": 3}))
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
        bm = BlockManager(midstep_config())
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.checkpointers_at(seq, INTERVAL + BLOCK, MIN_FORK, aimed=False)

    def test_the_prompt_s_checkpoints_space_the_decode_ones(self):
        """`last_checkpoint_pos` is the decode spacing rule's only input.

        A prompt that filed a midstep checkpoint at its end and left the
        watermark at 0 would let the first decode boundary keep another one
        immediately, which is what the interval exists to prevent.
        """
        bm = BlockManager(midstep_config())
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
        bm = BlockManager(midstep_config())
        bm.state_caches = (*bm.state_caches, StubStateCache(successor_room=0))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.checkpoint_cut(seq, 0, 44) == 32
        assert bm.checkpointers_at(seq, 32)

    def test_interval_zero_reserves_nothing(self):
        """0 is off for the midstep path too, as it is for the other three."""
        bm = BlockManager(midstep_config(state_checkpoint_interval_tokens=0))
        seq = stateful_seq(PROMPT)
        bm.allocate(seq, bm.can_allocate(seq))
        assert bm.midstep_positions(seq, 0, 44) == []

    def test_minus_one_reserves_the_anchor_alone(self):
        """The two changes compose: no grid, no cuts, and the reuse still there."""
        bm = BlockManager(midstep_config(state_checkpoint_interval_tokens=-1))
        first = stateful_seq(PROMPT)
        bm.allocate(first, bm.can_allocate(first))
        assert forward_midstep(bm, first) == [36]

        second = stateful_seq(PROMPT)
        assert bm.can_allocate(second) == 9

    def test_a_stateless_model_reserves_nothing(self):
        bm = BlockManager(
            midstep_config(
                pool_entries={}, state_transfer_kind="none", state_fork_tokens=0
            )
        )
        cold = Sequence(PROMPT, BLOCK, has_per_req_cache=False)
        bm.can_allocate(cold)
        assert cold.block_hashes == []
        assert bm.midstep_positions(cold, 0, 44) == []


#: The two backend modules `gdn_backends` re-imports under the aiter stub.
#: Evicted before the import so the fixture gets real source rather than a
#: copy some earlier test already bound to real aiter.
_UNDER_TEST = (
    "atom.model_ops.attentions.gdn_attn",
    "atom.model_ops.attentions.kimi_mla_gdn_attn",
)


@pytest.fixture
def gdn_backends():
    """`(GDNStateMixin, _KimiMLAGDNCommon)`, importable without a GPU.

    Both modules do `from aiter import ...` at module scope, and importing
    `aiter` anywhere runs its arch probe, which on a CPU-only box raises
    (`0 active drivers`) and then falls back to a `jax` import that is not
    installed. The two dtype/transfer declarations under test are plain Python
    on `atom` classes and need none of that, so `aiter` is stubbed for the
    duration of the import.

    A finder rather than a fixed `sys.modules` list: the transitive set is 19
    submodules today and is aiter's business, not this test's, so enumerating
    it would turn an unrelated aiter refactor into a failure here. Everything
    it fabricates is a MagicMock, so any test that leaned on real aiter
    behaviour through it would be asserting against a mock rather than
    silently passing — but nothing here does: the assertions read
    `_state_dtypes` and `state_transfer`, which touch only `torch` and
    `atom.model_engine.state_pool`.

    Scoped and restored. `tests/test_pp.py` installs its stubs at module
    scope and leaves them, which is invisible when it runs alone and makes
    collection order matter when it does not.
    """
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            mock = MagicMock(name=f"{self.__name__}.{name}")
            setattr(self, name, mock)
            return mock

    class _Loader(importlib.abc.Loader):
        def create_module(self, spec):
            return _Stub(spec.name)

        def exec_module(self, module):
            pass

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "aiter" or fullname.startswith("aiter."):
                return importlib.machinery.ModuleSpec(
                    fullname, _Loader(), is_package=True
                )
            return None

    def _fake(mod):
        """A `sys.modules` entry no import system produced.

        Other test files install bare `ModuleType`s over real `atom` modules
        (`atom.utils.forward_context`, `atom.model_ops.attention_gdn`, ... —
        six of them by the time the suite reaches this file) and never take
        them out. Any of those on the import path below re-raises as
        `ImportError: cannot import name X (unknown location)`, so they get
        evicted here and restored after, exactly like the aiter ones. Keyed on
        the absence of `__file__`/`__path__` rather than on a list of names,
        since the set is other files' business and grows.
        """
        return (
            mod is not None
            and getattr(mod, "__file__", None) is None
            and not hasattr(mod, "__path__")
        )

    finder = _Finder()
    # Drop any real aiter already imported, the atom modules that closed over
    # it, and any atom module some earlier test left stubbed, so the imports
    # below run against real source.
    stale = [
        k
        for k in sys.modules
        if k.split(".")[0] == "aiter"
        or k.startswith(_UNDER_TEST)
        or (k.split(".")[0] == "atom" and _fake(sys.modules[k]))
    ]
    saved = {k: sys.modules[k] for k in stale}
    for k in stale:
        del sys.modules[k]
    # Snapshot after the eviction: anything `atom` that appears between here
    # and the `finally` was imported while aiter was a mock.
    preexisting = {k for k in sys.modules if k.split(".")[0] == "atom"}
    sys.meta_path.insert(0, finder)
    try:
        from atom.model_ops.attentions.gdn_attn import GDNStateMixin
        from atom.model_ops.attentions.kimi_mla_gdn_attn import _KimiMLAGDNCommon

        yield GDNStateMixin, _KimiMLAGDNCommon
    finally:
        sys.meta_path.remove(finder)
        # Everything imported under the stub must go, not just the two entry
        # points: the aiter mocks reached whatever `atom` modules were pulled
        # in transitively, and a later test importing one would get a class
        # holding MagicMocks. Those are real modules with a real `__file__`,
        # so they are identified by the snapshot rather than by `_fake`.
        # Restoring `saved` last puts the other files' stubs back, so this
        # fixture is not observable either way.
        for k in [
            k
            for k in sys.modules
            if k.split(".")[0] == "aiter"
            or (k.split(".")[0] == "atom" and k not in preexisting)
        ]:
            del sys.modules[k]
        sys.modules.update(saved)


class TestMidstepExactnessPremise:
    """The interior-checkpoint path is only lossless while two dtypes agree.

    A checkpoint taken mid-prompt is sliced out of the chunk kernel's `h`,
    which is `k.new_empty(...)` — the activation dtype. The alternative it
    replaced, cutting the prefill so the position became a step end, stores an
    fp32 final state. Those differ, and the only reason the substitution is
    exact is that the destination pool is the activation dtype too, so both
    paths round identically on the way in. Verified on GPU in
    `tests/test_gdn_state_checkpoint_gpu.py`; this pins the premise where CI
    can see it, since that file skips without a device.

    Widen the pool without widening `h` and nothing breaks loudly: the writes
    still land, the shapes still match, and cached requests quietly resume from
    a state carrying bf16 rounding the uncached ones do not. That is an
    accuracy regression visible only as an eval delta between runs that hit the
    cache and runs that miss, which is close to the hardest kind to trace back.
    """

    @staticmethod
    def _dtypes(gdn_cls, model_type):
        """`_state_dtypes` without constructing a builder.

        The method reads only `model_runner.config`, so a namespace standing in
        for the runner is enough and keeps this off the GPU.
        """
        runner = SimpleNamespace(
            config=SimpleNamespace(
                torch_dtype=torch.bfloat16,
                hf_config=SimpleNamespace(model_type=model_type),
            )
        )
        return gdn_cls._state_dtypes(SimpleNamespace(model_runner=runner))

    def test_gdn_pool_matches_the_activation_dtype(self, gdn_backends):
        """Both halves, because a checkpoint writes both."""
        gdn_cls, _ = gdn_backends

        assert self._dtypes(gdn_cls, "qwen3_next") == (torch.bfloat16, torch.bfloat16)

    def test_kimi_is_the_one_pool_wider_than_h(self, gdn_backends):
        """And is excluded from the midstep path — for an unrelated reason.

        KDA is off it because `chunk_kda` never exposes per-chunk states at all
        (`_KimiMLAGDNCommon.state_transfer`). That its pool would also break the
        exactness argument is a second, independent reason, so porting
        `chunk_kda_paged` to lift the first would not be enough on its own.
        """
        gdn_cls, _ = gdn_backends

        assert self._dtypes(gdn_cls, "kimi_linear") == (torch.bfloat16, torch.float32)

    def test_midstep_backends_are_the_ones_with_a_matching_pool(self, gdn_backends):
        """The rule, rather than today's two answers to it.

        A backend that declares `readable_midstep` and allocates a pool wider
        than its activations gets silent per-request accuracy drift. Asserted
        as an invariant over the classes that declare it, so a third GDN-like
        backend added later has to satisfy it rather than merely resemble one.
        """
        gdn_cls, kimi_cls = gdn_backends

        for cls, model_type in (
            (gdn_cls, "qwen3_next"),
            (kimi_cls, "kimi_linear"),
        ):
            midstep = cls.state_transfer(SimpleNamespace()).readable_midstep
            dtypes = self._dtypes(gdn_cls, model_type)
            uniform = set(dtypes) == {torch.bfloat16}
            assert not midstep or uniform, (
                f"{cls.__name__} reads checkpoints out of bf16 `h` but pools "
                f"state at {dtypes}"
            )
