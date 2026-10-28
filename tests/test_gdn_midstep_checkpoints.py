# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Midstep state checkpoints on a grid coarser than the hash block, and on PAGE.

`TestMidstepCheckpoints` in `test_state_checkpoint.py` runs the mechanism with
a grid of one hash block, where every block boundary inside a forward is
readable. GDN's is not: its chunk kernel leaves a state every 64 tokens of the
*forward*, against a 16-token hash block. These tests pin what that changes --
which positions survive, where they move to, and that the scheduler and the
runner agree on both -- and the PAGE coordinator's scratch-slot lifecycle that
Qwen3.5 runs on.
"""

import random
from types import SimpleNamespace

import pytest
from test_state_checkpoint import (
    BLOCK,
    MIN_FORK,
    PAGED_COPY_SPEC,
    PROMPT,
    demand_config,
    make_block_manager,
    stateful_seq,
)

from atom.model_engine.scheduler import ScheduledBatch
from atom.model_engine.state_runtime import (
    StateRuntime,
    StateTransfer,
    midstep_checkpoint_targets,
    midstep_target_ok,
    snap_midstep_position,
)

#: Four hash blocks, as GDN's 64 is four of its 16-token blocks.
GRID = 4 * BLOCK
#: GDN's `conv_kernel - 1`.
MIN_OFFSET = 3

FORK_RUNTIME = StateRuntime(
    transfer=StateTransfer.fork(
        MIN_FORK, readable_midstep=True, midstep_grid=GRID, midstep_min_offset=3
    )
)
PAGED_RUNTIME = StateRuntime(
    transfer=StateTransfer.copy(
        PAGED_COPY_SPEC.layout_id,
        readable_midstep=True,
        midstep_grid=GRID,
        midstep_min_offset=MIN_OFFSET,
    ),
    checkpoint_spec=PAGED_COPY_SPEC,
)


def admitted(bm, tokens=PROMPT):
    seq = stateful_seq(list(tokens))
    bm.allocate(seq, bm.can_allocate(seq))
    return seq


# ── the shared predicate ─────────────────────────────────────────────────────


class TestMidstepGeometry:
    def test_interior_positions_must_sit_on_the_forward_s_own_grid(self):
        # A forward from 16: its states are at 16 + 64k, not at 64k.
        assert midstep_target_ok(80, 16, 200, 64, 3)
        assert not midstep_target_ok(64, 16, 200, 64, 3)
        assert not midstep_target_ok(128, 16, 200, 64, 3)

    def test_the_end_is_always_readable(self):
        assert midstep_target_ok(197, 16, 197, 64, 3)

    def test_the_conv_window_needs_this_forward_s_tokens(self):
        assert not midstep_target_ok(18, 16, 18, 64, 3)
        assert midstep_target_ok(19, 16, 19, 64, 3)

    def test_outside_the_forward_is_never_readable(self):
        assert not midstep_target_ok(16, 16, 200, 64, 3)
        assert not midstep_target_ok(208, 16, 200, 64, 3)

    def test_snapping_goes_back_never_forward(self):
        assert snap_midstep_position(7504, 0, 7513, 64) == 7488
        assert snap_midstep_position(7488, 0, 7513, 64) == 7488
        assert snap_midstep_position(7513, 0, 7513, 64) == 7513
        assert snap_midstep_position(8200, 16, 9000, 64) == 8144

    def test_snapped_positions_are_always_writable_and_findable(self):
        """Property: whatever survives is on the forward's grid and a block edge.

        `midstep_positions` keeps a snapped position only if it passes both;
        this pins that the two tests together admit nothing the kernel cannot
        read, for any forward that starts on a block boundary.
        """
        rng = random.Random(0)
        hbs, grid = 16, 64
        for _ in range(20000):
            start = rng.randrange(0, 4096) * hbs
            end = start + rng.randrange(1, 20000)
            pos = rng.randrange(start + 1, end + 1) // hbs * hbs
            if pos <= start:
                continue
            snapped = snap_midstep_position(pos, start, end, grid)
            if snapped % hbs or not midstep_target_ok(snapped, start, end, grid, 3):
                continue
            assert (snapped - start) % grid == 0 or snapped == end
            assert snapped % hbs == 0
            assert start < snapped <= pos
            assert pos - snapped < grid


# ── the runner's targets ─────────────────────────────────────────────────────


class TestMidstepTargets:
    def test_rows_follow_the_batch_not_the_state_list(self):
        """Row 0 holds no state; the two prefills behind it are rows 1 and 2.

        `saves` and `runtime_slots` are per state-holding seq, the token
        counts per batch row. Indexing both with one `i` read row 0's counts
        for the first state seq and pointed the kernel at the wrong
        `cu_seqlens` entry.
        """
        cached = [0, 16, 32]
        sched = [10, 200, 100]
        targets = midstep_checkpoint_targets(
            state_rows=[1, 2],
            num_cached_tokens=cached,
            num_scheduled_tokens=sched,
            runtime_slots=[5, 6],
            saves=[[(9, 80), (10, 216)], [(11, 132)]],
            num_slots=16,
            grid=64,
            min_offset=3,
        )
        assert targets == [
            (1, 9, 64, 0, 5),
            (1, 10, 200, 1, 5),
            (2, 11, 100, 1, 6),
        ]

    @pytest.mark.parametrize(
        "save, why",
        [
            ((99, 80), "slot outside the pool"),
            ((9, 64), "interior position off the forward's grid"),
            ((9, 17), "conv window reaches before the forward"),
            ((9, 300), "past the forward's end"),
        ],
    )
    def test_an_unwritable_reservation_raises(self, save, why):
        del why  # names the case in the parametrize list
        with pytest.raises(RuntimeError):
            midstep_checkpoint_targets([0], [16], [200], [5], [[save]], 16, 64, 3)

    def test_misaligned_lists_raise(self):
        with pytest.raises(RuntimeError):
            midstep_checkpoint_targets([0, 1], [0, 0], [64, 64], [5], [[]], 16, 64, 3)

    def test_scheduled_batch_carries_the_rows(self):
        stateless = stateful_seq(list(range(8)))
        stateless.has_per_req_cache = False
        a, b = stateful_seq(list(range(8))), stateful_seq(list(range(8)))
        a.state_slots, b.state_slots = [3], [4]
        a.midstep_reservations = [(7, 8, 0)]
        batch = ScheduledBatch(
            seqs={0: stateless, 1: a, 2: b},
            num_scheduled_tokens=[8, 8, 8],
            total_tokens_num=24,
            num_cached_tokens=[0, 0, 0],
        )
        assert batch.state_rows == [1, 2]
        assert batch.state_slots_committed == [3, 4]
        assert batch.state_save_all == [[(7, 8)], []]


# ── the scheduler's positions ────────────────────────────────────────────────


class TestMidstepGrid:
    def bm(self, runtime=FORK_RUNTIME, **overrides):
        return make_block_manager(demand_config(**overrides), state_runtime=runtime)

    def test_off_grid_positions_move_back_onto_it(self):
        """Rung 32 is on the grid; the anchor at 36 is not and joins it."""
        bm = self.bm()
        seq = admitted(bm)
        assert seq.checkpoint_end_pos == 36
        assert [p for p, _h in bm.midstep_positions(seq, 0, 44)] == [32]

    def test_the_anchor_keeps_its_role_after_moving(self):
        """Filed where it moved to, and ranked as the anchor there.

        `publish_midstep` demotes every position that is not the anchor. Had it
        compared against the unsnapped 36, the anchor it actually filed at 32
        would go to the LRU head as a guess.
        """
        bm = self.bm()
        seq = admitted(bm)
        bm.plan_midstep(seq, 0, 44)
        assert seq.midstep_anchor_pos == 32
        bm.hash_blocks(seq, 44)
        assert not bm.state._speculative
        assert seq.midstep_anchor_pos == 0

    def test_a_forward_from_a_hit_reads_on_its_own_grid(self):
        """From 4, the states are at 20, 36 -- not at 16, 32."""
        bm = self.bm(state_checkpoint_interval_tokens=-1)
        seq = admitted(bm)
        seq.checkpoint_end_pos = 40
        assert [p for p, _h in bm.midstep_positions(seq, 4, 44)] == [36]

    def test_the_end_of_the_forward_is_kept_exactly(self):
        bm = self.bm()
        seq = admitted(bm)
        assert [p for p, _h in bm.midstep_positions(seq, 0, 36)] == [32, 36]

    def test_a_forward_too_short_for_the_conv_window_keeps_nothing(self):
        bm = self.bm(state_checkpoint_interval_tokens=-1)
        seq = admitted(bm)
        seq.checkpoint_end_pos = 36
        assert bm.midstep_positions(seq, 34, 36) == []
        assert bm.checkpoint_funnel()["midstep_positions_unreadable"] == 1

    def test_the_reuse_survives_the_move(self):
        """The second request resumes at 32 instead of 36: one block more work."""
        bm = self.bm()
        first = admitted(bm)
        bm.plan_midstep(first, 0, 44)
        bm.hash_blocks(first, 44)
        assert bm.can_allocate(stateful_seq(PROMPT)) == 32 // BLOCK


class TestAlignMidstepChunk:
    def bm(self):
        return make_block_manager(demand_config(), state_runtime=FORK_RUNTIME)

    def test_a_budget_cut_chunk_ends_on_a_block(self):
        bm = self.bm()
        seq = admitted(bm)
        assert bm.align_midstep_chunk(seq, 0, 23) == 20
        assert bm.align_midstep_chunk(seq, 4, 19) == 16

    def test_the_final_chunk_is_left_alone(self):
        bm = self.bm()
        seq = admitted(bm)
        assert bm.align_midstep_chunk(seq, 20, 24) == 24

    def test_a_chunk_shorter_than_the_next_boundary_is_left_alone(self):
        bm = self.bm()
        seq = admitted(bm)
        assert bm.align_midstep_chunk(seq, 20, 3) == 3

    def test_an_unreadable_backend_is_left_alone(self):
        bm = make_block_manager(demand_config())
        seq = admitted(bm)
        assert bm.align_midstep_chunk(seq, 0, 23) == 23


# ── PAGE: the path Qwen3.5 runs on ───────────────────────────────────────────


class TestPagedMidstep:
    """Reserve a scratch slot, write it in the forward, store it a batch later.

    The scheduler pass after a forward is where an ordinary PAGE boundary is
    begun; a midstep one is begun there too. The pass after that is where
    `complete_previous_state_batch` turns it READY and hands the scratch back.
    """

    def bm(self, **overrides):
        overrides.setdefault("pool_entries", {"state": 8})
        return make_block_manager(
            demand_config(**overrides), state_runtime=PAGED_RUNTIME
        )

    def test_the_coordinator_is_the_midstep_class(self):
        bm = self.bm()
        assert bm.paged_state_checkpoints.readable_midstep
        # The cut and the keep both defer to the midstep path.
        seq = admitted(bm)
        assert bm.checkpoint_cut(seq, 0, 44) == 0
        assert bm.checkpointers_at(seq, 32) == []

    def test_a_reservation_borrows_a_scratch_slot(self):
        bm = self.bm()
        seq = admitted(bm)
        free = bm.state.num_free()
        # PAGE needs no successor room, so the anchor is the last hashable
        # block, 40. A forward over the whole prompt snaps it to 32, onto the
        # rung; one that ends at 40 reads it exactly.
        bm.plan_midstep(seq, 0, 44)
        assert [p for _s, p, _h in seq.midstep_reservations] == [32]
        bm.plan_midstep(seq, 0, 40)
        positions = [p for _s, p, _h in seq.midstep_reservations]
        assert positions == [32, 40]
        assert bm.state.num_free() == free - 2
        assert seq.state_slot not in [s for s, _p, _h in seq.midstep_reservations]

    def test_the_image_is_not_findable_before_its_copy_has_run(self):
        bm = self.bm()
        coord = bm.paged_state_checkpoints
        seq = admitted(bm)
        free = bm.state.num_free()
        bm.plan_midstep(seq, 0, 40)
        scratch = [s for s, _p, _h in seq.midstep_reservations]
        h32 = seq.block_hashes[32 // BLOCK - 1]

        bm.hash_blocks(seq, 40)  # postprocess of the forward that wrote them
        assert not coord.store.contains(h32)

        bm.complete_previous_state_batch()  # next pass: nothing issued yet
        assert not coord.store.contains(h32)
        ops = bm.take_state_maintenance_ops()
        assert [op.src_slot for op in ops.checkpoint_stores] == scratch
        assert not coord.store.contains(h32)
        assert bm.state.num_free() == free - 2  # still held: copy not issued

        bm.complete_previous_state_batch()  # the pass after: copies issued
        assert coord.store.contains(h32)
        assert bm.state.num_free() == free

    def test_the_reuse_is_there_without_the_cuts(self):
        bm = self.bm()
        seq = admitted(bm)
        bm.plan_midstep(seq, 0, 44)
        bm.hash_blocks(seq, 44)
        bm.complete_previous_state_batch()
        bm.take_state_maintenance_ops()
        bm.complete_previous_state_batch()
        # The anchor at 40 was read at 32, so the resume starts there.
        assert bm.can_allocate(stateful_seq(PROMPT)) == 32 // BLOCK

    def test_a_cancelled_forward_hands_its_scratch_back_vacant(self):
        bm = self.bm()
        seq = admitted(bm)
        free = bm.state.num_free()
        bm.plan_midstep(seq, 0, 44)
        bm.deallocate(seq)
        assert bm.state.num_free() == free + 1  # the seq's own slot as well
        assert bm.take_state_maintenance_ops().checkpoint_stores == ()

    def test_a_stored_hash_is_not_reserved_again(self):
        bm = self.bm()
        first = admitted(bm)
        bm.plan_midstep(first, 0, 44)
        bm.hash_blocks(first, 44)
        second = admitted(bm)
        bm.plan_midstep(second, 40, 44)
        # Only 40 < p <= 44 is covered, and 40's image is already queued.
        assert second.midstep_reservations == []

    def test_no_scratch_slot_drops_the_checkpoint_not_the_request(self):
        bm = self.bm(pool_entries={"state": 2})
        seq = admitted(bm)
        bm.plan_midstep(seq, 0, 40)
        assert [p for _s, p, _h in seq.midstep_reservations] == [32]
        assert bm.paged_state_checkpoints.checkpoints_dropped == 1

    def test_stores_beyond_the_copy_budget_wait_for_the_next_batch(self):
        """The runner stages two copies per sequence; one forward can publish
        more than that, so the rest ride the next batch, scratch still held."""
        bm = self.bm(max_num_seqs=1)
        coord = bm.paged_state_checkpoints
        seqs = [admitted(bm), admitted(bm, [t + 1000 for t in PROMPT])]
        for seq in seqs:
            bm.plan_midstep(seq, 0, 40)
            assert len(seq.midstep_reservations) == 2
            bm.hash_blocks(seq, 40)
        first = bm.take_state_maintenance_ops().checkpoint_stores
        assert len(first) == 2
        bm.complete_previous_state_batch()
        second = bm.take_state_maintenance_ops().checkpoint_stores
        assert len(second) == 2
        assert {op.src_slot for op in first}.isdisjoint(op.src_slot for op in second)
        assert coord.checkpoints_kept == 4

    def test_the_copy_budget_is_one_definition(self, monkeypatch):
        """The coordinator's per-batch budget is the runner's staging capacity,
        read from one function, so the two cannot drift apart."""
        from atom.model_engine import block_manager, page_unit_checkpoint

        assert page_unit_checkpoint.checkpoint_copy_ops_per_batch(3) == 6
        coord = self.bm(max_num_seqs=3).paged_state_checkpoints
        assert coord._max_ops_per_batch == 6
        monkeypatch.setattr(
            block_manager, "checkpoint_copy_ops_per_batch", lambda n: 7 * n
        )
        coord = self.bm(max_num_seqs=3).paged_state_checkpoints
        assert coord._max_ops_per_batch == 21

    def test_the_runner_stages_the_same_budget(self, monkeypatch):
        backends = pytest.importorskip(
            "atom.model_ops.attentions.backends", reason="needs aiter and torch"
        )
        sized = []
        monkeypatch.setattr(
            backends, "DescriptorStaging", lambda n, device: sized.append(n)
        )
        monkeypatch.setattr(backends, "checkpoint_copy_ops_per_batch", lambda n: 7 * n)
        builder = SimpleNamespace(
            _checkpoint_copy_plan=lambda: SimpleNamespace(num_spans=2),
            _checkpoint_descriptor_device=lambda: "cpu",
            model_runner=SimpleNamespace(config=SimpleNamespace(max_num_seqs=3)),
        )
        backends.AttentionMetadataBuilder._checkpoint_staging(builder)
        assert sized == [7 * 3 * 2]  # budget x spans per copy
