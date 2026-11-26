# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Copy state a forward rebuilds within `rebuild_tokens` and holds for
`trailing_slack` tokens past a boundary (a sliding-window ring)."""

from types import SimpleNamespace

import pytest
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_SPEC,
    ckpt_config,
    make_block_manager,
    stateful_seq,
)

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_runtime import (
    FORK,
    StateRuntime,
    StateTransfer,
)

# 40-token prompts on 4-token blocks: a hit at block b leaves 40 - 4b tokens.
REBUILD = 12
SLACK = 8


def _window_bm(*, rebuild=REBUILD, slack=SLACK, interval=-1):
    runtime = StateRuntime(
        transfer=StateTransfer.copy(
            PAGED_COPY_SPEC.layout_id,
            rebuild_tokens=rebuild,
            trailing_slack=slack,
        ),
        checkpoint_spec=PAGED_COPY_SPEC,
    )
    return make_block_manager(
        ckpt_config(
            num_kvcache_blocks=160,
            pool_entries={"state": 8},
            state_checkpoint_interval_tokens=interval,
        ),
        state_runtime=runtime,
    )


def _seq(n=40):
    return stateful_seq(list(range(100, 100 + n)))


class TestTransfer:
    def test_the_wire_carries_rebuild_and_slack(self):
        transfer = StateTransfer.copy("w", rebuild_tokens=12, trailing_slack=8)
        assert StateTransfer.from_wire(transfer.to_wire()) == transfer

    @pytest.mark.parametrize("field", ["rebuild_tokens", "trailing_slack"])
    def test_only_copy_state_rebuilds_or_trails(self, field):
        with pytest.raises(ValueError, match="only a copy transfer"):
            StateTransfer(FORK, 8, **{field: 4})


class TestResumableHit:
    def test_a_hit_a_rebuild_back_needs_no_image(self):
        bm = _window_bm()
        seq = _seq()
        hashes = list(range(1, 11))
        # 40 - 4 * 7 = 12 >= REBUILD; block 8 leaves 8 and needs an image.
        assert bm.paged_state_checkpoints.resumable_hit(seq, 10, hashes) == 7

    def test_an_image_inside_the_window_is_taken(self, monkeypatch):
        bm = _window_bm()
        seq = _seq()
        hashes = list(range(1, 11))
        coordinator = bm.paged_state_checkpoints
        monkeypatch.setattr(coordinator, "_reachable", lambda h: h == 9)
        assert coordinator.resumable_hit(seq, 10, hashes) == 9

    def test_without_rebuild_tokens_only_images_count(self):
        bm = _window_bm(rebuild=0)
        assert (
            bm.paged_state_checkpoints.resumable_hit(_seq(), 10, list(range(10))) == 0
        )


class TestResumeFresh:
    def test_a_rebuild_boundary_resumes_from_a_fresh_slot(self):
        bm = _window_bm()
        seq = _seq()
        assert bm._state_resume_hash(seq, -1, 0, 55, 7 * BLOCK) == -1
        assert bm._state_resume_hash(seq, -1, 0, 55, 8 * BLOCK) == 55

    def test_a_joint_rebuild_boundary_is_fresh_and_secured(self):
        bm = _window_bm()
        seq = _seq()
        assert bm._state_resume_hash(seq, 77, 7 * BLOCK, 55, 0) == -1
        assert bm._state_resume_hash(seq, 77, 8 * BLOCK, 55, 0) == 77
        seq.offload_joint.boundary_tokens = 7 * BLOCK
        assert bm._state_leg_secured(seq)

    def test_admitting_a_rebuild_hit_restores_nothing(self):
        bm = _window_bm()
        first = _seq()
        bm.allocate(first, bm.can_allocate(first))
        bm.hash_blocks(first, first.num_prompt_tokens)
        bm.deallocate(first)
        # Same 40-token prompt plus 20 new tokens: the 40-token hit is 20 back.
        second = stateful_seq(list(range(100, 160)))
        hit = bm.can_allocate(second)
        assert hit == 10
        bm.allocate(second, hit)
        assert second.num_cached_tokens == 40
        assert not bm.paged_state_checkpoints.restore_queued_for(second.state_slot)


class TestTrailingAnchor:
    def test_a_chunk_ending_inside_the_slack_is_not_cut(self):
        bm = _window_bm()
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        anchor = seq.checkpoint_end_pos
        assert 0 < anchor < seq.num_prompt_tokens <= anchor + SLACK
        assert bm.checkpoint_cut(seq, 0, seq.num_prompt_tokens) == 0

    def test_a_chunk_ending_past_the_slack_is_cut_at_the_anchor(self):
        bm = _window_bm(slack=1)
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        anchor = seq.checkpoint_end_pos
        assert bm.checkpoint_cut(seq, 0, seq.num_prompt_tokens) == anchor

    def test_the_forward_past_the_anchor_files_it_under_the_anchor_hash(self):
        bm = _window_bm()
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        anchor = seq.checkpoint_end_pos
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        anchor_hash = bm._chain_to(seq, [], anchor // BLOCK)[-1]
        pending = bm.paged_state_checkpoints._pending
        assert list(pending) == [(id(seq), anchor_hash)]
        assert seq.last_checkpoint_pos == anchor

    def test_without_slack_an_overshooting_forward_files_nothing(self):
        bm = _window_bm(slack=0)
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        assert not bm.paged_state_checkpoints._pending


class TestCopiedAfterTheForward:
    """A scheduled forward that runs past an anchor copies it right after."""

    def _scheduled(self, bm):
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        bm.checkpoint_after_forward(seq, seq.num_prompt_tokens)
        return seq, bm._chain_to(seq, [], seq.checkpoint_end_pos // BLOCK)[-1]

    def test_the_batch_carrying_the_forward_carries_the_copy(self):
        bm = _window_bm()
        seq, _ = self._scheduled(bm)
        ops = bm.take_state_maintenance_ops()
        assert [op.src_slot for op in ops.checkpoint_stores_after] == [seq.state_slot]
        assert not ops.checkpoint_stores and not ops.empty
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        assert not bm.paged_state_checkpoints._pending, "filed once"

    def test_the_image_is_ready_at_the_next_schedule_with_no_batch_between(self):
        bm = _window_bm()
        seq, h = self._scheduled(bm)
        bm.take_state_maintenance_ops()
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        bm.deallocate(seq)  # a P/D producer's request, freed after its transfer
        bm.complete_previous_state_batch()
        assert bm.paged_state_checkpoints.store.contains(h)

    def test_without_slack_nothing_is_filed(self):
        bm = _window_bm(slack=0)
        self._scheduled(bm)
        assert not bm.take_state_maintenance_ops().checkpoint_stores_after


class TestPipelineStages:
    """PP stages holding none of a rebuildable state join with no bytes."""

    @pytest.fixture(scope="class")
    def runner_module(self):
        from aiter_stub import atom_without_aiter

        with atom_without_aiter() as load:
            yield load("atom.model_engine.model_runner")

    def test_they_take_the_stateful_stages_units(self, runner_module, monkeypatch):
        image = ("w", 9000, 4096, 3)
        monkeypatch.setattr(
            runner_module,
            "get_pp_group",
            lambda: SimpleNamespace(world_size=4, cpu_group=None),
        )

        def gather(out, obj, group=None):
            out[:] = [None, None, None, image]

        monkeypatch.setattr(
            runner_module.torch.distributed, "all_gather_object", gather
        )
        monkeypatch.setattr(runner_module, "published_page_unit_bytes", lambda _: 4096)
        runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
        runner.attn_metadata_builder = object()
        transfer = StateTransfer.copy("w", rebuild_tokens=REBUILD, trailing_slack=SLACK)

        spec = runner._agree_pipeline_checkpoint_spec(None, transfer)

        assert spec.slot_bytes == spec.image_bytes == 0
        assert spec.units_per_checkpoint == 3 and spec.op_bytes == 9000

    def test_only_a_joint_image_may_hold_no_bytes(self):
        spec = PagedStateCheckpointSpec(
            page_unit_bytes=4096,
            slot_bytes=0,
            image_bytes=0,
            layout_id="joint",
            pipeline_units=3,
            pipeline_op_bytes=9000,
        )
        assert spec.units_per_checkpoint == 3 and spec.op_bytes == 9000
        with pytest.raises(ValueError):
            PagedStateCheckpointSpec(4096, 0, 0, "alone")


def test_a_rebuildable_state_keeps_no_interval_rungs():
    assert _window_bm(interval=BLOCK).state_checkpoint_interval_tokens == -1
    assert _window_bm(interval=0).state_checkpoint_interval_tokens == 0
    assert (
        _window_bm(rebuild=0, interval=BLOCK).state_checkpoint_interval_tokens == BLOCK
    )


class TestFinishedState:
    """A finished request's slot is kept as its last hashed block's image."""

    def _finished(self, bm, generated=10):
        seq = _seq()
        bm.allocate(seq, bm.can_allocate(seq))
        bm.hash_blocks(seq, seq.num_prompt_tokens)
        coordinator = bm.paged_state_checkpoints
        coordinator.take_checkpoint_ops()  # the anchor's store rides a batch
        coordinator.complete_previous_batch()
        for token in range(generated):
            seq.append_token(7 + token)
            bm.may_append(seq)
        bm.hash_decode_blocks(
            seq, seq.num_tokens - 1, next_forward_tokens=0, snapshot_step=0
        )
        return seq

    def test_the_slot_is_stored_then_handed_back(self):
        bm = _window_bm()
        seq = self._finished(bm)
        slot, coordinator = seq.state_slot, bm.paged_state_checkpoints
        boundary = seq.num_hashed_tokens // BLOCK
        free = bm.state.num_free()
        bm.keep_finished_state(seq)
        bm.deallocate(seq)
        assert bm.state.num_free() == free, "the kept slot is not freed"
        stores, _ = coordinator.take_checkpoint_ops()
        assert [op.src_slot for op in stores] == [slot]
        h = bm._chain_to(seq, [], boundary)[-1]
        assert coordinator.store.contains_or_pending(h)
        coordinator.complete_previous_batch()
        assert bm.state.num_free() == free + 1

    def test_nothing_is_kept_without_slack_or_past_it(self):
        bm = _window_bm(slack=0)
        seq = self._finished(bm)
        bm.keep_finished_state(seq)
        assert not bm.paged_state_checkpoints._scratch_queued
        bm = _window_bm(slack=1)
        seq = self._finished(bm, generated=10)
        bm.keep_finished_state(seq)
        assert not bm.paged_state_checkpoints._scratch_queued

    def test_a_pd_request_keeps_nothing(self):
        from atom.model_engine.scheduler import Scheduler

        kept = []
        scheduler = SimpleNamespace(
            block_manager=SimpleNamespace(keep_finished_state=kept.append)
        )
        pd = SimpleNamespace(kv_transfer_params={"transfer_id": "t-1"})
        local = SimpleNamespace(kv_transfer_params=None)
        Scheduler._keep_finished_state(scheduler, pd)
        Scheduler._keep_finished_state(scheduler, local)
        assert kept == [local]
