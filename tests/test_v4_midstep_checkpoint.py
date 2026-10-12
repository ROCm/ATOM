# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Midstep state checkpoints for PAGE-copy (DeepSeek-V4) state.

A readable PAGE class takes prompt checkpoints inside an uncut prefill forward:
each one gets a scratch Active Slot the runner fills, and the image is stored
out of that slot. The contract checked here is that this changes *when* the
checkpoints are taken, never *which* ones exist -- a later request resumes
exactly where it would have after the chunk was cut onto every rung.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig  # noqa: F401  (fixture side effects)

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.sequence import Sequence
from atom.model_engine.state_pool import StateSlotPool
from atom.model_engine.state_runtime import StateRuntime, StateTransfer
from atom.model_ops.v4_kernels.midstep import build_midstep_plan
from test_state_checkpoint import (
    BLOCK,
    PAGED_COPY_RUNTIME,
    PAGED_COPY_SPEC,
    ckpt_config,
)

READABLE_COPY_RUNTIME = StateRuntime(
    transfer=StateTransfer.copy(PAGED_COPY_SPEC.layout_id, readable_midstep=True),
    checkpoint_spec=PAGED_COPY_SPEC,
)
PROMPT = list(range(44))  # 11 blocks of 4


# ── plan bookkeeping (host) ──────────────────────────────────────────────


def test_plan_replays_each_ring_window_up_to_its_checkpoint():
    # Row 0: chunk (100, 110], checkpoint at 105. Row 1: chunk (40, 60],
    # checkpoints at 52 and 60 (the chunk end is a legal position too).
    plan = build_midstep_plan(
        saves=[[(7, 105)], [(9, 52), (11, 60)]],
        chunk_starts=[100, 40],
        cu_seqlens_q=np.array([0, 10, 30]),
        src_slots=[3, 4],
        ratios_overlap=[(4, True), (128, False)],
        swa_gather_rows=6,
    )
    assert plan.dst_slots.tolist() == [7, 9, 11]
    assert plan.src_slots.tolist() == [3, 4, 4]
    # SWA: the last min(pos - start, 6) rows before each checkpoint.
    assert plan.swa_cu.tolist() == [0, 5, 11, 17]
    assert plan.swa_rows.tolist() == (
        [0, 1, 2, 3, 4] + list(range(16, 22)) + list(range(24, 30))
    )
    # CSA (ratio 4, overlap -> K_pool 8): positions [max(start, pos-8), pos).
    csa = plan.compress_plans[4]
    assert csa[csa[:, 1] == 0][:, 2].tolist() == list(range(100, 105))
    assert csa[csa[:, 1] == 1][:, 2].tolist() == list(range(44, 52))
    assert csa[csa[:, 1] == 2][:, 2].tolist() == list(range(52, 60))
    # ragged id is the flat token row of that position in this forward.
    assert csa[csa[:, 1] == 1][:, 0].tolist() == list(range(14, 22))
    # HCA (ratio 128, no overlap): the whole chunk prefix before each position.
    hca = plan.compress_plans[128]
    assert hca[hca[:, 1] == 2][:, 2].tolist() == list(range(40, 60))


def test_plan_is_none_without_reservations():
    assert (
        build_midstep_plan([[], []], [0, 0], np.array([0, 4, 8]), [0, 1], [(4, True)])
        is None
    )


def test_plan_refuses_a_checkpoint_this_forward_cannot_produce():
    # Publishing it would file a findable image of the wrong state.
    with pytest.raises(ValueError, match="outside this forward's chunk"):
        build_midstep_plan(
            [[(7, 120)]], [100], np.array([0, 10]), [3], [(4, True)]
        )


# ── coordinator: scratch slot lifecycle ──────────────────────────────────


def _coordinator(num_units=40, num_slots=8):
    pool = StateSlotPool(num_slots, hash_block_size=BLOCK)
    c = PagedStateCheckpointCoordinator(
        BlockPool(num_units), PAGED_COPY_SPEC, enabled=True, readable_midstep=True
    )
    c.bind_slot_pool(pool)
    return c, pool


SEQ = SimpleNamespace(has_per_req_cache=True, state_slot=0)


def test_published_midstep_is_stored_from_its_scratch_slot_and_then_freed():
    c, pool = _coordinator()
    res = c.reserve_midstep(SEQ, [(8, 101), (16, 102)])
    scratch = [slot for slot, _p, _h in res]
    assert len(set(scratch)) == 2 and not any(pool.is_free(g) for g in scratch)

    c.publish_midstep(res, SEQ)
    stores, _ = c.take_checkpoint_ops()
    assert sorted(op.src_slot for op in stores) == sorted(scratch)
    # Read by the store this batch issues: held for exactly that batch.
    assert all(pool.is_pinned(g) for g in scratch)
    c.complete_previous_batch()
    pool.release_pins()
    assert all(pool.is_free(g) for g in scratch)
    assert c.contains(101) and c.contains(102)
    assert c.checkpoints_kept == 2


def test_reserve_skips_a_boundary_already_stored_or_queued():
    c, _pool = _coordinator()
    res = c.reserve_midstep(SEQ, [(8, 101)])
    c.publish_midstep(res, SEQ)
    assert c.reserve_midstep(SEQ, [(8, 101)]) == []  # queued for storing
    c.take_checkpoint_ops()
    c.complete_previous_batch()
    assert c.reserve_midstep(SEQ, [(8, 101)]) == []  # stored


def test_cancelled_reservation_returns_its_slot_unindexed():
    c, pool = _coordinator()
    res = c.reserve_midstep(SEQ, [(8, 101)])
    c.cancel_midstep(res)
    assert pool.is_free(res[0][0]) and not pool.holds_checkpoint(res[0][0])
    assert c.take_checkpoint_ops() == ((), ())


def test_unstorable_midstep_releases_its_slot_and_counts_dropped():
    c, pool = _coordinator(num_units=2)  # an image needs 3 units
    res = c.reserve_midstep(SEQ, [(8, 101)])
    c.publish_midstep(res, SEQ)
    stores, _ = c.take_checkpoint_ops()
    assert stores == ()
    assert pool.is_free(res[0][0])
    assert c.checkpoints_dropped == 1


def test_reserve_stops_when_the_slot_pool_is_exhausted():
    c, _pool = _coordinator(num_slots=1)
    res = c.reserve_midstep(SEQ, [(8, 101), (16, 102)])
    assert [h for _g, _p, h in res] == [101]
    assert c.checkpoints_dropped == 1


def test_unreadable_coordinator_reserves_nothing():
    pool = StateSlotPool(4, hash_block_size=BLOCK)
    c = PagedStateCheckpointCoordinator(BlockPool(40), PAGED_COPY_SPEC, enabled=True)
    c.bind_slot_pool(pool)
    assert c.reserve_midstep(SEQ, [(8, 101)]) == []


# ── block manager: same checkpoints, no cuts ─────────────────────────────


def _bm(runtime):
    # Enough Active Slots for one scratch slot per rung of a whole prompt.
    return BlockManager(
        ckpt_config(pool_entries={"state": 32}), state_runtime=runtime
    )


def _forward_midstep(bm: BlockManager, seq: Sequence) -> list[int]:
    """One uncut forward over the prompt, checkpoints taken midstep."""
    start = seq.num_cached_tokens
    end = seq.num_prompt_tokens
    assert bm.checkpoint_cut(seq, start, end) == 0
    bm.plan_midstep(seq, start, end)
    positions = [p for _g, p, _h in seq.midstep_reservations]
    bm.hash_blocks(seq, end - start, start_tokens=start)
    seq.num_cached_tokens = end
    return positions


def _forward_cut_draining(bm: BlockManager, seq: Sequence) -> list[int]:
    """The cut path as the scheduler runs it: one drain after every forward."""
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
        _flush(bm)
    return cuts


def _stored(bm: BlockManager) -> set[int]:
    return set(bm.paged_state_checkpoints.store.hash_to_checkpoint)


def _flush(bm: BlockManager) -> None:
    bm.take_state_maintenance_ops()
    bm.complete_previous_state_batch()


def _resume_hit(bm: BlockManager, tokens: list[int]) -> int:
    seq = Sequence(tokens, BLOCK, has_per_req_cache=True)
    return bm.can_allocate(seq, record=False)


def test_midstep_keeps_the_same_resume_points_without_cutting():
    # Cut path: the ladder shortens the chunk onto each rung it keeps.
    cut_bm = _bm(PAGED_COPY_RUNTIME)
    cut_seq = Sequence(PROMPT, BLOCK, has_per_req_cache=True)
    cut_bm.allocate(cut_seq, cut_bm.can_allocate(cut_seq))
    cuts = _forward_cut_draining(cut_bm, cut_seq)
    assert cuts, "the cut path must actually cut for this to compare anything"

    # Midstep path: one forward, the same checkpoints reserved inside it.
    ms_bm = _bm(READABLE_COPY_RUNTIME)
    ms_seq = Sequence(PROMPT, BLOCK, has_per_req_cache=True)
    ms_bm.allocate(ms_seq, ms_bm.can_allocate(ms_seq))
    positions = _forward_midstep(ms_bm, ms_seq)
    _flush(ms_bm)
    assert positions, "nothing was reserved"
    assert ms_seq.midstep_reservations == []

    # The same boundaries were checkpointed either way...
    assert _stored(ms_bm) == _stored(cut_bm)
    # ...so the next turn resumes at the same boundary either way.
    continuation = PROMPT + list(range(500, 512))
    assert _resume_hit(ms_bm, continuation) == _resume_hit(cut_bm, continuation) > 0
    # And so does an identical re-send.
    assert _resume_hit(ms_bm, PROMPT) == _resume_hit(cut_bm, PROMPT)


def test_midstep_reserves_the_prompt_end_anchor():
    bm = _bm(READABLE_COPY_RUNTIME)
    seq = Sequence(PROMPT, BLOCK, has_per_req_cache=True)
    bm.allocate(seq, bm.can_allocate(seq))
    anchor = seq.checkpoint_end_pos
    assert anchor > 0
    assert anchor in _forward_midstep(bm, seq)


def _run_turn(bm, tokens, budget, midstep):
    """Admit `tokens` and prefill it in `budget`-token chunks, as the scheduler
    does; the cut path additionally shortens each chunk onto its rung."""
    seq = Sequence(tokens, BLOCK, has_per_req_cache=True)
    bm.allocate(seq, bm.can_allocate(seq))
    while seq.num_cached_tokens < seq.num_prompt_tokens:
        start = seq.num_cached_tokens
        end = min(seq.num_prompt_tokens, start + budget)
        if midstep:
            assert bm.checkpoint_cut(seq, start, end) == 0
            bm.plan_midstep(seq, start, end)
        else:
            end = bm.checkpoint_cut(seq, start, end) or end
        bm.hash_blocks(seq, end - start, start_tokens=start)
        seq.num_cached_tokens = end
        _flush(bm)
    return seq


@pytest.mark.parametrize("budget", [8, 12, 16, 64])
def test_multi_turn_chunked_prefill_matches_the_cut_path(budget):
    turns = [PROMPT, PROMPT + list(range(500, 520)), PROMPT + list(range(500, 540))]
    cut_bm, ms_bm = _bm(PAGED_COPY_RUNTIME), _bm(READABLE_COPY_RUNTIME)
    for tokens in turns:
        assert _resume_hit(ms_bm, tokens) == _resume_hit(cut_bm, tokens)
        _run_turn(cut_bm, tokens, budget, midstep=False)
        _run_turn(ms_bm, tokens, budget, midstep=True)
        assert _stored(ms_bm) == _stored(cut_bm)
