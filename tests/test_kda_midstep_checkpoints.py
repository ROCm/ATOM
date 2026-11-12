# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 midstep KDA checkpoints (ATOM_K3_KDA_STATE_MIDSTEP=1) on #2591's framework.

The framework places the positions and borrows their scratch slots;
test_gdn_midstep_checkpoints.py pins that for any readable backend. What is
Kimi-K3's own lives in `atom.models.kimi_k3.midstep`, which imports no AITER,
so it is pinned here on the CPU:

  * the switch, off by default, and its refusal without FlashKDA;
  * the transfer it declares: readable, on FlashKDA's 32-token grid, at least
    `conv_kernel - 1` tokens into the forward;
  * the targets one prefill step yields -- several per forward, rows that hold
    no state, an end-of-forward position;
  * how a KDA layer writes them: an interior snapshot and its conv window into
    the scratch slot, an end position copied whole from the runtime slot.
"""

import pytest
import torch
from test_state_checkpoint import (
    PROMPT,
    demand_config,
    make_block_manager,
    make_scheduler,
    stateful_seq,
)

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_runtime import StateRuntime, StateTransfer

# From the submodule: `from atom.models.kimi_k3 import midstep` would ask the
# package, which resolves names from `model` and so imports AITER.
from atom.models.kimi_k3.midstep import (
    FLASH_KDA_CHUNK,
    kda_midstep_enabled,
    kda_midstep_transfer,
    kda_snapshots,
    refuse_midstep_under_pipeline,
    require_flash_kda_snapshots,
)
from atom.utils import envs

KDA_SPEC = PagedStateCheckpointSpec(10, 25, "kda-paged-state-v1:test", image_bytes=25)
#: Kimi-K3's `linear_conv_kernel_dim`.
CONV_KERNEL = 4
CONV_LEN = CONV_KERNEL - 1


def window_rows(snaps):
    """The flat conv-input rows each interior target's window reads, as
    `write_state_checkpoints` addresses them: `conv_len` rows ending at
    `cu_seqlens[row] + off`."""
    cu = snaps.cu_seqlens.tolist()
    rows = []
    for row, off, end in zip(
        snaps.rows.tolist(), snaps.offs.tolist(), snaps.is_end.tolist(), strict=True
    ):
        if not end:
            rows.extend(range(cu[row] + off - CONV_LEN, cu[row] + off))
    return rows


GRID = FLASH_KDA_CHUNK
# PROMPT is 44 tokens; a PAGE class needs no successor room, so its prompt-end
# anchor is the last hittable block's end.
ANCHOR = 40


@pytest.fixture(autouse=True)
def _clean_switches(monkeypatch):
    for name in (
        "ATOM_K3_KDA_STATE_MIDSTEP",
        "AITER_FDA_ENABLE",
        "ATOM_STATE_CHECKPOINT_DEMAND",
    ):
        monkeypatch.delenv(name, raising=False)


def k3_runtime(midstep_on: bool) -> StateRuntime:
    """What the Kimi-K3 backend's `state_transfer()` declares, either way."""
    transfer = (
        kda_midstep_transfer(KDA_SPEC.layout_id, CONV_KERNEL)
        if midstep_on
        else StateTransfer.copy(KDA_SPEC.layout_id)
    )
    return StateRuntime(transfer=transfer, checkpoint_spec=KDA_SPEC)


def k3_config(**overrides):
    """The MLPerf recipe's placement: no ladder, no demand rung, the anchor only."""
    overrides.setdefault("state_checkpoint_interval_tokens", -1)
    overrides.setdefault("state_checkpoint_demand", False)
    return demand_config(**overrides)


def admitted(bm, tokens=PROMPT):
    seq = stateful_seq(list(tokens))
    hit = bm.can_allocate(seq)
    assert hit >= 0
    bm.allocate(seq, hit)
    return seq


def saves_of(seq):
    """`ScheduledBatch.state_save_all`'s entry for one seq."""
    return [(slot, pos) for slot, pos, _h in seq.midstep_reservations]


# ── the switch and the transfer ──────────────────────────────────────────────


def test_the_switch_is_off_by_default():
    assert envs.ATOM_K3_KDA_STATE_MIDSTEP is False
    assert kda_midstep_enabled() is False


def test_the_switch_needs_flash_kda(monkeypatch):
    monkeypatch.setenv("ATOM_K3_KDA_STATE_MIDSTEP", "1")
    assert kda_midstep_enabled() is True
    monkeypatch.setenv("AITER_FDA_ENABLE", "0")
    with pytest.raises(ValueError, match="AITER_FDA_ENABLE"):
        kda_midstep_enabled()


def test_an_aiter_without_snapshot_arguments_is_refused_at_startup():
    def with_snapshots(q, k, v, *, snapshot_targets=None, snapshot_out=None):
        pass

    def without(q, k, v, *, initial_state=None):
        pass

    def half(q, k, v, *, snapshot_targets=None):
        pass

    def catch_all(q, k, v, **kwargs):
        pass

    require_flash_kda_snapshots(with_snapshots)
    with pytest.raises(RuntimeError, match="no snapshot_targets, snapshot_out"):
        require_flash_kda_snapshots(without)
    with pytest.raises(RuntimeError, match="no snapshot_out$"):
        require_flash_kda_snapshots(half)
    with pytest.raises(RuntimeError, match="aiter#7413"):
        require_flash_kda_snapshots(catch_all)


def test_the_switch_is_refused_under_pipeline_parallelism():
    refuse_midstep_under_pipeline(1)
    with pytest.raises(RuntimeError, match=r"pipeline_parallel_size=2\)"):
        refuse_midstep_under_pipeline(2)


def test_the_transfer_reads_midstep_on_the_flash_kda_grid():
    transfer = kda_midstep_transfer(KDA_SPEC.layout_id, CONV_KERNEL)
    assert transfer.copies and transfer.paged_layout_id == KDA_SPEC.layout_id
    assert transfer.readable_midstep
    assert transfer.midstep_grid == 32
    assert transfer.midstep_min_offset == CONV_LEN
    assert StateTransfer.from_wire(transfer.to_wire()) == transfer


# ── switch off: nothing changes ──────────────────────────────────────────────


def test_off_the_anchor_still_cuts_the_chunk():
    bm = make_block_manager(k3_config(), state_runtime=k3_runtime(False))
    assert bm.paged_state_checkpoints.readable_midstep is False
    seq = admitted(bm)
    assert bm.checkpoint_cut(seq, 0, len(PROMPT)) == ANCHOR
    bm.plan_midstep(seq, 0, len(PROMPT))
    assert seq.midstep_reservations == []


@pytest.mark.parametrize("on", [False, True])
def test_one_scheduler_pass(on, monkeypatch):
    """Off: the anchor cuts the prompt into two forwards, as before. On: one
    forward covers it, and its anchor, moved onto the 32-token grid, becomes
    one snapshot target."""
    if on:
        monkeypatch.setenv("ATOM_K3_KDA_STATE_MIDSTEP", "1")
    scheduler = make_scheduler(
        k3_config(), state_runtime=k3_runtime(kda_midstep_enabled())
    )
    scheduler.add(stateful_seq(PROMPT))
    batch, _scheduled = scheduler.schedule()
    (n,) = [int(x) for x in batch.num_scheduled_tokens]
    if not on:
        assert n == ANCHOR
        assert batch.state_save_all == [[]]
        return
    assert n == len(PROMPT)
    ((slot, pos),) = batch.state_save_all[0]
    assert pos == ANCHOR // GRID * GRID
    snaps = kda_snapshots(
        batch.state_save_all,
        batch.state_rows,
        batch.num_cached_tokens,
        batch.num_scheduled_tokens,
        batch.state_slots_committed,
        num_slots=8,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == ((0, pos),)
    assert snaps.slots.tolist() == [slot]
    assert snaps.runtime_slots.tolist() == batch.state_slots_committed
    assert window_rows(snaps) == list(range(pos - CONV_LEN, pos))
    assert snaps.is_end.tolist() == [0]


# ── switch on: what one forward's positions become ───────────────────────────


def test_several_positions_in_one_forward_become_several_targets():
    """A demand rung and the anchor inside one 100-token forward: two interior
    targets, each with its own scratch slot and conv window."""
    bm = make_block_manager(demand_config(), state_runtime=k3_runtime(True))
    seq = admitted(bm, list(range(100)))
    # As `_record_checkpoint_demand` leaves it after a refused hit at 32.
    seq.checkpoint_demand_pos = 32
    bm.plan_midstep(seq, 0, 100)
    saves = saves_of(seq)
    assert [pos for _slot, pos in saves] == [32, 96]
    snaps = kda_snapshots(
        [saves],
        [0],
        [0],
        [100],
        [seq.state_slot],
        num_slots=8,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == ((0, 32), (0, 96))
    assert snaps.slots.tolist() == [slot for slot, _pos in saves]
    assert seq.state_slot not in snaps.slots.tolist()
    assert snaps.runtime_slots.tolist() == [seq.state_slot] * 2
    assert window_rows(snaps) == [29, 30, 31, 93, 94, 95]
    assert snaps.is_end.tolist() == [0, 0] and snaps.h_rows.tolist() == [0, 1]


def test_an_interior_and_an_end_position_in_one_forward():
    """A rung at 32 is a recurrence snapshot; the anchor at the forward's end
    is the runtime slot's own state, so it is a copy, not a target."""
    bm = make_block_manager(demand_config(), state_runtime=k3_runtime(True))
    seq = admitted(bm)
    bm.plan_midstep(seq, 0, ANCHOR)
    saves = saves_of(seq)
    assert [pos for _slot, pos in saves] == [32, ANCHOR]
    snaps = kda_snapshots(
        [saves],
        [0],
        [0],
        [ANCHOR],
        [seq.state_slot],
        num_slots=8,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == ((0, 32),)
    assert snaps.slots.tolist() == [slot for slot, _pos in saves]
    assert snaps.offs.tolist() == [32, ANCHOR]
    assert snaps.is_end.tolist() == [0, 1]
    assert snaps.runtime_slots.tolist() == [seq.state_slot] * 2


def test_targets_follow_the_batch_row_of_each_state_holding_seq():
    """Row 1 holds no state, so the second save list is row 2's, and its conv
    window sits after rows 0 and 1 in the step's flat token layout. Positions
    are absolute; row 2 resumes at 64, so its 96 is 32 tokens into the step."""
    snaps = kda_snapshots(
        [[], [(5, 96)]],
        state_rows=[0, 2],
        num_cached_tokens=[0, 0, 64],
        num_scheduled_tokens=[40, 10, 64],
        runtime_slots=[1, 2],
        num_slots=8,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == ((2, 32),)
    assert snaps.slots.tolist() == [5] and snaps.runtime_slots.tolist() == [2]
    assert snaps.rows.tolist() == [2] and snaps.cu_seqlens.tolist() == [0, 40, 50, 114]
    assert window_rows(snaps) == [79, 80, 81]


def test_a_step_without_saves_builds_nothing():
    kw = {"num_slots": 8, "conv_len": CONV_LEN, "device": "cpu"}
    assert kda_snapshots(None, [0], [0], [64], [1], **kw) is None
    assert kda_snapshots([[]], [0], [0], [64], [1], **kw) is None


def test_a_position_off_the_grid_is_an_error_not_a_skip():
    """The scheduler publishes every reserved hash after the forward."""
    with pytest.raises(RuntimeError, match="not writable"):
        kda_snapshots(
            [[(5, 40)]],
            [0],
            [0],
            [64],
            [1],
            num_slots=8,
            conv_len=CONV_LEN,
            device="cpu",
        )


def test_a_full_state_pool_drops_the_checkpoint():
    """#2591's reservation is best-effort: with no scratch slot left, the
    position is dropped and counted, and the chunk is not cut for it."""
    bm = make_block_manager(
        demand_config(pool_entries={"state": 2}), state_runtime=k3_runtime(True)
    )
    seq = admitted(bm)
    assert bm.checkpoint_cut(seq, 0, ANCHOR) == 0
    bm.plan_midstep(seq, 0, ANCHOR)
    assert [pos for _slot, pos in saves_of(seq)] == [32]
    assert bm.paged_state_checkpoints.checkpoints_dropped == 1


# ── how a KDA layer writes them ──────────────────────────────────────────────
#
# One `write_state_checkpoints` launch per layer (the kernel itself is pinned
# in test_kda_midstep_ckpt_kernel.py, which needs Triton).


def test_interior_and_end_targets_share_one_launch():
    """Interior targets first, in aiter's snapshot order, each reading its row
    of `snapshot_out`; the anchor at the forward's end reads the runtime slot."""
    snaps = kda_snapshots(
        [[(4, 32), (5, 64), (6, 96)]],
        [0],
        [0],
        [96],
        [1],
        num_slots=8,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == ((0, 32), (0, 64))
    assert snaps.slots.tolist() == [4, 5, 6]
    assert snaps.offs.tolist() == [32, 64, 96]
    assert snaps.is_end.tolist() == [0, 0, 1]
    assert snaps.h_rows.tolist()[:2] == [0, 1]
    assert snaps.runtime_slots.tolist() == [1, 1, 1]
    ssm, conv = torch.zeros(8, 2, 4, 4), torch.zeros(8, 5, CONV_LEN + 2)
    snap_out, mixed_qkv = torch.zeros(2, 2, 4, 4), torch.zeros(96, 5)
    args = snaps.checkpoint_args(snap_out, ssm, mixed_qkv, conv)
    assert args["h"] is snap_out and args["h_rows"] is snaps.h_rows
    assert args["ssm_state"] is ssm and args["conv_state"] is conv
    assert args["x"] is mixed_qkv and args["cu_seqlens"] is snaps.cu_seqlens
    assert args["chunk_offsets"] is None and args["chunk_size"] is None
    assert args["conv_state_len"] == CONV_LEN


def test_an_end_position_alone_reads_the_pool():
    snaps = kda_snapshots(
        [[(4, 64)]],
        [0],
        [0],
        [64],
        [1],
        num_slots=6,
        conv_len=CONV_LEN,
        device="cpu",
    )
    assert snaps.targets == () and snaps.is_end.tolist() == [1]
    ssm = torch.zeros(6, 2, 4, 4)
    args = snaps.checkpoint_args(
        None, ssm, torch.zeros(64, 5), torch.zeros(6, 5, CONV_LEN)
    )
    assert args["h"] is ssm
