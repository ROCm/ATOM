# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 midstep KDA checkpoints (ATOM_K3_KDA_MIDSTEP_CKPT=1) on #2591's framework.

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
)
from atom.utils import envs

KDA_SPEC = PagedStateCheckpointSpec(10, 25, "kda-paged-state-v1:test", image_bytes=25)
#: Kimi-K3's `linear_conv_kernel_dim`.
CONV_KERNEL = 4
CONV_LEN = CONV_KERNEL - 1
GRID = FLASH_KDA_CHUNK
# PROMPT is 44 tokens; a PAGE class needs no successor room, so its prompt-end
# anchor is the last hittable block's end.
ANCHOR = 40


@pytest.fixture(autouse=True)
def _clean_switches(monkeypatch):
    for name in (
        "ATOM_K3_KDA_MIDSTEP_CKPT",
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
    assert envs.ATOM_K3_KDA_MIDSTEP_CKPT is False
    assert kda_midstep_enabled() is False


def test_the_switch_needs_flash_kda(monkeypatch):
    monkeypatch.setenv("ATOM_K3_KDA_MIDSTEP_CKPT", "1")
    assert kda_midstep_enabled() is True
    monkeypatch.setenv("AITER_FDA_ENABLE", "0")
    with pytest.raises(ValueError, match="AITER_FDA_ENABLE"):
        kda_midstep_enabled()


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
        monkeypatch.setenv("ATOM_K3_KDA_MIDSTEP_CKPT", "1")
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
    assert snaps.conv_rows.tolist() == list(range(pos - CONV_LEN, pos))
    assert snaps.end_slots is None


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
    assert snaps.conv_rows.tolist() == [29, 30, 31, 93, 94, 95]
    assert snaps.end_slots is None


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
    assert snaps.end_slots.tolist() == [saves[1][0]]
    assert snaps.end_runtime_slots.tolist() == [seq.state_slot]


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
    assert snaps.conv_rows.tolist() == [79, 80, 81]


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


def _planes(slots=6, heads=2, width=4, dim=5, rollback=2):
    torch.manual_seed(0)
    ssm = torch.randn(slots, heads, width, width)
    conv = torch.randn(slots, dim, CONV_LEN + rollback)
    return ssm, conv


def test_write_puts_each_snapshot_and_its_conv_window_in_its_slot():
    ssm, conv = _planes()
    before_ssm, before_conv = ssm.clone(), conv.clone()
    mixed_qkv = torch.randn(80, conv.shape[1])
    snaps = kda_snapshots(
        [[(4, 32), (5, 64)]],
        [0],
        [0],
        [80],
        [1],
        num_slots=6,
        conv_len=CONV_LEN,
        device="cpu",
    )
    snap_out = torch.randn(2, *ssm.shape[1:])
    snaps.write(ssm, conv, snap_out, mixed_qkv)
    assert torch.equal(ssm[4], snap_out[0]) and torch.equal(ssm[5], snap_out[1])
    assert torch.equal(conv[4, :, :CONV_LEN], mixed_qkv[29:32].T)
    assert torch.equal(conv[5, :, :CONV_LEN], mixed_qkv[61:64].T)
    # The rollback columns a prefill leaves alone come from the runtime slot.
    assert torch.equal(conv[4, :, CONV_LEN:], before_conv[1, :, CONV_LEN:])
    assert torch.equal(conv[5, :, CONV_LEN:], before_conv[1, :, CONV_LEN:])
    for slot in (0, 1, 2, 3):
        assert torch.equal(ssm[slot], before_ssm[slot])
        assert torch.equal(conv[slot], before_conv[slot])


def test_write_copies_an_end_position_from_the_runtime_slot():
    ssm, conv = _planes()
    before_ssm, before_conv = ssm.clone(), conv.clone()
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
    assert snaps.targets == () and snaps.end_slots.tolist() == [4]
    snaps.write(ssm, conv, None, torch.randn(64, conv.shape[1]))
    assert torch.equal(ssm[4], before_ssm[1])
    assert torch.equal(conv[4], before_conv[1])
    for slot in (0, 1, 2, 3, 5):
        assert torch.equal(ssm[slot], before_ssm[slot])
        assert torch.equal(conv[slot], before_conv[slot])
