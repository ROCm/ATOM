# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""PP mailbox (atom/distributed/pp_mailbox.py) without a GPU: which AttnRes bank
blocks cross a hop, the slot layout, and the NCCL hop's bank trimming.

The send / recv paths need HIP streams and peer memory; they are exercised by
``tools/pp_mailbox_selftest.py`` on the hardware.
"""

import importlib
import math

import pytest
import torch
from aiter_stub import atom_without_aiter

from atom.distributed.pp_mailbox import HopLayout, PPMailbox, closed_bank_blocks

K3_LAYERS, K3_BLOCK, K3_HIDDEN = 93, 12, 7168
K3_BLOCKS = math.ceil(K3_LAYERS / K3_BLOCK)  # 8
DSPARK_TAPS = (2, 23, 47, 71, 89)


@pytest.mark.parametrize(
    "first_layer, closed",
    [
        (0, 0),
        (1, 1),
        (7, 1),
        (12, 1),
        (13, 2),
        (24, 2),
        (25, 3),
        (38, 4),
        (51, 5),
        (63, 6),
        (74, 7),
        (85, 8),
        (92, 8),
    ],
)
def test_closed_bank_blocks(first_layer, closed):
    assert closed_bank_blocks(first_layer, K3_BLOCK, K3_BLOCKS) == closed


def test_closed_bank_blocks_without_attn_res():
    assert closed_bank_blocks(40, None, 0) == 0


def test_closed_blocks_match_attn_res():
    """A stage starting at L hands on, and reads before writing, only the bank
    slots below ``closed_bank_blocks(L)``: checked against AttnRes's own
    maybe_close_block (pure torch; the module's AITER imports are stand-ins on a
    CPU runner), for a fixed bank read through ``window`` and for one that grows
    by a slot per closed block."""
    with atom_without_aiter() as load:
        AttnRes = load("atom.model_ops.attention_residual").AttnRes
        if hasattr(AttnRes, "window"):
            _check_bank_rule(AttnRes)
        else:
            _check_growing_bank_rule(AttnRes)


def _site(AttnRes, layer_idx):
    stub = torch.nn.Identity()
    return AttnRes(stub, stub, enabled=True, block_size=K3_BLOCK, layer_idx=layer_idx)


def _check_growing_bank_rule(AttnRes):
    """A bank that grows by one slot per closed block: a stage starting at L
    receives exactly ``closed_bank_blocks(L)`` slots, all of them read, and
    hands on exactly its end's."""
    for first in range(1, K3_LAYERS):
        bank = torch.zeros(1, 0, 1)
        for layer in range(first):  # the stages before this one
            bank, _ = _site(AttnRes, layer).maybe_close_block(torch.ones(1, 1), bank)
        assert bank.shape[1] == closed_bank_blocks(first, K3_BLOCK, K3_BLOCKS)
        for end in range(first + 1, K3_LAYERS + 1):
            out = bank
            for layer in range(first, end):
                out, _ = _site(AttnRes, layer).maybe_close_block(torch.ones(1, 1), out)
            assert out.shape[1] == closed_bank_blocks(end, K3_BLOCK, K3_BLOCKS)


def _check_bank_rule(AttnRes):
    def site(layer_idx):
        return _site(AttnRes, layer_idx)

    for first in range(K3_LAYERS):
        for end in range(first + 1, K3_LAYERS + 1):
            bank = torch.zeros(1, K3_BLOCKS, 1)
            written, read_unwritten = set(), set()
            for layer in range(first, end):
                attn = site(layer)  # self_attention_attn_res: layer_idx = layer
                before = bank.clone()
                bank, _ = attn.maybe_close_block(torch.ones(1, 1), bank)
                written |= {
                    b
                    for b in range(K3_BLOCKS)
                    if not torch.equal(before[:, b], bank[:, b])
                }
                for s in (attn, site(layer + 1)):  # mlp_attn_res: layer_idx = layer + 1
                    read_unwritten |= set(range(s.window(bank).shape[1])) - written
            assert read_unwritten <= set(
                range(closed_bank_blocks(first, K3_BLOCK, K3_BLOCKS))
            )
            lo = closed_bank_blocks(first, K3_BLOCK, K3_BLOCKS)
            hi = closed_bank_blocks(end, K3_BLOCK, K3_BLOCKS)
            assert written == set(range(lo, hi))


def _layout(closed, aux_taps, *, rows=16384, slots=2):
    return HopLayout(rows, K3_HIDDEN, K3_BLOCKS, closed, aux_taps * K3_HIDDEN, 2, slots)


def test_hop_layout_sizes_and_offsets():
    lay = _layout(closed=3, aux_taps=2)
    assert lay.row_elems == K3_HIDDEN * (1 + 3) + 2 * K3_HIDDEN
    assert lay.slot_bytes % (2 << 20) == 0
    assert lay.slot_bytes >= 16384 * lay.row_elems * 2
    assert lay.ring_bytes == 2 * lay.slot_bytes
    off = lay.offsets()
    assert off["hidden"] == 0
    assert off["bank"] == 16384 * K3_HIDDEN * 2
    assert off["aux"] == off["bank"] + 16384 * 3 * K3_HIDDEN * 2
    assert off["aux"] + 16384 * lay.aux_cols * 2 <= lay.slot_bytes


def test_equal_mla_split_payload_per_hop():
    """12,13,13,13,12,11,11,8 (#813): what each hop carries, in H-wide rows, against
    today's NCCL payload (hidden + the whole 8-block bank + the aux taps)."""
    split = (12, 13, 13, 13, 12, 11, 11, 8)
    starts = [sum(split[:s]) for s in range(1, len(split))]
    mailbox, nccl = [], []
    for first in starts:
        taps = sum(t < first for t in DSPARK_TAPS)
        closed = closed_bank_blocks(first, K3_BLOCK, K3_BLOCKS)
        mailbox.append(1 + closed + taps)
        nccl.append(1 + K3_BLOCKS + taps)
    assert starts == [12, 25, 38, 51, 63, 74, 85]
    assert mailbox == [3, 6, 7, 9, 10, 12, 13]
    assert nccl == [10, 11, 11, 12, 12, 13, 13]
    assert sum(mailbox) / sum(nccl) < 0.75


def _trimmer(out_closed, in_closed, full_bank=True):
    """A PPMailbox carrying only what trim / expand read (no HIP, no group)."""
    box = object.__new__(PPMailbox)
    box.trim_nccl_bank = True
    box.total_blocks = K3_BLOCKS
    box.full_bank = full_bank
    box.out_layout = None if out_closed is None else _layout(out_closed, 0)
    box.in_layout = None if in_closed is None else _layout(in_closed, 0)
    return box


def test_nccl_hop_trims_and_restores_the_bank():
    rows, hidden = 5, 16
    bank = torch.randn(rows, K3_BLOCKS, hidden)
    sent = {"hidden_states": torch.randn(rows, hidden), "block_residual": bank}
    owned = _trimmer(out_closed=3, in_closed=None).trim_for_nccl(sent)
    assert owned == ("block_residual",)
    assert sent["block_residual"].shape == (rows, 3, hidden)
    assert sent["block_residual"].is_contiguous()
    received = {"block_residual": sent["block_residual"].clone()}
    _trimmer(out_closed=None, in_closed=3).expand_from_nccl(received)
    assert received["block_residual"].shape == (rows, K3_BLOCKS, hidden)
    assert torch.equal(received["block_residual"][:, :3], bank[:, :3])


def test_nccl_hop_leaves_a_growing_bank_as_sent():
    """A bank that grows by a slot per closed block already holds only closed
    blocks: nothing to trim, nothing to rebuild."""
    rows, hidden = 5, 16
    bank = torch.randn(rows, 3, hidden)
    sent = {"block_residual": bank}
    assert (
        _trimmer(out_closed=3, in_closed=None, full_bank=False).trim_for_nccl(sent)
        == ()
    )
    assert sent["block_residual"] is bank
    _trimmer(out_closed=None, in_closed=3, full_bank=False).expand_from_nccl(sent)
    assert sent["block_residual"] is bank


def test_nccl_hop_keeps_a_bank_it_cannot_trim():
    bank = torch.randn(2, K3_BLOCKS, 4)
    sent = {"block_residual": bank}
    assert _trimmer(out_closed=K3_BLOCKS, in_closed=None).trim_for_nccl(sent) == ()
    assert sent["block_residual"] is bank
    off = _trimmer(out_closed=3, in_closed=None)
    off.trim_nccl_bank = False
    assert off.trim_for_nccl(sent) == ()


def test_build_is_off_unless_asked(monkeypatch):
    monkeypatch.delenv("ATOM_PP_MAILBOX", raising=False)
    monkeypatch.delenv("ATOM_PP_TRIM_BANK", raising=False)
    # off: returns before touching aiter or the PP group
    assert PPMailbox.build(config=None, model=None, relay=None, device=None) is None


def test_env_defaults(monkeypatch):
    for name in (
        "ATOM_PP_MAILBOX",
        "ATOM_PP_MAILBOX_SLOTS",
        "ATOM_PP_MAILBOX_CHECK",
        "ATOM_PP_TRIM_BANK",
    ):
        monkeypatch.delenv(name, raising=False)
    envs = importlib.import_module("atom.utils.envs")
    assert envs.ATOM_PP_MAILBOX is False
    assert envs.ATOM_PP_MAILBOX_SLOTS == 2
    assert envs.ATOM_PP_MAILBOX_CHECK is False
    assert envs.ATOM_PP_TRIM_BANK is False
    monkeypatch.setenv("ATOM_PP_MAILBOX_SLOTS", "0")
    with pytest.raises(ValueError):
        _ = envs.ATOM_PP_MAILBOX_SLOTS
