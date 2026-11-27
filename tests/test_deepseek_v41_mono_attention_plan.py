# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""I3 (DESIGN_v3 4.9) for the V4.1 mono attention: for any token's selection
bound and window count, every chunk the merge stage polls (AS) is one the score
stage writes this step, every key slot is in a written chunk, and a 16-slot
block holds keys exactly when its first slot does (the PV's skip) -- in both
slot layouts, a build whose keys include a selection and one whose do not (the
DSpark draft: its window and its block, every slot the window's). The functions
under test are the ones the kernel calls (``mono/attention_plan.py``)."""

import random

import pytest

from atom.models.deepseek_v41.mono import attention_plan as aplan
from atom.models.deepseek_v41.mono.config import ATTENTION_KEYS_MAX
from atom.mono.dsa_indexer.paged import plan as ip

LAYOUTS = [aplan.selection_slots(True), aplan.selection_slots(False)]


def tokens(sel_slots):
    """(n_sel, count) pairs of a layout: the edges crossed, then random ones. A
    graph's pad row is (0, 0); a layout without a selection has n_sel 0."""
    window_max = aplan.window_slots(sel_slots)
    bounds = [0, 1, aplan.CH - 1, aplan.CH, ip.TOPK - 1, ip.TOPK, ip.TOPK + 1, 1 << 20]
    counts = [0, 1, aplan.CH - 1, aplan.CH, aplan.CH + 1, window_max - 1, window_max]
    rng = random.Random(0)
    pairs = [(b, c) for b in bounds for c in counts]
    pairs += [
        (rng.randrange(0, 1 << 21), rng.randrange(0, window_max + 1))
        for _ in range(500)
    ]
    return [(ip.selected(b) if sel_slots else 0, c) for b, c in pairs]


def written(n_sel, count, sel_slots, chunk_written=aplan.chunk_written):
    return {
        ch for ch in range(aplan.CHUNKS) if chunk_written(ch, n_sel, count, sel_slots)
    }


def polled(n_sel, count, sel_slots):
    return {
        aplan.polled_chunk(ch, n_sel, count, sel_slots) for ch in range(aplan.CHUNKS)
    }


@pytest.mark.parametrize("sel_slots", LAYOUTS)
def test_every_polled_chunk_is_written(sel_slots):
    for n_sel, count in tokens(sel_slots):
        assert n_sel <= sel_slots and count <= aplan.window_slots(sel_slots)
        assert polled(n_sel, count, sel_slots) <= written(n_sel, count, sel_slots)


@pytest.mark.parametrize("sel_slots", LAYOUTS)
def test_every_key_slot_is_in_a_written_chunk(sel_slots):
    for n_sel, count in tokens(sel_slots):
        keys = [
            s
            for s in range(ATTENTION_KEYS_MAX)
            if aplan.slot_valid(s, n_sel, count, sel_slots)
        ]
        assert len(keys) == n_sel + count, (n_sel, count)
        assert {s // aplan.CH for s in keys} <= written(n_sel, count, sel_slots)


@pytest.mark.parametrize("sel_slots", LAYOUTS)
def test_a_block_holds_keys_when_its_first_slot_does(sel_slots):
    for n_sel, count in tokens(sel_slots):
        for b in range(0, ATTENTION_KEYS_MAX, 16):
            first = aplan.slot_valid(b, n_sel, count, sel_slots)
            assert all(
                aplan.slot_valid(s, n_sel, count, sel_slots) <= first
                for s in range(b, b + 16)
            ), (n_sel, count, b)


def test_a_draft_row_keeps_every_window_and_block_key():
    """The DSpark draft's row (``write_draft_step_meta``): its window (128) and
    its whole block (5): every one a valid slot. With the selection's slots
    reserved, its last keys -- the block's own -- would have no slot."""
    count = 128 + 5
    draft = aplan.selection_slots(False)
    assert (
        sum(aplan.slot_valid(s, 0, count, draft) for s in range(ATTENTION_KEYS_MAX))
        == count
    )
    target = aplan.selection_slots(True)
    held = sum(aplan.slot_valid(s, 0, count, target) for s in range(ATTENTION_KEYS_MAX))
    assert held == aplan.window_slots(target) < count


def test_without_the_window_chunk_floor_a_pad_row_polls_an_unwritten_chunk():
    """Positive control: the score's written chunks without the first window
    chunk always written leave a pad row's, which the merge polls for every
    chunk, unwritten."""
    sel_slots = aplan.selection_slots(True)
    first = aplan.window_chunk(sel_slots)

    def unfloored(chunk, n_sel, count, sel_slots):
        return (chunk * aplan.CH < n_sel) | (
            (chunk >= first) & ((chunk - first) * aplan.CH < count)
        )

    assert not polled(0, 0, sel_slots) <= written(
        0, 0, sel_slots, chunk_written=unfloored
    )


def test_merge_heads_divide_a_tile():
    for tokens_ in range(1, 49):
        assert 16 % aplan.merge_heads(tokens_) == 0
