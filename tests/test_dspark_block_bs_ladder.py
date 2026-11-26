# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM_DSPARK_BLOCK_BS_LADDER: the batch a DSpark block runs at."""

from types import SimpleNamespace

import pytest

pytest.importorskip("aiter")
from atom.spec_decode.dspark_proposer import DSparkProposer


def _proposer(max_seqs=64, lossless=None):
    p = object.__new__(DSparkProposer)
    p.config = SimpleNamespace(max_num_seqs=max_seqs)
    p.lossless = lossless
    return p


def test_off_keeps_the_live_batch(monkeypatch):
    monkeypatch.delenv("ATOM_DSPARK_BLOCK_BS_LADDER", raising=False)
    assert _proposer()._pinned_draft_bs(7) == 7


@pytest.mark.parametrize(
    "live, want", [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16), (16, 16), (17, 17)]
)
def test_the_smallest_entry_at_or_above(monkeypatch, live, want):
    monkeypatch.setenv("ATOM_DSPARK_BLOCK_BS_LADDER", "16, 1,2,4,8")
    assert _proposer()._pinned_draft_bs(live) == want


def test_entries_above_max_num_seqs_are_dropped(monkeypatch):
    monkeypatch.setenv("ATOM_DSPARK_BLOCK_BS_LADDER", "4,128")
    assert _proposer(max_seqs=8)._pinned_draft_bs(5) == 5


def test_lossless_sampling_is_not_pinned(monkeypatch):
    monkeypatch.setenv("ATOM_DSPARK_BLOCK_BS_LADDER", "8")
    assert _proposer(lossless=object())._pinned_draft_bs(3) == 3
