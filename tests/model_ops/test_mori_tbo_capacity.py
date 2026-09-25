# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Capacity of the second TBO ubatch MoRI op (ATOM_MORI_TBO_HALF_BUFFERS).

Halving is only sound while no ubatch can carry more than half of a rank's
token budget. MoRI does not bound-check its input, so a ubatch past the op's
capacity writes off the end of the symmetric buffers instead of failing. These
pin the split that makes half enough, and the cases that must keep full size.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

import torch

import atom.model_ops.fused_moe.mori_prepare_finalize as mpf
from atom.utils.tbo.ubatch_splitting import maybe_create_ubatch_slices
from atom.utils.tbo.ubatching import _precompute_prefill_token_split

MAX_TOKENS = 16384


def _config(**kw):
    base = {"enable_tbo_decode": False, "prefill_context_parallel_size": 1}
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def half_on(monkeypatch):
    monkeypatch.setattr(mpf, "_TBO_HALF_BUFFERS", True)
    monkeypatch.setenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "1")


def test_off_keeps_the_full_budget(monkeypatch):
    monkeypatch.setattr(mpf, "_TBO_HALF_BUFFERS", False)
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config()) == MAX_TOKENS


@pytest.mark.parametrize("budget,expected", [(16384, 8192), (16383, 8192), (1, 1)])
def test_token_split_prefill_gets_half(half_on, budget, expected):
    assert mpf.tbo_max_tokens_per_rank(budget, _config()) == expected


@pytest.mark.parametrize(
    "config",
    [
        _config(enable_tbo_decode=True),
        _config(prefill_context_parallel_size=2),
    ],
)
def test_uneven_splits_keep_full_size(half_on, config):
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, config) == MAX_TOKENS


def test_request_boundary_split_keeps_full_size(half_on, monkeypatch):
    monkeypatch.setenv("ATOM_TBO_PREFILL_TOKEN_SPLIT", "0")
    assert mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config()) == MAX_TOKENS


@pytest.mark.parametrize(
    "lens",
    [
        [MAX_TOKENS],
        [MAX_TOKENS - 1],
        [1, MAX_TOKENS - 1],
        [MAX_TOKENS - 1, 1],
        [3, 5000, 11381],
        [2],
    ],
)
def test_no_token_split_ubatch_exceeds_half(half_on, lens):
    """Both the DP-reduced sizes and the realised slices, forced or not: a rank
    below the min-token bar is still split when a peer clears it."""
    capacity = mpf.tbo_max_tokens_per_rank(MAX_TOKENS, _config())
    toks = np.asarray(lens, dtype=np.int32)
    _, can_split, ub0, ub1 = _precompute_prefill_token_split(toks, len(lens), 8192)
    assert can_split and max(ub0, ub1) <= capacity
    slices = maybe_create_ubatch_slices(
        num_reqs=len(lens),
        num_tokens=int(toks.sum()),
        is_prefill=True,
        num_scheduled_tokens=toks,
        force=True,
    )
    widths = [s.token_slice.stop - s.token_slice.start for s in slices]
    assert widths == [ub0, ub1]


def _pf_with_slots(capacities):
    from unittest.mock import patch

    with patch.object(mpf, "MORI_AVAILABLE", True):
        return mpf.MoriPrepareAndFinalize(
            [object() for _ in capacities],
            max_tokens_per_rank=capacities[0],
            num_dispatchers=8,
            dispatch_format=mpf.MoriDispatchFormat(
                dtype=torch.bfloat16, quant_type=None, scale_dim=0, scale_type_size=4
            ),
            low_latency=False,
            internode=False,
            op_max_tokens_per_rank=capacities,
        )


def test_an_oversized_ubatch_is_refused_before_dispatch(monkeypatch):
    """Slot 1 (the second ubatch) holds half; slot 0 is also the sync op."""
    pf = _pf_with_slots((16, 8))
    monkeypatch.setattr(pf, "_current_tbo_slot", lambda: 1)
    a1 = torch.zeros(9, 32, dtype=torch.bfloat16)
    ids = torch.zeros(9, 6, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="holds 8 per rank"):
        pf.prepare_async(a1, torch.ones(9, 6), ids, 384, None, False)


def test_slot_zero_keeps_the_full_budget(monkeypatch):
    pf = _pf_with_slots((16, 8))
    monkeypatch.setattr(pf, "_current_tbo_slot", lambda: 0)
    taken = []
    monkeypatch.setattr(
        pf, "_prepare_async_comm_stream", lambda *a, **k: taken.append(a[0].shape)
    )
    a1 = torch.zeros(9, 32, dtype=torch.bfloat16)
    pf.prepare_async(
        a1, torch.ones(9, 6), torch.zeros(9, 6, dtype=torch.int32), 384, None, False
    )
    assert taken == [(9, 32)]


def test_capacities_need_one_entry_per_op(monkeypatch):
    pf = _pf_with_slots((16, 8))
    monkeypatch.setattr(mpf, "MORI_AVAILABLE", True)
    with pytest.raises(ValueError, match="one entry per op"):
        type(pf)(
            [object(), object()],
            max_tokens_per_rank=16,
            num_dispatchers=8,
            dispatch_format=pf.dispatch_format,
            low_latency=False,
            internode=False,
            op_max_tokens_per_rank=(16,),
        )


def test_slot_capacities_halve_only_the_second_slot(half_on):
    """What moe.py requests each all2all handle slot with."""
    assert mpf.mori_op_capacities(MAX_TOKENS, 2, _config()) == [MAX_TOKENS, 8192]
    assert mpf.mori_op_capacities(MAX_TOKENS, 2, _config(enable_tbo_decode=True)) == [
        MAX_TOKENS,
        MAX_TOKENS,
    ]


def test_without_tbo_there_is_one_full_slot(half_on, monkeypatch):
    asked = []
    monkeypatch.setattr(mpf, "tbo_max_tokens_per_rank", lambda *a: asked.append(a) or 1)
    assert mpf.mori_op_capacities(MAX_TOKENS, 1, _config()) == [MAX_TOKENS]
    assert asked == []  # no "slot 1 sized for ..." log without a slot 1


def test_half_buffers_off_keeps_every_slot_full(monkeypatch):
    monkeypatch.setattr(mpf, "_TBO_HALF_BUFFERS", False)
    assert mpf.mori_op_capacities(MAX_TOKENS, 2, _config()) == [MAX_TOKENS] * 2
