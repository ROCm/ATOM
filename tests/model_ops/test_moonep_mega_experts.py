# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract tests for MoonEP balancing inside MegaMoE."""

import copy
from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs AITER's MoE interfaces")

import torch

import atom.model_ops.fused_moe.moonep_mega_experts as mme

EPR, B, WORLD = 48, 8, 8
WIDE = (WORLD * (EPR + B), 4096)
NARROW = (WORLD * EPR, mme._MEGA_DECODE_MTPR)


class _FakePool:
    def __init__(self):
        window = torch.arange((EPR + B) * 2, dtype=torch.float32).view(EPR + B, 2)
        self.local = [window.clone() for _ in mme._ADOPTED]
        self.home = [w[:EPR] for w in self.local]
        self.calls = []

    def prefetch(self, selected, resident):
        self.calls.append((selected.clone(), resident.clone()))


class _FakeMega:
    def __init__(self, w1):
        self.w1, self.calls = w1, []

    def forward(self, x, wts, ids, **kwargs):
        self.calls.append(ids.clone())
        return x


class _FakePlanner:
    def __init__(self, **kwargs):
        self.kwargs, self.states = kwargs, []

    def new_slot_state(self):
        state = {
            name: torch.full((WORLD, B), -1, dtype=torch.int32)
            for name in ("held", "prev", "placed")
        }
        self.states.append(state)
        return state

    def plan(self, ids, state):
        state["placed"][0, 2] = 100  # what the planner placed on rank 0
        return ids + 1000  # stands in for the virtual ids


def _experts(monkeypatch, *, balance, fast_path=True, unified=True, tokens=16):
    import aiter.ops.flydsl.kernels.moonep_plan as plan

    monkeypatch.setenv("ATOM_MEGA_DECODE_FAST_PATH", "1" if fast_path else "0")
    monkeypatch.setattr(plan, "MoonEPPlanner", _FakePlanner)
    monkeypatch.setattr(mme, "_PLANNERS", {})
    built = {}

    def fake_build(*, experts, mtpr, w1, **_):
        mega = built.setdefault((experts, mtpr), _FakeMega(w1))
        mega.w1 = w1
        return mega

    monkeypatch.setattr(mme, "get_or_build_mega_moe", fake_build)
    monkeypatch.setattr(mme, "is_plugin_mode", lambda: False)
    context = SimpleNamespace(
        running_tokens_are_unified=unified, running_tokens=tokens, is_prefill=balance
    )
    monkeypatch.setattr(
        mme, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    import atom.utils.tbo.ubatching as ub

    monkeypatch.setattr(ub, "tbo_active", lambda: False)

    obj = mme.MoonEPMegaExperts.__new__(mme.MoonEPMegaExperts)
    obj._layer = SimpleNamespace()
    obj._model_dim, obj._inter_dim, obj._mtpr, obj._quant = 8, 8, 4096, "a8w4"
    obj._rank, obj._world_size, obj._experts_per_rank = 0, WORLD, EPR
    obj._prefetch_slots, obj._slot_state, obj._mask_pad_rows = B, None, False
    obj._pool, obj._parts = _FakePool(), ((False, False),) * len(mme._ADOPTED)
    obj._should_balance = lambda: balance
    return obj, built


def _call(obj, ids):
    x = torch.zeros(ids.shape[0], 8, dtype=torch.bfloat16)
    return obj(hidden_states=x, topk_weights=torch.ones(ids.shape), topk_ids=ids)


def test_balanced_prefill_runs_planned_ids_and_fills_the_slots(monkeypatch):
    obj, built = _experts(monkeypatch, balance=True)
    logical = torch.tensor([[0, 383], [50, 7]], dtype=torch.int64)

    _call(obj, logical)

    wide = built[WIDE]
    assert torch.equal(wide.calls[0], logical.to(torch.int32) + 1000)
    assert wide.w1.shape[0] == EPR + B
    [planner] = mme._PLANNERS.values()
    assert planner.kwargs == {
        "rank": 0,
        "world_size": WORLD,
        "experts": WORLD * EPR,
        "slots": B,
        "max_routes": 4096 * 2,
    }
    [(selected, resident)] = obj._pool.calls  # one launch fills every part
    assert selected.tolist()[2] == 100 and resident.tolist() == [-1] * B
    assert not built[NARROW].calls


def test_each_layer_keeps_its_own_slot_state(monkeypatch):
    first, _ = _experts(monkeypatch, balance=True)
    second = copy.copy(first)
    ids = torch.tensor([[0, 1]], dtype=torch.int32)

    _call(first, ids)
    _call(first, ids)
    [planner] = mme._PLANNERS.values()
    assert len(planner.states) == 1

    _call(second, ids)
    assert len(planner.states) == 2
    assert second._slot_state is not first._slot_state


def test_unified_decode_runs_the_resident_instance(monkeypatch):
    obj, built = _experts(monkeypatch, balance=False)
    logical = torch.tensor([[0, 383], [50, 7]], dtype=torch.int32)

    _call(obj, logical)

    assert torch.equal(built[NARROW].calls[0], logical)
    assert built[NARROW].w1.shape[0] == EPR
    assert not built[WIDE].calls
    assert not obj._pool.calls and not mme._PLANNERS


@pytest.mark.parametrize("fast_path, unified", [(False, True), (True, False)])
def test_other_passes_keep_experts_home_in_the_wide_instance(
    monkeypatch, fast_path, unified
):
    obj, built = _experts(
        monkeypatch, balance=False, fast_path=fast_path, unified=unified
    )
    logical = torch.tensor([[0, 383, -1], [50, 7, -1]], dtype=torch.int64)

    _call(obj, logical)

    # Owner o's expert k is virtual expert o * (EPR + B) + k; -1 stays.
    home = torch.tensor([[0, 383 + 7 * B, -1], [50 + B, 7, -1]], dtype=torch.int32)
    assert torch.equal(built[WIDE].calls[0], home)
    assert not obj._pool.calls and not mme._PLANNERS


@pytest.mark.parametrize(
    "is_prefill, unified, expect",
    [
        (True, False, True),  # every prefill balances, however small
        (False, True, False),  # the whole group decoding never balances
        (False, False, True),  # decoding while a DP peer prefills balances too
    ],
)
def test_only_unified_decode_skips_balancing(monkeypatch, is_prefill, unified, expect):
    context = SimpleNamespace(is_prefill=is_prefill, running_tokens_are_unified=unified)
    forward = SimpleNamespace(context=context)
    monkeypatch.setattr(mme, "get_forward_context", lambda: forward)
    obj = mme.MoonEPMegaExperts.__new__(mme.MoonEPMegaExperts)

    assert obj._should_balance() is expect


@pytest.mark.parametrize(
    "world, rank, device, slots, match",
    [
        (2, 0, 0, B, "EP4 and EP8"),
        (4, 1, 5, B, "cuda:1"),
        (8, 0, 0, 0, "prefetch_slots"),
        (8, 0, 0, 65, "prefetch_slots"),
    ],
)
def test_rejects_unsupported_layouts(monkeypatch, world, rank, device, slots, match):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: device)
    with pytest.raises(ValueError, match=match):
        mme.MoonEPMegaExperts(
            None,
            model_dim=8,
            inter_dim=8,
            mtpr=4096,
            rank=rank,
            world_size=world,
            num_experts=384,
            prefetch_slots=slots,
        )


def test_rejects_eplb(monkeypatch):
    config = SimpleNamespace(eplb_enable=True)
    monkeypatch.setattr(mme, "get_current_atom_config", lambda: config)
    with pytest.raises(ValueError, match="EPLB"):
        mme.MoonEPMegaExperts.for_layer(None, None, model_dim=8, inter_dim=8)
