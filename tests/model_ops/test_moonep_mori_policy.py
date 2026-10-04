# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract tests for MoonEP planning layered over MoRI dispatch."""

import importlib
from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs AITER's MoE interfaces")

import torch

import atom.model_ops.fused_moe.moonep_prepare_finalize as mpf


class _FakeTransport:
    """Stands in for MoriPrepareAndFinalize: records the ids it is handed."""

    def __init__(self):
        self.prepared = []
        self.finalized = []

    def num_dispatchers(self):
        return 2

    def max_num_tokens_per_rank(self):
        return 16

    def prepare(self, a1, weights, ids, *args):
        self.prepared.append((a1, weights, ids))
        return a1, None, None, ids, weights

    def finalize(self, output, fused, weights, ids, apply_router_weight_on_input):
        self.finalized.append(ids)
        return fused


class _FakePolicy:
    def __init__(self, plan):
        self.result = plan
        self.calls = 0

    def plan(self, ids):
        self.calls += 1
        return self.result


def _policy_prepare_finalize(*, prefill: bool):
    obj = mpf.MoonEPPrepareAndFinalize.__new__(mpf.MoonEPPrepareAndFinalize)
    obj._num_experts = 8
    obj._experts_per_rank = 4
    obj._prefetch_slots = 1
    obj._rank = 0
    obj._world_size = 2
    obj._active_ids = None
    obj._active_plan = None
    obj._transport = _FakeTransport()
    obj._is_prefill = lambda: prefill
    return obj


def test_virtual_mask_maps_home_and_prefetch_ids_onto_one_window():
    mask = mpf._make_virtual_expert_mask(
        rank=1,
        world_size=2,
        experts_per_rank=4,
        prefetch_slots=2,
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [0] * 6 + [1] * 6
    # fused_moe's local index is cumsum(mask) - 1: home slots land on window
    # rows [0, EPR) and prefetch slots on [EPR, EPR + B).
    assert (mask.cumsum(0) - 1)[6:12].tolist() == [0, 1, 2, 3, 4, 5]


@pytest.mark.parametrize(
    "is_prefill, unified, expected",
    [(False, True, False), (True, True, True), (False, False, True)],
)
def test_phase_is_decode_only_when_every_rank_decodes(
    monkeypatch, is_prefill, unified, expected
):
    context = SimpleNamespace(is_prefill=is_prefill, running_tokens_are_unified=unified)
    monkeypatch.setattr(
        mpf, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    obj = mpf.MoonEPPrepareAndFinalize.__new__(mpf.MoonEPPrepareAndFinalize)
    assert obj._is_prefill() is expected


def test_prefill_dispatches_and_combines_with_the_same_planned_ids():
    obj = _policy_prepare_finalize(prefill=True)
    logical = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    planned = torch.tensor([[0, 9], [2, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.75, 0.25], [0.6, 0.4]], dtype=torch.float32)
    policy = _FakePolicy(SimpleNamespace(planned_topk_ids=planned))
    obj._prefill_policy = lambda ids: policy

    result = obj.prepare(torch.randn(2, 8), weights, logical, 8, None, False, None)
    fused = torch.randn(2, 8)
    out = obj.finalize(torch.empty(2, 8), fused, weights, logical, False)

    transport = obj._transport
    assert policy.calls == 1
    _, sent_weights, sent_ids = transport.prepared[0]
    assert sent_ids is planned and sent_weights is weights
    assert result[3] is planned
    assert transport.finalized == [planned]
    assert out is fused
    assert obj._active_ids is None and obj._active_plan is None


def test_decode_sends_every_expert_to_its_owner_resident_slot():
    obj = _policy_prepare_finalize(prefill=False)
    logical = torch.tensor([[0, 7], [3, 4]], dtype=torch.int64)
    weights = torch.tensor([[0.8, 0.2], [0.55, 0.45]], dtype=torch.float32)
    obj._prefill_policy = lambda ids: pytest.fail("decode entered PrefillPolicy")

    obj.prepare(torch.randn(2, 8), weights, logical, 8, None, False, None)
    assert obj._active_plan is None
    obj.finalize(torch.empty(2, 8), torch.randn(2, 8), weights, logical, False)

    # EPR=4, B=1: expert e -> owner * 5 + e % 4.
    virtual = torch.tensor([[0, 8], [3, 5]], dtype=torch.int32)
    assert torch.equal(obj._transport.prepared[0][2], virtual)
    assert torch.equal(obj._transport.finalized[0], virtual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_decode_owner_ids_kernel_matches_reference(dtype):
    # DSV4-Pro EP8 geometry: 384 experts, EPR=48, B=8, top-6, plus invalid ids.
    ids = torch.randint(0, 384, (1000, 6), dtype=dtype)
    ids[::97, 2] = -1
    expected = mpf._owner_virtual_ids(ids, 48, 8)
    got = mpf._owner_virtual_ids(ids.cuda(), 48, 8)
    assert got.dtype == torch.int32
    assert torch.equal(got.cpu(), expected)
    assert torch.equal(expected[5, :], (ids[5] + ids[5] // 48 * 8).to(torch.int32))
    assert int(expected[0, 2]) == -1


class _FakePool:
    def __init__(self, local, experts_per_rank):
        self.local = local
        self.home = local[:experts_per_rank]
        self.selected = None

    def prefetch(self, selected):
        self.selected = selected.clone()


def _experts_prepare_finalize(monkeypatch, *, prefill: bool):
    obj = _policy_prepare_finalize(prefill=prefill)
    obj._active_ids = torch.zeros(1, 2, dtype=torch.int32)
    obj._active_plan = (
        SimpleNamespace(experts_to_copy=torch.tensor([[6], [-1]], dtype=torch.int32))
        if prefill
        else None
    )
    obj._virtual_masks = {}

    p1 = _FakePool(torch.empty(5, 8, 8), obj._experts_per_rank)
    p2 = _FakePool(torch.empty(5, 8, 8), obj._experts_per_rank)
    obj._pools = (
        (p1, False, False),
        (p2, False, False),
        (None, False, False),
        (None, False, False),
        (None, False, False),
        (None, False, False),
    )

    calls = []

    def fake_fused_moe(rows, w1, w2, weights, ids, mask, **kwargs):
        calls.append((w1, w2, weights, ids, mask, kwargs))
        return torch.full_like(rows, len(calls), dtype=torch.float32)

    fused_moe_module = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_module, "fused_moe", fake_fused_moe)
    return obj, p1, p2, calls


def test_prefill_experts_run_once_over_the_home_plus_prefetch_window(monkeypatch):
    obj, p1, p2, calls = _experts_prepare_finalize(monkeypatch, prefill=True)
    rows = torch.randn(3, 8)
    weights = torch.tensor([[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]], dtype=torch.float32)
    ids = torch.tensor([[0, 4], [1, 2], [4, 3]], dtype=torch.int32)
    out = obj.run_dispatched_experts(
        rows,
        p1.home,
        p2.home,
        topk_weights=weights,
        topk_ids=ids,
        expert_mask=None,
        num_local_tokens=None,
    )

    assert len(calls) == 1
    w1, w2, sent_weights, sent_ids, mask, _ = calls[0]
    assert w1 is p1.local and w2 is p2.local
    assert sent_weights is weights and sent_ids is ids
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert p1.selected.tolist() == [6]
    assert p2.selected.tolist() == [6]
    assert torch.equal(out, torch.full_like(rows, 1, dtype=torch.float32))


def test_decode_experts_use_the_same_window_and_skip_prefetch(monkeypatch):
    obj, p1, p2, calls = _experts_prepare_finalize(monkeypatch, prefill=False)
    rows = torch.randn(2, 8)
    weights = torch.tensor([[0.8, 0.2], [0.55, 0.45]], dtype=torch.float32)
    ids = torch.tensor([[0, 8], [3, 5]], dtype=torch.int32)
    obj.run_dispatched_experts(
        rows,
        p1.home,
        p2.home,
        topk_weights=weights,
        topk_ids=ids,
        expert_mask=None,
        num_local_tokens=None,
    )

    assert len(calls) == 1
    w1, w2, _, _, mask, _ = calls[0]
    assert w1 is p1.local and w2 is p2.local
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    assert p1.selected is None and p2.selected is None
