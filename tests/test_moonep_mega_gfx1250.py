# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MoonEP on the fused MoRI v2 transport (ATOM_ENABLE_MOONEP=1): the DP-agreed
balance gate, the per-expert pool shapes, and the forward hand-off of virtual
ids and pool windows. The MegaMoE op and the MoonEP layer are fakes."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")
pytest.importorskip("triton", reason="MegaMoE imports its Triton row-zeroing kernel")

import torch

import atom.model_ops.fused_moe.moonep_mega_gfx1250 as moonep
import atom.model_ops.fused_moe.mori_v2_prepare_finalize as mv2
import atom.utils.forward_context as fc

TOPK = 9
DIM = 16


def _context(monkeypatch, context):
    monkeypatch.setattr(
        moonep, "get_forward_context", lambda: SimpleNamespace(context=context)
    )


def _step(*, prefill, unified, across_dp, running=0):
    return SimpleNamespace(
        is_prefill=prefill,
        running_tokens_are_unified=unified,
        running_tokens_across_dp=across_dp,
        running_tokens=running,
    )


def test_no_context_balances(monkeypatch):
    _context(monkeypatch, None)
    assert moonep.moonep_balance_step()


def test_unified_decode_stays_home(monkeypatch):
    _context(monkeypatch, _step(prefill=False, unified=True, across_dp=(512,) * 4))
    assert not moonep.moonep_balance_step()


@pytest.mark.parametrize(
    "across_dp, balanced",
    [((16384, 0, 0, 0), True), ((4096, 128, 128, 128), False)],
)
def test_prefilling_step_needs_the_group_token_sum(monkeypatch, across_dp, balanced):
    monkeypatch.setenv("MOONEP_MIN_BALANCE_TOKENS", "16384")
    _context(monkeypatch, _step(prefill=False, unified=False, across_dp=across_dp))
    assert moonep.moonep_balance_step() is balanced


def test_zero_threshold_balances_every_prefilling_step(monkeypatch):
    monkeypatch.setenv("MOONEP_MIN_BALANCE_TOKENS", "0")
    _context(monkeypatch, _step(prefill=True, unified=True, across_dp=None, running=1))
    assert moonep.moonep_balance_step()


def test_expert_rows():
    assert moonep._expert_rows(torch.empty(65, 4096, 3584), 65) == (4096, 3584)
    assert moonep._expert_rows(torch.empty(65 * 4096, 224), 65) == (4096, 224)
    with pytest.raises(ValueError):
        moonep._expert_rows(torch.empty(100, 8), 65)


def test_virtual_experts(monkeypatch):
    monkeypatch.setenv("MOONEP_PREFETCH_SLOTS", "8")
    assert moonep.moonep_virtual_experts(260, 4) == 4 * (65 + 8)


def test_forward_hands_mega_the_virtual_ids_and_pool_windows(monkeypatch):
    seen = {}

    class FakeMega:
        max_tokens_per_rank = 256

        def __call__(self, x, weights, ids, **kwargs):
            seen["ids"] = ids.clone()
            seen["w1"], seen["w1_scale"] = kwargs["w1"], kwargs["w1_scale"]
            seen["w2"], seen["w2_scale"] = kwargs["w2"], kwargs["w2_scale"]
            return torch.zeros_like(x)

    windows = [torch.full((1,), float(i)) for i in range(4)]

    class FakeMoonEP:
        def route(self, ids, *, balance):
            seen["balance"] = balance
            return ids + 1000

    monkeypatch.setattr(moonep, "moonep_balance_step", lambda: True)
    monkeypatch.setattr(
        mv2.MoriV2ModularKernel, "_assert_recipe_matches", lambda *a: None
    )
    monkeypatch.setattr(
        fc, "get_forward_context", lambda: SimpleNamespace(context=None)
    )
    kernel = mv2.MoriV2ModularKernel.__new__(mv2.MoriV2ModularKernel)
    kernel._recv_bound = lambda arena_rows: None
    layer_moonep = FakeMoonEP()
    layer_moonep.windows = windows
    pf = SimpleNamespace(
        mega=FakeMega(),
        mask_pad_rows=False,
        num_dispatchers=lambda: 4,
        combine_quant_for_step=lambda: "bf16",
        moonep=layer_moonep,
    )
    object.__setattr__(kernel, "prepare_finalize", pf)
    ids = (torch.arange(8 * TOPK) % 260).reshape(8, TOPK)
    kernel.forward(
        torch.ones(8, DIM, dtype=torch.bfloat16),
        torch.zeros(1),
        torch.zeros(1),
        torch.ones(8, TOPK),
        ids,
        w1_scale=torch.zeros(1),
        w2_scale=torch.zeros(1),
    )
    assert seen["balance"] is True
    assert torch.equal(seen["ids"], ids.to(torch.int32) + 1000)
    assert [seen[k] for k in ("w1", "w1_scale", "w2", "w2_scale")] == windows


def test_moonep_needs_the_fused_transport(monkeypatch):
    # The EP group is looked up at call time; a stand-in module keeps this test
    # independent of how (or whether) aiter.dist is importable here.
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_ep_group=lambda: SimpleNamespace(ranks=[0])),
    )
    monkeypatch.setenv("ATOM_ENABLE_MOONEP", "1")
    monkeypatch.setattr(mv2, "_resolve_transport", lambda: "gather")
    moe = SimpleNamespace(num_experts=260)
    manager = SimpleNamespace(world_size=4, rank=0)
    with pytest.raises(ValueError, match="fused MegaMoE transport"):
        mv2.make_mori_v2_prepare_finalize(moe, manager)
