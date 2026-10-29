"""MoriV2ModularKernel's Triton-experts branch: MegaMoE as the transport, ATOM's
Triton experts in between. No GPU: MegaMoE and triton_mega_moe are stand-ins,
so what is checked is the branch itself and what it hands triton_mega_moe."""

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.fused_moe import mori_v2_prepare_finalize as mv2

TOKENS, TOPK, EPR, WORLD, MAX_TOKENS = 4, 2, 3, 4, 16


def _mega(dispatch="mori", combine="fused_scatter"):
    recipe = SimpleNamespace(
        dispatch=SimpleNamespace(value=dispatch),
        combine=SimpleNamespace(value=combine),
    )
    return SimpleNamespace(recipe=recipe, max_tokens_per_rank=MAX_TOKENS)


def _forward(monkeypatch, mega, context):
    calls = []

    def triton_mega_moe(*args, **kwargs):
        calls.append(kwargs)
        return args[1]

    monkeypatch.setitem(
        sys.modules,
        "atom.model_ops.fused_moe_triton",
        SimpleNamespace(triton_mega_moe=triton_mega_moe),
    )
    monkeypatch.setattr(
        mv2, "get_forward_context", lambda: SimpleNamespace(context=context)
    )
    monkeypatch.setattr(mv2, "get_dp_group", lambda: SimpleNamespace(world_size=2))
    prepare_finalize = SimpleNamespace(mega=mega, num_dispatchers=lambda: WORLD)
    kernel = mv2.MoriV2ModularKernel(prepare_finalize)
    experts = {
        name: torch.zeros(1)
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_scale",
            "w2_scale",
            "w13_swizzle_layout",
            "w2_swizzle_layout",
        )
    }
    kernel.forward(
        torch.zeros(TOKENS, 8),
        torch.zeros(EPR, 8, 8),
        torch.zeros(EPR, 8, 8),
        torch.zeros(TOKENS, TOPK),
        torch.zeros(TOKENS, TOPK, dtype=torch.int32),
        moe_extra_args={"triton_experts": experts},
    )
    return calls


def test_decode_step_reaches_triton_mega_moe(monkeypatch):
    context = SimpleNamespace(
        running_tokens_across_dp=[3, 5, 2, 4],
        running_tokens=4,
        is_prefill=False,
        running_tokens_are_unified=True,
    )
    (call,) = _forward(monkeypatch, _mega(), context)
    # What the group sent, below the arena's WORLD * MAX_TOKENS rows.
    assert call["recv_token_bound"] == 14
    assert call["m_eff"] == 4 * 2
    assert call["n_local_experts"] == EPR


def test_no_context_keeps_the_full_arena(monkeypatch):
    (call,) = _forward(monkeypatch, _mega(), None)
    assert call["recv_token_bound"] is None and call["m_eff"] is None


@pytest.mark.parametrize(
    "dispatch,combine,message",
    [
        ("compact", "fused_scatter", "ATOM_MEGA_STAGE1_FUSED=0"),
        ("mori", "mori", "ATOM_MEGA_STAGE2_FUSED=1"),
    ],
)
def test_unsupported_pairings_are_refused(monkeypatch, dispatch, combine, message):
    with pytest.raises(RuntimeError, match=message):
        _forward(monkeypatch, _mega(dispatch, combine), None)
