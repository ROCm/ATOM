# SPDX-License-Identifier: MIT
"""The fused router also writes the MoE input as token-major MX rows, and the
grouped MoE takes them inside `moe_router_quant_rows` (CPU).

`FusedMoE.forward_impl_fused_router` is the production method, imported
against the stub AITER in `tests/aiter_stub.py`; the aiter router, the row
buffers and the hand-off context are replaced by recorders.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs

ROWS, K = 64, 256


@pytest.fixture
def moe_mod():
    with atom_under_stubs() as load:
        yield load("atom.model_ops.moe")


class _Log(list):
    """The ordered event log, with the recorders' knobs on ``state``."""


@pytest.fixture
def calls(moe_mod, monkeypatch):
    """Recorders for the aiter calls; returns the ordered event log."""
    log = _Log()
    rows_obj = SimpleNamespace(quant_rows=("q4", "s4"), quant_rows8=("q8", "s8"))
    state = {"buffers": rows_obj}

    def router(x, w, bias, topk, split_k, **kw):
        log.append(("router", sorted(kw)))
        return (torch.ones(x.shape[0], topk), torch.zeros(x.shape[0], topk))

    def buffers(rows, k, device, topk):
        log.append(("buffers", rows, k, topk))
        return state["buffers"]

    @contextmanager
    def hand_off(topk_ids, rows):
        log.append(("enter", rows is rows_obj))
        yield
        log.append(("exit",))

    monkeypatch.setattr(moe_mod, "_router_split_k", lambda *a: 8)
    monkeypatch.setattr(moe_mod, "_flydsl_moe_router_topk", router)
    monkeypatch.setattr(moe_mod, "_router_quant_rows_buffers", buffers)
    monkeypatch.setattr(moe_mod, "_moe_router_quant_rows", hand_off)
    monkeypatch.setattr(moe_mod, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(moe_mod, "pass_scheduled_tokens_gpu", lambda n: n)
    log.state = state
    return log


@pytest.fixture
def layer(calls):
    """The attributes `forward_impl_fused_router` reads off its layer."""

    def forward_impl(hidden, logits, mxfp8_copy_out=None, routing=None):
        calls.append(("forward_impl", routing is not None, mxfp8_copy_out))
        return hidden

    return SimpleNamespace(
        sp_mega_moe=None,
        top_k=4,
        renormalize=True,
        e_score_correction_bias=None,
        forward_impl=forward_impl,
    )


def test_the_router_rows_go_to_the_grouped_moe(moe_mod, calls, layer):
    moe_mod.FusedMoE.forward_impl_fused_router(
        layer,
        torch.zeros(ROWS, K, dtype=torch.bfloat16),
        torch.zeros(128, K, dtype=torch.bfloat16),
    )
    assert calls == [
        ("buffers", ROWS, K, 4),
        (
            "router",
            [
                "need_renorm",
                "num_valid_rows",
                "quant_rows",
                "quant_rows8",
                "routed_scaling_factor",
            ],
        ),
        ("enter", True),
        ("forward_impl", True, None),
        ("exit",),
    ]


@pytest.mark.parametrize("why", ["no buffers", "mxfp8 copy"])
def test_router_rows_not_taken(moe_mod, calls, layer, why):
    if why == "no buffers":
        calls.state["buffers"] = None
    copy = ("p", "s") if why == "mxfp8 copy" else None
    moe_mod.FusedMoE.forward_impl_fused_router(
        layer,
        torch.zeros(ROWS, K, dtype=torch.bfloat16),
        torch.zeros(128, K, dtype=torch.bfloat16),
        mxfp8_copy_out=copy,
    )
    router = [c for c in calls if c[0] == "router"]
    assert router == [
        ("router", ["need_renorm", "num_valid_rows", "routed_scaling_factor"])
    ]
    assert not [c for c in calls if c[0] == "enter"]
    assert calls[-1] == ("forward_impl", True, copy)
    # Asked for buffers only where no copy is requested.
    assert bool([c for c in calls if c[0] == "buffers"]) == (why == "no buffers")


def test_after_router_hook_runs_before_the_rows_are_handed_off(moe_mod, calls, layer):
    """The routed-first hook still runs right after the router launch, before
    the grouped MoE inside the hand-off."""
    with moe_mod.moe_after_router_hook(lambda: calls.append(("hook",))):
        moe_mod.FusedMoE.forward_impl_fused_router(
            layer,
            torch.zeros(ROWS, K, dtype=torch.bfloat16),
            torch.zeros(128, K, dtype=torch.bfloat16),
        )
    kinds = [c[0] for c in calls]
    assert kinds.index("router") < kinds.index("hook") < kinds.index("enter")
