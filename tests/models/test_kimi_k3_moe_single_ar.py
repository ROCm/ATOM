# SPDX-License-Identifier: MIT
"""Kimi-K3 latent MoE at TP > 1: the routed experts and the shared down_proj
write their partial sums into one buffer, which one all-reduce reduces; and
`moe_forward` writes its output into a caller's tensor."""

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from aiter_stub import atom_without_aiter
from torch import nn

HIDDEN, LATENT, EXPERTS, SHARED_INTER, TOKENS, TP = 16, 8, 6, 4, 5, 2


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fn = lambda x: x

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@pytest.fixture
def block(monkeypatch):
    """A real `KimiSparseMoeBlock` at TP 2 with real BF16 linears, after the
    post-load pass merged its front GEMMs; the experts and the routed norm are
    stubs, and the all-reduce stands for ranks that hold the same partials.
    Small integer weights and inputs keep every GEMM exact."""
    pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")
    from atom.model_ops import kimi_k3 as kimi_ops
    from atom.model_ops import linear
    from atom.models.kimi_k3 import model as kimi

    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    for name in ("FusedMoE", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: TP)
    atom_config = SimpleNamespace(
        enable_tbo=False,
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    monkeypatch.setattr(kimi, "get_current_atom_config", lambda: atom_config)
    config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        routed_expert_hidden_size=LATENT,
        moe_intermediate_size=SHARED_INTER,
        num_shared_experts=2,
        moe_renormalize=True,
        moe_router_activation_func="sigmoid",
        latent_moe_use_norm=True,
        rms_norm_eps=1e-6,
        hidden_act="situ",
    )
    moe = kimi.KimiSparseMoeBlock(config, prefix="layers.1.block_sparse_moe")
    gen = torch.Generator().manual_seed(0)
    for p in moe.parameters():
        p.data = torch.randint(-2, 3, p.shape, generator=gen).to(p.dtype)

    def experts(routed_input, router_logits, out=None):
        y = routed_input + router_logits.sum(-1, keepdim=True)
        return y if out is None else out.copy_(y)

    def situ(x, beta, linear_beta, quant=False):
        gate, up = x.chunk(2, dim=-1)
        return gate - up, None

    def mm(self, inp, weights, bias=None, otype=None, out=None, **_):
        y = F.linear(inp.double(), weights.double()).to(otype or inp.dtype)
        return y if out is None else out.copy_(y)

    reduces = []

    def all_reduce(x):
        reduces.append(x)
        return x * TP

    moe.experts.fn = experts
    monkeypatch.setattr(kimi_ops, "situ_and_mul_maybe_quant", situ)
    monkeypatch.setattr(type(kimi.tgemm), "mm", mm)
    monkeypatch.setattr(kimi, "tensor_model_parallel_all_reduce", all_reduce)
    for _, module in moe.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()
    assert moe.front_proj.merged_weight is not None
    x = torch.randint(-2, 3, (TOKENS, HIDDEN), generator=gen).bfloat16()
    return SimpleNamespace(moe=moe, x=x, reduces=reduces)


def test_one_all_reduce_gives_what_two_did(block, monkeypatch):
    moe = block.moe
    routed, shared = moe.split_moe_forward(block.x)
    (partials,) = block.reduces
    block.reduces.clear()

    # A shared down_proj that cannot write into a caller's tensor keeps the
    # two all-reduces.
    monkeypatch.setattr(moe.shared_experts.down_proj, "supports_out", lambda: False)
    want_routed, want_shared = moe.split_moe_forward(block.x)
    latent_partial, shared_partial = block.reduces

    assert latent_partial.shape == (TOKENS, LATENT)
    assert torch.equal(
        partials, torch.cat([latent_partial.flatten(), shared_partial.flatten()])
    )
    assert torch.equal(routed, want_routed)
    assert torch.equal(shared, want_shared)


@pytest.fixture(scope="module")
def moe_module():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.moe")


class _Method:
    """A quant method; `apply` records its keywords."""

    def __init__(self, takes_out: bool) -> None:
        self.apply_takes_out = takes_out
        self.calls = []

    def select_experts(self, layer, x, router_logits):
        return router_logits, router_logits.long()

    def place_experts(self, layer, topk_weights, topk_ids, gathered):
        return topk_weights, topk_ids

    def apply(self, *, x, **kwargs):
        self.calls.append(kwargs)
        y = x * 2
        return y if kwargs.get("out") is None else kwargs["out"].copy_(y)


@pytest.mark.parametrize("takes_out", [True, False])
def test_moe_forward_writes_out(moe_module, monkeypatch, takes_out):
    """`out` reaches a method that takes it and no other, and holds the output
    either way; a call without it returns the method's own tensor."""
    moe = moe_module
    method = _Method(takes_out)
    layer = SimpleNamespace(
        quant_method=method,
        sp_mega_moe=None,
        moe_parallel_config=SimpleNamespace(dp_logical_ratio=1, pcp_merge=False),
        reduce_results=False,
        global_num_experts=4,
        expert_map=None,
        activation=None,
        apply_router_weight_on_input=False,
        prefix="experts",
    )
    layer.forward_impl = partial(moe.FusedMoE.forward_impl, layer)
    layer._run_experts = partial(moe.FusedMoE._run_experts, layer)
    context = SimpleNamespace(compilation_config=SimpleNamespace())
    context.compilation_config.static_forward_context = {"experts": layer}
    monkeypatch.setattr(moe, "get_current_atom_config", lambda: context)
    monkeypatch.setattr(moe, "get_dp_group", lambda: SimpleNamespace(world_size=1))
    step = SimpleNamespace(is_dummy_run=False, pcp_split=False)
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=step)
    )
    monkeypatch.setattr(moe, "pass_scheduled_tokens_gpu", lambda rows: None)
    x, router_logits = torch.randn(3, 4), torch.rand(3, 2)

    want = moe.moe_forward(x, router_logits, "experts")
    assert "out" not in method.calls[-1]
    out = torch.empty_like(x)
    assert moe.moe_forward(x, router_logits, "experts", out) is out
    assert method.calls[-1].get("out") is (out if takes_out else None)
    assert torch.equal(out, want)
