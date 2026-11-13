# SPDX-License-Identifier: MIT
"""Kimi-K3 latent MoE: after the loader's post-load pass, `split_moe_forward`
runs the shared gate_up, the router gate and the latent down-projection as one
GEMM over their stacked weights, and every consumer sees what the three GEMMs
gave it."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("aiter", reason="atom.models.kimi_k3.model imports aiter")

from atom.model_ops import kimi_k3 as kimi_ops
from atom.model_ops import linear
from atom.models.kimi_k3 import model as kimi

HIDDEN, LATENT, EXPERTS, SHARED_INTER, TOKENS = 16, 8, 6, 4, 5


class _Stub(nn.Module):
    """Stands in for a sub-module class; the test sets `fn` after `__init__`."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.fn = lambda x: x

    def forward(self, *args):
        return self.fn(*args)


@pytest.fixture
def block(monkeypatch):
    """A real `KimiSparseMoeBlock` at TP 1 with real BF16 linears; the experts
    and the routed norm are stubs. Small integer weights and inputs keep every
    GEMM exact, so the merged and separate paths must agree bit for bit."""
    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    for name in ("FusedMoE", "RMSNorm"):
        monkeypatch.setattr(kimi, name, _Stub)
    monkeypatch.setattr(kimi, "get_tensor_model_parallel_world_size", lambda: 1)
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

    seen = {}

    def experts(routed_input, router_logits):
        seen["routed_input"], seen["router_logits"] = routed_input, router_logits
        return routed_input + router_logits.sum(-1, keepdim=True)

    def situ(x, beta, linear_beta, quant=False):
        seen["gate_up"] = x
        gate, up = x.chunk(2, dim=-1)
        return gate - up, None

    moe.experts.fn = experts
    monkeypatch.setattr(kimi_ops, "situ_and_mul_maybe_quant", situ)
    gemms = []

    def mm(self, inp, weights, bias=None, otype=None, **_):
        gemms.append(weights)
        return F.linear(inp.double(), weights.double()).to(otype or inp.dtype)

    monkeypatch.setattr(type(kimi.tgemm), "mm", mm)
    x = torch.randint(-2, 3, (TOKENS, HIDDEN), generator=gen).bfloat16()
    return SimpleNamespace(moe=moe, x=x, seen=seen, gemms=gemms)


def _post_load(model):
    """The loader's post-load pass: every module's hook, parents first."""
    for _, module in model.named_modules():
        if hasattr(module, "process_weights_after_loading"):
            module.process_weights_after_loading()


def _run(b):
    b.seen.clear()
    b.gemms.clear()
    routed, shared = b.moe.split_moe_forward(b.x)
    return routed, shared, dict(b.seen), list(b.gemms)


def test_one_front_gemm_feeds_what_three_did(block):
    moe = block.moe
    want_routed, want_shared, want_seen, want_gemms = _run(block)

    _post_load(moe)
    merged = moe.front_proj.merged_weight
    assert merged is not None
    for lin in (moe.shared_experts.gate_up_proj, moe.gate, moe.routed_expert_down_proj):
        assert lin.weight.untyped_storage().data_ptr() == merged.data_ptr()
    routed, shared, seen, gemms = _run(block)

    assert merged.shape == (2 * 2 * SHARED_INTER + EXPERTS + LATENT, HIDDEN)
    assert len(gemms) == len(want_gemms) - 2
    assert sum(w is merged for w in gemms) == 1
    for key in ("router_logits", "routed_input", "gate_up"):
        assert torch.equal(seen[key], want_seen[key]), key
    # The fused MoE kernels take their tuned path only on contiguous rows.
    assert seen["routed_input"].is_contiguous()
    assert torch.equal(routed, want_routed)
    assert torch.equal(shared, want_shared)


def test_layout_a_linear_settles_on_load_keeps_three_gemms(block):
    """A linear whose own hook pads its output rows is no plain GEMM any more:
    the merge, run after it, sees that and leaves the block on its three GEMMs."""
    moe = block.moe
    want_routed, want_shared, _, want_gemms = _run(block)

    moe.gate.output_row_alignment = 4  # EXPERTS rows pad to 8 in its hook
    _post_load(moe)
    routed, shared, seen, gemms = _run(block)

    assert moe.gate.is_output_padded
    assert moe.front_proj.merged_weight is None
    assert len(gemms) == len(want_gemms)
    assert seen["router_logits"].shape == (TOKENS, EXPERTS)
    assert torch.equal(routed, want_routed)
    assert torch.equal(shared, want_shared)
