# SPDX-License-Identifier: MIT
"""Image bias changes expert selection without changing V4 routing weights."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("ROCm router required", allow_module_level=True)

from atom.models.deepseek_v4 import MoE as V4MoE
from atom.models.deepseek_v41.moe import MoE
from atom.utils import forward_context


@pytest.fixture
def metadata(monkeypatch):
    metadata = SimpleNamespace(image_mask=None)
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        forward_context.ForwardContext(attn_metadata=metadata),
    )
    return metadata


@pytest.fixture
def moe(metadata):
    layer = MoE.__new__(MoE)
    nn.Module.__init__(layer)
    layer.gate = nn.Module()
    layer.gate.e_score_correction_bias = torch.linspace(-2, 2, 384, device="cuda")
    layer.gate.bias_vl = -layer.gate.e_score_correction_bias
    layer.routed_scaling_factor = 1.5
    layer.experts = SimpleNamespace(custom_routing_function=layer._topk)
    return layer


@pytest.mark.parametrize("rows", [1, 17, 129])
@pytest.mark.parametrize("population", ["text", "image", "mixed"])
def test_image_router_against_unbiased_score_oracle(moe, metadata, rows, population):
    torch.manual_seed(17)
    logits = torch.randn(rows, 384, device="cuda")
    hidden = torch.empty(rows, 8, device="cuda")
    mask = torch.arange(rows, device="cuda") % 2 == 0
    if population != "mixed":
        mask.fill_(population == "image")
    metadata.image_mask = mask
    weights, ids = moe._topk(hidden, logits, 6, True)
    scores = torch.nn.functional.softplus(logits).sqrt()
    bias = torch.where(
        mask[:, None], moe.gate.bias_vl, moe.gate.e_score_correction_bias
    )
    expected_ids = (scores + bias).topk(6, dim=-1).indices
    expected_weights = scores.gather(-1, expected_ids)
    expected_weights *= 1.5 / (expected_weights.sum(-1, keepdim=True) + 1e-20)
    order = ids.argsort(-1)
    expected_order = expected_ids.argsort(-1)
    torch.testing.assert_close(
        ids.gather(-1, order).long(), expected_ids.gather(-1, expected_order)
    )
    torch.testing.assert_close(
        weights.gather(-1, order), expected_weights.gather(-1, expected_order)
    )


@pytest.mark.parametrize("fail", [False, True])
def test_fixed_hook_reads_live_mask_and_keeps_v4_executor(
    moe, metadata, monkeypatch, fail
):
    calls = []
    hidden = torch.randn(1, 17, 8, device="cuda")
    logits = torch.randn(17, 384, device="cuda")
    route = moe.experts.custom_routing_function

    def execute(self, flat):
        assert self.experts.custom_routing_function is route
        _, ids = route(flat, logits, 6, True)
        calls.append(ids.clone())
        if metadata.image_mask is not None and fail:
            raise RuntimeError("executor failed")
        return flat + 1

    monkeypatch.setattr(V4MoE, "forward", execute)
    metadata.image_mask = torch.ones((1, 17), dtype=torch.bool, device="cuda")
    if fail:
        with pytest.raises(RuntimeError, match="executor failed"):
            moe(hidden)
    else:
        torch.testing.assert_close(moe(hidden), hidden + 1)
    metadata.image_mask = None
    torch.testing.assert_close(moe(hidden), hidden + 1)
    assert moe.experts.custom_routing_function is route
    assert len(calls) == 2 and not torch.equal(calls[0], calls[1])


@pytest.mark.parametrize("fail", [False, True])
def test_offline_forward_scopes_the_same_routing_metadata(metadata, fail):
    from atom.models.deepseek_v41.model import DeepseekV41ForCausalLM

    mask = torch.tensor([[False, True]], device="cuda")
    old = forward_context.get_forward_context().attn_metadata
    finished = []
    cache = SimpleNamespace(
        position=0, begin_step=lambda *args: object(), finish_step=finished.append
    )

    def hidden(token_ids, cache, step, embeddings, **kwargs):
        assert forward_context.get_forward_context().attn_metadata.image_mask is mask
        assert kwargs["image_mask"] is mask
        if fail:
            raise RuntimeError("layer failed")
        return torch.ones((1, 2, 8), device="cuda")

    model = SimpleNamespace(
        forward_hidden=hidden,
        norm=lambda x: x,
        head=SimpleNamespace(get_logits=lambda x: x),
    )
    tokens = torch.zeros((1, 2), dtype=torch.int64, device="cuda")
    if fail:
        with pytest.raises(RuntimeError, match="layer failed"):
            DeepseekV41ForCausalLM.forward(model, tokens, cache, image_mask=mask)
    else:
        result = DeepseekV41ForCausalLM.forward(model, tokens, cache, image_mask=mask)
        assert result.shape == (1, 8) and len(finished) == 1
    assert forward_context.get_forward_context().attn_metadata is old
