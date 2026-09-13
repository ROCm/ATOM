# SPDX-License-Identifier: MIT
"""Shared AITER GEMMs preserve routing and activation contracts across graph replay."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.model_ops.blockscale import native_quant_linear, quantize_fp4
from atom.model_ops.deepseek_v41.moe import Expert
from atom.model_ops.deepseek_v41.moe_aiter import AiterExperts
from atom.model_ops.deepseek_v41.moe_eager import execute_experts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ROCm GPU required"
)


class Projection(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight, self.weight_scale = quantize_fp4(
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.03
        )

    def forward(self, x, otype=torch.bfloat16):
        return native_quant_linear(
            x, self.weight, self.weight_scale, weight_group_rows=1, dtype=otype
        )


def experts(first):
    return {
        str(i): Expert(
            Projection(192, 128),
            Projection(128, 192),
            Projection(192, 128),
            swiglu_limit=10.0,
        )
        for i in range(first, first + 4)
    }


@pytest.mark.parametrize("tokens", [0, 1, 3, 17, 32, 513])
@pytest.mark.parametrize("first", [0, 4])
def test_aiter_matches_eager_with_remote_routes(tokens, first):
    torch.manual_seed(62)
    modules = experts(first)
    adapted = AiterExperts(modules, 8)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    indices = torch.rand(tokens, 8, device="cuda").topk(4, dim=-1).indices
    weights = torch.rand(tokens, 4, device="cuda")
    expected = execute_experts(
        x, weights, indices, modules, 8, SimpleNamespace(world_size=1)
    )
    actual = adapted(x, weights, indices)
    assert torch.isfinite(actual).all()
    error = (actual - expected).norm()
    assert error <= 0.005 * expected.norm().clamp_min(1e-30)


def test_aiter_graph_reads_new_expert_ids():
    torch.manual_seed(45)
    modules = experts(0)
    adapted = AiterExperts(modules, 8)
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(3, 4, device="cuda")
    indices = torch.tensor([[0, 1, 2, 3], [7, 6, 4, 5], [0, 7, 1, 6]], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        adapted(x, weights, indices)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = adapted(x, weights, indices)
        for shift in [1, 3, 7]:
            indices.copy_((indices + shift) % 8)
            x.normal_()
            graph.replay()
            expected = adapted(x, weights, indices)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.cuda.current_stream().wait_stream(stream)


def test_aiter_rejects_incomplete_scale_rows_before_pointer_dispatch():
    modules = experts(0)
    modules["1"].w2.weight_scale = modules["1"].w2.weight_scale[:, :-1].contiguous()
    with pytest.raises(ValueError, match="group32 weights"):
        AiterExperts(modules, 8)


def test_aiter_native_weight_arenas_preserve_eager_views():
    modules = experts(0)
    original = {
        (name, projection, field): getattr(getattr(module, projection), field)
        .view(torch.uint8)
        .clone()
        for name, module in modules.items()
        for projection in ("w1", "w2", "w3")
        for field in ("weight", "weight_scale")
    }
    adapted = AiterExperts(modules, 8)
    for (name, projection, field), expected in original.items():
        current = getattr(getattr(modules[name], projection), field)
        torch.testing.assert_close(current.view(torch.uint8), expected, rtol=0, atol=0)
        arena = getattr(
            adapted,
            ("down" if projection == "w2" else "up")
            + ("_scale" if field == "weight_scale" else ""),
        )
        assert (
            current.untyped_storage().data_ptr() == arena.untyped_storage().data_ptr()
        )


def test_aiter_no_local_routes_return_zero():
    adapted = AiterExperts(experts(0), 8)
    x = torch.randn(3, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.ones(3, 4, device="cuda")
    ids = torch.tensor([[4, 5, 6, 7]] * 3, device="cuda")
    assert torch.equal(
        adapted(x, weights, ids), torch.zeros_like(x, dtype=torch.float32)
    )
