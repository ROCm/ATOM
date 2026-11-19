# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise V-up's output layout on CPU with only the GPU BMMs stubbed."""

import ast
from pathlib import Path

import pytest
import torch


@pytest.fixture
def projection():
    # Module-level AITER/Triton imports prevent a normal import on the CPU gate.
    # Execute the production method bodies, removing only the trace decorator.
    path = Path(__file__).resolve().parents[1] / "atom/model_ops/attention_mla.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MLAAttention"
    )
    names = {"_v_up_proj", "_v_up_proj_and_o_proj"}
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in methods} == names
    for method in methods:
        method.decorator_list = []
    namespace = {"torch": torch}
    exec(  # noqa: S102 - execute only the checked-in production method bodies
        compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    attn = type("Projection", (), {name: namespace[name] for name in names})()
    attn.num_heads = 3
    attn.kv_lora_rank = 8
    attn.v_head_dim = 4
    return attn, namespace


def _stub_bmm(namespace, branch, weights, scale):
    seen = {}

    def write_output(x, weight, output):
        assert weight is weights
        seen.update(x=x, output=output)
        result = torch.bmm(x.float(), weight.float().transpose(1, 2))
        output.copy_(result.transpose(0, 1))
        return output

    def bf16(x, weight, *, YQ):
        seen["YQ"] = YQ
        write_output(x, weight, YQ.transpose(0, 1))
        return YQ

    def fp8(x, weight, weight_scale, *, group_size, YQ, transpose_bm):
        assert weight_scale is scale
        assert group_size == 128 and transpose_bm
        return write_output(x, weight, YQ)

    def fp4(x, weight, weight_scale, *, y, transpose_bm, prequant, y_scale):
        assert weight_scale is scale
        assert transpose_bm and prequant and y_scale is None
        return write_output(x, weight, y)

    def unexpected(*args, **kwargs):
        pytest.fail("V-up dispatched to the wrong BMM branch")

    kernels = {
        "bf16": "_aiter_triton_bf16_bmm",
        "fp8": "_aiter_triton_fp8_bmm",
        "fp4": "batched_gemm_a16wfp4",
    }
    namespace.update({name: unexpected for name in kernels.values()})
    namespace[kernels[branch]] = {"bf16": bf16, "fp8": fp8, "fp4": fp4}[branch]
    namespace["is_rocm_aiter_fp4bmm_enabled"] = lambda: branch == "fp4"
    return seen


def _inputs(tokens, heads, rank, v_dim, noncontiguous=False):
    width = 2 * rank if noncontiguous else rank
    x = (torch.arange(tokens * heads * width) % 7).to(torch.bfloat16)
    x = x.reshape(tokens, heads, width)
    if noncontiguous:
        x = x[..., 1::2]
        assert not x.is_contiguous() and x.storage_offset() == 1
    weights = (torch.arange(heads * v_dim * rank) % 3).to(torch.bfloat16)
    return x, weights.reshape(heads, v_dim, rank)


@pytest.mark.parametrize("branch", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_v_up_output_is_contiguous_and_o_proj_receives_a_view(
    projection, branch, noncontiguous
):
    attn, namespace = projection
    tokens, heads, rank, v_dim = 5, attn.num_heads, attn.kv_lora_rank, attn.v_head_dim
    x, weights = _inputs(tokens, heads, rank, v_dim, noncontiguous)
    scale = None if branch == "bf16" else torch.ones(heads)
    attn.W_V, attn.W_V_scale = weights, scale
    seen = _stub_bmm(namespace, branch, weights, scale)

    def o_proj(flat):
        output = seen["output"]
        assert output.shape == (tokens, heads, v_dim)
        assert output.dtype == torch.bfloat16
        assert output.stride() == (heads * v_dim, v_dim, 1)
        assert output.is_contiguous()
        assert flat.shape == (tokens, heads * v_dim)
        assert flat.is_contiguous()
        assert flat.data_ptr() == output.data_ptr()
        if branch == "bf16":
            yq = seen["YQ"]
            assert yq.shape == (heads, tokens, v_dim)
            assert yq.stride() == (v_dim, heads * v_dim, 1)
            assert yq.data_ptr() == output.data_ptr()
        return flat

    attn.o_proj = o_proj
    result = attn._v_up_proj_and_o_proj(x)
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16).reshape(tokens, -1))


@pytest.mark.parametrize("branch", ["bf16", "fp8", "fp4"])
def test_v_up_output_uses_the_dcp_head_and_weight_overrides(projection, branch):
    attn, namespace = projection
    tokens, heads = 5, 2 * attn.num_heads
    x, weights = _inputs(tokens, heads, attn.kv_lora_rank, attn.v_head_dim, True)
    scale = None if branch == "bf16" else torch.ones(heads)
    attn.W_V = torch.empty(attn.num_heads, attn.v_head_dim, attn.kv_lora_rank)
    attn.W_V_scale = None if branch == "bf16" else torch.zeros(attn.num_heads)
    seen = _stub_bmm(namespace, branch, weights, scale)

    result = attn._v_up_proj(x, W_V=weights, W_V_scale=scale, num_heads=heads)

    assert result.shape == (tokens, heads, attn.v_head_dim)
    assert result.is_contiguous()
    assert result.data_ptr() == seen["output"].data_ptr()
    expected = torch.einsum("tnl,nvl->tnv", x.float(), weights.float())
    assert torch.equal(result, expected.to(torch.bfloat16))
