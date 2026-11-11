# SPDX-License-Identifier: MIT
"""Both attention paths retain wo_a's fused quantization after composition."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("fused_attention", [False, True])
@pytest.mark.parametrize("fused_woa", [False, True])
@pytest.mark.parametrize("leading_shape", [(1, 3), (2, 2)])
def test_attention_output_projection(fused_attention, fused_woa, leading_shape):
    source = (
        Path(__file__).resolve().parents[1] / "atom/models/deepseek_v41/attention.py"
    )
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Attention"
    )
    methods = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_project_out", "_project_quantized_out"}
    ]
    assert len(methods) == 2
    rows = torch.Size(leading_shape).numel()
    groups, rank, head_dim = 2, 64, 32
    x = torch.empty(rows, groups, 8 * head_dim, dtype=torch.float8_e4m3fn)
    scale = torch.empty(rows, groups, 8 * head_dim // 32, dtype=torch.float8_e8m0fnu)
    y = torch.arange(rows * groups * rank).reshape(rows, groups, rank).bfloat16()
    y8 = (y % 16).to(torch.float8_e4m3fn)
    y_scale = torch.full((rows, groups, rank // 32), 127, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    inverse = Mock(return_value=(x, scale))
    plain = Mock(return_value=y)
    fused = Mock(return_value=(y8, y_scale))
    scope = {
        "inverse_rope_group_quant": inverse,
        "batched_gemm_a8w8_mxscale_bpreshuffle": plain,
        "batched_gemm_a8w8_mxscale_bpreshuffle_fused_quant": fused,
    }
    exec(  # noqa: S102 - execute production methods without GPU-only imports.
        compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"),
        scope,
    )
    model = SimpleNamespace(
        heads=16,
        head_dim=head_dim,
        groups=groups,
        o_rank=rank,
        _fuse_woa_quant=fused_woa,
        wo_a=SimpleNamespace(
            weight=torch.empty(groups * rank, 8 * head_dim),
            weight_scale=torch.empty(groups * rank // 32, 8 * head_dim // 32),
        ),
        wo_b=Mock(return_value=object()),
    )
    project = scope["_project_quantized_out"]
    model._project_quantized_out = project.__get__(model)
    if fused_attention:
        result = model._project_quantized_out(x, scale, leading_shape, torch.bfloat16)
        inverse.assert_not_called()
    else:
        output = torch.empty(
            *leading_shape, model.heads, head_dim, dtype=torch.bfloat16
        )
        rope = SimpleNamespace(cos_cache=object(), sin_cache=object())
        positions = torch.arange(rows).view(leading_shape)
        result = scope["_project_out"](model, output, rope, positions)
        inverse.assert_called_once()
        assert inverse.call_args.args[0].shape == (rows, model.heads, head_dim)
        assert torch.equal(inverse.call_args.args[1], positions.flatten())
    called, unused = (fused, plain) if fused_woa else (plain, fused)
    called.assert_called_once()
    unused.assert_not_called()
    args, kwargs = called.call_args
    assert args[0] is x and args[2] is scale
    assert args[1].shape == (groups, rank, 8 * head_dim)
    assert args[1].data_ptr() == model.wo_a.weight.data_ptr()
    assert args[3].shape == (groups, rank // 32, 8 * head_dim // 32)
    assert args[3].data_ptr() == model.wo_a.weight_scale.data_ptr()
    assert kwargs == ({} if fused_woa else {"dtype": torch.bfloat16})
    model.wo_b.assert_called_once()
    args, kwargs = model.wo_b.call_args
    expected = y8 if fused_woa else y
    assert args[0].shape == (*leading_shape, groups * rank)
    assert args[0].dtype == expected.dtype
    assert torch.equal(
        args[0].view(torch.uint8), expected.view(*leading_shape, -1).view(torch.uint8)
    )
    if fused_woa:
        assert kwargs["x_scale"].shape == (*leading_shape, groups * rank // 32)
        assert kwargs["x_scale"].data_ptr() == y_scale.data_ptr()
    else:
        assert not kwargs
    assert result is model.wo_b.return_value
