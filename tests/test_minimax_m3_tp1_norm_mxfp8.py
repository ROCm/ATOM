# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3 at TP 1: the norm in front of a native MXFP8 linear writes that
linear's E4M3 + E8M0 input in the same aiter launch."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest
import torch

pytest.importorskip("aiter")

from aiter import dtypes

from atom.model_ops import blockscale, layernorm
from atom.models.minimax_m3 import model as m3_model


@pytest.mark.parametrize("emit_bf16", [False, True])
def test_helper_is_one_aiter_mxfp8_norm(monkeypatch, emit_bf16):
    """Residual add + Gemma norm + group-32 E8M0 quant in one aiter call; the
    bf16 row only when asked."""
    fused = Mock()
    monkeypatch.setattr(layernorm, "fused_qk_rmsnorm_group_quant", fused)
    hidden_states = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(64)), variance_epsilon=1e-6
    )

    result = layernorm.gemma_rms_norm_mxfp8_quant(
        hidden_states, residual, norm, emit_bf16=emit_bf16
    )

    assert fused.call_count == 1
    (values, scales, x, weight, eps), kwargs = fused.call_args
    assert x.shape == (6, 64) and x.data_ptr() == hidden_states.data_ptr()
    assert weight.data_ptr() == norm.weight.data_ptr() and eps == 1e-6
    assert values.shape == (6, 64) and values.dtype == dtypes.fp8
    assert "transpose_scale" not in kwargs  # row-major scales
    assert scales.dtype == torch.float8_e8m0fnu and scales.shape == (6, 2)
    assert kwargs["q_residual"].data_ptr() == residual.data_ptr()
    assert kwargs["group_size"] == 32 and kwargs["gemma_norm"] is True
    assert (kwargs["q_out_unquantized"] is not None) == emit_bf16
    assert len(result) == (4 if emit_bf16 else 3)
    assert result[0].shape == (2, 3, 64) and result[1].shape == (2, 3, 2)
    assert result[-1].data_ptr() == kwargs["q_res_out"].data_ptr()
    if emit_bf16:
        assert result[2].data_ptr() == kwargs["q_out_unquantized"].data_ptr()


def test_moe_feeds_the_shared_mlp_its_given_mxfp8_rows():
    """With the post-attention norm's MXFP8 rows given, the shared MLP takes
    them (and their scales); the router and the routed experts the bf16 rows."""
    hidden = torch.randn(4, 64)
    shared_x, shared_x_scale = torch.zeros(4, 64), torch.zeros(4, 2)
    moe = SimpleNamespace(
        gate=MagicMock(return_value="logits"),
        experts=MagicMock(return_value=torch.zeros(4, 64)),
        fuse_shared_experts=False,
        routed_scaling_factor=1.0,
        shared_experts=MagicMock(return_value=torch.ones(4, 64)),
    )

    out = m3_model.MiniMaxM3MoE.forward(moe, hidden, shared_x, shared_x_scale)

    assert torch.equal(out, torch.ones(4, 64))
    kwargs = moe.experts.call_args.kwargs
    assert (
        kwargs["hidden_states"].data_ptr() == hidden.data_ptr()
        and kwargs["router_logits"] == "logits"
    )
    (x,), kwargs = moe.shared_experts.call_args
    assert x.data_ptr() == shared_x.data_ptr()
    assert kwargs["x_scale"].data_ptr() == shared_x_scale.data_ptr()


def _ulps(a: torch.Tensor, b: torch.Tensor) -> int:
    """Largest distance in bf16 units of last place between a and b."""
    a, b = (x.view(torch.int16).int() for x in (a, b))
    a, b = (torch.where(x < 0, -(x & 0x7FFF), x) for x in (a, b))
    return (a - b).abs().max().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 255, 256, 4096, 32768])
def test_helper_is_quantize_fp8_of_its_normed_row(rows):
    """M3.1's hidden 6144 at decode and prefill row counts: the fp8 values and
    scales are quantize_fp8 of the bf16 row the launch writes, and that row is
    the norm's (within 1 bf16 ulp: the norm's launch shape is aiter's)."""
    torch.manual_seed(rows)
    # bf16 like the model's weights (aiter wants the weight in the input dtype)
    norm = layernorm.GemmaRMSNorm(6144, eps=1e-6).to("cuda", torch.bfloat16)
    norm.weight.data.normal_(0.0, 0.1)
    x = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda") * 3
    residual = torch.randn_like(x)

    normed, residual_ref = norm.forward_cuda(x, residual)
    values, scales, row, residual_out = layernorm.gemma_rms_norm_mxfp8_quant(
        x, residual, norm, emit_bf16=True
    )
    values_ref, scales_ref = blockscale.quantize_fp8(row)

    assert torch.equal(values.view(torch.uint8), values_ref.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), scales_ref.view(torch.uint8))
    assert _ulps(row, normed) <= 1
    assert torch.equal(residual_out.view(torch.int16), residual_ref.view(torch.int16))
    # the launch without the bf16 row (input norm, dense MLP) writes the same
    values2, scales2, residual_out2 = layernorm.gemma_rms_norm_mxfp8_quant(
        x, residual, norm
    )
    assert torch.equal(values2.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(scales2.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(residual_out2.view(torch.int16), residual_out.view(torch.int16))


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize(
    "flag, tp, mxfp8, on",
    [
        ("1", 1, True, True),
        ("0", 1, True, False),
        ("1", 2, True, False),
        ("1", 1, False, False),
    ],
)
def test_layer_takes_the_mxfp8_norm_only_at_tp1_with_the_flag(
    monkeypatch, moe, flag, tp, mxfp8, on
):
    """ATOM_M3_TP1_NORM_MXFP8=1 at TP 1 with native MXFP8 consumers: both norms
    take the one launch; qkv and the dense gate_up get its scales, a MoE layer
    hands the bf16 row to the router / routed experts and the MXFP8 row to the
    shared expert. Otherwise both norms keep the two-launch path."""
    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8", flag)
    monkeypatch.setattr(m3_model, "get_tensor_model_parallel_world_size", lambda: tp)
    x, bf16, fp8 = torch.zeros(2, 64), torch.ones(2, 64), torch.full((2, 64), 2.0)
    one_launch = Mock(
        side_effect=lambda h, r, norm, emit_bf16=False: (
            (fp8, "scale", bf16, x) if emit_bf16 else (fp8, "scale", x)
        )
    )
    two_launch = Mock(return_value=(bf16, x))
    monkeypatch.setattr(m3_model, "gemma_rms_norm_mxfp8_quant", one_launch)
    monkeypatch.setattr(m3_model, "fused_allreduce_gemma_rms_norm", two_launch)
    linear = SimpleNamespace(native_a8_group_rows=32, runs_bf16=not mxfp8)
    mlp = Mock(return_value=x, gate_up_proj=linear)
    takes = m3_model.MiniMaxM3MoE.takes_shared_mxfp8_input
    ffn = Mock(return_value=x, takes_shared_mxfp8_input=lambda: takes(ffn))
    ffn.shared_experts = mlp
    layer = SimpleNamespace(
        self_attn=Mock(return_value=x, qkv_proj=linear),
        mlp=mlp,
        block_sparse_moe=ffn,
        is_moe_layer=moe,
        input_layernorm="in_norm",
        post_attention_layernorm="post_norm",
    )

    m3_model.MiniMaxM3DecoderLayer.forward(layer, None, x, x)

    assert one_launch.call_count == (2 if on else 0)
    assert two_launch.call_count == (0 if on else 2)
    norms = ["in_norm", "post_norm"]
    assert [
        c.args[2] for c in (one_launch if on else two_launch).call_args_list
    ] == norms
    scale = "scale" if on else None
    assert layer.self_attn.call_args.kwargs["hidden_states_scale"] == scale
    if not on:
        assert (ffn if moe else mlp).call_args == call(bf16)
    elif moe:
        assert ffn.call_args == call(bf16, shared_x=fp8, shared_x_scale="scale")
    else:
        assert mlp.call_args == call(fp8, x_scale="scale")
