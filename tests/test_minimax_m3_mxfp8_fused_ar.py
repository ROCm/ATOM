# SPDX-License-Identifier: MIT
"""MiniMax-M3 native MXFP8 fused all-reduce: eligibility and the helper (CPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_without_aiter
from conftest import gpu_stack_import


@pytest.fixture
def mods():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            layernorm=load("atom.model_ops.layernorm"),
            blockscale=load("atom.model_ops.blockscale"),
        )


def _linear(**kw):
    base = {"params_dtype": torch.float8_e4m3fn, "native_a8_group_rows": 1}
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize(
    "kw, eligible",
    [
        ({}, True),
        ({"params_dtype": torch.float8_e4m3fnuz}, True),
        ({"quant": "per_Token"}, False),
        ({"params_dtype": torch.float4_e2m1fn_x2}, False),
        ({"native_a8_group_rows": None}, False),
        ({"runs_bf16": True}, False),
    ],
)
def test_only_native_group32_fp8_linears_consume_mxfp8(kw, eligible):
    model = gpu_stack_import("atom.models.minimax_m3.model")  # imports FlyDSL
    quant = getattr(model.QuantType, kw.pop("quant", "per_1x32"))
    assert model._linear_consumes_mxfp8(_linear(quant_type=quant, **kw)) is eligible


@pytest.mark.parametrize(
    "api, gfx, want",
    [(True, "gfx1250", True), (False, "gfx1250", False), (True, "gfx950", False)],
)
def test_the_fused_op_is_used_where_aiter_has_it_on_gfx1250(
    mods, monkeypatch, api, gfx, want
):
    ln = mods.layernorm
    op = Mock() if api else None
    monkeypatch.setattr(
        ln, "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant", op
    )
    monkeypatch.setattr(ln, "get_gfx_runtime", lambda: gfx)
    assert ln.fused_allreduce_gemma_rms_norm_mxfp8_supported() is want


@pytest.mark.parametrize("tp", [4, 1])
def test_the_helper_returns_values_scales_then_residual(mods, monkeypatch, tp):
    ln = mods.layernorm
    monkeypatch.setattr(ln, "get_tensor_model_parallel_world_size", lambda: tp)
    values, scales, residual_out, normed = object(), object(), object(), object()
    fused = Mock(return_value=(values, residual_out, scales))
    monkeypatch.setattr(
        ln, "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant", fused
    )
    quantize = Mock(return_value=(values, scales))
    monkeypatch.setattr(mods.blockscale, "quantize_fp8", quantize)
    norm = Mock(return_value=(normed, residual_out))
    norm.weight, norm.variance_epsilon = object(), 1e-6
    x = torch.randn(3, 64)
    residual = torch.randn_like(x)

    out = ln.fused_allreduce_gemma_rms_norm_mxfp8_quant(x, residual, norm)

    assert out == (values, scales, residual_out)
    if tp > 1:
        fused.assert_called_once_with(
            x, residual, norm.weight, norm.variance_epsilon, gemma_norm=True
        )
        quantize.assert_not_called()
    else:
        norm.assert_called_once_with(x, residual)
        quantize.assert_called_once_with(normed)
