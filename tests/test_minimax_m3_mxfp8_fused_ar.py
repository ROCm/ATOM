# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MiniMax-M3 native MXFP8 fused all-reduce: the gate and the helper (CPU)."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture
def mods():
    with atom_without_aiter() as load:
        yield SimpleNamespace(
            layernorm=load("atom.model_ops.layernorm"),
            blockscale=load("atom.model_ops.blockscale"),
        )


def _linear(**kw):
    base = {
        "params_dtype": torch.float8_e4m3fn,
        "native_a8_group_rows": 1,
        "runs_bf16": False,
    }
    return SimpleNamespace(**{**base, **kw})


@pytest.mark.parametrize(
    "kw, api, gfx, want",
    [
        ({}, True, "gfx1250", True),
        ({"native_a8_group_rows": 32}, True, "gfx1250", True),
        ({}, False, "gfx1250", False),
        ({}, True, "gfx950", False),
        ({"params_dtype": torch.float8_e4m3fnuz}, True, "gfx1250", False),
        ({"params_dtype": torch.float4_e2m1fn_x2}, True, "gfx1250", False),
        ({"native_a8_group_rows": None}, True, "gfx1250", False),
        ({"runs_bf16": True}, True, "gfx1250", False),
    ],
)
def test_the_fused_op_feeds_only_native_e4m3fn_group32_linears_on_gfx1250(
    mods, monkeypatch, kw, api, gfx, want
):
    ln = mods.layernorm
    op = Mock() if api else None
    monkeypatch.setattr(
        ln, "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant", op
    )
    monkeypatch.setattr(ln, "get_gfx_runtime", lambda: gfx)
    assert ln.fused_allreduce_gemma_rms_norm_mxfp8_supported(_linear(**kw)) is want


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
