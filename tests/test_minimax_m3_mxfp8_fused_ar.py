# SPDX-License-Identifier: MIT
"""CPU-only routing tests for MiniMax-M3 native MXFP8 fused all-reduce."""

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


def _gpu_stack_import(name):
    try:
        return importlib.import_module(name)
    except Exception:
        try:
            importlib.import_module("aiter")
            importlib.import_module("triton")
        except Exception as exc:  # noqa: BLE001 - any failure means no GPU stack
            pytest.skip(
                f"{name} needs the aiter/triton stack, absent here: {exc}",
                allow_module_level=True,
            )
        if not torch.cuda.is_available():
            pytest.skip(
                f"{name} needs a GPU because aiter probes the arch at import",
                allow_module_level=True,
            )
        raise


layernorm = _gpu_stack_import("atom.model_ops.layernorm")
minimax_m3 = _gpu_stack_import("atom.models.minimax_m3.model")
blockscale = _gpu_stack_import("atom.model_ops.blockscale")


@pytest.mark.parametrize(
    "params_dtype",
    [torch.float8_e4m3fn, torch.float8_e4m3fnuz],
)
def test_native_group32_fp8_linear_is_eligible(params_dtype):
    linear = SimpleNamespace(
        quant_type=minimax_m3.QuantType.per_1x32,
        params_dtype=params_dtype,
        native_a8_group_rows=1,
        runs_bf16=False,
    )
    assert minimax_m3._linear_consumes_mxfp8(linear)


@pytest.mark.parametrize(
    "linear",
    [
        SimpleNamespace(
            quant_type=minimax_m3.QuantType.per_Token,
            params_dtype=torch.float8_e4m3fn,
            native_a8_group_rows=1,
        ),
        SimpleNamespace(
            quant_type=minimax_m3.QuantType.per_1x32,
            params_dtype=torch.float4_e2m1fn_x2,
            native_a8_group_rows=1,
        ),
        SimpleNamespace(
            quant_type=minimax_m3.QuantType.per_1x32,
            params_dtype=torch.float8_e4m3fn,
            native_a8_group_rows=None,
        ),
        SimpleNamespace(
            quant_type=minimax_m3.QuantType.per_1x32,
            params_dtype=torch.float8_e4m3fn,
            native_a8_group_rows=32,
            runs_bf16=True,
        ),
        SimpleNamespace(),
    ],
)
def test_non_native_mxfp8_linear_is_ineligible(linear):
    assert not minimax_m3._linear_consumes_mxfp8(linear)


def test_tp_helper_returns_values_scales_then_residual(monkeypatch):
    monkeypatch.setattr(
        layernorm,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    values, residual_out, scales = object(), object(), object()
    fused = Mock(return_value=(values, residual_out, scales))
    monkeypatch.setattr(
        layernorm,
        "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant",
        fused,
    )
    hidden_states = torch.randn(3, 64)
    residual = torch.randn_like(hidden_states)
    norm = SimpleNamespace(weight=object(), variance_epsilon=1e-6)

    result = layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
        hidden_states,
        residual,
        norm,
    )

    assert result == (values, scales, residual_out)
    fused.assert_called_once_with(
        hidden_states,
        residual,
        norm.weight,
        norm.variance_epsilon,
        gemma_norm=True,
    )


def test_tp_helper_names_the_required_aiter_api(monkeypatch):
    monkeypatch.setattr(
        layernorm,
        "get_tensor_model_parallel_world_size",
        lambda: 4,
    )
    monkeypatch.setattr(
        layernorm,
        "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant",
        None,
    )

    with pytest.raises(
        RuntimeError,
        match="tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant",
    ):
        layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
            object(),
            object(),
            object(),
        )


def test_single_rank_helper_norms_then_quantizes(monkeypatch):
    monkeypatch.setattr(
        layernorm,
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    normed, residual_out = object(), object()
    norm = Mock(return_value=(normed, residual_out))
    values, scales = object(), object()
    quantize = Mock(return_value=(values, scales))
    monkeypatch.setattr(blockscale, "quantize_fp8", quantize)

    hidden_states = object()
    residual = object()
    result = layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
        hidden_states,
        residual,
        norm,
    )

    assert result == (values, scales, residual_out)
    norm.assert_called_once_with(hidden_states, residual)
    quantize.assert_called_once_with(normed)
