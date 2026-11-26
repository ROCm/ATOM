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
quantization = _gpu_stack_import("atom.model_ops.blockscale_kernels.quantization")


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


def test_single_rank_helper_norms_then_quantizes_by_default(monkeypatch):
    """TP 1 without ATOM_M3_TP1_NORM_MXFP8: the norm, then quantize_fp8."""
    monkeypatch.delenv("ATOM_M3_TP1_NORM_MXFP8", raising=False)
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    normed, residual_out = object(), object()
    norm = Mock(return_value=(normed, residual_out))
    values, scales = object(), object()
    quantize = Mock(return_value=(values, scales))
    monkeypatch.setattr(blockscale, "quantize_fp8", quantize)

    hidden_states, residual = torch.zeros(2, 64, dtype=torch.bfloat16), object()
    result = layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
        hidden_states, residual, norm
    )

    assert result == (values, scales, residual_out)
    norm.assert_called_once_with(hidden_states, residual)
    quantize.assert_called_once_with(normed)


def test_single_rank_helper_keeps_two_launches_past_max_rows(monkeypatch):
    """ATOM_M3_TP1_NORM_MXFP8 past ATOM_M3_TP1_NORM_MXFP8_MAX_ROWS rows (a
    prefill chunk): the norm, then quantize_fp8; the caller asks no bf16 row."""
    import aiter.ops.fused_qk_rmsnorm_group_quant as fqk

    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8", "1")
    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8_MAX_ROWS", "4")
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    fused = Mock()
    monkeypatch.setattr(fqk, "fused_qk_rmsnorm_group_quant", fused)
    normed, residual_out = object(), object()
    norm = Mock(return_value=(normed, residual_out))
    quantize = Mock(return_value=(object(), object()))
    monkeypatch.setattr(blockscale, "quantize_fp8", quantize)
    hidden_states = torch.zeros(5, 64, dtype=torch.bfloat16)

    layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(hidden_states, object(), norm)

    assert not fused.called
    quantize.assert_called_once_with(normed)
    assert layernorm.tp1_norm_mxfp8_one_launch(4)
    assert not layernorm.tp1_norm_mxfp8_one_launch(5)
    with pytest.raises(AssertionError):
        layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
            hidden_states, object(), norm, emit_bf16=True
        )


@pytest.mark.parametrize("emit_bf16", [False, True])
def test_single_rank_helper_is_one_aiter_mxfp8_norm(monkeypatch, emit_bf16):
    """TP 1 with ATOM_M3_TP1_NORM_MXFP8: residual add + the Gemma norm + its
    group-32 MXFP8 quant in one aiter launch (fp8 values + E8M0 scales), the
    bf16 row only when asked."""
    import aiter.ops.fused_qk_rmsnorm_group_quant as fqk

    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8", "1")
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    fused = Mock()
    monkeypatch.setattr(fqk, "fused_qk_rmsnorm_group_quant", fused)
    hidden_states = torch.randn(2, 3, 64, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden_states)
    norm = SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(64)), variance_epsilon=1e-6
    )

    result = layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
        hidden_states, residual, norm, emit_bf16=emit_bf16
    )

    (values, scales, x, weight, eps), kwargs = fused.call_args
    assert fused.call_count == 1 and eps == 1e-6
    assert weight.data_ptr() == norm.weight.data_ptr()
    assert x.shape == (6, 64) and x.data_ptr() == hidden_states.data_ptr()
    assert values.dtype == quantization.FP8_DTYPE and values.shape == (6, 64)
    assert scales.dtype == torch.float8_e8m0fnu and scales.shape == (6, 2)
    assert kwargs["q_residual"].data_ptr() == residual.data_ptr()
    assert kwargs["group_size"] == 32 and kwargs["gemma_norm"] is True
    assert (kwargs["q_out_unquantized"] is not None) == emit_bf16
    assert len(result) == (4 if emit_bf16 else 3)
    assert result[0].shape == (2, 3, 64) and result[1].shape == (2, 3, 2)
    assert result[-1].data_ptr() == kwargs["q_res_out"].data_ptr()
    if emit_bf16:
        assert result[2].data_ptr() == kwargs["q_out_unquantized"].data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("rows", [1, 255, 256, 4096, 32768])
def test_single_rank_helper_bytes_are_norm_then_quantize_fp8(monkeypatch, rows):
    """M3.1's hidden 6144 at decode and prefill row counts (255/256 straddle the
    gfx1250 switch to the interleaved launch): the one-launch MXFP8 norm writes
    the bytes of the norm followed by quantize_fp8, and the same bf16 row."""
    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8", "1")
    monkeypatch.setenv("ATOM_M3_TP1_NORM_MXFP8_MAX_ROWS", "1000000")
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    torch.manual_seed(rows)
    # bf16 like the model's weights (aiter wants the weight in the input dtype)
    norm = layernorm.GemmaRMSNorm(6144, eps=1e-6).to("cuda", torch.bfloat16)
    norm.weight.data.normal_(0.0, 0.1)
    x = torch.randn(rows, 6144, dtype=torch.bfloat16, device="cuda") * 3
    residual = torch.randn_like(x)

    normed, residual_ref = norm.forward_cuda(x, residual)
    values_ref, scales_ref = blockscale.quantize_fp8(normed)
    values, scales, row, residual_out = (
        layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(
            x, residual, norm, emit_bf16=True
        )
    )

    assert torch.equal(values.view(torch.uint8), values_ref.view(torch.uint8))
    assert torch.equal(scales.view(torch.uint8), scales_ref.view(torch.uint8))
    assert torch.equal(row.view(torch.int16), normed.view(torch.int16))
    assert torch.equal(residual_out.view(torch.int16), residual_ref.view(torch.int16))
    # the launch without the bf16 row (input norm, dense MLP) writes the same
    values2, scales2, residual_out2 = (
        layernorm.fused_allreduce_gemma_rms_norm_mxfp8_quant(x, residual, norm)
    )
    assert torch.equal(values2.view(torch.uint8), values.view(torch.uint8))
    assert torch.equal(scales2.view(torch.uint8), scales.view(torch.uint8))
    assert torch.equal(residual_out2.view(torch.int16), residual_out.view(torch.int16))
