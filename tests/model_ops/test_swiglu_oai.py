# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU coverage for fused SwiGLU-OAI native group32 MXFP8 quantization."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("SwiGLU-OAI requires a ROCm GPU", allow_module_level=True)

_MXFP8_ROWS = (1, 8, 16, 32, 64, 96, 128, 192, 240, 256)


def _edge_rows(gate_up: torch.Tensor) -> None:
    """Fill the first eight rows with quantization and activation edge cases."""
    intermediate = gate_up.shape[-1] // 2
    gate, up = gate_up[:, :intermediate], gate_up[:, intermediate:]
    gate[0], up[0] = 0.0, 0.0
    gate[1] = -60.0
    gate[2], up[2] = 1e4, -1e4
    gate[3, ::7] = float("nan")
    gate[4, ::5] = float("-inf")
    up[5, ::3] = float("inf")
    gate[6] = -51.5
    gate[7] = -45.0


def _mxfp8_input(rows: int, intermediate: int = 768) -> torch.Tensor:
    # Shared I=768 and dense I=3072 are both per-rank MiniMax-M3 TP4 shapes.
    torch.manual_seed(1702 + rows)
    gate_up = (
        torch.randn(rows, 2 * intermediate, device="cuda", dtype=torch.float32) * 3
    )
    if rows >= 8:
        _edge_rows(gate_up)
    return gate_up.to(torch.bfloat16)


def _assert_bitwise_mxfp8(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    (quantized, scale), (quantized_ref, scale_ref) = actual, expected
    assert quantized.dtype == quantized_ref.dtype == torch.float8_e4m3fn
    assert scale.dtype == scale_ref.dtype == torch.float8_e8m0fnu
    assert quantized.shape == quantized_ref.shape
    assert (
        scale.shape
        == scale_ref.shape
        == (
            *quantized.shape[:-1],
            quantized.shape[-1] // 32,
        )
    )
    assert quantized.is_contiguous() and scale.is_contiguous()
    assert torch.equal(quantized.view(torch.uint8), quantized_ref.view(torch.uint8))
    assert torch.equal(scale.view(torch.uint8), scale_ref.view(torch.uint8))


@pytest.mark.parametrize("rows", _MXFP8_ROWS)
def test_swiglu_mxfp8_is_bitwise_split_group32_quant(rows: int) -> None:
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split, swiglu_oai_split_mxfp8

    gate_up = _mxfp8_input(rows)
    expected = quantize_fp8(swiglu_oai_split(gate_up, 1.702, 1.0, 7.0))
    actual = swiglu_oai_split_mxfp8(gate_up, 1.702, 1.0, 7.0)
    _assert_bitwise_mxfp8(actual, expected)


@pytest.mark.parametrize("rows", [1, 32, 256])
def test_dense_swiglu_mxfp8_is_bitwise_split_group32_quant(rows: int) -> None:
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split, swiglu_oai_split_mxfp8

    gate_up = _mxfp8_input(rows, intermediate=3072)
    expected = quantize_fp8(swiglu_oai_split(gate_up, 1.702, 1.0, 7.0))
    actual = swiglu_oai_split_mxfp8(gate_up, 1.702, 1.0, 7.0)
    _assert_bitwise_mxfp8(actual, expected)


def test_swiglu_mxfp8_cuda_graph_replay_is_bitwise() -> None:
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split, swiglu_oai_split_mxfp8

    static = _mxfp8_input(96)
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            swiglu_oai_split_mxfp8(static, 1.702, 1.0, 7.0)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = swiglu_oai_split_mxfp8(static, 1.702, 1.0, 7.0)
    for seed in (19, 23):
        torch.manual_seed(seed)
        static.copy_(
            torch.randn_like(static, dtype=torch.float32).mul_(4).to(torch.bfloat16)
        )
        graph.replay()
        expected = quantize_fp8(swiglu_oai_split(static, 1.702, 1.0, 7.0))
        _assert_bitwise_mxfp8(captured, expected)


def test_swiglu_mxfp8_output_is_accepted_by_native_down_projection() -> None:
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale
    from torch import nn

    from atom.model_ops.blockscale import native_quant_linear, quantize_fp8
    from atom.model_ops.swiglu_oai import swiglu_oai_split_mxfp8
    from atom.model_ops.utils import shuffle_weights

    quantized, scale = swiglu_oai_split_mxfp8(_mxfp8_input(8), 1.702, 1.0, 7.0)
    weight, weight_scale = quantize_fp8(
        torch.randn(6144, 768, device="cuda", dtype=torch.bfloat16)
    )
    weight = nn.Parameter(weight, requires_grad=False)
    shuffle_weights(weight)
    weight_scale = shuffle_mxfp8_bpreshuffle_scale(weight_scale)
    out = native_quant_linear(
        quantized,
        weight,
        weight_scale,
        x_scale=scale,
        weight_group_rows=1,
        weight_preshuffled=True,
    )
    assert out.shape == (8, 6144)
    assert out.dtype == torch.bfloat16


def test_swiglu_mxfp8_support_does_not_probe_arch_while_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from atom.model_ops import swiglu_oai

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    def fail_if_called() -> bool:
        raise AssertionError(
            "architecture probing must stay outside the compiled graph"
        )

    monkeypatch.setattr(swiglu_oai, "swiglu_oai_mxfp8_arch_supported", fail_if_called)
    assert swiglu_oai.swiglu_oai_mxfp8_supported(_mxfp8_input(1))


def test_swiglu_mxfp8_rejects_ineligible_layout_dtype_and_shape() -> None:
    from atom.model_ops.swiglu_oai import swiglu_oai_mxfp8_supported

    gate_up = _mxfp8_input(8)
    assert not swiglu_oai_mxfp8_supported(gate_up.to(torch.float16))
    assert not swiglu_oai_mxfp8_supported(gate_up[:, :-2])
    assert not swiglu_oai_mxfp8_supported(gate_up[:, ::2])
    assert not swiglu_oai_mxfp8_supported(gate_up[:0])
