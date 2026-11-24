# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import pytest
import torch
from atomrl.engine.inference import fp8_weight_quantizer as quantizer
from atomrl.engine.training import fp8_weight_quantizer as training_quantizer


def _expand_block_scales(
    scales: torch.Tensor,
    shape: tuple[int, int],
    block_size: tuple[int, int],
) -> torch.Tensor:
    return scales.repeat_interleave(block_size[0], dim=0).repeat_interleave(
        block_size[1],
        dim=1,
    )[: shape[0], : shape[1]]


def test_per_block_quantizer_returns_vllm_dequant_scale(monkeypatch) -> None:
    monkeypatch.setattr(quantizer, "_is_rocm", lambda: False)
    weight = torch.tensor(
        [[-4.0, -2.0], [1.0, 3.0]],
        dtype=torch.float32,
    )

    fp8_weight, weight_scale_inv = quantizer._per_block_cast_to_fp8(
        weight,
        block_size=(2, 2),
    )

    expected_scale = weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
    assert weight_scale_inv.item() == pytest.approx(expected_scale.item())
    reconstructed = fp8_weight.float() * _expand_block_scales(
        weight_scale_inv,
        tuple(weight.shape),
        (2, 2),
    )
    torch.testing.assert_close(reconstructed, weight, rtol=0.06, atol=0.06)


def test_rocm_per_block_quantizer_casts_directly_to_fnuz(monkeypatch) -> None:
    # Patch the fnuz predicate, not _is_rocm: not all of ROCm is fnuz (gfx950 is OCP
    # e4m3). See test_gfx950_per_block_quantizer_uses_ocp_e4m3 below.
    monkeypatch.setattr(quantizer, "_is_fp8_fnuz", lambda: True)
    weight = torch.tensor(
        [[-4.0, -2.0], [1.0, 3.0]],
        dtype=torch.float32,
    )

    fp8_weight, weight_scale_inv = quantizer._per_block_cast_to_fp8(
        weight,
        block_size=(2, 2),
    )

    assert fp8_weight.dtype == torch.float8_e4m3fnuz
    # vLLM intentionally uses 224 instead of torch.finfo(...).max (240)
    # for ROCm fnuz accuracy.
    expected_scale = weight.abs().max() / 224.0
    assert weight_scale_inv.item() == pytest.approx(expected_scale.item())
    reconstructed = fp8_weight.float() * _expand_block_scales(
        weight_scale_inv,
        tuple(weight.shape),
        (2, 2),
    )
    torch.testing.assert_close(reconstructed, weight, rtol=0.06, atol=0.06)


def test_quantizer_replaces_only_terminal_weight_suffix() -> None:
    assert hasattr(quantizer, "_scale_name_for_weight")
    assert (
        quantizer._scale_name_for_weight(
            "model.layers.0.self_attn.indexer.weights_proj.weight"
        )
        == "model.layers.0.self_attn.indexer.weights_proj.weight_scale_inv"
    )


@pytest.mark.parametrize(
    "name",
    [
        "embed.weight",
        "head.weight",
        "layers.0.ffn.gate.weight",
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.compressor.wgate.weight",
        "layers.2.attn.indexer.compressor.wkv.weight",
        "layers.2.attn.indexer.compressor.wgate.weight",
        "layers.2.attn.indexer.weights_proj.weight",
    ],
)
def test_quantizer_skips_redhat_non_fp8_weights(name: str) -> None:
    assert quantizer._should_quantize(name) is False


def test_training_quantizer_skips_dsv4_indexer_score_projection() -> None:
    name = "model.layers.2.self_attn.indexer.weights_proj.weight"
    assert training_quantizer._should_quantize(name) is False


def test_gfx950_per_block_quantizer_uses_ocp_e4m3(monkeypatch) -> None:
    """On ROCm arches that are not fnuz (gfx950/MI355X) the format is OCP e4m3, max 448.

    Gating on "is ROCm" quantised gfx950 weights as fnuz with a 224 cap, a different
    format and a 2x-too-small range versus what TE and vLLM use on that arch.
    """
    monkeypatch.setattr(quantizer, "_is_rocm", lambda: True)
    monkeypatch.setattr(quantizer, "_is_fp8_fnuz", lambda: False)
    weight = torch.tensor([[-4.0, -2.0], [1.0, 3.0]], dtype=torch.float32)

    fp8_weight, weight_scale_inv = quantizer._per_block_cast_to_fp8(
        weight, block_size=(2, 2)
    )

    assert fp8_weight.dtype == torch.float8_e4m3fn
    expected_scale = weight.abs().max() / torch.finfo(torch.float8_e4m3fn).max
    assert weight_scale_inv.item() == pytest.approx(expected_scale.item())


def test_pow2_scales_round_up_not_to_nearest() -> None:
    """pow2_scales must round the dequant multiplier UP, to match TE.

    TE's compute_scale_from_amax() truncates the mantissa of FP8_MAX/amax, so the
    reciprocal it hands the GEMM is rounded up. Round-to-nearest would disagree with
    TE on roughly half of all blocks, by exactly one power of two.
    """
    weight = torch.zeros(2, 2, dtype=torch.float32)
    weight[0, 0] = 3.0  # amax/448 lands strictly between two powers of two

    _, continuous = quantizer._per_block_cast_to_fp8(weight, (2, 2), pow2_scales=False)
    _, pow2 = quantizer._per_block_cast_to_fp8(weight, (2, 2), pow2_scales=True)

    # exactly a power of two, and >= the continuous scale (rounded up, never down)
    mant = pow2.view(torch.int32) & 0x007FFFFF
    assert int(mant.abs().sum()) == 0
    assert (pow2 >= continuous).all()
    assert torch.allclose(pow2, torch.exp2(torch.ceil(torch.log2(continuous))))


def test_per_block_cast_rounds_through_fp32_not_bf16(monkeypatch) -> None:
    """The cast must not double-round through BF16.

    Dividing by the scale in BF16 rounds the quotient to 8 mantissa bits before the
    e4m3 cast rounds it again to 3. That is never closer to the original than a
    single rounding out of FP32, and it made pre-quantized weight sync disagree with
    vLLM's online ``per_block_cast_to_fp8`` on 3.01% of Qwen3-8B's elements. Pin the
    FP32 path: with it, the two are bit-identical.
    """
    monkeypatch.setattr(quantizer, "_is_fp8_fnuz", lambda: False)
    torch.manual_seed(0)
    weight = torch.randn(128, 128).to(torch.bfloat16)

    fp8_weight, scale_inv = quantizer._per_block_cast_to_fp8(
        weight,
        block_size=(128, 128),
    )

    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = scale_inv.reshape(())
    # .float() explicitly: a 0-dim tensor is a "wrapped number" for type promotion,
    # so ``bf16_tensor * zero_dim_fp32`` yields BF16 and would quietly re-introduce
    # the double rounding this test exists to catch. The implementation is safe
    # because its reciprocal always carries the (bm, bn, 1, 1) block-grid shape.
    expected = (
        (weight.float() * scale.reciprocal())
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    assert torch.equal(fp8_weight.view(torch.uint8), expected.view(torch.uint8))

    # The fixture has to actually exercise the difference, or this test passes
    # for the wrong reason.
    bf16_divide = (
        (weight / scale.to(weight.dtype))
        .clamp(-fp8_max, fp8_max)
        .to(torch.float8_e4m3fn)
    )
    assert bf16_divide.dtype == torch.float8_e4m3fn
    differing = int((bf16_divide.view(torch.uint8) != expected.view(torch.uint8)).sum())
    assert differing > 0, "fixture no longer triggers the double rounding"

    # Every one of those differences is a loss for the BF16 divide, never a gain.
    original = weight.float()
    err_fp32 = (expected.float() * scale - original).abs()
    err_bf16 = (bf16_divide.float() * scale - original).abs()
    assert float(err_fp32.pow(2).mean()) < float(err_bf16.pow(2).mean())
    assert bool((err_fp32 <= err_bf16).all())


def test_per_block_cast_chunking_does_not_change_the_result(monkeypatch) -> None:
    """The FP32 chunk size is a memory knob, not a numerics knob."""
    monkeypatch.setattr(quantizer, "_is_fp8_fnuz", lambda: False)
    torch.manual_seed(1)
    weight = torch.randn(384, 256).to(torch.bfloat16)

    whole, whole_scale = quantizer._per_block_cast_to_fp8(weight)
    # One block-row per chunk instead of all of them at once.
    monkeypatch.setattr(quantizer, "_FP32_CHUNK_ELEMS", 1)
    chunked, chunked_scale = quantizer._per_block_cast_to_fp8(weight)

    assert torch.equal(whole.view(torch.uint8), chunked.view(torch.uint8))
    assert torch.equal(whole_scale, chunked_scale)
