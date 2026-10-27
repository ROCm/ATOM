# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: the fused indexer's quantization helper.

``_block128_fp8`` is new math with no coverage anywhere else -- the fused
TP4 kernel's own tests exercise the kernel side of this format
(``tests/models/glm52/kernels/test_glm5_tp4_index_fp8.py``), not ATOM's
host-side quantizer. These tests run CPU-only (no GPU op is used; aiter's
dtype constants import fine without a device)."""

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from aiter import dtypes

from atom.models.glm52.mono.weights import _block128_fp8, fp32_vector
from atom.mono.runtime.consensus import MonoUnsupported


def test_single_block_shape_and_dtype():
    matrix = torch.randn(128, 128, dtype=torch.bfloat16)
    fp8, scale = _block128_fp8(matrix, name="t")
    assert fp8.dtype is dtypes.fp8
    assert tuple(fp8.shape) == (128, 128)
    assert fp8.is_contiguous()
    assert scale.dtype is torch.float32
    assert tuple(scale.shape) == (1, 1)


def test_multi_block_scale_shape_matches_block_grid():
    # w_index_k is (128, 6144) -> scale (1, 48); w_index_q is (4096, 2048) ->
    # scale (32, 16). Shape is `(rows // 128, cols // 128)`.
    matrix = torch.randn(256, 384, dtype=torch.bfloat16)
    _, scale = _block128_fp8(matrix, name="t")
    assert tuple(scale.shape) == (2, 3)


def test_round_trip_is_close_within_fp8_precision():
    torch.manual_seed(0)
    matrix = torch.randn(128, 128, dtype=torch.bfloat16)
    fp8, scale = _block128_fp8(matrix, name="t")
    dequant = fp8.to(torch.float32) * scale.view(1, 1)
    rel_error = (dequant - matrix.to(torch.float32)).abs().amax() / matrix.to(
        torch.float32
    ).abs().amax()
    assert rel_error < 0.2  # e4m3 is ~2-3 mantissa bits; a loose bound


def test_each_block_gets_its_own_scale_not_a_global_one():
    # Review-focus property: one block at a much larger magnitude than
    # another must not blow out the other block's precision -- this is
    # exactly what distinguishes block scaling from
    # dynamic_per_batched_tensor_quant (one scalar for the whole tensor).
    matrix = torch.zeros(256, 128, dtype=torch.bfloat16)
    matrix[:128, :] = 1.0  # small-magnitude block
    matrix[128:, :] = 1000.0  # large-magnitude block
    fp8, scale = _block128_fp8(matrix, name="t")
    assert scale[0, 0].item() != pytest.approx(scale[1, 0].item())
    dequant_small = fp8[:128, :].to(torch.float32) * scale[0, 0]
    # The small block must still resolve to ~1.0, not be crushed to 0 by a
    # scale sized for the large block.
    assert dequant_small.abs().mean().item() > 0.5


def test_misaligned_shape_is_refused():
    matrix = torch.randn(130, 128, dtype=torch.bfloat16)
    with pytest.raises(MonoUnsupported):
        _block128_fp8(matrix, name="t")


def test_non_contiguous_is_refused():
    matrix = torch.randn(128, 256, dtype=torch.bfloat16)[:, ::2]
    with pytest.raises(MonoUnsupported):
        _block128_fp8(matrix, name="t")


def test_fp32_vector_refuses_wrong_dtype():
    with pytest.raises(MonoUnsupported):
        fp32_vector(torch.randn(128, dtype=torch.bfloat16), "g", 128)


def test_fp32_vector_refuses_wrong_shape():
    with pytest.raises(MonoUnsupported):
        fp32_vector(torch.randn(64, dtype=torch.float32), "g", 128)


def test_fp32_vector_accepts_the_expected_shape():
    tensor = torch.randn(128, dtype=torch.float32)
    assert fp32_vector(tensor, "g", 128) is tensor
