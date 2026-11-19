# SPDX-License-Identifier: MIT
"""SiTUv2 with E8M0 scales: byte-identical to situ_and_mul + down_proj's
per_1x128_e8m0_quant."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import dtypes

from atom.model_ops.kimi_k3.activations import (
    situ_and_mul,
    situ_and_mul_fp8_e8m0_quant,
)
from atom.model_ops.linear import per_1x128_e8m0_quant

# Kimi-K3's activation_situ_beta / activation_situ_linear_beta
BETA, LINEAR_BETA = 4.0, 25.0
# Decode and prefill row counts.
ROWS = [1, 8, 15, 33, 135, 1009, 8065, 16384]


def assert_same_bytes(actual, expected):
    (q, s), (q_ref, s_ref) = actual, expected
    assert q.dtype == q_ref.dtype and s.dtype == s_ref.dtype == dtypes.fp8_e8m0
    assert q.shape == q_ref.shape and s.shape == s_ref.shape
    assert s.stride() == s_ref.stride()
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))
    mismatch = (q.view(torch.uint8) != q_ref.view(torch.uint8)).nonzero()
    assert (
        mismatch.numel() == 0
    ), f"{len(mismatch)} fp8 bytes differ, first {mismatch[:4]}"


def reference(x, linear_beta, transpose_scale):
    d = x.shape[-1] // 2
    act = situ_and_mul(x, BETA, linear_beta).reshape(-1, d)
    return per_1x128_e8m0_quant(act, dtypes.fp8, transpose_scale)


# d per TP8 rank: shared expert 768, layer-0 dense 4224 (a 1-group last tile).
@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("d", [768, 4224])
@pytest.mark.parametrize("rows", ROWS)
def test_situ_e8m0_matches_situ_and_mul_then_down_proj_quant(rows, d, transpose_scale):
    torch.manual_seed(20261107)
    x = torch.randn(rows, 2 * d, dtype=torch.bfloat16, device="cuda") * 3
    if rows >= 3:
        x[0, :128] = 0  # all-zero groups
        x[1, 128:256] = 0
        x[2, d + 256 : d + 384] = 0  # up 0 -> an all-zero group

    expected = reference(x, LINEAR_BETA, transpose_scale)
    actual = situ_and_mul_fp8_e8m0_quant(x, BETA, LINEAR_BETA, transpose_scale)
    assert_same_bytes(actual, expected)


def test_situ_e8m0_without_linear_beta():
    torch.manual_seed(20261108)
    x = torch.randn(135, 2 * 768, dtype=torch.bfloat16, device="cuda") * 3
    assert_same_bytes(
        situ_and_mul_fp8_e8m0_quant(x, BETA, None, True), reference(x, None, True)
    )
