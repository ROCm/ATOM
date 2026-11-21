# SPDX-License-Identifier: MIT
"""MLA out gate with E8M0 scales: byte-identical to sigmoid_mul + o_proj's
per_1x128_e8m0_quant."""

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("ROCm GPU required", allow_module_level=True)

from aiter import dtypes

from atom.model_ops.linear import per_1x128_e8m0_quant
from atom.model_ops.triton_fused_sigmoid_mul_quant import (
    sigmoid_mul,
    sigmoid_mul_fp8_e8m0_group_quant,
)

# Decode, tile edges, both sides of the 4096-row tile switch, max prefill.
ROWS = [1, 8, 15, 33, 128, 135, 1009, 4095, 4096, 8065, 16384]


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


def with_edge_groups(x):
    """All-zero, power-of-two, huge and tiny groups."""
    if x.shape[0] >= 4 and x.shape[1] >= 512:
        x[0, :128] = 0
        x[1, 128:256] = 448.0
        x[2, 256:384] *= 2.0**40
        x[3, 384:512] *= 2.0**-40
    return x


@pytest.mark.parametrize("transpose_scale", [True, False])
@pytest.mark.parametrize("rows", ROWS)
def test_out_gate_e8m0_matches_sigmoid_mul_then_o_proj_quant(rows, transpose_scale):
    torch.manual_seed(20261103)
    n = 1536  # o_proj input per TP8 rank
    attn = with_edge_groups(
        torch.randn(rows, n, dtype=torch.bfloat16, device="cuda") * 4
    )
    gate = torch.randn(rows, n, dtype=torch.bfloat16, device="cuda") * 4

    expected = per_1x128_e8m0_quant(
        sigmoid_mul(attn, gate), dtypes.fp8, transpose_scale=transpose_scale
    )
    actual = sigmoid_mul_fp8_e8m0_group_quant(attn, gate, transpose_scale)
    assert_same_bytes(actual, expected)


def test_out_gate_e8m0_reads_row_strided_inputs():
    torch.manual_seed(20261104)
    rows, n = 135, 1536
    storage = torch.randn(rows, 2 * n, dtype=torch.bfloat16, device="cuda")
    attn, gate = storage[:, :n], storage[:, n:]
    assert not attn.is_contiguous()

    expected = per_1x128_e8m0_quant(
        sigmoid_mul(attn.contiguous(), gate.contiguous()), dtypes.fp8, True
    )
    assert_same_bytes(sigmoid_mul_fp8_e8m0_group_quant(attn, gate, True), expected)


def test_out_gate_e8m0_saturates_an_inf_group():
    rows, n = 8, 1536
    attn = torch.randn(rows, n, dtype=torch.bfloat16, device="cuda")
    attn[3, 200] = float("inf")
    gate = torch.full_like(attn, 10.0)

    expected = per_1x128_e8m0_quant(sigmoid_mul(attn, gate), dtypes.fp8, True)
    assert_same_bytes(sigmoid_mul_fp8_e8m0_group_quant(attn, gate, True), expected)
