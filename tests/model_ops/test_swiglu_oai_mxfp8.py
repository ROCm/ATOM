# SPDX-License-Identifier: MIT
"""`swiglu_oai_split_mxfp8` writes what `quantize_fp8` makes of `swiglu_oai_split`.

The fused kernel exists so that a native group32 A8 `down_proj` (MXFP8) can
skip its own `quantize_fp8`; that is only a valid substitution if the E4M3
values and the E8M0 group scales are the same bytes the separate path writes.
The shapes cover the M3.1 per-rank intermediates (768 shared expert, 3072 dense
MLP), a width below one tile, row counts from decode to a prefill chunk, an
all-zero group, a value past the SwiGLU clamp, and a row-strided input (a
padded gate_up sliced back to its width).
"""

import pytest
import torch

if not torch.cuda.is_available() or torch.version.hip is None:
    pytest.skip("ROCm GPU required", allow_module_level=True)

pytest.importorskip("aiter")

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.swiglu_oai import (
    swiglu_oai_split,
    swiglu_oai_split_mxfp8,
)

ALPHA, BETA, LIMIT = 1.702, 1.0, 7.0


def _separate(gate_up):
    return quantize_fp8(swiglu_oai_split(gate_up, alpha=ALPHA, beta=BETA, limit=LIMIT))


def _assert_same_bytes(got, want):
    (q, s), (rq, rs) = got, want
    assert q.shape == rq.shape and s.shape == rs.shape
    assert torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), rs.view(torch.uint8))


@pytest.mark.parametrize("rows", [1, 2, 16, 64, 1355, 4120])
@pytest.mark.parametrize("inter", [96, 768, 3072])
def test_matches_separate_quant(rows, inter):
    torch.manual_seed(rows * 7 + inter)
    gate_up = (torch.randn(rows, 2 * inter, device="cuda") * 4).to(torch.bfloat16)
    if rows > 2:
        gate_up[1, :32] = 0  # an all-zero group: the amax floor decides its scale
        gate_up[2, 5] = 1e4  # past the clamp
    got = swiglu_oai_split_mxfp8(gate_up, alpha=ALPHA, beta=BETA, limit=LIMIT)
    _assert_same_bytes(got, _separate(gate_up))


def test_row_strided_input():
    full = (torch.randn(16, 2 * 768 + 64, device="cuda") * 4).to(torch.bfloat16)
    gate_up = full[:, : 2 * 768]
    assert not gate_up.is_contiguous()
    got = swiglu_oai_split_mxfp8(gate_up, alpha=ALPHA, beta=BETA, limit=LIMIT)
    _assert_same_bytes(got, _separate(gate_up))


def test_leading_dims_and_empty():
    gate_up = torch.randn(2, 3, 2 * 768, device="cuda").to(torch.bfloat16)
    q, s = swiglu_oai_split_mxfp8(gate_up, alpha=ALPHA, beta=BETA, limit=LIMIT)
    assert q.shape == (2, 3, 768) and s.shape == (2, 3, 24)
    _assert_same_bytes(
        (q.reshape(6, 768), s.reshape(6, 24)), _separate(gate_up.reshape(6, -1))
    )
    q, s = swiglu_oai_split_mxfp8(
        gate_up[:0].reshape(0, 2 * 768), alpha=ALPHA, beta=BETA, limit=LIMIT
    )
    assert q.shape == (0, 768) and s.shape == (0, 24)


def test_rejects_intermediate_not_multiple_of_32():
    gate_up = torch.randn(4, 2 * 48, device="cuda").to(torch.bfloat16)
    with pytest.raises(ValueError):
        swiglu_oai_split_mxfp8(gate_up, alpha=ALPHA, beta=BETA, limit=LIMIT)
