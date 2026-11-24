# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""swiglu_oai_quant_fp8 must return the bytes of swiglu_oai_split followed by quantize_fp8."""

import pytest
import torch

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.mx8_fused import swiglu_oai_quant_fp8
from atom.model_ops.swiglu_oai import swiglu_oai_split

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("inter", [768, 3072])  # shared expert / dense + draft MLP per rank at TP4
@pytest.mark.parametrize("rows", [1, 7, 8, 64, 513, 4096])
@pytest.mark.parametrize("limit", [7.0, None])
def test_swiglu_oai_quant_matches_unfused(inter, rows, limit):
    torch.manual_seed(rows + inter)
    gate_up = torch.randn(rows, 2 * inter, device="cuda", dtype=torch.bfloat16) * 4
    gate_up[0, :8] = 0  # an all-zero-ish group start: the amax floor path
    q_ref, s_ref = quantize_fp8(swiglu_oai_split(gate_up, 1.702, 1.0, limit))
    q, s = swiglu_oai_quant_fp8(gate_up, 1.702, 1.0, limit)
    assert q.dtype == q_ref.dtype and s.dtype == s_ref.dtype
    assert q.shape == q_ref.shape and s.shape == s_ref.shape
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s.view(torch.uint8), s_ref.view(torch.uint8))


def test_swiglu_oai_quant_keeps_leading_dims():
    gate_up = torch.randn(2, 3, 2 * 768, device="cuda", dtype=torch.bfloat16)
    q, s = swiglu_oai_quant_fp8(gate_up, 1.702, 1.0, 7.0)
    assert q.shape == (2, 3, 768) and s.shape == (2, 3, 24)
