# SPDX-License-Identifier: MIT
"""Bound V4 RMSNorm rounding and preserve downstream quantization semantics."""

import pytest
import torch

pytest.importorskip("aiter", reason="RMSNorm is an AITER kernel")

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.layernorm import RMSNorm
from tests.models.deepseek_v41 import oracle_kernels


@pytest.mark.parametrize("dim", [128, 512, 1280, 5120])
@pytest.mark.parametrize("length", [1, 257])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_norm_reference_rounding_and_quantization(reference, single_rank, dim, length):
    torch.manual_seed(451)
    with reference.set_dtype(torch.bfloat16):
        expected_norm = reference.RMSNorm(dim, 1e-20).cuda()
        actual_norm = RMSNorm(dim, 1e-20).cuda()
        weights = (torch.randn(dim, dtype=torch.float32) * 0.1 + 1).bfloat16()
        expected_norm.weight.data.copy_(weights)
        actual_norm.weight.data.copy_(weights)
        hidden = torch.randn(2, length, dim, device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = expected_norm(hidden)
            actual = actual_norm(hidden)
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2**-16)
        actual_q, actual_s = quantize_fp8(actual)
        expected_q, expected_s = oracle_kernels.act_quant(
            actual, 32, "ue8m0", torch.float8_e8m0fnu
        )
        assert torch.equal(actual_q.view(torch.uint8), expected_q.view(torch.uint8))
        assert torch.equal(actual_s.view(torch.uint8), expected_s.view(torch.uint8))


@pytest.mark.parametrize("dim,length", [(128, 7), (512, 257), (1280, 24), (5120, 6)])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_shared_norm_layout_and_fp64_formula(single_rank, dim, length):
    device = "cuda"
    torch.manual_seed(452)
    norm = RMSNorm(dim, 1e-20).to(device=device, dtype=torch.bfloat16)
    norm.weight.data.copy_((torch.randn(dim, device=device) * 0.1 + 1).bfloat16())
    storage = torch.randn(2, length * 2, dim, device=device).bfloat16()
    hidden = storage[:, ::2]
    before = storage.clone()
    values = hidden.double()
    expected = (
        norm.weight.double()
        * values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.eps)
    )
    with torch.inference_mode():
        actual = norm(hidden)
    assert actual.shape == hidden.shape
    assert actual.dtype == hidden.dtype
    assert torch.equal(storage, before)
    torch.testing.assert_close(actual, expected.bfloat16(), rtol=1 / 128, atol=2**-12)
