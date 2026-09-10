# SPDX-License-Identifier: MIT
"""Regression for HIP per-tensor quantization reading partial vector rows."""

import pytest
import torch

from atom.model_ops.attention_mla import quant_fp8_per_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires ROCm GPU")
@pytest.mark.parametrize("tokens", [1, 5, 7, 88, 8192])
@pytest.mark.parametrize("dim", [128, 192])
def test_quant_does_not_read_adjacent_storage(tokens, dim):
    n = tokens * 16 * dim
    storage = torch.full((n + 128,), 64, device="cuda", dtype=torch.bfloat16)
    x = storage[:n].view(tokens, 16, dim)
    x.fill_(1)
    quantized, scale = quant_fp8_per_tensor(x)
    torch.testing.assert_close(
        scale, torch.full_like(scale, 1 / torch.finfo(quantized.dtype).max), rtol=1e-6, atol=0
    )
    torch.testing.assert_close(quantized.float() * scale, x.float(), rtol=1e-6, atol=0)


@pytest.mark.parametrize("shape", [(0, 16, 192), (3, 7)])
def test_empty_or_partial_vector_input(shape):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    y, scale = quant_fp8_per_tensor(x)
    assert y.shape == x.shape
    assert scale.shape == (1,)
    assert torch.isfinite(y.float() * scale).all()
