# SPDX-License-Identifier: MIT

import pytest
import torch
from aiter import QuantType, dtypes, get_hip_quant

from atom.model_ops.kimi_k3.activations import (
    situ_and_mul,
    situ_and_mul_maybe_quant,
    situ_and_mul_mxfp4_quant,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 31])
def test_situ_mxfp4_quant_matches_standalone_bytes(m):
    torch.manual_seed(20261003)
    d = 6144
    beta, linear_beta = 2.0, 1.5
    x = torch.randn(m, 2 * d, device="cuda", dtype=torch.bfloat16)

    actual, actual_scale = situ_and_mul_mxfp4_quant(x, beta, linear_beta)
    activation = situ_and_mul(x, beta, linear_beta)
    expected, expected_scale = get_hip_quant(QuantType.per_1x32)(
        activation,
        quant_dtype=dtypes.fp4x2,
        shuffle=False,
    )
    dispatched, dispatched_scale = situ_and_mul_maybe_quant(
        x,
        beta,
        linear_beta,
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
    )

    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(actual_scale.view(torch.uint8), expected_scale.view(torch.uint8))
    assert torch.equal(dispatched.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(
        dispatched_scale.view(torch.uint8), actual_scale.view(torch.uint8)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_situ_mxfp4_fusion_keeps_prefill_on_standalone_path():
    x = torch.randn(32, 2 * 6144, device="cuda", dtype=torch.bfloat16)
    out, scale = situ_and_mul_maybe_quant(
        x,
        2.0,
        1.5,
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
    )
    assert out.dtype == torch.bfloat16
    assert scale is None
