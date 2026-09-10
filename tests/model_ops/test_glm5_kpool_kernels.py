# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU parity checks for the GLM-5.3 pooled-indexer kernels."""

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="requires the AITER runtime")
pytest.importorskip("triton", reason="requires Triton")

from atom.model_ops.glm5_next import kpool

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@needs_gpu
def test_pool_and_rotate_matches_the_reference():
    torch.manual_seed(7)
    keys = torch.randn(17, 4, 128, device="cuda", dtype=torch.bfloat16)
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128, device="cuda")

    got = kpool.pool_and_rotate(keys, gates, ape).float()
    expected = (
        kpool.hadamard128_ref(
            kpool.pool_compress_ref(keys, gates, ape).to(torch.bfloat16).float()
        )
        .to(torch.bfloat16)
        .float()
    )

    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


@needs_gpu
def test_query_quant_uses_aiters_native_fp8_contract():
    torch.manual_seed(11)
    query = torch.randn(65, 128, device="cuda", dtype=torch.bfloat16)

    got_q, got_scale = kpool.fwht128_quant_fp8(query)
    rotated = kpool.hadamard128_ref(query).to(torch.bfloat16).float()
    expected_q, expected_scale = kpool.quant_fp8_ue8m0_ref(rotated)

    assert got_q.dtype == aiter.dtypes.fp8
    assert kpool.FP8_MAX == float(torch.finfo(aiter.dtypes.fp8).max)
    torch.testing.assert_close(
        got_scale.cpu(), expected_scale[:, None].cpu(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        (got_q.float() * got_scale).cpu(),
        (expected_q.float() * expected_scale[:, None]).cpu(),
        rtol=2e-2,
        atol=2e-2,
    )
