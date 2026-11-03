# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`fused_cumsum_kkt` with the gates computed inside (`gating=`) against
`fused_gdn_gating` followed by `fused_cumsum_kkt`.

The gates are the same expressions, so beta and the gates are the same bits;
the cumsum over each program's two heads can group its additions differently
from the cumsum over all heads, so g_cumsum, A and the scales are compared
within fp32 rounding."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

from atom.model_ops.attention_gdn import fused_gdn_gating
from atom.model_ops.fla_ops.fused_cumsum_kkt import fused_cumsum_kkt

DEV = "cuda"
SEQLENS = [1, 2, 3, 4, 9, 64, 70, 129, 517, 3, 1000]


@pytest.mark.parametrize("k_heads", [16, 4])
def test_kkt_gating_matches_gating_kernel(k_heads):
    v_heads, T = 2 * k_heads, sum(SEQLENS)
    g = torch.Generator(device="cpu").manual_seed(k_heads)
    rows = torch.randn(T, 4096 + 2 * v_heads, generator=g).to(torch.bfloat16).to(DEV)
    b, a = rows[:, 4096 : 4096 + v_heads], rows[:, 4096 + v_heads :]
    A_log = torch.rand(v_heads, generator=g).to(DEV)
    dt_bias = torch.randn(v_heads, generator=g).to(DEV)
    k = (
        torch.nn.functional.normalize(
            torch.randn(1, T, k_heads, 128, generator=g), dim=-1
        )
        .to(torch.bfloat16)
        .to(DEV)
    )
    cu = torch.tensor([0] + SEQLENS, device=DEV).cumsum(0).to(torch.int32)

    gate, beta = fused_gdn_gating(A_log, a, b, dt_bias)
    ref = fused_cumsum_kkt(gate, k, beta, cu_seqlens=cu)
    got = fused_cumsum_kkt(None, k, None, cu_seqlens=cu, gating=(a, b, A_log, dt_bias))

    assert torch.equal(got[2], ref[2]), "beta differs"
    torch.testing.assert_close(got[0], ref[0], atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(got[1], ref[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(got[3], ref[3], atol=1e-6, rtol=1e-5)
