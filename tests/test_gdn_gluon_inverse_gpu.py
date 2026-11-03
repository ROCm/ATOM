# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Gluon gfx1250 `fused_inverse_recompute_gfx1250` against the Triton
`fused_inverse_recompute`. Same math; the fp32 dots of the inverse and the bf16
products add in another order, so a few w and u values differ by bf16
rounding (more steps where a sum cancels)."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 Gluon kernel", allow_module_level=True)

from atom.model_ops.fla_ops.fused_cumsum_kkt import fused_cumsum_kkt
from atom.model_ops.fla_ops.fused_inverse_recompute import fused_inverse_recompute
from atom.model_ops.fla_ops.fused_inverse_recompute_gfx1250 import (
    fused_inverse_recompute_gfx1250,
)

SEQLENS = [1, 2, 3, 4, 9, 16, 17, 33, 64, 70, 129, 517, 3, 1000]


@pytest.mark.parametrize("k_heads", [16, 4])
def test_gluon_inverse_matches_triton(k_heads):
    v_heads, T = 2 * k_heads, sum(SEQLENS)
    g = torch.Generator(device="cpu").manual_seed(k_heads)
    k = (
        torch.nn.functional.normalize(
            torch.randn(1, T, k_heads, 128, generator=g), dim=-1
        )
        .to(torch.bfloat16)
        .cuda()
    )
    v = torch.randn(1, T, v_heads, 128, generator=g).to(torch.bfloat16).cuda()
    gate = (-torch.rand(1, T, v_heads, generator=g) * 0.3).cuda()
    beta = torch.rand(1, T, v_heads, generator=g).to(torch.bfloat16).cuda()
    cu = torch.tensor([0] + SEQLENS).cumsum(0).to(torch.int32).cuda()
    _, A, beta_hm, scale_hm = fused_cumsum_kkt(gate, k, beta, cu_seqlens=cu)

    w_ref, u_ref = fused_inverse_recompute(k, v, A, beta_hm, scale_hm, cu_seqlens=cu)
    w, u = fused_inverse_recompute_gfx1250(k, v, A, beta_hm, scale_hm, cu, None)
    for got, ref in ((w, w_ref), (u, u_ref)):
        torch.testing.assert_close(got.float(), ref.float(), atol=2e-3, rtol=1.6e-2)
        assert (got != ref).float().mean().item() < 1e-3
