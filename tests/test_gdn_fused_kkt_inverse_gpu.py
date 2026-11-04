# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused gating and WY against separate KKT and inverse kernels."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from atom.model_ops.fla_ops.fused_cumsum_kkt import fused_cumsum_kkt
from atom.model_ops.fla_ops.fused_inverse_recompute_gfx1250 import (
    fused_inverse_recompute_gfx1250,
)
from atom.model_ops.fla_ops.fused_kkt_inverse_gfx1250 import fused_kkt_inverse_gfx1250
from atom.model_ops.fla_ops.index import prepare_chunk_indices


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fused_kkt_inverse_and_padded_plan(index_dtype):
    lens = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 517]
    tokens = sum(lens)
    gen = torch.Generator(device="cpu").manual_seed(7)
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, 16, 128, generator=gen), dim=-1
    ).to(device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, tokens, 32, 128, generator=gen).to(
        device="cuda", dtype=torch.bfloat16
    )
    rows = torch.randn(tokens, 96, generator=gen).to(
        device="cuda", dtype=torch.bfloat16
    )
    a, b = rows[:, 32:64], rows[:, 64:]
    gating = (
        a,
        b,
        torch.rand(32, generator=gen).cuda(),
        torch.randn(32, generator=gen).cuda(),
    )
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device="cuda", dtype=index_dtype
    )
    g, A, beta, scale = fused_cumsum_kkt(None, k, None, cu_seqlens=cu, gating=gating)
    w, u = fused_inverse_recompute_gfx1250(k, v, A, beta, scale, cu, None)
    got = fused_kkt_inverse_gfx1250(k, v, cu, None, gating)
    assert torch.equal(got[3], beta)
    torch.testing.assert_close(got[0], g, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(got[4], scale, atol=1e-6, rtol=1e-5)
    for actual, expected in zip(got[1:3], (w, u)):
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=2e-3, rtol=1.6e-2
        )
        assert (actual != expected).float().mean().item() < 1e-3

    indices = prepare_chunk_indices(cu, 64)
    padded = torch.full((len(indices) + 5, 2), -1, device="cuda", dtype=index_dtype)
    padded[: len(indices)] = indices
    plan = SimpleNamespace(
        chunk_indices_64=padded,
        nt64_max=len(padded),
        nt64=torch.tensor([len(indices)], device="cuda", dtype=torch.int32),
    )
    planned = fused_kkt_inverse_gfx1250(k, v, cu, plan, gating)
    for actual, expected in zip(planned, got):
        assert torch.equal(actual, expected)
