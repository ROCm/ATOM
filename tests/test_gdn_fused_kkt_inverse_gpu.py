# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused gating, KKT and inverse/WY against an fp64 reference; the decay rows chunk_h reads.

The gates, beta and the scales are the same expressions as the separate KKT kernel's. w and u are
compared with the fp64 WY factors per (chunk, head), also where (I + A) is close to the all-ones
lower matrix (near-parallel keys, beta = 1, slow decay), where a bf16 inverse cancels.
"""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
if torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250":
    pytest.skip("gfx1250 kernels", allow_module_level=True)

from atom.model_ops.fla_ops.fused_cumsum_kkt import fused_cumsum_kkt
from atom.model_ops.fla_ops.fused_kkt_inverse_gfx1250 import fused_kkt_inverse_gfx1250
from atom.model_ops.fla_ops.index import prepare_chunk_indices

DEV = "cuda"


def _wy_fp64(k, v, cu, gating, beta_hm):
    """fp64 chunk gates, w and u [NT, H, 64, D] (zero past each sequence end) and the valid mask."""
    a, _, A_log, dt_bias = gating
    x = a.double() + dt_bias.double()
    gate = -torch.exp(A_log.double()) * torch.where(
        x <= 20.0, torch.log1p(torch.exp(x)), x
    )
    starts, lens = [], []
    for s in range(cu.numel() - 1):
        for t0 in range(int(cu[s]), int(cu[s + 1]), 64):
            starts.append(t0)
            lens.append(min(64, int(cu[s + 1]) - t0))
    ar = torch.arange(64, device=DEV)
    valid = ar[None, :] < torch.tensor(lens, device=DEV)[:, None]
    idx = torch.where(valid, torch.tensor(starts, device=DEV)[:, None] + ar[None, :], 0)
    m = valid[:, None, :].double()
    gc = (gate[idx].permute(0, 2, 1) * m).cumsum(-1)
    beta = beta_hm[0].double().T[idx].permute(0, 2, 1) * m
    kh = (
        k[0]
        .double()[idx]
        .repeat_interleave(v.shape[2] // k.shape[2], dim=2)
        .permute(0, 2, 1, 3)
    )
    vh = v[0].double()[idx].permute(0, 2, 1, 3)
    low = torch.tril(torch.ones(64, 64, device=DEV, dtype=torch.bool), -1)
    decay = torch.exp(torch.where(low, gc[..., :, None] - gc[..., None, :], 0.0))
    A = torch.where(low, (kh @ kh.transpose(-1, -2)) * decay, 0.0) * beta[..., :, None]
    eye = torch.eye(64, device=DEV, dtype=torch.float64)
    Ai = torch.linalg.solve_triangular(eye + A, eye.expand_as(A), upper=False)
    w = Ai @ ((beta * torch.exp(gc))[..., None] * kh)
    u = Ai @ (beta[..., None] * vh)
    return w, u, idx, valid


def _chunk_errors(got, ref, idx, valid):
    """rel-L2 error of each (chunk, head) of a [1, T, H, D] kernel output."""
    x = got[0].double()[idx].permute(0, 2, 1, 3) * valid[:, None, :, None]
    return (x - ref).flatten(2).norm(dim=-1) / ref.flatten(2).norm(dim=-1).clamp_min(
        1e-30
    )


def _gating(tokens, gen, A_log=None):
    rows = torch.randn(tokens, 96, generator=gen).to(device=DEV, dtype=torch.bfloat16)
    a, b = rows[:, 32:64], rows[:, 64:]
    A_log = torch.rand(32, generator=gen) if A_log is None else A_log
    return a, b, A_log.to(DEV), torch.randn(32, generator=gen).to(DEV)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fused_kkt_inverse_and_padded_plan(index_dtype):
    # single tokens, partial chunks, chunk boundaries; gates fast and slow enough for both gating branches
    lens = [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129, 517]
    tokens = sum(lens)
    gen = torch.Generator(device="cpu").manual_seed(7)
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, 16, 128, generator=gen), dim=-1
    ).to(device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, tokens, 32, 128, generator=gen).to(
        device=DEV, dtype=torch.bfloat16
    )
    gating = _gating(tokens, gen)
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device=DEV, dtype=index_dtype
    )
    g, _, beta, scale = fused_cumsum_kkt(None, k, None, cu_seqlens=cu, gating=gating)
    got = fused_kkt_inverse_gfx1250(k, v, cu, None, gating, True)
    g_k, decay, w_k, u_k, beta_k, scale_k = got
    assert torch.equal(beta_k, beta)
    torch.testing.assert_close(g_k, g, atol=1e-5, rtol=1e-6)
    torch.testing.assert_close(scale_k, scale, atol=1e-6, rtol=1e-5)
    w, u, idx, valid = _wy_fp64(k, v, cu, gating, beta_k)
    assert _chunk_errors(w_k, w, idx, valid).max() < 3e-3
    assert _chunk_errors(u_k, u, idx, valid).max() < 3e-3

    # decay per (chunk, head): exp(g_last - g_i) (0 past the sequence end), then exp(g_last)
    indices = prepare_chunk_indices(cu, 64)
    expected = torch.zeros_like(decay)
    for chunk, (seq, i) in enumerate(indices.tolist()):
        start = cu[seq].item() + 64 * i
        end = min(start + 64, cu[seq + 1].item())
        g_c = g_k[0, start:end].T
        expected[chunk, :, 0, : end - start] = torch.exp(g_c[:, -1:] - g_c)
        expected[chunk, :, 1] = torch.exp(g_c[:, -1:])
    torch.testing.assert_close(decay, expected, atol=1e-6, rtol=1e-5)
    without = fused_kkt_inverse_gfx1250(k, v, cu, None, gating, False)
    assert without[1] is None
    for i in (0, 2, 3, 4, 5):
        assert torch.equal(without[i], got[i])

    padded = torch.full((len(indices) + 5, 2), -1, device=DEV, dtype=index_dtype)
    padded[: len(indices)] = indices
    plan = SimpleNamespace(
        chunk_indices_64=padded,
        nt64_max=len(padded),
        nt64=torch.tensor([len(indices)], device=DEV, dtype=torch.int32),
    )
    planned = fused_kkt_inverse_gfx1250(k, v, cu, plan, gating, True)
    assert torch.equal(planned[1][: len(indices)], decay)
    for i in (0, 2, 3, 4, 5):
        assert torch.equal(planned[i], got[i])


@pytest.mark.parametrize("spread", [0.0, 0.05, 0.2])
def test_near_parallel_keys(spread):
    """Keys k0 + spread * noise, beta = 1 and a gate of ~-0.02 a token: A ~ the all-ones lower matrix
    (a real layer has such heads). Its inverse is small, so w and u must stay at the fp64 error level.
    """
    lens = [64, 200, 37, 1000]
    tokens = sum(lens)
    gen = torch.Generator(device="cpu").manual_seed(11)
    k0 = torch.randn(1, 1, 16, 128, generator=gen)
    k = torch.nn.functional.normalize(
        k0 + spread * torch.randn(1, tokens, 16, 128, generator=gen), dim=-1
    ).to(device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, tokens, 32, 128, generator=gen).to(
        device=DEV, dtype=torch.bfloat16
    )
    rows = torch.empty(tokens, 96, device=DEV, dtype=torch.bfloat16)
    rows[:, 32:64] = -4.0  # softplus(-4 + dt_bias) ~ 0.02
    rows[:, 64:] = 8.0  # beta = bf16(sigmoid(8)) = 1
    gating = (
        rows[:, 32:64],
        rows[:, 64:],
        torch.zeros(32, device=DEV),
        torch.zeros(32, device=DEV),
    )
    cu = torch.tensor(
        [0, *torch.tensor(lens).cumsum(0).tolist()], device=DEV, dtype=torch.int32
    )
    _, _, w_k, u_k, beta_k, _ = fused_kkt_inverse_gfx1250(k, v, cu, None, gating, False)
    assert torch.equal(beta_k, torch.ones_like(beta_k))
    w, u, idx, valid = _wy_fp64(k, v, cu, gating, beta_k)
    assert _chunk_errors(w_k, w, idx, valid).max() < 3e-3
    assert _chunk_errors(u_k, u, idx, valid).max() < 3e-3
