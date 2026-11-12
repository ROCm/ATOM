# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`latent_tail` against `KimiSparseMoeBlock._apply_routed_norm_and_up_proj`:
the eight TP ranks' column shares, concatenated, are the block's routed RMSNorm
and up-projection plus the shared output, rounded to bf16 once. The bound is
one bf16 rounding of the exact product of the input the block hands its
up-projection: the block's GEMM itself sums bf16 partials, so its output is
further than that from the exact product."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU kernel")

L, H, TP = 3584, 7168, 8
N = H // TP
EPS = 1e-6


def _bf16_ulp(t):
    return torch.exp2(torch.floor(torch.log2(t.abs().clamp_min(1e-30))) - 7)


@pytest.fixture
def block(monkeypatch):
    pytest.importorskip("aiter")
    from atom.model_ops import layernorm, linear

    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(linear, "get_tp_group", lambda: group)
    torch.manual_seed(0)
    norm = layernorm.RMSNorm(L, eps=EPS)
    norm.weight.data = (1 + 0.1 * torch.randn(L)).bfloat16()
    up = linear.ReplicatedLinear(L, H, bias=False, quant_config=None)
    up.weight.data = 0.02 * torch.randn(H, L, dtype=torch.bfloat16)
    norm, up = norm.cuda(), up.cuda()
    fed = []

    def up_proj(x):
        fed.append(x)
        return up(x)

    return SimpleNamespace(
        routed_expert_norm=norm,
        fuse_routed_triton_mxfp4_quant=False,
        routed_expert_up_proj=up_proj,
        up=up,
        fed=fed,
    )


def _shares(block, latent, shared):
    from atom.model_ops.kimi_k3.latent_tail import latent_tail

    w, norm = block.up.weight, block.routed_expert_norm
    cols = [slice(r * N, (r + 1) * N) for r in range(TP)]
    return [latent_tail(latent, norm.weight, EPS, w[c], shared[:, c]) for c in cols]


def test_column_shares_are_the_routed_tail_plus_shared_rounded_once(block):
    from atom.models.kimi_k3.model import KimiSparseMoeBlock

    for T in range(1, 65):
        latent = 3 * torch.randn(T, L, device="cuda", dtype=torch.bfloat16)
        shared = 0.5 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
        block.fed.clear()
        KimiSparseMoeBlock._apply_routed_norm_and_up_proj(block, latent)
        exact = (block.fed[0].double() @ block.up.weight.double().T).float()
        want = exact + shared.float()

        got = torch.cat(_shares(block, latent, shared), 1).float()

        scale = torch.stack([exact.abs(), shared.float().abs(), want.abs()]).amax(0)
        ulps = (got - want).abs() / _bf16_ulp(scale)
        assert ulps.max() <= 1, f"T={T}: {ulps.max():.3f} bf16 roundings off"


def test_a_captured_tail_replays_its_eager_result(block):
    T = 32
    latent = 3 * torch.randn(T, L, device="cuda", dtype=torch.bfloat16)
    shared = 0.5 * torch.randn(T, H, device="cuda", dtype=torch.bfloat16)
    eager = _shares(block, latent, shared)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _shares(block, latent, shared)

    for _ in range(3):
        for out in captured:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert all(torch.equal(c, e) for c, e in zip(captured, eager))
