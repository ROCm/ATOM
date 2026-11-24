# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.moe.router_replayer import RouterReplayer
from torch import nn


class _DummyMoE(nn.Module):
    def __init__(self, dim: int = 4, num_experts: int = 3) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def test_replay_injects_distributions() -> None:
    model = _DummyMoE()
    x = torch.randn(2, 4)
    baseline = model(x).detach()
    injected = torch.full_like(baseline, 7.25)
    replayer = RouterReplayer()
    replayer.install_hooks(model, {0: injected.cpu()})
    try:
        out = model(x)
        assert torch.allclose(out, injected.to(out.device))
    finally:
        replayer.remove_hooks()
