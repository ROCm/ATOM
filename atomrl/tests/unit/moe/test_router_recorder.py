# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.moe.router_recorder import RouterRecorder
from torch import nn


class _DummyMoE(nn.Module):
    """Minimal module matching ``iter_moe_modules`` heuristics."""

    def __init__(self, dim: int = 4, num_experts: int = 3) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_experts)]
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gate(x)


def test_install_and_remove_hooks() -> None:
    model = _DummyMoE()
    rec = RouterRecorder()
    rec.install_hooks(model)
    assert len(rec._handles) > 0
    rec.remove_hooks()
    assert len(rec._handles) == 0


def test_record_logits() -> None:
    model = _DummyMoE()
    rec = RouterRecorder()
    rec.install_hooks(model)
    x = torch.randn(2, 4)
    logits = model(x)
    dists = rec.get_distributions()
    assert len(dists) >= 1
    layer0 = dists[0]
    assert layer0.shape == logits.shape
    assert torch.allclose(layer0, logits.detach().float().cpu())
    rec.remove_hooks()


def test_clear() -> None:
    model = _DummyMoE()
    rec = RouterRecorder()
    rec.install_hooks(model)
    model(torch.randn(1, 4))
    assert len(rec.get_distributions()) >= 1
    rec.clear()
    assert rec.get_distributions() == {}
    rec.remove_hooks()
