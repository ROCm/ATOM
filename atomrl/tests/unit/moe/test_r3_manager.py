# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.core.config import R3Config
from atomrl.core.protocol import DataProto
from atomrl.moe.r3_manager import R3Manager
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


def test_lifecycle_record_transfer_replay_cleanup() -> None:
    cfg = R3Config(enabled=True, record_router_logits=True, replay_mode="distribution")
    mgr = R3Manager(cfg)
    model = _DummyMoE()
    x = torch.randn(2, 4)

    with mgr.record_phase(model):
        logits = model(x)
    recorded = mgr.recorder.get_distributions()
    assert len(recorded) >= 1
    assert torch.allclose(recorded[0], logits.detach().float().cpu())

    batch = DataProto(tensors={"reward": torch.zeros(2)})
    transferred = R3Manager.transfer_distributions(batch, recorded)
    assert transferred.has_router_distributions()

    with mgr.replay_phase(model, recorded):
        replayed = model(x)
    assert torch.allclose(replayed, recorded[0].to(replayed.device))

    mgr.clear()
    assert mgr.recorder.get_distributions() == {}


def test_disabled_r3() -> None:
    cfg = R3Config(enabled=False, record_router_logits=True)
    mgr = R3Manager(cfg)
    model = _DummyMoE()
    with mgr.record_phase(model):
        model(torch.randn(1, 4))
    assert mgr.recorder.get_distributions() == {}
