# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""MoE R3 record/replay alignment using local router stubs."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from atomrl.core.config import R3Config
from atomrl.core.protocol import DataProto
from atomrl.moe.r3_manager import R3Manager
from torch import nn

pytestmark = pytest.mark.moe


class SparseMoeStub(nn.Module):
    """Minimal module name-matched for :func:`iter_moe_modules` discovery."""

    def __init__(self, dim: int = 32, num_experts: int = 8) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return router logits only so R3 hooks match a single tensor output."""
        return self.gate(x)


def _router_logits(output: torch.Tensor) -> torch.Tensor:
    return output


def test_r3_reduces_kl() -> None:
    """Replayed router logits match the recording; natural forward on new inputs does not."""
    torch.manual_seed(7)
    dim, experts = 32, 8
    model = SparseMoeStub(dim, experts)
    x0 = torch.randn(2, 4, dim)
    x1 = torch.randn(2, 4, dim)

    with torch.no_grad():
        recorded_logits = _router_logits(model(x0))
        natural_logits = _router_logits(model(x1))

        p = F.log_softmax(recorded_logits, dim=-1)
        q_natural = F.softmax(natural_logits, dim=-1)
        kl_natural = F.kl_div(p, q_natural, reduction="batchmean", log_target=False)

        r3 = R3Manager(
            R3Config(
                enabled=True, record_router_logits=True, replay_mode="distribution"
            )
        )
        with r3.replay_phase(model, {0: recorded_logits}):
            replayed_logits = _router_logits(model(x1))

        q_replay = F.softmax(replayed_logits, dim=-1)
        kl_replay = F.kl_div(p, q_replay, reduction="batchmean", log_target=False)

    assert float(kl_replay) < float(kl_natural)
    assert float(kl_replay) < 1e-3


def test_router_distributions_preserved() -> None:
    """Round-trip router tensors through DataProto helpers and R3Manager.transfer_distributions."""
    base = DataProto(tensors={"mask": torch.ones(2, 5, dtype=torch.long)})
    logits = torch.randn(2, 8, dtype=torch.float64)
    base.add_router_distributions(3, logits)

    recorded = base.get_router_distributions()
    assert 3 in recorded
    transferred = R3Manager.transfer_distributions(
        DataProto(tensors={"mask": torch.ones(2, 5, dtype=torch.long)}),
        recorded,
    )
    got = transferred.get_router_distributions()[3]
    assert torch.equal(got, recorded[3].cpu())
