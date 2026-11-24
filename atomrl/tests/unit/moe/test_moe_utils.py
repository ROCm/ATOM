# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from __future__ import annotations

import torch
from atomrl.moe.moe_utils import (
    check_expert_utilization,
    compute_load_balance_loss,
    compute_router_entropy,
)


def test_load_balance_loss() -> None:
    num_experts = 4
    logits = torch.randn(8, num_experts)
    loss = compute_load_balance_loss(logits, num_experts=num_experts, top_k=2)
    assert loss.ndim == 0
    assert float(loss) >= 0.0


def test_router_entropy() -> None:
    logits = torch.randn(6, 5)
    ent = compute_router_entropy(logits)
    assert ent.ndim == 0
    assert float(ent) > 0.0


def test_expert_utilization() -> None:
    num_experts = 3
    logits = torch.randn(10, num_experts)
    util = check_expert_utilization(logits, num_experts)
    assert util["num_tokens"] == 10
    assert util["num_experts"] == num_experts
    assert len(util["mean_softmax_mass_per_expert"]) == num_experts
