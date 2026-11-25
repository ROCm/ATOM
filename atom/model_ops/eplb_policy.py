# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""What `EPLBManager` asks of an expert placement policy.

The manager owns the schedule (when a rebalance may run), the group's load
statistics (`ExpertLoadMonitor`) and the migration; a policy decides which
layers it places, what statistics it needs and the placement a rebalance
moves to. `placement_policy` in `EPLBConfig` names one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


def physical_load_to_logical_load(
    physical_load: torch.Tensor,
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
) -> torch.Tensor:
    """Fold [layers, physical] load into [layers, logical] by live placement."""
    assert physical_load.dim() == 2
    assert physical_to_logical_map.shape == physical_load.shape
    out = torch.zeros(
        (physical_load.shape[0], num_logical_experts),
        dtype=physical_load.dtype,
        device=physical_load.device,
    )
    idx = physical_to_logical_map.to(torch.int64)
    valid = idx >= 0
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    out.scatter_add_(
        1, safe_idx, torch.where(valid, physical_load, torch.zeros_like(physical_load))
    )
    return out


@dataclass(frozen=True)
class PlanContext:
    """The placement a rebalance starts from and the layers it may move."""

    live: Any  # ExpertLocationMetadata: the live placement
    moe_layers: dict[int, Any]  # placed layer id -> its EP MoE module
    num_nodes: int
    dp_size: int  # row sets the EP group serves a step (1: every rank's alike)


@dataclass(frozen=True)
class PlacementPlan:
    """A rebalance's new placement ([layers, ...] like the live one) and the
    layers whose experts move. No layer to move ends rebalancing, for
    ``stop_reason``."""

    physical_to_logical: torch.Tensor
    logical_to_physical: torch.Tensor
    replica_count: torch.Tensor
    layer_ids: list[int]
    stop_reason: str = ""
    rearrange_ms: float = 0.0  # computing the placement, its loads excluded
    logical_load: torch.Tensor | None = None  # the load planned on, for logs


class PlacementPolicy(ABC):
    """A placement policy; `EPLBManager` drives it through these members."""

    # Place every EP MoE layer, past a model's `eplb_layer_end` too.
    places_all_layers: bool = False
    # Rebalance only when the live placement's balancedness is below the
    # manager's threshold; a policy that is not gated rebalances on schedule.
    gated: bool = True

    def check_deployment(
        self,
        *,
        num_logical: int,
        num_physical: int,
        ep_size: int,
        dp_size: int,
        multi_node: bool,
    ) -> None:
        """Refuse at start a MoE layout or parallel deployment the policy
        cannot place."""

    def start(self, monitor: Any, live: Any) -> None:
        """Switch on the statistics the policy plans on, once the live
        placement exists."""

    def stop(self, monitor: Any) -> None:
        """Switch the statistics off: rebalancing stopped, no plan reads them
        again."""

    @abstractmethod
    def plan(self, monitor: Any, ctx: PlanContext) -> PlacementPlan | None:
        """The next placement from the group's loads, or None when there is
        nothing to plan on (the rebalance is skipped and not counted). Every
        rank derives the same one from the same reduced loads."""
