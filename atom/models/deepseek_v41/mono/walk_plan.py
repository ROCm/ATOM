# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FP4 score stage's FULL-layer task plan, a plan a compression ratio,
laid out on the host while the step's metadata is staged
(``index_plan.fill_walk_plan``) and published with it: the step builder calls
``WalkPlanner`` once its rows' visibility and batch ids are on the host."""

import torch

from atom.model_ops.attentions.deepseek_v41.metadata import (
    StepPlan,
    visible_buffer_name,
)
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import MAX_ROWS
from atom.mono.plan.execution import BLOCKS
from atom.utils import CpuGpuBuffer


def walk_plan_name(ratio: int) -> str:
    """The step buffer holding ratio ``ratio``'s plan ([BLOCKS, 4] int32)."""
    return f"v41_mono_walk_plan_{ratio}"


class WalkPlanner:
    """A V4.1 step planner (``add_step_planner``): each FULL ratio's plan over
    the step's rows, for a step mono can take (at most ``MAX_ROWS`` rows)."""

    def __init__(self, ratios) -> None:
        self.ratios = tuple(ratios)

    def buffers(self, device, publication_group: str) -> dict:
        return {
            walk_plan_name(ratio): CpuGpuBuffer(
                BLOCKS,
                4,
                dtype=torch.int32,
                device=device,
                pin_memory=torch.device(device).type != "cpu",
                publication_group=publication_group,
            )
            for ratio in self.ratios
        }

    def __call__(self, buffers, rows: int) -> StepPlan:
        """Lay the plans out of ``buffers``' staged rows: the rows each plan
        publishes, by buffer name."""
        if rows > MAX_ROWS:
            return StepPlan({})
        owners = buffers["batch_id_per_q_token"].np[:rows]
        for ratio in self.ratios:
            ip.fill_walk_plan(
                buffers[visible_buffer_name(ratio)].np[:rows],
                owners,
                buffers[walk_plan_name(ratio)].np,
            )
        return StepPlan({walk_plan_name(ratio): BLOCKS for ratio in self.ratios})
