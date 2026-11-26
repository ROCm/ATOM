# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Composing blocks into one layer program, and binding its pointer tables.

A layer recipe is an ordered list of blocks. Each block's stages run in that order, and blocks talk through
channels named in their constructors, so a recipe reads like the layer's dataflow.
"""

from dataclasses import dataclass, field

import torch

from atom.models.qwen38.mono.kernels.blocks.base import Block
from atom.models.qwen38.mono.kernels.kernel import MonoKernel
from atom.models.qwen38.mono.kernels.program import PlanError, Program, Scenario
from atom.models.qwen38.mono.kernels.target import MI355X


@dataclass
class Schedule:
    """Placement on top of the blocks' stages: ``place`` (stage name -> first CTA) and ``insert``
    ((stage name, block): the block's stages go right after that stage in program order).
    """

    place: dict[str, int] = field(default_factory=dict)
    insert: list[tuple[str, Block]] = field(default_factory=list)


def compose(
    name: str, scenario: Scenario, blocks: list[Block], schedule: Schedule | None = None
) -> Program:
    """The program of ``blocks``' stages in order with ``schedule`` applied (placements wrap at the grid)."""
    target = MI355X
    stages = []
    for block in blocks:
        stages += block.stages(scenario)
    if schedule:
        for after, block in schedule.insert:
            idx = [i for i, s in enumerate(stages) if s.name == after]
            if not idx:
                raise PlanError(
                    f"schedule: no stage {after} to insert {block.name} after"
                )
            stages[idx[0] + 1 : idx[0] + 1] = block.stages(scenario)
        names = {s.name for s in stages}
        for stage in schedule.place:
            if stage not in names:
                raise PlanError(f"schedule: no stage {stage}")
        for s in stages:
            if s.name in schedule.place:
                s.base = schedule.place[s.name] % target.grid
    return Program(name, scenario, stages, target=target)


class LayerKernel:
    """A composed, checked layer kernel plus how to fill its pointer table."""

    def __init__(self, program: Program, blocks: list[Block]):
        self.blocks = blocks
        self.kernel = MonoKernel(program)
        self.program = self.kernel.program

    def table(self, packed: dict[str, torch.Tensor], runtime: dict) -> torch.Tensor:
        """The pointer table from packed weights and runtime args (inputs, outputs, caches, metadata)."""
        both = dict(packed)
        both.update(runtime)
        return self.kernel.table(both)
