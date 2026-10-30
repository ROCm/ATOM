# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Logical tasks and resident CTA placement for the gfx1250 K2 schedule.

The optional c1 placement follows gfx950 attn_post's first_task mechanism:
score at 0, PV at 64, WOA at 160, then WOB after the WOA task range. The
wave32 arithmetic tiles and validated 320-CTA grid remain the backend's.
Payload/ready slots are always indexed by logical task, never physical CTA.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionExecutionPlan:
    blocks: int
    score_tasks: int
    pv_tasks: int
    woa_tasks: int
    score_base: int = 0
    pv_base: int = 0
    woa_base: int = 0
    wob_base: int = 0

    def task(self, bid, base):
        if base % self.blocks == 0:
            return bid
        return (bid - base % self.blocks + self.blocks) % self.blocks

    def owner(self, task, base):
        return (task + base) % self.blocks


def attention_execution_plan(rows, heads, placed=False):
    if rows not in (1, 2) or heads not in (16, 32):
        raise ValueError("gfx1250 attention plan supports rows1/2, TP2/TP4 heads")
    blocks = 5120 // 16
    tiles = heads // 16
    woa_tasks = heads // 8 * 1024 // 32
    placed = placed and rows == 1
    return AttentionExecutionPlan(
        blocks, rows * tiles * 8, rows * tiles * 16, woa_tasks,
        pv_base=64 if placed else 0,
        woa_base=160 if placed else 0,
        wob_base=160 + woa_tasks if placed else 0,
    )
