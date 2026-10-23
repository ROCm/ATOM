# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Deployment-sized QSA scoring and fixed-LDS exact selection tree."""

from dataclasses import dataclass

from atom.models.qwen4_exp_mono.shapes import (
    QSA_COMPRESS_RATIO,
    QSA_INDEX_BUDGET,
)

# Preserve the accepted 8K kernel's 2048-key LDS selection. Larger deployments
# add lists, not LDS. Winners remain in logical-index order at every level.
INDEX_SEGMENT = 2048
INDEX_GROUPS_PER_TASK = 16
SELECTED_GROUPS = QSA_INDEX_BUDGET // QSA_COMPRESS_RATIO
SELECT_FAN = INDEX_SEGMENT // SELECTED_GROUPS


@dataclass(frozen=True)
class QsaPlan:
    max_model_len: int

    def __post_init__(self):
        if self.max_model_len <= 0:
            raise ValueError("QSA deployment capacity must be positive")

    @property
    def columns(self) -> int:
        groups = -(-self.max_model_len // QSA_COMPRESS_RATIO)
        return -(-groups // INDEX_GROUPS_PER_TASK) * INDEX_GROUPS_PER_TASK

    @property
    def score_tasks(self) -> int:
        return self.columns // INDEX_GROUPS_PER_TASK

    @property
    def lists(self) -> tuple[int, ...]:
        count = -(-self.columns // INDEX_SEGMENT)
        levels = [count]
        while count > 1:
            count = -(-count // SELECT_FAN)
            levels.append(count)
        return tuple(levels)
