# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank-local verdicts every rank of a CPU (gloo) process group takes alike.

Every rank must call these at the same point; ``group`` None is a single rank.
"""

import torch
import torch.distributed as dist


def group_max(value: int, group) -> int:
    """The largest ``value`` any rank of ``group`` passed."""
    if group is None:
        return value
    flag = torch.tensor([value], dtype=torch.int64, device="cpu")
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return int(flag.item())


def group_min(value: int, group) -> int:
    """The smallest ``value`` any rank of ``group`` passed."""
    return -group_max(-value, group)


def group_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true."""
    return not group_max(int(not ok), group)


def agree_kv_budget(
    free: int,
    available_for_kv_budget: int,
    groups,
) -> tuple[int, int]:
    """Minimum free memory and KV budget over each group larger than one rank.

    Returns ``(free, available_for_kv_budget)``, the values every rank of the
    groups takes alike. Callers must do any check that can raise on the
    returned values, not on their own ``free``.
    """
    for group in groups:
        if group.world_size > 1:
            free = group_min(free, group.cpu_group)
            available_for_kv_budget = group_min(
                available_for_kv_budget, group.cpu_group
            )
    return free, available_for_kv_budget
