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
    flag = torch.tensor([value], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return int(flag.item())


def group_agree(ok: bool, group) -> bool:
    """True when every rank of ``group`` passed ``ok`` true."""
    return not group_max(int(not ok), group)
