# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Rank-local verdicts every rank of a CPU (gloo) process group takes alike.

Every rank must call these at the same point; ``group`` None is a single rank.
"""

from typing import NamedTuple

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


class AgreedKvBudget(NamedTuple):
    """Free memory and KV budget every rank sizes its pool from."""

    free: int
    available_for_kv_budget: int
    free_for_kv: int
    available_for_kv: int


def agree_kv_budget(
    free: int,
    available_for_kv_budget: int,
    connector_bytes: int,
    groups,
) -> AgreedKvBudget:
    """Minimum free memory and KV budget over each group larger than one rank.

    ``plan_pools`` sizes the block count from ``available_for_kv``. The
    connector reserve is checked against the agreed free memory, so a rank
    that is short raises here on every rank or on none.
    """
    for group in groups:
        if group.world_size > 1:
            free = group_min(free, group.cpu_group)
            available_for_kv_budget = group_min(
                available_for_kv_budget, group.cpu_group
            )
    # Free HBM has to cover the buffers the connector allocates after the
    # KV cache is sized, or the cache is planned into memory that is not there.
    free_for_kv = free - connector_bytes
    if connector_bytes and free_for_kv <= 0:
        raise RuntimeError(
            f"Only {free / (1 << 30):.2f}GB is free on the GPU, less than "
            f"the {connector_bytes / (1 << 30):.2f}GB the KV connector "
            "buffers need after the KV cache is sized; free GPU memory or "
            "turn those buffers off (ATOM_PD_MLA_STAGING=0 / "
            "ATOM_PD_MLA_LANDING=0)."
        )
    return AgreedKvBudget(
        free,
        available_for_kv_budget,
        free_for_kv,
        min(available_for_kv_budget, free_for_kv),
    )
