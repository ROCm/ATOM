# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bind resolved rank ownership to native or plugin process groups once."""

import logging
from dataclasses import replace

from atom.moe_config import get_moe_execution_plan
from atom.parallel_mapping import get_parallel_mapping

logger = logging.getLogger("atom")


def bind_worker_parallel_mapping(config, *, global_rank: int) -> None:
    from aiter.dist import parallel_state as groups

    topology = get_parallel_mapping(config)
    rank = topology.for_rank(global_rank)
    plan = replace(get_moe_execution_plan(config), mapping=rank.moe)
    checks = [
        ("tp", groups.get_tp_group(), topology.tp_size, rank.tp_rank),
        ("dp", groups.get_dp_group(), topology.dp_size, rank.dp_rank),
    ]
    # TP-only vLLM reuse does not install a PCP coordinator. AITER exposes
    # the singleton PCP dimension through these accessors in that case.
    pcp_size = groups.get_prefill_context_model_parallel_world_size()
    pcp_rank = groups.get_prefill_context_model_parallel_rank()
    if (pcp_size, pcp_rank) != (topology.pcp_size, rank.pcp_rank):
        raise ValueError("Resolved PCP mapping must match the process group")
    if pcp_size > 1:
        checks.append(("pcp", groups.get_pcp_group(), pcp_size, pcp_rank))
    if topology.pp_size > 1:
        checks.append(("pp", groups.get_pp_group(), topology.pp_size, rank.pp_rank))
    if plan.uses_routed_dispatch:
        checks.append(("ep", groups.get_ep_group(), rank.moe.ep_size, rank.moe.ep_rank))
    for dimension, group, size, index in checks:
        if (group.world_size, group.rank_in_group) != (size, index):
            raise ValueError(
                f"Resolved {dimension.upper()} mapping must match the process group: "
                f"mapping={index}/{size}, group={group.rank_in_group}/{group.world_size}"
            )
        expected = topology.group_ranks(global_rank, dimension)
        if tuple(group.ranks) != expected:
            raise ValueError(
                f"Resolved {dimension.upper()} group members/order must match the mapping: "
                f"mapping={expected}, group={tuple(group.ranks)}"
            )
    if getattr(config, "enable_tbo", False) and not plan.supports_tbo:
        raise ValueError(
            f"TBO is unsupported with resolved MoE transport {plan.transport!r}"
        )
    config.worker_parallel_mapping = rank
    config.worker_moe_execution_plan = plan
    logger.info(
        "Resolved worker mapping: rank=%d TP=%d/%d DP=%d/%d PCP=%d/%d PP=%d/%d "
        "MoE-TP=%d/%d EP=%d/%d backend=%s transport=%s (%s)",
        global_rank,
        rank.tp_rank,
        topology.tp_size,
        rank.dp_rank,
        topology.dp_size,
        rank.pcp_rank,
        topology.pcp_size,
        rank.pp_rank,
        topology.pp_size,
        rank.moe.tp_rank,
        rank.moe.tp_size,
        rank.moe.ep_rank,
        rank.moe.ep_size,
        plan.implementation,
        plan.transport,
        plan.reason,
    )
