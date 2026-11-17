# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Logical parallel topology resolved at the configuration boundary.

The native and plugin launchers consume this topology to create groups. Rank
ownership is a pure projection of DP x PP x PCP x TP; it never comes from a
process group. Physical widths are explicit for simulated deployments.
"""

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from atom.model_engine.capabilities import pcp_group_size_of
from atom.moe_config import MoEParallelMapping, get_moe_backend_config

if TYPE_CHECKING:
    from atom.config import Config


@dataclass(frozen=True)
class ParallelRankMapping:
    global_rank: int
    tp_rank: int
    dp_rank: int
    pp_rank: int
    pcp_rank: int


@dataclass(frozen=True)
class ParallelMapping:
    tp_size: int
    tp_world_size: int
    dp_size: int
    local_dp_size: int
    pp_size: int
    pcp_size: int
    dp_rank_offset: int
    dp_logical_size: int
    enable_ep: bool
    flatten_dp: bool
    merge_pcp: bool
    enable_dp_attention: bool = False
    dp_isolated: bool = False

    def __post_init__(self):
        for name in ("tp_size", "tp_world_size", "dp_size", "pp_size", "pcp_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one")
        if not 1 <= self.tp_world_size <= self.tp_size:
            raise ValueError("Physical TP width must be within the logical TP width")
        if not 1 <= self.local_dp_size <= self.dp_size:
            raise ValueError("Local DP width must be within the global DP width")
        if not 0 <= self.dp_rank_offset < self.dp_size:
            raise ValueError("DP rank offset must be within the global DP width")
        if self.dp_rank_offset + self.local_dp_size > self.dp_size:
            raise ValueError("Local DP ranks must fit within the global DP width")
        # Validate logical expert widths even before a worker is launched.
        self.moe_for_rank(0)

    def worker_topology(self) -> "ParallelMapping":
        """RLHF's launch DP ranks each own an independent TP-only world."""
        if not self.dp_isolated:
            return self
        return replace(
            self,
            dp_size=1,
            local_dp_size=1,
            dp_rank_offset=0,
            dp_logical_size=1,
            dp_isolated=False,
        )

    @property
    def stage_size(self) -> int:
        return self.tp_world_size * self.pcp_size

    @property
    def world_size(self) -> int:
        return self.dp_size * self.pp_size * self.stage_size

    def global_rank(self, stage_rank: int, dp_rank: int, pp_rank: int = 0) -> int:
        if not 0 <= stage_rank < self.stage_size:
            raise ValueError("Worker rank must be within its TP x PCP stage")
        if not 0 <= dp_rank < self.dp_size or not 0 <= pp_rank < self.pp_size:
            raise ValueError("Worker DP/PP rank is outside the resolved topology")
        return (dp_rank * self.pp_size + pp_rank) * self.stage_size + stage_rank

    def for_rank(self, global_rank: int) -> ParallelRankMapping:
        if not 0 <= global_rank < self.world_size:
            raise ValueError("Global rank is outside the resolved topology")
        outer, tp_rank = divmod(global_rank, self.tp_world_size)
        outer, pcp_rank = divmod(outer, self.pcp_size)
        dp_rank, pp_rank = divmod(outer, self.pp_size)
        return ParallelRankMapping(global_rank, tp_rank, dp_rank, pp_rank, pcp_rank)

    def moe_for_rank(self, global_rank: int) -> MoEParallelMapping:
        rank = self.for_rank(global_rank)
        return MoEParallelMapping.make(
            tp_size=self.tp_size,
            tp_rank=rank.tp_rank,
            dp_size=self.dp_size,
            dp_rank=rank.dp_rank,
            local_dp_size=self.local_dp_size,
            dp_logical_size=self.dp_logical_size,
            enable_ep=self.enable_ep,
            flatten_dp=self.flatten_dp,
            pcp_size=self.pcp_size,
            pcp_rank=rank.pcp_rank,
            merge_pcp=self.merge_pcp,
            enable_dp_attention=self.enable_dp_attention,
        )

    def group_ranks(self, global_rank: int, dimension: str) -> tuple[int, ...]:
        rank = self.for_rank(global_rank)
        dimensions = {
            "tp": {"tp"},
            "dp": {"dp"},
            "pp": {"pp"},
            "pcp": {"pcp"},
            # AITER's transport group spans DP x PCP x TP. Expert checkpoint
            # sharding may have a different width/order; do not change either.
            "ep": {"dp", "pcp", "tp"},
        }
        varied = dimensions[dimension]
        return tuple(
            self.global_rank(pcp * self.tp_world_size + tp, dp, pp)
            for dp in (range(self.dp_size) if "dp" in varied else (rank.dp_rank,))
            for pp in (range(self.pp_size) if "pp" in varied else (rank.pp_rank,))
            for pcp in (range(self.pcp_size) if "pcp" in varied else (rank.pcp_rank,))
            for tp in (range(self.tp_world_size) if "tp" in varied else (rank.tp_rank,))
        )


def resolve_parallel_mapping(
    config: "Config", *, visible_devices: int | None = None
) -> ParallelMapping:
    """Normalize native DP-attention and already-translated plugin topology once."""
    parallel = config.parallel_config
    native = config.plugin_config is None or not config.plugin_config.is_plugin_mode
    tp = config.tensor_parallel_size
    dp = parallel.data_parallel_size
    local_dp = parallel.data_parallel_size_local or dp
    offset = parallel.data_parallel_rank if native and parallel.is_multinode_dp else 0
    dp_attention = config.enable_dp_attention
    if native and dp_attention:
        dp *= tp
        local_dp *= tp
        offset *= tp
        tp = 1
    logical_dp = config.dp_logical_size or dp
    physical_tp = tp
    if native and config.fake_eplb and visible_devices:
        if dp_attention and visible_devices < local_dp:
            if dp != local_dp:
                raise ValueError(
                    "Simulated DP-attention requires a single-node topology"
                )
            if dp % visible_devices:
                raise ValueError(
                    "Simulated DP width must divide by visible device count"
                )
            dp = local_dp = visible_devices
        elif not dp_attention:
            physical_tp = min(tp, visible_devices)
    sp = config.sequence_parallel_size
    return ParallelMapping(
        tp_size=tp,
        tp_world_size=physical_tp,
        dp_size=dp,
        local_dp_size=local_dp,
        pp_size=config.pipeline_parallel_size,
        pcp_size=pcp_group_size_of(config),
        dp_rank_offset=offset,
        dp_logical_size=logical_dp,
        enable_ep=config.enable_expert_parallel,
        flatten_dp=dp_attention or config.moe_ep_flatten_tp_across_dp,
        merge_pcp=get_moe_backend_config(config).pcp_merge or sp > 1,
        enable_dp_attention=dp_attention,
        dp_isolated=(
            config.dp_isolation_device_map_env is not None
            and (dp > 1 or config.dp_isolation_device_map_env in os.environ)
        ),
    )


def get_parallel_mapping(config: "Config") -> ParallelMapping:
    """Return Config's saved topology, resolving it once when first requested."""
    cached = config.parallel_mapping
    if cached is None:
        cached = resolve_parallel_mapping(config)
        config.parallel_mapping = cached
    return cached


def apply_parallel_mapping(config: "Config") -> None:
    """Populate legacy runtime fields from the saved mapping, without re-resolving."""
    mapping = get_parallel_mapping(config)
    config.tensor_parallel_size = mapping.tp_size
    config.parallel_config.data_parallel_size = mapping.dp_size
    config.parallel_config.data_parallel_size_local = mapping.local_dp_size
    config.parallel_config.data_parallel_rank = mapping.dp_rank_offset
    config.dp_logical_size = (
        mapping.dp_logical_size if mapping.dp_logical_size != mapping.dp_size else 0
    )
