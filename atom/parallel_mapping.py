# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Logical parallel topology resolved at the configuration boundary.

The native and plugin launchers consume this topology to create groups. Rank
ownership is a pure projection of DP x PP x PCP x TP; it never comes from a
process group. Physical widths are explicit for simulated deployments.
"""

from dataclasses import dataclass

from atom.moe_config import MoEParallelMapping, get_moe_backend_config


@dataclass(frozen=True)
class ParallelRankMapping:
    global_rank: int
    tp_rank: int
    dp_rank: int
    pp_rank: int
    pcp_rank: int
    moe: MoEParallelMapping


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
        self.for_rank(0)

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
        moe = MoEParallelMapping.make(
            tp_size=self.tp_size,
            tp_rank=tp_rank,
            dp_size=self.dp_size,
            dp_rank=dp_rank,
            local_dp_size=self.local_dp_size,
            dp_logical_size=self.dp_logical_size,
            enable_ep=self.enable_ep,
            flatten_dp=self.flatten_dp,
            pcp_size=self.pcp_size,
            pcp_rank=pcp_rank,
            merge_pcp=self.merge_pcp,
        )
        return ParallelRankMapping(
            global_rank, tp_rank, dp_rank, pp_rank, pcp_rank, moe
        )

    def group_ranks(self, global_rank: int, dimension: str) -> tuple[int, ...]:
        rank = self.for_rank(global_rank)
        dimensions = {
            "tp": {"tp"},
            "dp": {"dp"},
            "pp": {"pp"},
            "pcp": {"pcp"},
            "ep": {"tp"}
            | ({"dp"} if self.flatten_dp else set())
            | ({"pcp"} if self.merge_pcp else set()),
        }
        varied = dimensions[dimension]
        return tuple(
            self.global_rank(pcp * self.tp_world_size + tp, dp, pp)
            for dp in (range(self.dp_size) if "dp" in varied else (rank.dp_rank,))
            for pp in (range(self.pp_size) if "pp" in varied else (rank.pp_rank,))
            for pcp in (range(self.pcp_size) if "pcp" in varied else (rank.pcp_rank,))
            for tp in (range(self.tp_world_size) if "tp" in varied else (rank.tp_rank,))
        )


def resolve_parallel_mapping(config, *, visible_devices: int | None = None):
    """Normalize native DP-attention and already-translated plugin topology once."""
    parallel = getattr(config, "parallel_config", None)
    plugin = getattr(config, "plugin_config", None)
    native = not getattr(plugin, "is_plugin_mode", False)
    tp = getattr(config, "tensor_parallel_size", 1)
    dp = getattr(parallel, "data_parallel_size", 1)
    local_dp = getattr(parallel, "data_parallel_size_local", None) or dp
    offset = (
        getattr(parallel, "data_parallel_rank", 0)
        if native and getattr(parallel, "is_multinode_dp", False)
        else 0
    )
    dp_attention = getattr(config, "enable_dp_attention", False)
    if native and dp_attention:
        dp *= tp
        local_dp *= tp
        offset *= tp
        tp = 1
    logical_dp = getattr(config, "dp_logical_size", 0) or dp
    physical_tp = tp
    if native and getattr(config, "fake_eplb", False) and visible_devices:
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
    sp = getattr(config, "sequence_parallel_size", 1)
    return ParallelMapping(
        tp_size=tp,
        tp_world_size=physical_tp,
        dp_size=dp,
        local_dp_size=local_dp,
        pp_size=getattr(config, "pipeline_parallel_size", 1),
        pcp_size=max(getattr(config, "prefill_context_parallel_size", 1), sp),
        dp_rank_offset=offset,
        dp_logical_size=logical_dp,
        enable_ep=getattr(config, "enable_expert_parallel", False),
        flatten_dp=dp_attention
        or getattr(config, "moe_ep_flatten_tp_across_dp", False),
        merge_pcp=get_moe_backend_config(config).pcp_merge or sp > 1,
    )


def get_parallel_mapping(config) -> ParallelMapping:
    """Saved topology; lightweight CPU callers may resolve it explicitly here."""
    cached = getattr(config, "parallel_mapping", None)
    if cached is None:
        cached = resolve_parallel_mapping(config)
        config.parallel_mapping = cached
    return cached


def apply_parallel_mapping(config) -> None:
    """Populate legacy runtime fields from the saved mapping, without re-resolving."""
    mapping = get_parallel_mapping(config)
    config.tensor_parallel_size = mapping.tp_size
    config.parallel_config.data_parallel_size = mapping.dp_size
    config.parallel_config.data_parallel_size_local = mapping.local_dp_size
    config.parallel_config.data_parallel_rank = mapping.dp_rank_offset
    config.dp_logical_size = (
        mapping.dp_logical_size if mapping.dp_logical_size != mapping.dp_size else 0
    )
