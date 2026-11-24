# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Ray cluster initialization and resource pool management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import ray

from atomrl.core.config import ClusterConfig

logger = logging.getLogger(__name__)


@dataclass
class ResourcePool:
    """A pool of GPU resources for a worker group.

    ``process_on_nodes`` models complex topology layouts (e.g. [4,4,2]).
    If omitted, we default to a single-node layout of ``num_gpus`` workers.
    """

    name: str
    num_gpus: int
    num_cpus: int = 0
    node_indices: list[int] = field(default_factory=list)
    colocate_with: str | None = None
    process_on_nodes: list[int] = field(default_factory=list)
    max_colocate_count: int = 1
    detached: bool = False
    topology_tags: dict[str, str] = field(default_factory=dict)
    use_placement_groups: bool = False

    def __post_init__(self) -> None:
        if not self.process_on_nodes:
            self.process_on_nodes = [max(0, int(self.num_gpus))]

    @property
    def world_size(self) -> int:
        return sum(self.process_on_nodes)

    def split(
        self, split_size: int | list[int], suffix: str | None = None
    ) -> list[ResourcePool]:
        """Split current pool into sub-pools by worker counts."""
        if isinstance(split_size, int):
            if split_size <= 0 or self.world_size % split_size != 0:
                raise ValueError(
                    "split_size must be a positive divisor of pool world_size."
                )
            split_list = [split_size] * (self.world_size // split_size)
        else:
            split_list = list(split_size)
            if any(v <= 0 for v in split_list) or sum(split_list) != self.world_size:
                raise ValueError(
                    "split_size list must be positive and sum to pool world_size."
                )

        out: list[ResourcePool] = []
        for idx, sz in enumerate(split_list):
            name = (
                f"{self.name}_split_{idx}"
                if suffix is None
                else f"{self.name}_{suffix}_{idx}"
            )
            out.append(
                ResourcePool(
                    name=name,
                    num_gpus=sz,
                    num_cpus=sz,
                    node_indices=self.node_indices,
                    colocate_with=self.colocate_with,
                    process_on_nodes=[sz],
                    max_colocate_count=self.max_colocate_count,
                    detached=self.detached,
                    topology_tags=dict(self.topology_tags),
                )
            )
        return out

    def merge_with(self, other: ResourcePool, name: str | None = None) -> ResourcePool:
        """Merge two compatible pools into one."""
        if self.max_colocate_count != other.max_colocate_count:
            raise ValueError("Cannot merge pools with different max_colocate_count.")
        merged_name = name or f"{self.name}_{other.name}"
        return ResourcePool(
            name=merged_name,
            num_gpus=self.num_gpus + other.num_gpus,
            num_cpus=self.num_cpus + other.num_cpus,
            node_indices=self.node_indices + other.node_indices,
            colocate_with=self.colocate_with or other.colocate_with,
            process_on_nodes=self.process_on_nodes + other.process_on_nodes,
            max_colocate_count=self.max_colocate_count,
            detached=self.detached and other.detached,
            topology_tags={**self.topology_tags, **other.topology_tags},
        )


class RayCluster:
    """Manages Ray cluster lifecycle and resource pools."""

    def __init__(self, config: ClusterConfig) -> None:
        self.config = config
        self._pools: dict[str, ResourcePool] = {}
        self._initialized = False

    def init(self) -> None:
        """Initialize the Ray cluster."""
        if self._initialized:
            return

        ray_kwargs: dict = {}
        if self.config.ray_address:
            ray_kwargs["address"] = self.config.ray_address
        else:
            ray_kwargs["num_gpus"] = self.config.num_nodes * self.config.gpus_per_node

        ray.init(**ray_kwargs, ignore_reinit_error=True)
        self._initialized = True
        logger.info(
            "Ray cluster initialized: %d nodes x %d GPUs",
            self.config.num_nodes,
            self.config.gpus_per_node,
        )

    def shutdown(self) -> None:
        if self._initialized:
            ray.shutdown()
            self._initialized = False

    def create_pool(
        self,
        name: str,
        num_gpus: int,
        colocate_with: str | None = None,
        process_on_nodes: list[int] | None = None,
        max_colocate_count: int = 1,
        detached: bool = False,
        topology_tags: dict[str, str] | None = None,
    ) -> ResourcePool:
        """Create a named resource pool."""
        explicit_pg = process_on_nodes is not None
        process_layout = process_on_nodes[:] if explicit_pg else [num_gpus]
        pool = ResourcePool(
            name=name,
            num_gpus=num_gpus,
            num_cpus=num_gpus,
            colocate_with=colocate_with,
            process_on_nodes=process_layout,
            max_colocate_count=max_colocate_count,
            detached=detached,
            topology_tags=dict(topology_tags or {}),
            use_placement_groups=explicit_pg,
        )
        self._pools[name] = pool
        logger.info(
            "Created resource pool '%s' with %d GPUs (layout=%s, colocate=%s)",
            name,
            num_gpus,
            process_layout,
            colocate_with,
        )
        return pool

    def get_pool(self, name: str) -> ResourcePool:
        if name not in self._pools:
            raise KeyError(f"Resource pool '{name}' not found")
        return self._pools[name]

    @property
    def total_gpus(self) -> int:
        return self.config.num_nodes * self.config.gpus_per_node

    def map_topology(self, mapping: dict[str, str]) -> dict[str, ResourcePool]:
        """Map role names to existing pools for complex topology plans."""
        resolved: dict[str, ResourcePool] = {}
        for role, pool_name in mapping.items():
            resolved[role] = self.get_pool(pool_name)
        return resolved
