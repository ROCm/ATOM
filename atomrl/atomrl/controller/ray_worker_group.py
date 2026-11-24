# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Ray WorkerGroup: manages a set of Ray actors for a single role."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import ray

from atomrl.controller.dispatch import (
    DispatchMode,
    collect_proto,
    collect_with_mask,
    dispatch_proto,
)
from atomrl.controller.ray_cluster import ResourcePool
from atomrl.controller.worker_group_factory import resolve_worker_class
from atomrl.core.protocol import DataProto

logger = logging.getLogger(__name__)


@dataclass
class _SpawnSpec:
    prefix: str
    group: RayWorkerGroup


def _prefixed_method(prefix: str, method: str) -> str:
    if not prefix:
        return method
    return f"{prefix}_{method}"


def _placement_node_ips(
    topology_tags: dict[str, str],
    group_count: int,
) -> list[str]:
    """Resolve an optional node pin for each positive placement group."""
    node_ip = str(topology_tags.get("node_ip", "") or "").strip()
    node_ips_raw = str(topology_tags.get("node_ips", "") or "").strip()
    if node_ip and node_ips_raw:
        raise ValueError("topology_tags cannot define both node_ip and node_ips")
    if node_ips_raw:
        node_ips = [value.strip() for value in node_ips_raw.split(",") if value.strip()]
        if len(node_ips) != group_count:
            raise ValueError(
                "topology_tags.node_ips must contain one entry per positive "
                f"process_on_nodes group: got {len(node_ips)}, expected {group_count}"
            )
        return node_ips
    return [node_ip] * group_count


class RayWorkerGroup:
    """A group of Ray actor workers for a single RL role.

    Handles creation, dispatch, collection, and lifecycle of workers.
    """

    def __init__(
        self,
        worker_cls: type,
        pool: ResourcePool,
        num_workers: int,
        worker_kwargs: dict | None = None,
        role_key: str | None = None,
        dispatch_mode: DispatchMode | str = DispatchMode.DP_COMPUTE_PROTO,
        detached: bool | None = None,
        worker_names: list[str] | None = None,
        method_prefix: str = "",
    ) -> None:
        if role_key is not None:
            worker_cls = resolve_worker_class(role_key)
        self.worker_cls = worker_cls
        self.pool = pool
        self.num_workers = num_workers if num_workers > 0 else pool.world_size
        self.worker_kwargs = worker_kwargs or {}
        self.dispatch_mode = DispatchMode(dispatch_mode)
        self.detached = pool.detached if detached is None else detached
        self.method_prefix = method_prefix
        self._actors: list[ray.actor.ActorHandle] = []
        self._worker_names: list[str] = (
            worker_names[:] if worker_names is not None else []
        )
        self._lazy_dispatch_state: dict[str, Any] = {}
        self._spawned_groups: dict[str, _SpawnSpec] = {}
        self._dp_rank_mapping: list[int] | None = None
        self._collect_mask: list[bool] | None = None

    @property
    def dp_rank_mapping(self) -> list[int] | None:
        """Return the current public DP dispatch mapping, if configured."""
        return (
            list(self._dp_rank_mapping) if self._dp_rank_mapping is not None else None
        )

    def start(self) -> None:
        """Create and start all workers in this group."""
        if self._actors:
            return

        if self.detached and self._worker_names:
            self._actors = [ray.get_actor(name) for name in self._worker_names]
            return

        if self.pool.num_gpus <= 0:
            gpus_per_worker = 0.0
        else:
            base = self.pool.num_gpus / max(1, self.num_workers)
            gpus_per_worker = min(
                1.0, max(base, 1.0 / max(1, self.pool.max_colocate_count))
            )

        use_pg = getattr(self.pool, "use_placement_groups", False)
        if use_pg:
            self._start_with_placement_groups(gpus_per_worker)
        else:
            self._start_simple(gpus_per_worker)

        logger.info(
            "Started %d workers of type %s (%s GPUs each, placement_groups=%s)",
            self.num_workers,
            self.worker_cls.__name__,
            gpus_per_worker,
            use_pg,
        )

    def _start_simple(self, gpus_per_worker: float) -> None:
        """Original start path — no placement groups."""
        if len([n for n in (self.pool.process_on_nodes or []) if int(n) > 0]) > 1:
            # Ray then picks rank->node on its own, which a TP>1 rollout cannot
            # use: VLLMReplicaManager requires ranks i*tp..i*tp+tp-1 to share a
            # node. Only _start_with_placement_groups pins that.
            logger.warning(
                "pool '%s' declares a multi-node layout %s but placement groups are "
                "off, so rank->node is left to Ray. A TP>1 rollout will fail with "
                "'replica spans N nodes'.",
                self.pool.name,
                self.pool.process_on_nodes,
            )

        RemoteWorker = ray.remote(
            num_gpus=gpus_per_worker,
            num_cpus=1,
        )(self.worker_cls)

        for rank in range(self.num_workers):
            actor_name = f"{self.pool.name}:{self.worker_cls.__name__}:{rank}"
            options_kwargs: dict[str, Any] = {}
            if self.detached:
                options_kwargs["name"] = actor_name
                options_kwargs["lifetime"] = "detached"
            actor_ctor = (
                RemoteWorker.options(**options_kwargs)
                if options_kwargs
                else RemoteWorker
            )
            actor = actor_ctor.remote(
                rank=rank,
                world_size=self.num_workers,
                **self.worker_kwargs,
            )
            self._actors.append(actor)
            if self.detached:
                self._worker_names.append(actor_name)

    def _start_with_placement_groups(self, gpus_per_worker: float) -> None:
        """verl-style start: create per-node placement groups with STRICT_PACK.

        One placement group per ``process_on_nodes`` entry, filled with
        consecutive ranks, so rank -> node is deterministic. A TP>1 rollout
        depends on that: ``VLLMReplicaManager.create`` requires ranks
        ``i*tp .. i*tp+tp-1`` to share a node and raises "replica i spans N
        nodes" otherwise.
        """
        from ray.util.placement_group import placement_group
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

        pgs = []
        positive_group_count = sum(
            1 for count in self.pool.process_on_nodes if count > 0
        )
        pinned_node_ips = _placement_node_ips(
            self.pool.topology_tags,
            positive_group_count,
        )
        placement_idx = 0
        for node_idx, count in enumerate(self.pool.process_on_nodes):
            if count <= 0:
                continue
            pinned_node_ip = pinned_node_ips[placement_idx]
            placement_idx += 1
            bundles = []
            for _ in range(count):
                bundle = {"GPU": 1, "CPU": 1}
                if pinned_node_ip:
                    # Ray exposes a unique built-in resource for every node.
                    # A fractional request pins all STRICT_PACK bundles without
                    # consuming the whole node resource per worker.
                    bundle[f"node:{pinned_node_ip}"] = 0.001
                bundles.append(bundle)
            pg = placement_group(
                bundles,
                strategy="STRICT_PACK",
                name=f"{self.pool.name}_pg_{node_idx}",
            )
            pgs.append((pg, count))

        ray.get([pg.ready() for pg, _ in pgs])
        logger.info(
            "Created %d placement groups for pool '%s': %s (node_ips=%s)",
            len(pgs),
            self.pool.name,
            [(pg.bundle_count, count) for pg, count in pgs],
            [node_ip or "auto" for node_ip in pinned_node_ips],
        )

        RemoteWorker = ray.remote(self.worker_cls)
        rank = 0
        for pg, count in pgs:
            for local_rank in range(count):
                actor_name = f"{self.pool.name}:{self.worker_cls.__name__}:{rank}"
                options_kwargs: dict[str, Any] = {
                    "num_gpus": gpus_per_worker,
                    "num_cpus": 1,
                    "scheduling_strategy": PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=local_rank,
                    ),
                }
                if self.detached:
                    options_kwargs["name"] = actor_name
                    options_kwargs["lifetime"] = "detached"
                actor = RemoteWorker.options(**options_kwargs).remote(
                    rank=rank,
                    world_size=self.num_workers,
                    **self.worker_kwargs,
                )
                self._actors.append(actor)
                if self.detached:
                    self._worker_names.append(actor_name)
                rank += 1

    def stop(self) -> None:
        """Stop all workers."""
        if self.detached:
            logger.info("Skip killing detached workers for pool '%s'.", self.pool.name)
            self._actors.clear()
            return
        for actor in self._actors:
            ray.kill(actor)
        self._actors.clear()

    def is_alive(self) -> bool:
        """Best-effort liveness check for all workers."""
        if not self._actors:
            return False
        refs = [actor.__ray_ready__.remote() for actor in self._actors]
        ready, _ = ray.wait(refs, num_returns=len(refs), timeout=2.0)
        return len(ready) == len(refs)

    def setup_dispatch_collect_info(self) -> None:
        """Query workers for dp_rank and is_collect, store for auto dispatch/collect.

        Uses method_prefix so fused workers route correctly (e.g. prefix="actor"
        queries ``actor_get_dp_rank`` / ``actor_get_is_collect``).
        """
        dp_ranks = self.execute_all_sync("get_dp_rank")
        collect_mask = self.execute_all_sync("get_is_collect")
        self._dp_rank_mapping = [int(r) for r in dp_ranks]
        self._collect_mask = [bool(c) for c in collect_mask]
        logger.info(
            "Auto dispatch info (prefix=%r): dp_rank_mapping=%s, collect_mask=%s",
            self.method_prefix,
            self._dp_rank_mapping,
            self._collect_mask,
        )

    def execute_all_async(
        self, method: str, *args: Any, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        target_method = _prefixed_method(self.method_prefix, method)
        return [getattr(a, target_method).remote(*args, **kwargs) for a in self._actors]

    def execute_all_sync(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        refs = self.execute_all_async(method, *args, **kwargs)
        return ray.get(refs)

    def execute_rank_zero_sync(self, method: str, *args: Any, **kwargs: Any) -> Any:
        return self.call_single(0, method, *args, **kwargs)

    def call_all(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on all workers and collect results."""
        return self.execute_all_sync(method, *args, **kwargs)

    def call_single(
        self, worker_idx: int, method: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Call a method on a single worker."""
        target_method = _prefixed_method(self.method_prefix, method)
        ref = getattr(self._actors[worker_idx], target_method).remote(*args, **kwargs)
        return ray.get(ref)

    def call_single_async(
        self, worker_idx: int, method: str, *args: Any, **kwargs: Any
    ) -> ray.ObjectRef:
        target_method = _prefixed_method(self.method_prefix, method)
        return getattr(self._actors[worker_idx], target_method).remote(*args, **kwargs)

    def call_with_timeout(
        self,
        worker_idx: int,
        method: str,
        timeout_s: float,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        ref = self.call_single_async(worker_idx, method, *args, **kwargs)
        ready, _ = ray.wait([ref], timeout=timeout_s)
        if not ready:
            raise TimeoutError(
                f"{method} on worker {worker_idx} timed out in {timeout_s}s"
            )
        return ray.get(ref)

    def dispatch_and_call(
        self,
        method: str,
        data: DataProto,
        mode: DispatchMode | str | None = None,
        mesh_mapping: list[int] | None = None,
        lazy_key: str | None = None,
        **kwargs: Any,
    ) -> DataProto:
        """Split data across workers, call method, and merge results.

        Priority for dispatch routing:
          1. Explicit ``mesh_mapping`` argument
          2. Auto ``_dp_rank_mapping`` (set by ``setup_dispatch_collect_info``)
          3. Plain split via ``dispatch_proto``
        """
        dispatch_mode = self.dispatch_mode if mode is None else DispatchMode(mode)
        use_auto = (
            mesh_mapping is None
            and self._dp_rank_mapping is not None
            and dispatch_mode
            in (DispatchMode.DP_COMPUTE_PROTO, DispatchMode.DP_COMPUTE)
        )

        if use_auto:
            dp_size = len(set(self._dp_rank_mapping))
            unique_chunks = data.split(dp_size)
            # ray.put each unique DP chunk once; TP peers share the ObjectRef.
            chunk_refs = [ray.put(c) for c in unique_chunks]
            obj_refs = [
                chunk_refs[self._dp_rank_mapping[i]] for i in range(self.num_workers)
            ]

            target_method = _prefixed_method(self.method_prefix, method)
            call_refs = [
                getattr(self._actors[i], target_method).remote(obj_refs[i], **kwargs)
                for i in range(self.num_workers)
            ]
            results = ray.get(call_refs)

            if self._collect_mask is not None:
                return collect_with_mask(results, self._collect_mask)
            return DataProto.merge(results)

        chunks = dispatch_proto(
            data,
            self.num_workers,
            mode=dispatch_mode,
            mesh_mapping=mesh_mapping,
            lazy_state=self._lazy_dispatch_state,
            lazy_key=lazy_key,
        )
        target_method = _prefixed_method(self.method_prefix, method)
        if not chunks:
            return DataProto()

        if len(chunks) == 1:
            call_refs = [
                getattr(self._actors[0], target_method).remote(chunks[0], **kwargs)
            ]
        elif len(chunks) == self.num_workers:
            call_refs = [
                getattr(self._actors[i], target_method).remote(chunks[i], **kwargs)
                for i in range(self.num_workers)
            ]
        else:
            raise ValueError(
                f"dispatch produced {len(chunks)} chunks for {self.num_workers} workers; "
                "expected 1 (rank-zero) or num_workers."
            )
        results = ray.get(call_refs)
        return collect_proto(results, mode=dispatch_mode)

    def spawn(self, prefixes: list[str]) -> dict[str, RayWorkerGroup]:
        """Create prefixed logical views over this worker group.

        Spawned groups share the same actor handles and differ only by method prefix.
        """
        spawned: dict[str, RayWorkerGroup] = {}
        for prefix in prefixes:
            if prefix in self._spawned_groups:
                spawned[prefix] = self._spawned_groups[prefix].group
                continue
            child = RayWorkerGroup(
                worker_cls=self.worker_cls,
                pool=self.pool,
                num_workers=self.num_workers,
                worker_kwargs=self.worker_kwargs,
                dispatch_mode=self.dispatch_mode,
                detached=self.detached,
                worker_names=self._worker_names,
                method_prefix=prefix,
            )
            child._actors = self._actors
            self._spawned_groups[prefix] = _SpawnSpec(prefix=prefix, group=child)
            spawned[prefix] = child
        return spawned

    def fuse(self, prefix: str) -> RayWorkerGroup:
        """Return a single prefixed logical group."""
        return self.spawn([prefix])[prefix]

    @classmethod
    def from_detached(
        cls,
        worker_cls: type,
        pool: ResourcePool,
        worker_names: list[str],
        *,
        worker_kwargs: dict | None = None,
        dispatch_mode: DispatchMode | str = DispatchMode.DP_COMPUTE_PROTO,
    ) -> RayWorkerGroup:
        group = cls(
            worker_cls=worker_cls,
            pool=pool,
            num_workers=len(worker_names),
            worker_kwargs=worker_kwargs,
            dispatch_mode=dispatch_mode,
            detached=True,
            worker_names=worker_names,
        )
        group.start()
        return group

    @property
    def actors(self) -> list[ray.actor.ActorHandle]:
        return self._actors

    @property
    def worker_names(self) -> list[str]:
        return self._worker_names[:]
