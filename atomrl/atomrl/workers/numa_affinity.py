# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""NUMA-aware CPU affinity for GPU-backed Ray workers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)
_DEFAULT_NODE_ROOT = Path("/sys/devices/system/node")


@dataclass(frozen=True)
class NumaBinding:
    """The GPU-local NUMA placement applied to the current process."""

    physical_gpu_id: int
    numa_node: int
    cpus: frozenset[int]


def parse_linux_cpu_list(value: str) -> set[int]:
    """Expand Linux cpulist syntax such as ``0-3,8,10-11``."""
    cpus: set[int] = set()
    for item in value.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid CPU range: {item}")
            cpus.update(range(start, end + 1))
        else:
            cpus.add(int(item))
    if not cpus:
        raise ValueError("CPU list is empty")
    return cpus


def resolve_physical_gpu_id(
    ray_gpu_ids: Iterable[Any],
    cuda_visible_devices: str | None,
) -> int:
    """Resolve the physical GPU assigned by Ray, with an environment fallback."""
    ray_ids = list(ray_gpu_ids)
    if ray_ids:
        return int(float(str(ray_ids[0])))
    if cuda_visible_devices:
        return int(float(cuda_visible_devices.split(",", 1)[0].strip()))
    raise ValueError("No Ray GPU assignment or CUDA_VISIBLE_DEVICES value")


def current_physical_gpu_id() -> int:
    """Return the physical GPU ID assigned to the current Ray worker."""
    try:
        import ray
    except (ImportError, OSError):
        ray_gpu_ids = []
    else:
        errors: list[type[BaseException]] = [
            AttributeError,
            OSError,
            RuntimeError,
        ]
        ray_error = getattr(getattr(ray, "exceptions", None), "RayError", None)
        if isinstance(ray_error, type) and issubclass(ray_error, BaseException):
            errors.append(ray_error)
        try:
            ray_gpu_ids = ray.get_gpu_ids()
        except tuple(errors):
            ray_gpu_ids = []
    return resolve_physical_gpu_id(
        ray_gpu_ids,
        os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


def _query_rocm_numa_topology() -> str:
    result = subprocess.run(
        ["rocm-smi", "--showtoponuma", "--json"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def _set_process_affinity(pid: int, cpus: set[int]) -> None:
    os.sched_setaffinity(pid, cpus)


def bind_current_process_to_gpu_numa(
    physical_gpu_id: int,
    *,
    enabled: bool = True,
    topology_json: str | None = None,
    node_root: Path = _DEFAULT_NODE_ROOT,
    set_affinity: Callable[[int, set[int]], None] = _set_process_affinity,
    logger: logging.Logger | None = None,
) -> NumaBinding | None:
    """Bind this process to CPUs local to its GPU; warn and continue on failure."""
    if not enabled:
        return None

    log = logger or _LOG
    try:
        topology = json.loads(
            topology_json if topology_json is not None else _query_rocm_numa_topology()
        )
        gpu_topology = topology[f"card{int(physical_gpu_id)}"]
        numa_value = gpu_topology.get(
            "(Topology) Numa Affinity",
            gpu_topology.get("(Topology) Numa Node"),
        )
        if numa_value is None:
            raise ValueError(f"NUMA node missing for GPU {physical_gpu_id}")
        numa_node = int(numa_value)
        cpus = parse_linux_cpu_list(
            (node_root / f"node{numa_node}" / "cpulist").read_text(encoding="utf-8")
        )
        set_affinity(0, cpus)
        binding = NumaBinding(
            physical_gpu_id=int(physical_gpu_id),
            numa_node=numa_node,
            cpus=frozenset(cpus),
        )
        log.info(
            "NUMA affinity: physical_gpu=%d numa_node=%d cpus=%s",
            binding.physical_gpu_id,
            binding.numa_node,
            _format_cpu_list(binding.cpus),
        )
        return binding
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        log.warning(
            "NUMA affinity unavailable for physical GPU %s; continuing unbound: %s",
            physical_gpu_id,
            exc,
        )
        return None


def _format_cpu_list(cpus: Iterable[int]) -> str:
    return ",".join(str(cpu) for cpu in sorted(cpus))
