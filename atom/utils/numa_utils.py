# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""NUMA binding for GPU worker processes.

Pin each GPU worker to the CPU cores and preferred memory of its GPU's local
NUMA node. The GPU->node mapping is auto-detected (``amdsmi`` -- the ROCm analog
of NVML -- with a sysfs fallback) so the operator needs no per-machine config;
an explicit per-rank node list can override it.

Public entry point: :func:`numa_bind_to_node`, called once at worker start.
Knobs live in :mod:`atom.utils.envs` (``ATOM_NUMA_BIND``, ``ATOM_AUTO_NUMA_BIND``,
``ATOM_NUMA_NODE``, ``ATOM_CRASH_ON_NUMA_BIND_FAILURE``).
"""

import ctypes
import glob
import logging
import os

from atom.utils import envs

logger = logging.getLogger("atom")


def _parse_cpulist(s: str) -> set[int]:
    """Parse a sysfs cpulist string (e.g. ``"0-63,128-191"``) into a set."""
    out: set[int] = set()
    for part in s.strip().split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def _node_cpus(node: int) -> set[int]:
    """CPUs belonging to NUMA ``node`` (kernel-reported, no topology guessing)."""
    with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
        return _parse_cpulist(f.read())


def _physical_index(gpu_id: int) -> int:
    """Map a logical GPU id to the physical index, honoring *_VISIBLE_DEVICES.

    System tools (amdsmi) and sysfs enumerate every physical GPU regardless of
    the visible-device mask, so a logical worker id must be translated back to
    the physical device it actually drives.
    """
    visible = os.environ.get("HIP_VISIBLE_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )
    if not visible:
        return gpu_id
    vis = [int(x) for x in visible.split(",") if x.strip() != ""]
    return vis[gpu_id] if gpu_id < len(vis) else gpu_id


def _query_node_amdsmi(gpu_id: int) -> int | None:
    """Physical GPU -> NUMA node via amdsmi (ROCm analog of NVML affinity)."""
    try:
        import amdsmi
    except Exception:
        return None
    phys = _physical_index(gpu_id)
    try:
        amdsmi.amdsmi_init()
        try:
            handles = amdsmi.amdsmi_get_processor_handles()
            if phys >= len(handles):
                return None
            node = int(amdsmi.amdsmi_topo_get_numa_node_number(handles[phys]))
            return node if node >= 0 else None
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug(f"amdsmi NUMA query failed for gpu {gpu_id}: {e}")
        return None


def _query_node_sysfs(gpu_id: int) -> int | None:
    """Fallback: physical GPU -> node via DRM cards sorted by PCI BDF.

    Assumes ROCm enumerates GPUs in PCI-BDF order, which a non-identity
    *_VISIBLE_DEVICES permutation can break -- so this is only used when amdsmi
    is unavailable. Prefer the explicit ``ATOM_NUMA_NODE`` list on such setups.
    """
    phys = _physical_index(gpu_id)
    cards: list[tuple[str, int]] = []
    for dev in glob.glob("/sys/class/drm/card*/device"):
        nn = os.path.join(dev, "numa_node")
        if not os.path.exists(nn):
            continue
        bdf = os.path.basename(os.path.realpath(dev))
        with open(nn) as f:
            cards.append((bdf, int(f.read())))
    cards.sort()
    if phys >= len(cards):
        return None
    node = cards[phys][1]
    return node if node >= 0 else None


def _resolve_node(gpu_id: int) -> int | None:
    """Explicit ``ATOM_NUMA_NODE`` wins; else auto-detect (amdsmi -> sysfs)."""
    explicit = [x for x in envs.ATOM_NUMA_NODE.split(",") if x.strip() != ""]
    if explicit:
        idx = gpu_id if gpu_id < len(explicit) else len(explicit) - 1
        return int(explicit[idx])
    if not envs.ATOM_AUTO_NUMA_BIND:
        return None
    node = _query_node_amdsmi(gpu_id)
    if node is None:
        node = _query_node_sysfs(gpu_id)
    return node


def _set_preferred_memory(node: int) -> None:
    """Best-effort memory binding via libnuma ``numa_set_preferred``.

    If libnuma is absent, first-touch on the pinned CPUs still lands memory on
    the local node, so this is an optimization, not a requirement.
    """
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
        if libnuma.numa_available() < 0:
            return
        libnuma.numa_set_preferred(ctypes.c_int(node))
    except Exception as e:
        logger.debug(f"numa_set_preferred({node}) skipped: {e}")


def numa_bind_to_node(gpu_id: int, label: str = "") -> None:
    """Bind the current process to its GPU's NUMA-local cores and memory.

    No-op unless ``ATOM_NUMA_BIND`` is enabled. Must run before any large
    allocation / native (e.g. mooncake RDMA) thread spawn so the affinity mask
    is inherited by child threads and Linux first-touch places memory on the
    node. The node's CPUs are intersected with the current affinity so an
    existing container cpuset is respected. On failure it warns (and raises only
    if ``ATOM_CRASH_ON_NUMA_BIND_FAILURE``).
    """
    if not envs.ATOM_NUMA_BIND:
        return
    tag = f" ({label})" if label else ""
    try:
        node = _resolve_node(gpu_id)
        if node is None or node < 0:
            raise RuntimeError(f"could not resolve NUMA node for gpu {gpu_id}")
        cpus = _node_cpus(node) & os.sched_getaffinity(0)
        if not cpus:
            raise RuntimeError(
                f"NUMA node {node} has no CPUs allowed by the current affinity"
            )
        os.sched_setaffinity(0, cpus)
        _set_preferred_memory(node)
        logger.info(f"NUMA bind{tag}: gpu={gpu_id} -> node {node} ({len(cpus)} cores)")
    except Exception as e:
        msg = (
            f"NUMA bind{tag} failed for gpu {gpu_id}: {e}. In docker add "
            f"--cap-add SYS_NICE, or set ATOM_NUMA_NODE explicitly."
        )
        if envs.ATOM_CRASH_ON_NUMA_BIND_FAILURE:
            raise RuntimeError(msg) from e
        logger.warning(msg)
