# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""NUMA binding for GPU worker processes.

Pin each GPU worker to the CPU cores and preferred memory of its GPU's local
NUMA node. The GPU->node mapping is auto-detected (``amdsmi`` -- the ROCm analog
of NVML -- with a sysfs fallback) so the operator needs no per-machine config;
an explicit per-rank node list can override it.

Public entry points: :func:`numa_bind_to_node`, called once at worker start,
and :func:`opt_out_of_numa_balancing`, called first thing in every serving
process. Knobs live in :mod:`atom.utils.envs` (``ATOM_NUMA_BIND``,
``ATOM_AUTO_NUMA_BIND``, ``ATOM_NUMA_NODE``, ``ATOM_CRASH_ON_NUMA_BIND_FAILURE``,
``ATOM_NUMA_BALANCING_OPT_OUT``).
"""

import ctypes
import glob
import logging
import os
import platform

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
    # ROCR first: it filters the device table before HIP sees it, so when both
    # are set HIP's indices are already relative to the filtered list. A recipe
    # that sets ROCR alone (the only correct way to split across NUMA nodes --
    # setting both cuts the count twice) would otherwise fall through to the
    # identity mapping and bind every rank to node 0.
    visible = (
        os.environ.get("ROCR_VISIBLE_DEVICES")
        or os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
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


# Memory-policy syscalls (glibc exports neither; libnuma's `numa_set_localalloc`
# sets MPOL_DEFAULT, which is the policy that NUMA balancing scans).
_SYS_MEMPOLICY = {"x86_64": (238, 239), "aarch64": (237, 236)}  # (set, get)
_MPOL_DEFAULT = 0
_MPOL_LOCAL = 4


def opt_out_of_numa_balancing(label: str = "") -> bool:
    """Keep the kernel's automatic NUMA balancing off this process's memory.

    The GPU-mapped host memory of a serving process (``hipHostMalloc``,
    torch's pinned memory, Engram's UVA tables, ``hipHostRegister``) is
    tracked by KFD through MMU notifiers (its SVM ranges -- with XNACK off as
    well -- and userptr BOs). NUMA balancing's scan (``task_numa_work``, 256 MB
    of the address space at a time, in the context of whichever thread of the
    process it lands on) marks those PTEs PROT_NONE; the notifier
    (``svm_range_cpu_invalidate_pagetables``, 293 of 298 measured; the userptr
    one the rest) quiesces the process: KFD removes every GPU queue of it
    through MES until its restore worker revalidates the range -- measured on
    MI455X TP4 c1 decode at ~4-5 ms per eviction, 13 evictions in 2.3 s of
    decode, each one stretching that rank's single-launch step -- and every
    peer, which spins on it -- from 3.5 to 7-9 ms: the ITL p99 tail. Nothing
    is migrated (the pages are pinned), so the scan buys nothing here.

    The scan skips a VMA whose policy -- its own, else the scanning thread's --
    lacks MPOL_F_MOF. The default task policy carries it; an explicit
    MPOL_LOCAL does not, and places pages exactly as the default does (the
    faulting CPU's node, falling back elsewhere). So this sets MPOL_LOCAL as
    the task policy, and only over MPOL_DEFAULT: a policy the operator chose
    (``numactl``) is kept, and ``numa_bind_to_node`` may still replace it with
    its own (also scan-free) preferred node.

    Per thread, inherited by threads and children created after it, so it runs
    before a process starts any. In a container the default seccomp profile
    refuses the mempolicy calls without CAP_SYS_NICE (``--cap-add SYS_NICE``):
    then it only warns, and the host's ``kernel.numa_balancing=0`` is the way. No-op where NUMA balancing is off, on another
    architecture, or with ``ATOM_NUMA_BALANCING_OPT_OUT=0``. Returns whether it
    set the policy.
    """
    if not envs.ATOM_NUMA_BALANCING_OPT_OUT:
        return False
    nrs = _SYS_MEMPOLICY.get(platform.machine())
    if nrs is None:
        return False
    try:
        with open("/proc/sys/kernel/numa_balancing") as f:
            if f.read().strip() in ("", "0"):
                return False
    except OSError:
        return False
    tag = f" ({label})" if label else ""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        mode = ctypes.c_int(-1)
        if (
            libc.syscall(
                ctypes.c_long(nrs[1]),
                ctypes.byref(mode),
                None,
                ctypes.c_ulong(0),
                None,
                ctypes.c_ulong(0),
            )
            != 0
        ):
            raise OSError(ctypes.get_errno(), "get_mempolicy")
        if mode.value != _MPOL_DEFAULT:
            return False
        if (
            libc.syscall(
                ctypes.c_long(nrs[0]),
                ctypes.c_int(_MPOL_LOCAL),
                None,
                ctypes.c_ulong(0),
            )
            != 0
        ):
            raise OSError(ctypes.get_errno(), "set_mempolicy")
    except Exception as e:  # noqa: BLE001 - a latency optimization, never fatal
        # EPERM: a container's default seccomp profile allows the mempolicy
        # calls only with CAP_SYS_NICE (docker run --cap-add SYS_NICE)
        hint = (
            " (needs CAP_SYS_NICE in a container: --cap-add SYS_NICE; or set "
            "kernel.numa_balancing=0 on the host)"
            if getattr(e, "errno", None) == 1
            else ""
        )
        logger.warning(f"NUMA balancing opt-out{tag} skipped: {e}{hint}")
        return False
    logger.debug(f"NUMA balancing opt-out{tag}: task memory policy MPOL_LOCAL")
    return True


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
