# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mooncake Store as the L2 of the in-process LMCache offload.

With ``LMCACHE_REMOTE_URL=mooncakestore://...`` every PP worker's LMCache
engine writes each chunk through its CPU pool (L1) into a pooled Mooncake
Store (L2) whose memory belongs to separate owner processes, and reads L1
misses back over RDMA. LMCache needs no change for that, but the worker does,
because each of the following fails silently or hangs otherwise:

* **THP L1.** An ionic NIC registers about 3 GiB of 4 KiB pages in all, and
  one 4 KiB page counts its whole MR against that budget: an 8 GiB THP buffer
  with a 2 MiB stretch of 4 KiB pages failed to register as one MR and
  registered as 1 GiB MRs (``MC_MAX_MR_SIZE``, which the atomesh launcher
  sets). LMCache's NUMA allocator first-touches 4 KiB pages, so its
  multi-GiB L1 cannot register at all, and a failed registration is only a
  warning, after which every L2 put fails with AddressNotRegistered while
  ``put_failed_count`` stays 0. The pool is therefore allocated here instead
  -- mmap, mbind(MPOL_BIND) to the GPU's node, MADV_HUGEPAGE, one first touch
  per 2 MiB, hipHostRegister -- and allocation fails unless every byte of it
  is a transparent huge page. 1 GiB MRs would tolerate a few 4 KiB pages; the
  strict check is a policy that keeps the L1 off the shared budget entirely.
  After the engine is built, the L1 must be such a region: LMCache takes other
  allocators when it has no NUMA mapping for the GPU, or for hugetlb, shared
  memory or P2P pools.
* **One NIC per rank, the GPU's own.** A requester listing several NICs, or
  owners sharing the requesters' NICs, stalled concurrent reads for 30-60 s
  and then failed them; one NIC per stage, disjoint from the owners', ran
  4 x 33 GB/s with no error. The NIC is the RDMA device closest to the GPU in
  the PCI tree. HIP numbers the GPUs in KFD order, not PCI order, so neither
  ``rdma<gpu>`` nor a BDF-sorted list names it reliably.
* **One RPC per lookup.** LMCache's Mooncake connector has no batched
  ``contains``, so a lookup asked the master once per chunk on the
  scheduler's synchronous path: 25-58 ms for 640-1280 chunks, against
  0.4-1 ms for one ``batch_is_exist``.
* **L2 gets never wait for room in the L1.** The connector's batched get
  allocates its L1 buffers on LMCache's storage event loop, busy-waiting until
  eviction makes room; that loop is also the thread that finishes the
  write-through puts whose L1 references block the eviction. On
  pit2-p03-g40 (8 GiB L1 per stage, agentic 1M traces at 16 sessions) three
  of four stages' loops spun there for good within 3 minutes, and every later
  L2 put and get of those workers hung. A get now takes what eviction frees at
  once and treats the rest of the batch as missing; LMCache keeps the prefix
  up to the first missing chunk, and the scheduler recomputes the rest.
* **Recipe checks.** Settings that LMCache or Mooncake ignore without a word,
  or whose wrong value only shows up as a corrupt or leaked object under load,
  are refused at startup.

The checks of the remote tier itself (backend present, connection up, L1
registered, one chunk round trip) live in
:mod:`atom.kv_transfer.offload.remote_check` and apply to any remote URL.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import json
import logging
import os
import platform
import re
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

from atom.utils import envs

logger = logging.getLogger("atom")

MOONCAKE_STORE_URL_SCHEME = "mooncakestore://"
HUGE_PAGE_BYTES = 2 * 1024 * 1024
# Below this a put that outlives its timeout drops its L1 reference while RDMA
# may still read the buffer (a corrupt object), and a get that times out leaks
# the L1 objects it allocated. LMCache's defaults are 1 s and 10 s.
MIN_TRANSFER_TIMEOUT_S = 30

_IB_SYSFS_ROOT = Path("/sys/class/infiniband")
_PCI_SYSFS_ROOT = Path("/sys/bus/pci/devices")
_NODE_SYSFS_ROOT = Path("/sys/devices/system/node")
_THP_SYSFS_ROOT = Path("/sys/kernel/mm/transparent_hugepage")
# The sysfs path component of a PCI host bridge, e.g. "pci0000:70".
_PCI_HOST_BRIDGE = re.compile(r"pci[0-9a-f]{4}:[0-9a-f]{2}")

# Extra-config keys that never reach Mooncake under the name given: its dict
# setup ignores the legacy names, ATOM picks the NIC per rank, and LMCache reads
# the transfer timeout only from the unprefixed key.
_IGNORED_EXTRA_CONFIG_KEYS = {
    "master_server_address": "use mooncake_master_server_addr",
    "mooncake_master_server_address": "use mooncake_master_server_addr",
    "device_name": (
        "ATOM sets mooncake_rdma_devices per rank; override it with "
        "ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES"
    ),
    "mooncake_device_name": (
        "ATOM sets mooncake_rdma_devices per rank; override it with "
        "ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES"
    ),
    "mooncake_rdma_devices": (
        "it would give every PP rank the same NICs; ATOM sets it per rank, "
        "override it with ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES"
    ),
    "mooncake_transfer_timeout": "use transfer_timeout",
}

_PROT_READ_WRITE = 0x1 | 0x2
_MAP_PRIVATE_ANONYMOUS = 0x02 | 0x20
_MAP_FAILED = ctypes.c_void_p(-1).value
_MADV_HUGEPAGE = 14
# Linux 6.1+: rebuild a range's 4 KiB pages as huge pages, compacting now.
_MADV_COLLAPSE = 25
# Seconds to wait before each retry of a collapse that found no free huge page:
# the four stages of a node allocate their pools at the same time.
_COLLAPSE_RETRY_DELAYS_S = (1.0, 2.0, 4.0, 8.0)
_MPOL_BIND = 2
_MPOL_MF_STRICT_MOVE = 0x1 | 0x2
_PR_GET_THP_DISABLE = 42
# mbind has no glibc wrapper and libnuma is not guaranteed.
_MBIND_SYSCALL = {"x86_64": 237, "aarch64": 235}
_ULONG_BITS = ctypes.sizeof(ctypes.c_ulong) * 8
_SMAPS_HEADER = re.compile(r"^([0-9a-f]+)-([0-9a-f]+) ")
# The first touch faults the pool in 1 GiB at a time and reports its progress
# when it has been slow for this long: on a node short of free memory every
# huge-page fault reclaims page cache through compaction first.
_TOUCH_SLICE_BYTES = 512 * HUGE_PAGE_BYTES
_TOUCH_PROGRESS_INTERVAL_S = 10.0


def uses_mooncake_store(cfg: Any) -> bool:
    """Whether LMCache's remote tier for ``cfg`` is a Mooncake Store."""
    remote_url = getattr(cfg, "remote_url", None)
    return isinstance(remote_url, str) and remote_url.startswith(
        MOONCAKE_STORE_URL_SCHEME
    )


def prepare_mooncake_store_l2(cfg: Any) -> None:
    """Validate the recipe, give this rank its NIC, patch LMCache's L1, lookup and get.

    Runs in each worker before its LMCache engine is built, since building it
    allocates the L1 pool and registers it with Mooncake.

    Raises:
        ValueError: The configuration would make the L2 fail silently, or no
            NIC can be chosen for this rank.
    """
    validate_mooncake_store_config(cfg)
    # Here, not only at allocation: a failure inside LMCache's storage manager
    # leaves threads behind.
    _require_thp_allowed()
    extra = cfg.extra_config
    protocol = str(extra.get("mooncake_protocol", extra.get("protocol"))).lower()
    device = None
    if protocol.strip() != "tcp":
        import torch

        device = requester_rdma_device(torch.cuda.current_device())
        extra["mooncake_rdma_devices"] = device
    pool = store_pool_of(device)
    if pool is not None:
        extra["mooncake_master_server_addr"] = pool.master
        extra["mooncake_metadata_server"] = pool.metadata
        cfg.remote_url = f"{MOONCAKE_STORE_URL_SCHEME}{pool.master}/"
    install_thp_pinned_allocator()
    install_batch_is_exist_lookup()
    install_non_blocking_l2_get_allocation()
    logger.info(
        "LMCache Mooncake Store L2: protocol=%s rdma_devices=%s master=%s, THP "
        "pinned L1, one batch_is_exist RPC per lookup, L2 gets without waiting "
        "for L1 room",
        protocol,
        device or "-",
        extra.get("mooncake_master_server_addr", "-"),
    )


def validate_mooncake_store_config(cfg: Any) -> None:
    """Refuse a Mooncake Store L2 recipe that would fail without an error.

    Raises:
        ValueError: Naming the setting and what to set instead.
    """
    if os.environ.get("MOONCAKE_CONFIG_PATH"):
        raise ValueError(
            "MOONCAKE_CONFIG_PATH is set: LMCache's Mooncake connector then reads "
            "that file instead of LMCACHE_EXTRA_CONFIG, so this rank's NIC and "
            "the checks below would not apply. Unset it."
        )
    qp_per_endpoint = os.environ.get("MC_NUM_QP_PER_EP", "")
    if qp_per_endpoint.strip() != "1":
        raise ValueError(
            f"MC_NUM_QP_PER_EP={qp_per_endpoint or '<unset, 2>'}: the Mooncake "
            "Store L2 needs MC_NUM_QP_PER_EP=1 in every Mooncake process -- "
            "these workers, the Store owners, and the decode side (P->D shares "
            "the variable, and peers with different QP counts cannot connect). "
            "With 2 QPs concurrent L2 reads stall for 30-60 s, then fail."
        )
    numa_mode = getattr(cfg, "numa_mode", None)
    if numa_mode not in ("auto", "manual"):
        raise ValueError(
            f"LMCACHE_NUMA_MODE={numa_mode!r}: the Mooncake Store L2 needs "
            "LMCACHE_NUMA_MODE=auto (or manual), the allocator ATOM replaces with "
            "a huge-page one; any other L1 holds 4 KiB pages the NIC cannot "
            "register."
        )
    if getattr(cfg, "local_cpu_use_hugepages", False):
        raise ValueError(
            "LMCACHE_LOCAL_CPU_USE_HUGEPAGES is set: LMCache then allocates the "
            "L1 from the reserved hugetlb pool with an allocator of its own, "
            "outside ATOM's huge-page allocator and its NUMA and THP checks. "
            "Unset it; ATOM's allocator gives the L1 transparent huge pages."
        )
    blocking_timeout = float(getattr(cfg, "blocking_timeout_secs", 0) or 0)
    if blocking_timeout < MIN_TRANSFER_TIMEOUT_S:
        raise ValueError(
            f"LMCACHE_BLOCKING_TIMEOUT_SECS={blocking_timeout:g}: set it to at "
            f"least {MIN_TRANSFER_TIMEOUT_S}; a get that times out leaks the L1 "
            "objects it allocated."
        )
    extra = getattr(cfg, "extra_config", None)
    if not isinstance(extra, dict):
        raise ValueError(  # noqa: TRY004 - a configuration error, like the rest
            "LMCACHE_EXTRA_CONFIG must be a JSON object carrying the mooncake_* "
            "setup keys of the Mooncake Store L2"
        )
    for key, instead in _IGNORED_EXTRA_CONFIG_KEYS.items():
        if key in extra:
            raise ValueError(
                f"LMCACHE_EXTRA_CONFIG key {key!r} would be ignored: {instead}"
            )
    if extra.get("save_chunk_meta", True):
        raise ValueError(
            'LMCACHE_EXTRA_CONFIG needs "save_chunk_meta": false (JSON false): '
            "the Mooncake L2 moves chunks zero-copy from and to the registered "
            "L1; with chunk metadata every put and get is staged through "
            "Mooncake's local buffer instead."
        )
    try:
        transfer_timeout = float(extra.get("transfer_timeout", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "LMCACHE_EXTRA_CONFIG transfer_timeout must be a number"
        ) from exc
    if transfer_timeout < MIN_TRANSFER_TIMEOUT_S:
        raise ValueError(
            f"LMCACHE_EXTRA_CONFIG transfer_timeout={transfer_timeout:g}: set it "
            f"to at least {MIN_TRANSFER_TIMEOUT_S}; a put that times out releases "
            "its L1 buffer while RDMA may still read it, storing a corrupt object."
        )
    segment = extra.get(
        "mooncake_global_segment_size", extra.get("global_segment_size")
    )
    if segment is None or str(segment).strip() != "0":
        raise ValueError(
            f"LMCACHE_EXTRA_CONFIG mooncake_global_segment_size={segment!r}: the "
            'workers only request, so set it to "0" (unset, LMCache gives each '
            "worker a 3.1 GiB segment of 4 KiB pages that dies with it)."
        )
    if not extra.get("mooncake_protocol", extra.get("protocol")):
        raise ValueError(
            "LMCACHE_EXTRA_CONFIG must name mooncake_protocol (rdma or tcp); "
            "unset, LMCache silently uses tcp."
        )


# ---------------------------------------------------------------------------
# One requester NIC per rank
# ---------------------------------------------------------------------------


def parse_device_list(value: str) -> list[str]:
    """Split a comma-separated device list, dropping blanks."""
    return [device.strip() for device in value.split(",") if device.strip()]


def requester_rdma_device(device_index: int) -> str:
    """Return the one RDMA device the Store client of this GPU's worker uses.

    ``ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES`` wins when set: one device per GPU
    ordinal, or one for all. Otherwise it is the GPU's NIC in the PCI tree.

    Raises:
        ValueError: No device can be chosen, it does not exist, or the Store
            owners on this node use it.
    """
    table = parse_device_list(envs.ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES)
    if table:
        if len(table) == 1:
            device = table[0]
        elif device_index < len(table):
            device = table[device_index]
        else:
            raise ValueError(
                f"ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES lists {len(table)} devices "
                f"and has none for GPU {device_index}"
            )
        origin = "ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES"
    else:
        bdf = gpu_pci_bdf(device_index)
        device = rail_rdma_device(bdf)
        origin = f"the PCI topology of GPU {device_index} ({bdf})"
    if not (_IB_SYSFS_ROOT / device).exists():
        raise ValueError(f"RDMA device {device!r} from {origin} does not exist")
    owner_devices = parse_device_list(envs.ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES)
    # With per-NIC pools the owners on this device are its own pool's, the
    # only ones this worker reads; store_pool_of checks the device has one.
    if device in owner_devices and not envs.ATOM_LMCACHE_MOONCAKE_POOLS.strip():
        raise ValueError(
            f"RDMA device {device!r} from {origin} is also a Store owner's "
            f"(ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES={','.join(owner_devices)}); "
            "owners and requesters on one NIC stall concurrent reads"
        )
    logger.info(
        "LMCache Mooncake Store L2: GPU %d uses RDMA device %s (from %s)",
        device_index,
        device,
        origin,
    )
    return device


class StorePool(NamedTuple):
    """The master of one per-NIC Store pool."""

    master: str
    metadata: str


def parse_store_pools(value: str) -> dict[str, StorePool]:
    """Parse ``ATOM_LMCACHE_MOONCAKE_POOLS``; blank means one shared pool.

    Raises:
        ValueError: Not a non-empty JSON object mapping each RDMA device to
            ``{"master": "host:port", "metadata": "<url>"}``.
    """
    if not value.strip():
        return {}
    try:
        table = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"ATOM_LMCACHE_MOONCAKE_POOLS is not JSON: {exc}") from exc
    if not isinstance(table, dict) or not table:
        raise ValueError(
            "ATOM_LMCACHE_MOONCAKE_POOLS must be a non-empty JSON object keyed "
            "by RDMA device"
        )
    pools = {}
    for device, pool in table.items():
        fields = pool if isinstance(pool, dict) else {}
        master, metadata = fields.get("master"), fields.get("metadata")
        if not (isinstance(master, str) and master and isinstance(metadata, str)):
            raise ValueError(
                f"ATOM_LMCACHE_MOONCAKE_POOLS[{device!r}] must be "
                '{"master": "host:port", "metadata": "<url>"}'
            )
        pools[device] = StorePool(master, metadata)
    return pools


def store_pool_of(device: str | None) -> StorePool | None:
    """Return the per-NIC pool of a worker's RDMA device, None without pools.

    Where owners must share the requesters' NICs (a node with only its own
    GPUs' NICs), each NIC gets a master of its own: the worker on a NIC reads
    only that pool's owners, and a NIC carries one requester and its own
    pool's owners instead of every owner's reads.

    Raises:
        ValueError: Pools are set, but the worker has no RDMA device (tcp)
            or no pool for its device.
    """
    pools = parse_store_pools(envs.ATOM_LMCACHE_MOONCAKE_POOLS)
    if not pools:
        return None
    if device is None:
        raise ValueError(
            "ATOM_LMCACHE_MOONCAKE_POOLS keys the Store pools by RDMA device; "
            "a tcp Store client has none"
        )
    pool = pools.get(device)
    if pool is None:
        raise ValueError(
            f"ATOM_LMCACHE_MOONCAKE_POOLS has no pool for RDMA device "
            f"{device!r} (pools: {', '.join(sorted(pools))})"
        )
    return pool


def gpu_pci_bdf(device_index: int) -> str:
    """PCI address of a CUDA/HIP device ordinal, as sysfs spells it."""
    import torch

    props = torch.cuda.get_device_properties(device_index)
    return (
        f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:"
        f"{props.pci_device_id:02x}.0"
    )


def rail_rdma_device(
    gpu_bdf: str,
    *,
    ib_root: Path = _IB_SYSFS_ROOT,
    pci_root: Path = _PCI_SYSFS_ROOT,
) -> str:
    """Return the ACTIVE RDMA device that shares the deepest PCI path with a GPU.

    On a rail-optimized node each GPU sits behind a PCIe switch with exactly
    one NIC, which shares more of the sysfs device path with that GPU than
    any other NIC does.

    Raises:
        ValueError: The GPU is not in sysfs, no ACTIVE device shares its PCI
            host bridge, or several are equally close.
    """
    gpu_node = pci_root / gpu_bdf
    if not gpu_node.exists():
        raise ValueError(f"GPU {gpu_bdf} is not in {pci_root}")
    gpu_path = gpu_node.resolve().parts
    host_bridge = next(
        (i for i, part in enumerate(gpu_path) if _PCI_HOST_BRIDGE.fullmatch(part)),
        None,
    )
    if host_bridge is None:
        raise ValueError(f"no PCI host bridge in GPU {gpu_bdf}'s path {gpu_node}")
    try:
        devices = sorted(ib_root.iterdir())
    except OSError as exc:
        raise ValueError(f"cannot list RDMA devices in {ib_root}") from exc
    closeness: dict[str, int] = {}
    for device in devices:
        if not _has_active_port(device):
            continue
        nic_path = (device / "device").resolve().parts
        shared = 0
        for gpu_part, nic_part in zip(gpu_path, nic_path):
            if gpu_part != nic_part:
                break
            shared += 1
        closeness[device.name] = shared
    best = max(closeness.values(), default=0)
    closest = [name for name, shared in closeness.items() if shared == best]
    # A NIC that does not share the host bridge component shares no hardware.
    if best <= host_bridge or len(closest) != 1:
        found = ", ".join(f"{n}:{s}" for n, s in sorted(closeness.items()))
        raise ValueError(
            f"cannot tell GPU {gpu_bdf}'s NIC from the PCI topology (ACTIVE "
            f"devices and shared path depth: {found or 'none'}); set "
            "ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES"
        )
    return closest[0]


def _has_active_port(device: Path) -> bool:
    for state_file in device.glob("ports/*/state"):
        try:
            state = state_file.read_text().partition(":")[0].strip()
        except OSError:
            continue
        if state == "4":  # IB_PORT_ACTIVE, also used by RoCE devices
            return True
    return False


# ---------------------------------------------------------------------------
# THP pinned L1
# ---------------------------------------------------------------------------


class _ThpRegion(NamedTuple):
    mapping: int  # mmap'ed address, 2 MiB below the region at most
    mapping_bytes: int
    region_bytes: int
    node: int


_thp_regions: dict[int, _ThpRegion] = {}
_thp_regions_lock = threading.Lock()
# L1 regions faulted at worker start, before the weights load, by node; the
# first L1 allocation on that node that fits takes its region.
_reserved_l1: dict[int, tuple[int, _ThpRegion]] = {}
_install_lock = threading.Lock()
# LMCache's own free, for a buffer it allocated before the patch.
_native_free_pinned_numa_ptr = None


@functools.cache
def _libc() -> ctypes.CDLL:
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libc.syscall.restype = ctypes.c_long
    return libc


def _os_error(call: str) -> OSError:
    err = ctypes.get_errno()
    return OSError(err, f"{call} failed: {os.strerror(err)}")


def _require_thp_allowed() -> None:
    # prctl(PR_SET_THP_DISABLE) is inherited across fork and exec, and it
    # makes the kernel ignore MADV_HUGEPAGE in this process.
    zero = ctypes.c_ulong(0)
    if _libc().prctl(ctypes.c_int(_PR_GET_THP_DISABLE), zero, zero, zero, zero) == 1:
        raise RuntimeError(
            "THP is disabled for this process (prctl PR_SET_THP_DISABLE, possibly "
            "inherited from its parent), so the Mooncake L2's L1 cannot be "
            "allocated on huge pages"
        )


def _mbind(address: int, length: int, node: int) -> None:
    syscall_nr = _MBIND_SYSCALL.get(platform.machine())
    if syscall_nr is None:
        raise RuntimeError(f"mbind: unsupported architecture {platform.machine()}")
    words = node // _ULONG_BITS + 1
    mask = (ctypes.c_ulong * words)()
    mask[node // _ULONG_BITS] = 1 << (node % _ULONG_BITS)
    # The kernel reads maxnode - 1 bits.
    rc = _libc().syscall(
        ctypes.c_long(syscall_nr),
        ctypes.c_void_p(address),
        ctypes.c_ulong(length),
        ctypes.c_long(_MPOL_BIND),
        mask,
        ctypes.c_ulong(words * _ULONG_BITS + 1),
        ctypes.c_ulong(_MPOL_MF_STRICT_MOVE),
    )
    if rc != 0:
        raise _os_error(f"mbind(MPOL_BIND, node {node})")


def numa_node_meminfo(node: int, root: Path = _NODE_SYSFS_ROOT) -> dict[str, int]:
    """A NUMA node's meminfo, kB fields in bytes; empty when it cannot be read."""
    try:
        text = (root / f"node{node}" / "meminfo").read_text()
    except OSError:
        return {}
    info = {}
    for line in text.splitlines():
        # "Node 0 MemFree:        838449356 kB"
        fields = line.split()
        if len(fields) < 4 or not fields[2].endswith(":"):
            continue
        try:
            value = int(fields[3])
        except ValueError:
            continue
        info[fields[2][:-1]] = value * 1024 if fields[-1] == "kB" else value
    return info


def _current_gpu_numa_node() -> int | None:
    """The NUMA node of this process's current GPU, None when sysfs has none."""
    import torch

    if not torch.cuda.is_available():
        return None
    bdf = gpu_pci_bdf(torch.cuda.current_device())
    try:
        node = int((_PCI_SYSFS_ROOT / bdf / "numa_node").read_text())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def _require_gpu_numa_node(node: int) -> None:
    # LMCache passes the node of its GPU-to-NUMA mapping, which a manual
    # mapping can point anywhere. The other node's memory belongs to the
    # processes bound there -- on a 1P1D node, the decode's.
    gpu_node = _current_gpu_numa_node()
    if gpu_node is not None and gpu_node != node:
        raise ValueError(
            f"LMCache asked for its L1 on NUMA node {node}, but this worker's GPU "
            f"is on node {gpu_node}; the Mooncake Store L2 keeps the L1 on the "
            "GPU's node (check LMCACHE_NUMA_MODE and gpu_to_numa_mapping)"
        )


def _warn_if_node_is_short(node: int, region_bytes: int) -> None:
    meminfo = numa_node_meminfo(node)
    free = meminfo.get("MemFree")
    if free is not None and free < region_bytes:
        logger.warning(
            "LMCache L1: NUMA node %d has %.1f GiB free for a %.1f GiB pool and "
            "%.1f GiB of page cache: every huge-page fault past the free memory "
            "reclaims page cache through compaction first, which can take "
            "minutes; drop the node's page cache before starting",
            node,
            free / 2**30,
            region_bytes / 2**30,
            meminfo.get("FilePages", 0) / 2**30,
        )


def _touch_every_huge_page(address: int, length: int, node: int) -> None:
    """Fault in each huge page of the range, reporting progress when slow.

    It writes 1, not 0: the kernel's shrinker of underused huge pages (6.12
    and later) splits an all-zero one, and may do so before hipHostRegister
    pins it.
    """
    import numpy as np

    pages = np.frombuffer(
        (ctypes.c_uint8 * length).from_address(address), dtype=np.uint8
    )
    started = last_report = time.monotonic()
    for offset in range(0, length, _TOUCH_SLICE_BYTES):
        pages[offset : offset + _TOUCH_SLICE_BYTES : HUGE_PAGE_BYTES] = 1
        now = time.monotonic()
        if now - last_report >= _TOUCH_PROGRESS_INTERVAL_S:
            last_report = now
            logger.info(
                "LMCache L1: %.1f of %.1f GiB faulted in on NUMA node %d after "
                "%.0f s (%.1f GiB free on the node)",
                min(offset + _TOUCH_SLICE_BYTES, length) / 2**30,
                length / 2**30,
                node,
                now - started,
                numa_node_meminfo(node).get("MemFree", 0) / 2**30,
            )


def _collapse_into_huge_pages(address: int, length: int) -> None:
    """Ask the kernel to rebuild the range's 4 KiB stretches as huge pages.

    A huge-page fault that finds no free 2 MiB block on the bound node falls
    back to 4 KiB pages; MADV_COLLAPSE compacts synchronously whatever the THP
    defrag setting, allocates under the range's mempolicy and skips what is
    already huge. It fails with ENOMEM or EAGAIN when compaction finds no huge
    page, which on pit2-p03-g40 happened to one of four stages allocating at
    once and not to the others, so it is retried a few times with a growing
    pause. Best effort: the caller counts again. It must run before the range
    is pinned, since pinned pages cannot move.
    """
    libc = _libc()
    for attempt, delay_s in enumerate((*_COLLAPSE_RETRY_DELAYS_S, None), start=1):
        if libc.madvise(address, length, _MADV_COLLAPSE) == 0:
            if attempt > 1:
                logger.info(
                    "LMCache L1: MADV_COLLAPSE succeeded on attempt %d", attempt
                )
            return
        error = ctypes.get_errno()
        logger.warning(
            "LMCache L1: madvise(MADV_COLLAPSE) of %.2f GiB failed on attempt %d: %s",
            length / 2**30,
            attempt,
            os.strerror(error),
        )
        if delay_s is None or error not in (errno.ENOMEM, errno.EAGAIN):
            return
        time.sleep(delay_s)


def _host_register(address: int, length: int) -> None:
    import torch

    status = torch.cuda.cudart().cudaHostRegister(address, length, 0)
    if int(status) != 0:
        raise RuntimeError(f"hipHostRegister of {length} bytes failed: {status}")


def _host_unregister(address: int) -> None:
    import torch

    status = torch.cuda.cudart().cudaHostUnregister(address)
    if int(status) != 0:
        raise RuntimeError(f"hipHostUnregister failed: {status}")


def anon_huge_page_bytes(start: int, end: int, smaps: str = "/proc/self/smaps") -> int:
    """Bytes of ``[start, end)`` backed by transparent huge pages.

    Raises:
        RuntimeError: A mapping straddles a boundary of the range, so its huge
            pages cannot be attributed to the range.
    """
    total_kb = 0
    in_range = False
    # A mapping's name is a path, and paths need not be UTF-8.
    with open(smaps, errors="replace") as f:
        for line in f:
            header = _SMAPS_HEADER.match(line)
            if header:
                low, high = int(header[1], 16), int(header[2], 16)
                in_range = low < end and high > start
                if in_range and (low < start or high > end):
                    raise RuntimeError(
                        f"mapping {low:#x}-{high:#x} straddles {start:#x}-{end:#x}"
                    )
            elif in_range and line.startswith("AnonHugePages:"):
                total_kb += int(line.split()[1])
    return total_kb * 1024


def _thp_settings() -> str:
    settings = []
    for name in ("enabled", "defrag"):
        try:
            settings.append(f"{name}={(_THP_SYSFS_ROOT / name).read_text().strip()}")
        except OSError:
            settings.append(f"{name}=?")
    return ", ".join(settings)


def _fault_thp_region(region_bytes: int, node: int) -> tuple[int, _ThpRegion]:
    """Map ``region_bytes`` on ``node`` and fault it as transparent huge pages.

    mmap, mbind(MPOL_BIND), MADV_HUGEPAGE, one first touch per 2 MiB, and
    MADV_COLLAPSE when that left 4 KiB pages. Not pinned, and not checked:
    the caller pins it and then counts its huge pages.

    Returns:
        (2 MiB-aligned address, region).
    """
    _warn_if_node_is_short(node, region_bytes)
    # One spare huge page lets the region start on a 2 MiB boundary.
    mapping_bytes = region_bytes + HUGE_PAGE_BYTES
    libc = _libc()
    mapping = libc.mmap(
        None, mapping_bytes, _PROT_READ_WRITE, _MAP_PRIVATE_ANONYMOUS, -1, 0
    )
    if mapping in (None, _MAP_FAILED):
        raise _os_error(f"mmap of {mapping_bytes} bytes")
    address = (mapping + HUGE_PAGE_BYTES - 1) & ~(HUGE_PAGE_BYTES - 1)
    try:
        _mbind(address, region_bytes, node)
        if libc.madvise(address, region_bytes, _MADV_HUGEPAGE) != 0:
            raise _os_error("madvise(MADV_HUGEPAGE)")
        _touch_every_huge_page(address, region_bytes, node)
        huge = anon_huge_page_bytes(address, address + region_bytes)
        if huge != region_bytes:
            # pit2-p03-g40, 48.6 GiB per stage next to a 768 GiB Store owner:
            # 2-27 MiB of each pool came back as 4 KiB pages.
            logger.info(
                "LMCache L1: %d bytes of the %.2f GiB pool on NUMA node %d are "
                "4 KiB pages after the first touch; collapsing them",
                region_bytes - huge,
                region_bytes / 2**30,
                node,
            )
            _collapse_into_huge_pages(address, region_bytes)
    except BaseException:
        libc.munmap(mapping, mapping_bytes)
        raise
    return address, _ThpRegion(mapping, mapping_bytes, region_bytes, node)


def reserve_thp_l1(size: int, numa_id: int) -> None:
    """Fault an L1 of ``size`` bytes on ``numa_id`` now; the L1 allocation takes it.

    For a worker before its weights load: a weight load pins its staging
    buffers and fills the page cache, pinned pages never move, and the node
    left behind can be too fragmented to compact into the L1's huge pages
    (pit2-p03-g35, next to 768 GiB of Store owners: 1.2 GiB of a 48.6 GiB L1
    stayed 4 KiB pages after MADV_COLLAPSE, with 642 GiB free). One reservation
    per node; a second one replaces the first.
    """
    size, node = int(size), int(numa_id)
    region_bytes = -(-size // HUGE_PAGE_BYTES) * HUGE_PAGE_BYTES
    started = time.monotonic()
    address, region = _fault_thp_region(region_bytes, node)
    with _thp_regions_lock:
        previous = _reserved_l1.pop(node, None)
        _reserved_l1[node] = (address, region)
    if previous is not None:
        _libc().munmap(previous[1].mapping, previous[1].mapping_bytes)
    huge = anon_huge_page_bytes(address, address + region_bytes)
    logger.info(
        "LMCache L1: reserved %.2f GiB on NUMA node %d before the weights load, "
        "%.2f GiB of it transparent huge pages (%.1f s)",
        region_bytes / 2**30,
        node,
        huge / 2**30,
        time.monotonic() - started,
    )


def _take_reserved_l1(node: int, region_bytes: int) -> tuple[int, _ThpRegion] | None:
    """The node's reserved region if it holds ``region_bytes``; else free it."""
    with _thp_regions_lock:
        reserved = _reserved_l1.pop(node, None)
    if reserved is None:
        return None
    if reserved[1].region_bytes >= region_bytes:
        return reserved
    logger.warning(
        "LMCache L1: the %.2f GiB reserved on NUMA node %d is smaller than the "
        "%.2f GiB pool; allocating the pool anew",
        reserved[1].region_bytes / 2**30,
        node,
        region_bytes / 2**30,
    )
    _libc().munmap(reserved[1].mapping, reserved[1].mapping_bytes)
    return None


def release_reserved_l1() -> None:
    """Free the reservations no L1 allocation took."""
    with _thp_regions_lock:
        reserved = list(_reserved_l1.items())
        _reserved_l1.clear()
    for node, (_, region) in reserved:
        logger.warning(
            "LMCache L1: no pool took the %.2f GiB reserved on NUMA node %d; "
            "freeing it",
            region.region_bytes / 2**30,
            node,
        )
        _libc().munmap(region.mapping, region.mapping_bytes)


def current_gpu_numa_node() -> int | None:
    """The NUMA node of the current GPU, None when the platform has none."""
    return _current_gpu_numa_node()


def alloc_thp_pinned_numa_ptr(size: int, numa_id: int = 0) -> int:
    """Pinned host memory on ``numa_id``, every byte of it a transparent huge page.

    A drop-in for ``lmcache.device_ops.alloc_pinned_numa_ptr``: the region
    :func:`reserve_thp_l1` faulted on the node when it is large enough, else
    a new one (mmap, mbind(MPOL_BIND), MADV_HUGEPAGE, one first touch per
    2 MiB, MADV_COLLAPSE when that left 4 KiB pages); then hipHostRegister.
    The region is ``size`` rounded up to 2 MiB at least. A node other than the
    current GPU's is refused before anything is allocated.

    Raises:
        ValueError: ``numa_id`` is not the current GPU's node.
        RuntimeError: Fewer than all bytes are huge pages (the message gives
            the THP settings), or pinning failed.
        OSError: A system call failed.
    """
    size, node = int(size), int(numa_id)
    if size <= 0:
        raise ValueError(f"pinned allocation size must be positive, got {size}")
    _require_thp_allowed()
    _require_gpu_numa_node(node)
    started = time.monotonic()
    reserved = _take_reserved_l1(node, -(-size // HUGE_PAGE_BYTES) * HUGE_PAGE_BYTES)
    if reserved is None:
        address, region = _fault_thp_region(
            -(-size // HUGE_PAGE_BYTES) * HUGE_PAGE_BYTES, node
        )
    else:
        address, region = reserved
    region_bytes = region.region_bytes
    registered = False
    try:
        # A reserved region waited through the weight load; a fresh one was
        # just collapsed.
        if (
            reserved is not None
            and anon_huge_page_bytes(address, address + region_bytes) != region_bytes
        ):
            _collapse_into_huge_pages(address, region_bytes)
        _host_register(address, region_bytes)
        registered = True
        huge = anon_huge_page_bytes(address, address + region_bytes)
        if huge != region_bytes:
            raise RuntimeError(
                f"the {region_bytes / 2**30:.2f} GiB LMCache L1 on NUMA node {node} "
                f"has {region_bytes - huge} bytes outside transparent huge pages "
                f"even after MADV_COLLAPSE ({_thp_settings()}); drop the node's "
                "page cache or free memory before starting: the Mooncake Store "
                "L2 needs an L1 of huge pages"
            )
    except BaseException:
        if registered:
            try:
                _host_unregister(address)
            except Exception:  # cleanup of a failed allocation
                logger.warning(
                    "hipHostUnregister after a failed allocation failed",
                    exc_info=True,
                )
        _libc().munmap(region.mapping, region.mapping_bytes)
        raise
    with _thp_regions_lock:
        _thp_regions[address] = region
    logger.info(
        "LMCache L1: %.2f GiB pinned on NUMA node %d, all transparent huge pages "
        "(%s, %.1f s)",
        region_bytes / 2**30,
        node,
        "reserved before the weights load" if reserved is not None else "allocated",
        time.monotonic() - started,
    )
    return address


def free_thp_pinned_numa_ptr(ptr: int, size: int | None = None) -> None:
    """Free a region of :func:`alloc_thp_pinned_numa_ptr`.

    A pointer it did not allocate goes to the allocator it replaced.
    """
    with _thp_regions_lock:
        region = _thp_regions.pop(int(ptr), None)
    if region is None:
        if _native_free_pinned_numa_ptr is None:
            raise ValueError(f"{int(ptr):#x} is not a pinned THP region")
        _native_free_pinned_numa_ptr(ptr, size)
        return
    try:
        _host_unregister(int(ptr))
    finally:
        _libc().munmap(region.mapping, region.mapping_bytes)


def install_thp_pinned_allocator() -> None:
    """Make LMCache allocate its NUMA-bound pinned pools on huge pages.

    Process-wide and idempotent. LMCache resolves
    ``lmcache.device_ops.alloc_pinned_numa_ptr`` at allocation time, so
    patching the instance attribute reaches every later pool.
    """
    global _native_free_pinned_numa_ptr

    import lmcache

    device_ops = lmcache.device_ops
    with _install_lock:
        if device_ops.alloc_pinned_numa_ptr is alloc_thp_pinned_numa_ptr:
            return
        _native_free_pinned_numa_ptr = device_ops.free_pinned_numa_ptr
        device_ops.alloc_pinned_numa_ptr = alloc_thp_pinned_numa_ptr
        device_ops.free_pinned_numa_ptr = free_thp_pinned_numa_ptr


def verify_thp_l1(engine: Any) -> None:
    """Fail unless the engine's L1 is a region of :func:`alloc_thp_pinned_numa_ptr`.

    LMCache sends its pool through the patched allocator only when it has a
    NUMA mapping for the GPU. When auto-detection fails it logs a warning and
    falls back to hipHostMalloc -- huge pages advised, the node preferred
    rather than bound, nothing checked -- and the hugetlb, shared-memory and
    P2P options use other allocators. Mooncake can register such a pool all
    the same (1 GiB MRs absorb a few 4 KiB pages), so only a region allocated
    here is known to be huge pages on the GPU's node. Runs after
    ``post_init``, which allocates the pool.

    Raises:
        RuntimeError: The engine has no pinned L1, or another allocator made it.
    """
    storage_manager = getattr(engine, "storage_manager", None)
    backends = getattr(storage_manager, "storage_backends", None) or {}
    allocator = getattr(backends.get("LocalCPUBackend"), "memory_allocator", None)
    buffer = getattr(getattr(allocator, "pin_allocator", None), "buffer", None)
    if buffer is None:
        raise RuntimeError(
            "LMCache built no pinned CPU pool (L1) for the Mooncake Store L2 to "
            f"register (allocator: {type(allocator).__name__})"
        )
    start = int(buffer.data_ptr())
    end = start + int(buffer.numel()) * int(buffer.element_size())
    with _thp_regions_lock:
        node = next(
            (
                region.node
                for address, region in _thp_regions.items()
                if address <= start and end <= address + region.region_bytes
            ),
            None,
        )
    if node is None:
        raise RuntimeError(
            f"the {(end - start) / 2**30:.2f} GiB LMCache L1 at {start:#x} did not "
            "come from ATOM's huge-page allocator, so it may hold 4 KiB pages or "
            "sit on the other NUMA node: LMCache takes that allocator only with a "
            "NUMA mapping for this GPU (see 'Failed to auto read NUMA mapping' "
            "above), and not for hugetlb, shm_name or P2P pools"
        )
    logger.info(
        "LMCache Mooncake Store L2: the %.2f GiB L1 is transparent huge pages "
        "bound to NUMA node %d",
        (end - start) / 2**30,
        node,
    )


# ---------------------------------------------------------------------------
# One batch_is_exist RPC per lookup
# ---------------------------------------------------------------------------


def count_stored_prefix(store: Any, keys: list) -> int:
    """How many leading ``keys`` the Store holds, asked in one RPC.

    ``batch_is_exist`` answers 1 (present), 0 (absent) or a negative error
    per key; an error ends the prefix like an absence.
    """
    if not keys:
        return 0
    hits = 0
    for result in store.batch_is_exist([key.to_string() for key in keys]):
        if result != 1:
            break
        hits += 1
    return hits


def _batched_contains_by_batch_is_exist(self, keys: list) -> int:
    return count_stored_prefix(self.store, keys)


def _supports_batched_contains(self) -> bool:
    return True


def install_batch_is_exist_lookup() -> None:
    """Give LMCache's Mooncake connector a one-RPC ``batched_contains``.

    Idempotent, and a no-op once LMCache's connector implements its own.
    """
    from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakestoreConnector,
    )

    with _install_lock:
        if (
            MooncakestoreConnector.support_batched_contains
            is not RemoteConnector.support_batched_contains
        ):
            return
        MooncakestoreConnector.support_batched_contains = _supports_batched_contains
        MooncakestoreConnector.batched_contains = _batched_contains_by_batch_is_exist


# ---------------------------------------------------------------------------
# L2 gets that never wait for room in the L1
# ---------------------------------------------------------------------------


class _AllocateWithoutWaiting:
    """LMCache's CPU backend as the Mooncake connector's gets see it.

    ``LocalCPUBackend.allocate`` busy-waits by default until eviction frees
    enough of the L1. The connector's batched get allocates on LMCache's
    storage event loop, which is also where the write-through puts finish and
    drop the L1 references that keep their chunks from being evicted: once the
    in-flight puts and the pinned chunks fill the L1, the loop waits for itself
    forever, and every later put and get of the worker hangs behind it.

    Here an allocation evicts what it can and otherwise returns None, and after
    one fails the rest of the batch is not allocated at all: LMCache keeps a
    retrieved prefix only up to the first missing chunk, so a later buffer
    would only evict L1 chunks to read data that is thrown away.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._batch_failed = False

    def start_batch(self) -> None:
        self._batch_failed = False

    def allocate(self, shapes, dtypes, fmt=None, eviction=True, busy_loop=True):
        del busy_loop  # never: the caller is LMCache's storage event loop
        if self._batch_failed:
            return None
        memory_obj = self._backend.allocate(
            shapes, dtypes, fmt, eviction=eviction, busy_loop=False
        )
        if memory_obj is None:
            self._batch_failed = True
        return memory_obj

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)


def install_non_blocking_l2_get_allocation() -> None:
    """Make the Mooncake connector's gets give up on a full L1 instead of waiting.

    Wraps ``MooncakestoreConnector.batched_get``, its only get entry point:
    each call starts a new batch for the connector's
    :class:`_AllocateWithoutWaiting` view of the CPU backend, created on first
    use. The allocations of a batch run before its first ``await``, so no other
    coroutine on the loop interleaves with them. Process-wide and idempotent.
    """
    from lmcache.v1.storage_backend.connector.mooncakestore_connector import (
        MooncakestoreConnector,
    )

    with _install_lock:
        upstream = MooncakestoreConnector.batched_get
        if getattr(upstream, "_atom_without_waiting", False):
            return

        @functools.wraps(upstream)
        async def batched_get(self, keys):
            backend = self.local_cpu_backend
            if not isinstance(backend, _AllocateWithoutWaiting):
                backend = _AllocateWithoutWaiting(backend)
                self.local_cpu_backend = backend
            backend.start_batch()
            return await upstream(self, keys)

        batched_get._atom_without_waiting = True
        MooncakestoreConnector.batched_get = batched_get
