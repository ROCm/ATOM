# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which RDMA device, and which Store pool, a worker's Store client uses.

* **One NIC per worker, the GPU's own.** Under one shared master, where every
  stage reads every owner, a requester listing several NICs, or owners on the
  requesters' NICs, stalled concurrent reads for 30-60 s and then failed them;
  one NIC per stage, disjoint from the owners', ran 4 x 33 GB/s with no error.
  The NIC is the RDMA device closest to the GPU in the PCI tree. HIP numbers
  the GPUs in KFD order, not PCI order, so neither ``rdma<gpu>`` nor a
  BDF-sorted list names it reliably.
* **Per-NIC pools.** Where the owners must share the stages' NICs (two nodes
  whose only NICs are their GPUs'), one master per NIC keeps each NIC to one
  stage and its own pool's owners: 4 x 37 GB/s with no retransmission across
  two nodes, where one master for the same 8 owners stalled a stage for good.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger("atom")

_IB_SYSFS_ROOT = Path("/sys/class/infiniband")
_PCI_SYSFS_ROOT = Path("/sys/bus/pci/devices")
# The sysfs path component of a PCI host bridge, e.g. "pci0000:70".
_PCI_HOST_BRIDGE = re.compile(r"pci[0-9a-f]{4}:[0-9a-f]{2}")


def parse_device_list(value: str) -> list[str]:
    """Split a comma-separated device list, dropping blanks."""
    return [device.strip() for device in value.split(",") if device.strip()]


def requester_rdma_device(device_index: int, cfg: Any) -> str:
    """Return the one RDMA device the Store client of this GPU's worker uses.

    ``cfg.rdma_devices`` wins when set: one device per GPU ordinal, or one for
    all. Otherwise it is the GPU's NIC in the PCI tree.

    Raises:
        ValueError: No device can be chosen, it does not exist, or, without
            per-NIC pools, the Store owners on this node use it
            (``cfg.owner_rdma_devices``).
    """
    table = list(cfg.rdma_devices)
    if table:
        if len(table) == 1:
            device = table[0]
        elif device_index < len(table):
            device = table[device_index]
        else:
            raise ValueError(
                f"mooncake_store.rdma_devices lists {len(table)} devices and "
                f"has none for GPU {device_index}"
            )
        origin = "mooncake_store.rdma_devices"
    else:
        bdf = gpu_pci_bdf(device_index)
        device = rail_rdma_device(bdf)
        origin = f"the PCI topology of GPU {device_index} ({bdf})"
    if not (_IB_SYSFS_ROOT / device).exists():
        raise ValueError(f"RDMA device {device!r} from {origin} does not exist")
    owner_devices = list(cfg.owner_rdma_devices)
    # With per-NIC pools the owners on this device are its own pool's, the
    # only ones this worker reads; store_pool_of checks the device has one.
    if device in owner_devices and not cfg.pools:
        raise ValueError(
            f"RDMA device {device!r} from {origin} is also a Store owner's "
            f"(mooncake_store.owner_rdma_devices={','.join(owner_devices)}); "
            "owners and requesters on one NIC stall concurrent reads"
        )
    logger.info(
        "Mooncake Store offload: GPU %d uses RDMA device %s (from %s)",
        device_index,
        device,
        origin,
    )
    return device


class StorePool(NamedTuple):
    """The master of one per-NIC Store pool."""

    master: str
    metadata: str


def parse_store_pools(value: Any) -> dict[str, StorePool]:
    """Parse ``mooncake_store.pools``; None or blank means one shared pool.

    Accepts the JSON object itself or its text.

    Raises:
        ValueError: Not a non-empty object mapping each RDMA device to
            ``{"master": "host:port", "metadata": "<url>"}``.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        if not value.strip():
            return {}
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"mooncake_store.pools is not JSON: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError(
            "mooncake_store.pools must be a non-empty JSON object keyed by RDMA "
            "device"
        )
    pools = {}
    for device, pool in value.items():
        fields = pool if isinstance(pool, Mapping) else {}
        master, metadata = fields.get("master"), fields.get("metadata")
        if not (
            isinstance(device, str)
            and device
            and isinstance(master, str)
            and master
            and isinstance(metadata, str)
            and metadata
        ):
            raise ValueError(
                f"mooncake_store.pools[{device!r}] must be "
                '{"master": "host:port", "metadata": "<url>"}'
            )
        pools[device] = StorePool(master, metadata)
    return pools


def store_pool_of(
    device: str | None, pools: Mapping[str, StorePool]
) -> StorePool | None:
    """Return the per-NIC pool of a worker's RDMA device, None without pools.

    Where owners must share the requesters' NICs (a node with only its own
    GPUs' NICs), each NIC gets a master of its own: the worker on a NIC reads
    only that pool's owners, and a NIC carries one requester and its own
    pool's owners instead of every owner's reads.

    Raises:
        ValueError: Pools are set, but the worker has no RDMA device (tcp)
            or no pool for its device.
    """
    if not pools:
        return None
    if device is None:
        raise ValueError(
            "mooncake_store.pools keys the Store pools by RDMA device; a tcp "
            "Store client has none"
        )
    pool = pools.get(device)
    if pool is None:
        raise ValueError(
            f"mooncake_store.pools has no pool for RDMA device {device!r} "
            f"(pools: {', '.join(sorted(pools))})"
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
            "mooncake_store.rdma_devices"
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
