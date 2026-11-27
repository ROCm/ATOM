# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which RDMA device a worker's Store client uses.

* **One NIC per worker, the GPU's own.** Under one shared master, where every
  stage reads every owner, a requester listing several NICs, or owners on the
  requesters' NICs, stalled concurrent reads for 30-60 s and then failed them;
  one NIC per stage, disjoint from the owners', ran 4 x 33 GB/s with no error.
  The NIC is the ACTIVE RDMA device behind the GPU's own PCIe switch: the one
  closest to the GPU in the PCI tree, sharing its root port and the switch's
  upstream port, and no closer to another GPU. A GPU whose own NIC is down or
  missing gets none, not a neighbour's. HIP numbers the GPUs in KFD order, not
  PCI order, so neither ``rdma<gpu>`` nor a BDF-sorted list names it reliably.
* **Per-NIC pools.** Where the owners must share the stages' NICs (two nodes
  whose only NICs are their GPUs'), one master per NIC keeps each NIC to one
  stage and its own pool's owners: 4 x 37 GB/s with no retransmission across
  two nodes, where one master for the same 8 owners stalled a stage for good.
  A worker uses its NIC's pool (``MooncakeStoreOffloadConfig.pool_of``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

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
        try:
            device = rail_rdma_device(bdf)
        except ValueError as exc:
            raise ValueError(f"{exc}; set mooncake_store.rdma_devices") from None
        origin = f"the PCI topology of GPU {device_index} ({bdf})"
    if not (_IB_SYSFS_ROOT / device).exists():
        raise ValueError(f"RDMA device {device!r} from {origin} does not exist")
    owner_devices = list(cfg.owner_rdma_devices)
    # With per-NIC pools the owners on this device are its own pool's, the
    # only ones this worker reads; `cfg.pool_of` checks the device has one.
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
    """Return the ACTIVE RDMA device behind the GPU's own PCIe switch.

    On a rail-optimized node each GPU sits behind a PCIe switch with exactly
    one NIC, which shares more of the sysfs device path with that GPU than
    any other NIC does. The device must share the GPU's root port and the
    switch's upstream port, and be no closer to another GPU (a PCI function of
    this GPU's vendor and class): with the GPU's own NIC down or missing, the
    closest ACTIVE one is a neighbour's.

    Raises:
        ValueError: The GPU is not in sysfs, no ACTIVE device shares its PCIe
            switch, several are equally close, or the closest is closer to
            another GPU.
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
    nic_paths: dict[str, tuple[str, ...]] = {}
    for device in devices:
        if not _has_active_port(device):
            continue
        nic_paths[device.name] = (device / "device").resolve().parts
        closeness[device.name] = _shared_depth(gpu_path, nic_paths[device.name])
    best = max(closeness.values(), default=0)
    closest = [name for name, shared in closeness.items() if shared == best]
    # Below the host bridge come the root port and the switch's upstream port;
    # a NIC on another root port is another GPU's, even under one host bridge.
    switch = host_bridge + 2
    if best <= switch or len(closest) != 1:
        found = ", ".join(f"{n}:{s}" for n, s in sorted(closeness.items()))
        raise ValueError(
            f"cannot tell GPU {gpu_bdf}'s NIC from the PCI topology (ACTIVE "
            f"devices and shared path depth: {found or 'none'})"
        )
    # A NIC closer to another GPU is that GPU's, as when this GPU's own is gone.
    for peer in _peer_gpu_paths(gpu_node, pci_root):
        if _shared_depth(peer, nic_paths[closest[0]]) > best:
            raise ValueError(
                f"cannot tell GPU {gpu_bdf}'s NIC from the PCI topology: the "
                f"closest ACTIVE device, {closest[0]}, is closer to GPU {peer[-1]}"
            )
    return closest[0]


def _shared_depth(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    shared = 0
    for a_part, b_part in zip(a, b):
        if a_part != b_part:
            break
        shared += 1
    return shared


def _peer_gpu_paths(gpu_node: Path, pci_root: Path) -> list[tuple[str, ...]]:
    """sysfs paths of the node's other GPUs: PCI functions of this GPU's vendor
    and class, so it assumes one GPU model per node."""
    kind = _pci_kind(gpu_node)
    if kind is None:
        raise ValueError(f"cannot read GPU {gpu_node.name}'s PCI vendor and class")
    try:
        nodes = list(pci_root.iterdir())
    except OSError as exc:
        raise ValueError(f"cannot list PCI devices in {pci_root}") from exc
    return [
        node.resolve().parts
        for node in nodes
        if node.name != gpu_node.name and _pci_kind(node) == kind
    ]


def _pci_kind(node: Path) -> tuple[str, ...] | None:
    try:
        return tuple((node / f).read_text().strip() for f in ("vendor", "class"))
    except OSError:
        return None


def _has_active_port(device: Path) -> bool:
    for state_file in device.glob("ports/*/state"):
        try:
            state = state_file.read_text().partition(":")[0].strip()
        except OSError:
            continue
        if state == "4":  # IB_PORT_ACTIVE, also used by RoCE devices
            return True
    return False
