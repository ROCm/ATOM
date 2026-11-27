# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which RDMA device sits behind a GPU's own PCIe switch, read from sysfs.

Shared by every RDMA user in ATOM: the Mooncake Store offload's requesters
(``kv_transfer/offload/mooncake_store/nic.py``), the Mooncake P/D connector
and the image embedding transfer, through ``atom/utils/mooncake_hca.py``.
HIP numbers GPUs in KFD order, not PCI order, so a GPU's NIC is found from
its PCI address, not from its ordinal.
"""

from __future__ import annotations

import re
from pathlib import Path

IB_SYSFS_ROOT = Path("/sys/class/infiniband")
PCI_SYSFS_ROOT = Path("/sys/bus/pci/devices")
# The sysfs path component of a PCI host bridge, e.g. "pci0000:70".
_PCI_HOST_BRIDGE = re.compile(r"pci[0-9a-f]{4}:[0-9a-f]{2}")


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
    ib_root: Path = IB_SYSFS_ROOT,
    pci_root: Path = PCI_SYSFS_ROOT,
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
