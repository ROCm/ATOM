# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The RDMA device nearest a GPU in the PCI tree.

On a rail-optimized node each GPU sits behind a PCIe switch with one NIC. HIP
numbers the GPUs in KFD order, not PCI order, so neither ``rdma<gpu>`` nor a
BDF-sorted list names that NIC reliably: on pit2-p03-g08 HIP GPUs 0..7 are the
GPUs of ``rdma3, 0, 2, 1, 7, 4, 6, 5`` (#3833). The sysfs paths do name it.

Used by the Mooncake Store client (``offload.mooncake_store.nic``) and the
Mooncake P/D connector (``disaggregation.mooncake.mooncake_connector``), so a
rank's Store and P/D traffic leave through the same NIC.
"""

from __future__ import annotations

import re
from pathlib import Path

# Read when a lookup runs, not bound as defaults, so tests can plant a tree.
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
    ib_root: Path | None = None,
    pci_root: Path | None = None,
) -> str:
    """Return the ACTIVE RDMA device that shares the deepest PCI path with a GPU.

    On a rail-optimized node each GPU sits behind a PCIe switch with exactly
    one NIC, which shares more of the sysfs device path with that GPU than
    any other NIC does. ``ib_root`` and ``pci_root`` default to
    ``IB_SYSFS_ROOT`` and ``PCI_SYSFS_ROOT``.

    Raises:
        ValueError: The GPU is not in sysfs, no ACTIVE device shares its PCI
            host bridge, or several are equally close.
    """
    ib_root = IB_SYSFS_ROOT if ib_root is None else ib_root
    pci_root = PCI_SYSFS_ROOT if pci_root is None else pci_root
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
        if not has_active_port(device):
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
            f"devices and shared path depth: {found or 'none'})"
        )
    return closest[0]


def has_active_port(device: Path) -> bool:
    """Whether an ``/sys/class/infiniband/<device>`` has a port in ACTIVE state."""
    for state_file in device.glob("ports/*/state"):
        try:
            state = state_file.read_text().partition(":")[0].strip()
        except OSError:
            continue
        if state == "4":  # IB_PORT_ACTIVE, also used by RoCE devices
            return True
    return False
