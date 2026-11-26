# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""Which HCAs a Mooncake TransferEngine registers on, and its transport.

Shared by every Mooncake user in ATOM: the KV connector
(``kv_transfer/disaggregation/mooncake``) and the image embedding transfer
(``atom/ec_transfer/mooncake.py``).
"""

from __future__ import annotations

import os
from pathlib import Path

IB_SYSFS_ROOT = Path("/sys/class/infiniband")


def ib_device_exists(device_name: str) -> bool:
    return (IB_SYSFS_ROOT / device_name).exists()


def auto_select_ib_device(phys_idx: int) -> str:
    # Older environments expose paired HCAs as rdmaN. Spur MI350 fabric exposes
    # them as ionic_N, so try ionic_N only when rdmaN is not present.
    rdma_device = f"rdma{phys_idx}"
    if ib_device_exists(rdma_device):
        return rdma_device
    ionic_device = f"ionic_{phys_idx}"
    if ib_device_exists(ionic_device):
        return ionic_device
    return rdma_device


def parse_ib_devices(configured_devices: str) -> list[str]:
    """Normalize a comma-separated Mooncake RDMA device filter."""
    return list(
        dict.fromkeys(
            device.strip() for device in configured_devices.split(",") if device.strip()
        )
    )


def select_ib_devices(
    protocol: str,
    configured_devices: str,
    phys_idx: int | None,
    *,
    enable_alternate_hca: bool = False,
    hca_count: int = 8,
) -> list[str]:
    """Resolve the HCAs on which Mooncake registers this rank's GPU memory."""
    if protocol.strip().lower() == "tcp":
        return []
    if configured_devices:
        return parse_ib_devices(configured_devices)
    if phys_idx is None:
        raise ValueError("physical GPU index is required for RDMA device selection")

    primary = auto_select_ib_device(phys_idx)
    devices = [primary]
    if enable_alternate_hca:
        if hca_count <= 0:
            raise ValueError("ib_hca_count must be a positive integer")
        for idx in range(hca_count):
            device = auto_select_ib_device(idx)
            if device not in devices and ib_device_exists(device):
                devices.append(device)
    return devices


def ib_device_filter(
    protocol: str,
    configured_devices: str,
    phys_idx: int | None,
    *,
    enable_alternate_hca: bool = False,
    hca_count: int = 8,
) -> str:
    """``select_ib_devices`` as the comma-separated filter
    ``TransferEngine.initialize`` takes.

    Mooncake's TCP transport requires an empty device list. Passing a usable
    HCA alongside ``protocol=tcp`` allows the transfer engine to activate RDMA
    as an alternate path, which violates the caller's explicit transport
    choice. RDMA-family transports retain the existing configured/automatic
    device selection.
    """
    return ",".join(
        select_ib_devices(
            protocol,
            configured_devices,
            phys_idx,
            enable_alternate_hca=enable_alternate_hca,
            hca_count=hca_count,
        )
    )


def configure_transport(protocol: str) -> None:
    """Make Mooncake honor ATOM's explicit transport selection.

    Legacy TransferEngine builds auto-discover installed HCAs independently
    of the device filter. ``MC_FORCE_TCP`` is Mooncake's supported override
    for preventing that implicit RDMA transport from being installed.
    """
    if protocol.strip().lower() == "tcp":
        os.environ["MC_FORCE_TCP"] = "true"
