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
from typing import Any

# The GPU's PCIe-local NIC; re-exported for the Store's callers and tests.
from atom.utils.pci_topology import IB_SYSFS_ROOT, gpu_pci_bdf, rail_rdma_device

logger = logging.getLogger("atom")

_IB_SYSFS_ROOT = IB_SYSFS_ROOT


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
