# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Map a process's logical GPU to the physical GPU index of system tools.

The physical index is amd-smi's and rocm-smi's: every AMD GPU on the node, in
PCI-bus order. It is the index the NIC pairing and the NUMA lookup use.

A ``*_VISIBLE_DEVICES`` entry is either an integer, taken as that index, or a
ROCm GPU UUID, ``GPU-<hex>``. The hex is the GPU's unique id, the value
``rocm-smi --showuniqueid`` prints after ``0x``, and a UUID is resolved through
the unique id the kernel publishes in sysfs. ROCr itself numbers integer
entries in KFD order, which on some nodes is not PCI-bus order; there only a
UUID names the same GPU for ROCr and for this index.
"""

import glob
import os
import re
from dataclasses import dataclass

# Read at call time so tests can point it at a fake tree.
_DRM_ROOT = "/sys/class/drm"
_AMD_PCI_VENDOR = "0x1002"
_CARD = re.compile(r"card\d+")
_UUID_PREFIX = "GPU-"

# ROCR first: it filters the device table before HIP sees it, so when both
# are set HIP's indices are already relative to the filtered list. A recipe
# that sets ROCR alone (the only correct way to split across NUMA nodes --
# setting both cuts the count twice) would otherwise fall through to the
# identity mapping.
_VISIBLE_DEVICES_ENVS = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


@dataclass(frozen=True)
class PhysicalGpu:
    bdf: str
    unique_id: int | None
    numa_node: int


def _read(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _read_int(path: str, base: int) -> int | None:
    text = _read(path)
    try:
        return int(text, base) if text else None
    except ValueError:
        return None


def physical_gpus() -> list[PhysicalGpu]:
    """Every AMD GPU on the node, in amd-smi's index order (PCI-bus order).

    Other DRM cards (a BMC's VGA, partition nodes, connectors) are skipped, so
    position ``i`` is amd-smi's GPU ``i``.
    """
    gpus: dict[str, PhysicalGpu] = {}
    for card in glob.glob(os.path.join(_DRM_ROOT, "card*")):
        if not _CARD.fullmatch(os.path.basename(card)):
            continue
        dev = os.path.join(card, "device")
        if _read(os.path.join(dev, "vendor")) != _AMD_PCI_VENDOR:
            continue
        numa_node = _read_int(os.path.join(dev, "numa_node"), 10)
        bdf = os.path.basename(os.path.realpath(dev)).lower()
        gpus[bdf] = PhysicalGpu(
            bdf=bdf,
            unique_id=_read_int(os.path.join(dev, "unique_id"), 16),
            numa_node=-1 if numa_node is None else numa_node,
        )
    return [gpus[bdf] for bdf in sorted(gpus)]


def physical_gpu_index(logical_idx: int) -> int:
    """Physical index of the process's GPU ``logical_idx`` (e.g. ``cuda:1``).

    Raises:
        ValueError: The mask has no entry for ``logical_idx``, or its entry is
            neither an integer nor the UUID of a GPU on this node.
    """
    name = next((n for n in _VISIBLE_DEVICES_ENVS if os.environ.get(n)), None)
    if name is None:
        return logical_idx
    value = os.environ[name]
    entries = [e.strip() for e in value.split(",") if e.strip()]
    if logical_idx >= len(entries):
        raise ValueError(f"{name}={value} has no entry for logical GPU {logical_idx}")
    entry = entries[logical_idx]
    try:
        return int(entry)
    except ValueError:
        pass
    problem = "is neither an integer nor a GPU-<hex> UUID"
    if entry[: len(_UUID_PREFIX)].upper() == _UUID_PREFIX:
        try:
            unique_id = int(entry[len(_UUID_PREFIX) :], 16)
        except ValueError:
            pass
        else:
            for idx, gpu in enumerate(physical_gpus()):
                if gpu.unique_id == unique_id:
                    return idx
            problem = f"matches no AMD GPU's unique_id under {_DRM_ROOT}"
    raise ValueError(f"{name} entry {entry!r} for logical GPU {logical_idx} {problem}")
