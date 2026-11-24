# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ROCm GPU UUIDs in *_VISIBLE_DEVICES resolve to amd-smi's index (GPU-free)."""

import sys
from types import SimpleNamespace

import pytest

from atom.utils import numa_utils, visible_devices
from atom.utils.visible_devices import physical_gpu_index, physical_gpus

# The DRM cards of a pit2 node: card numbers are not in PCI-bus order, so
# amd-smi's GPU 0 is card9 and card1 is GPU 3. Value: (BDF, unique_id, node).
_AMD_CARDS = {
    "card1": ("0000:75:00.0", "7c2431bef0a19e9c", 0),
    "card9": ("0000:05:00.0", "b9ebac5f14bf98f1", 0),
    "card17": ("0000:65:00.0", "ac7e36c9cf753afd", 0),
    "card25": ("0000:15:00.0", "7160f88cb991bd5a", 0),
    "card33": ("0000:f5:00.0", "e34f75a476cede3d", 1),
    "card41": ("0000:85:00.0", "57f5c23cb997acc7", 1),
    "card49": ("0000:e5:00.0", "b1a42e0d0b35ae5c", 1),
    "card57": ("0000:95:00.0", "9f0c86e17f55a44d", 1),
}
# amd-smi's order, which `rocm-smi --showbus` prints for the node.
_AMD_SMI_BDFS = sorted(bdf for bdf, _, _ in _AMD_CARDS.values())
_GPU3 = "GPU-7c2431bef0a19e9c"  # card1, 0000:75:00.0, node 0
_GPU4 = "GPU-57f5c23cb997acc7"  # card41, 0000:85:00.0, node 1


@pytest.fixture
def drm(tmp_path, monkeypatch):
    """A fake /sys/class/drm of a pit2 node, with no visible-device mask."""
    root = tmp_path / "drm"

    def card(name, target, **files):
        target.mkdir(parents=True)
        for key, value in files.items():
            (target / key).write_text(f"{value}\n")
        (root / name).mkdir(parents=True)
        (root / name / "device").symlink_to(target)

    for name, (bdf, unique_id, node) in _AMD_CARDS.items():
        card(
            name,
            tmp_path / "pci" / bdf,
            vendor="0x1002",
            unique_id=unique_id,
            numa_node=node,
        )
    # The BMC's VGA sits between the GPUs on the bus; it is no GPU of amd-smi.
    card("card0", tmp_path / "pci" / "0000:54:00.0", vendor="0x1a03", numa_node=0)
    (root / "card0-VGA-1").mkdir()
    # A partition node is a platform device with no vendor.
    card("card10", tmp_path / "platform" / "amdgpu_xcp_7")
    monkeypatch.setattr(visible_devices, "_DRM_ROOT", str(root))
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    return root


def test_physical_gpus_are_amd_smi_gpus(drm):
    assert [gpu.bdf for gpu in physical_gpus()] == _AMD_SMI_BDFS


@pytest.mark.parametrize(
    "env", ["ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"]
)
def test_integer_entries_are_physical_indices(drm, monkeypatch, env):
    monkeypatch.setenv(env, "6, 2,")
    assert [physical_gpu_index(i) for i in range(2)] == [6, 2]


def test_no_mask_is_identity(drm):
    assert physical_gpu_index(5) == 5


@pytest.mark.parametrize(
    "mask,expected",
    [
        (f"{_GPU3},{_GPU4}", [3, 4]),
        (f"{_GPU4},{_GPU3}", [4, 3]),
        (f"{_GPU3},5", [3, 5]),
        (f"7,{_GPU4.upper()}", [7, 4]),
    ],
)
def test_uuid_entries_resolve_to_amd_smi_index(drm, monkeypatch, mask, expected):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", mask)
    assert [physical_gpu_index(i) for i in range(2)] == expected


def test_rocr_mask_wins_over_hip_mask(drm, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", _GPU4)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    assert physical_gpu_index(0) == 4


@pytest.mark.parametrize(
    "mask,logical,entry",
    [
        ("GPU-00000000deadbeef", 0, "GPU-00000000deadbeef"),
        (f"{_GPU3},GPU-xyz", 1, "GPU-xyz"),
        ("card1", 0, "card1"),
        ("3", 1, "for logical GPU 1"),
    ],
)
def test_unresolvable_entry_is_named(drm, monkeypatch, mask, logical, entry):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", mask)
    with pytest.raises(ValueError, match=entry):
        physical_gpu_index(logical)


@pytest.fixture
def no_amdsmi(monkeypatch):
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.delenv("ATOM_NUMA_NODE", raising=False)
    monkeypatch.delenv("ATOM_AUTO_NUMA_BIND", raising=False)


def test_numa_node_of_uuid_mask_from_sysfs(drm, no_amdsmi, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", f"{_GPU3},{_GPU4}")
    assert [numa_utils._resolve_node(i) for i in range(2)] == [0, 1]


def test_numa_sysfs_fallback_skips_non_amd_cards(drm, no_amdsmi, monkeypatch):
    # The BMC's VGA at 0000:54 must not shift GPUs 2-7 by one.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4")
    assert numa_utils._resolve_node(0) == 1


def test_numa_node_of_uuid_mask_from_amdsmi(drm, monkeypatch):
    nodes = [bdf >= "0000:80" for bdf in _AMD_SMI_BDFS]
    amdsmi = SimpleNamespace(
        amdsmi_init=lambda: None,
        amdsmi_shut_down=lambda: None,
        amdsmi_get_processor_handles=lambda: list(range(len(nodes))),
        amdsmi_topo_get_numa_node_number=lambda handle: int(nodes[handle]),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", amdsmi)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", f"{_GPU4},{_GPU3}")
    assert [numa_utils._query_node_amdsmi(i) for i in range(2)] == [1, 0]
