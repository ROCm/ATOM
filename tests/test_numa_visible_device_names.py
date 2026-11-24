# SPDX-License-Identifier: MIT
"""*_VISIBLE_DEVICES entries given as ``GPU-<unique id>`` names map to physical
indices, so NUMA binding resolves a node for them as it does for numbers."""

import os

import pytest

from atom.utils import numa_utils


def _cards(tmp_path, ids):
    """A fake /sys/class/drm with one card per (bdf, unique id), created in
    reverse so directory order differs from PCI-BDF order."""
    paths = []
    for n, (bdf, uid) in reversed(list(enumerate(ids))):
        real = tmp_path / "pci" / bdf
        real.mkdir(parents=True)
        (real / "unique_id").write_text(uid + "\n")
        (real / "numa_node").write_text("0\n" if n < 2 else "1\n")
        card = tmp_path / "drm" / f"card{9 - n}"
        card.mkdir(parents=True)
        os.symlink(real, card / "device")
        paths.append(str(card / "device"))
    return paths


@pytest.fixture
def fake_sysfs(tmp_path, monkeypatch):
    ids = [
        ("0002:00:01.0", "aa01"),
        ("0002:00:02.0", "bb02"),
        ("0003:00:01.0", "cc03"),
        ("0003:00:02.0", "dd04"),
    ]
    paths = _cards(tmp_path, ids)
    monkeypatch.setattr(numa_utils.glob, "glob", lambda pattern: paths)
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    return ids


def test_numbers_are_taken_as_they_are(fake_sysfs, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    assert [numa_utils._physical_index(i) for i in range(2)] == [2, 3]


def test_names_map_to_pci_order(fake_sysfs, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-aa01,GPU-bb02,GPU-cc03,GPU-dd04")
    assert [numa_utils._physical_index(i) for i in range(4)] == [0, 1, 2, 3]


def test_a_subset_of_names_keeps_its_own_cards(fake_sysfs, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-DD04, GPU-bb02")
    assert [numa_utils._physical_index(i) for i in range(2)] == [3, 1]


def test_names_resolve_a_node_through_sysfs(fake_sysfs, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-aa01,GPU-bb02,GPU-cc03,GPU-dd04")
    assert [numa_utils._query_node_sysfs(i) for i in range(4)] == [0, 0, 1, 1]


def test_an_unknown_name_is_an_error(fake_sysfs, monkeypatch):
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "GPU-ffff")
    with pytest.raises(ValueError, match="no GPU with unique id"):
        numa_utils._physical_index(0)
