# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Mooncake Store L2 preparation of the in-process LMCache offload.

LMCache and Mooncake are faked; sysfs is a temporary tree shaped like a
rail-optimized MI355X node (pit2-p03-g40), where HIP numbers the GPUs in KFD
order rather than PCI order.
"""

from __future__ import annotations

import asyncio
import ctypes
import errno
import functools
import json
import logging
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload import mooncake_store_l2 as l2

# HIP ordinal -> GPU BDF, and the BDF of the NIC behind the same PCIe switch,
# as measured on pit2-p03-g40.
GPU_BDF = ["75", "05", "65", "15", "f5", "85", "e5", "95"]
NIC_BDF = {
    f"rdma{i}": bus
    for i, bus in enumerate(["09", "19", "69", "79", "89", "99", "e9", "f9"])
}


def _valid_extra(**overrides):
    extra = {
        "save_chunk_meta": False,
        "transfer_timeout": 60,
        "use_exists_sync": True,
        "mooncake_master_server_addr": "10.0.0.1:50051",
        "mooncake_metadata_server": "http://10.0.0.1:50080/metadata",
        "mooncake_protocol": "rdma",
        "mooncake_global_segment_size": "0",
        "mooncake_local_buffer_size": "67108864",
    }
    extra.update(overrides)
    return {k: v for k, v in extra.items() if v is not None}


def _cfg(**overrides):
    fields = {
        "remote_url": "mooncakestore://10.0.0.1:50051/",
        "numa_mode": "auto",
        "blocking_timeout_secs": 60,
        "extra_config": _valid_extra(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def store_env(monkeypatch):
    monkeypatch.setenv("MC_NUM_QP_PER_EP", "1")
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES", raising=False)
    monkeypatch.delenv("ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES", raising=False)
    monkeypatch.delenv("ATOM_LMCACHE_MOONCAKE_POOLS", raising=False)


def _root_complex(bus: str) -> str:
    return f"pci0000:{bus[0]}0"


def _make_node(root: Path, *, inactive=()):
    """A sysfs tree: per GPU one host bridge and one switch shared with a NIC."""
    devices = root / "devices"
    pci = root / "bus/pci/devices"
    ib = root / "class/infiniband"
    pci.mkdir(parents=True)
    ib.mkdir(parents=True)
    for bus in GPU_BDF:
        rc = _root_complex(bus)
        top = bus[0]
        gpu = devices / rc / f"0000:{top}0:01.1/0000:{top}1:00.0/0000:{top}2:00.0"
        gpu = gpu / f"0000:{bus}:00.0"
        gpu.mkdir(parents=True)
        (pci / f"0000:{bus}:00.0").symlink_to(gpu)
    for name, bus in NIC_BDF.items():
        top = bus[0]
        nic = devices / _root_complex(bus) / f"0000:{top}0:01.1/0000:{top}1:00.0"
        nic = nic / f"0000:{top}2:01.0/0000:{bus}:00.0"
        nic.mkdir(parents=True)
        node = ib / name
        (node / "ports/1").mkdir(parents=True)
        state = "1: DOWN" if name in inactive else "4: ACTIVE"
        (node / "ports/1/state").write_text(state + "\n")
        (node / "device").symlink_to(nic)
    return pci, ib


# --- recipe ---------------------------------------------------------------


def test_uses_mooncake_store_only_for_its_scheme():
    assert l2.uses_mooncake_store(_cfg())
    for url in (None, "", "lm://host:1", "redis://x", 7):
        assert not l2.uses_mooncake_store(SimpleNamespace(remote_url=url))
    assert not l2.uses_mooncake_store(SimpleNamespace())


def test_validate_accepts_the_recipe_and_legacy_protocol_key(store_env):
    l2.validate_mooncake_store_config(_cfg())
    legacy = _valid_extra(mooncake_protocol=None, protocol="rdma")
    legacy["mooncake_global_segment_size"] = 0
    l2.validate_mooncake_store_config(_cfg(extra_config=legacy))


@pytest.mark.parametrize(
    "env, cfg_overrides, message",
    [
        ({"MOONCAKE_CONFIG_PATH": "/x.json"}, {}, "MOONCAKE_CONFIG_PATH"),
        ({"MC_NUM_QP_PER_EP": "2"}, {}, "MC_NUM_QP_PER_EP=2"),
        ({"MC_NUM_QP_PER_EP": ""}, {}, "MC_NUM_QP_PER_EP=<unset"),
        ({}, {"numa_mode": None}, "LMCACHE_NUMA_MODE"),
        ({}, {"local_cpu_use_hugepages": True}, "LMCACHE_LOCAL_CPU_USE_HUGEPAGES"),
        ({}, {"blocking_timeout_secs": 10}, "LMCACHE_BLOCKING_TIMEOUT_SECS=10"),
        ({}, {"extra_config": None}, "JSON object"),
        ({}, {"extra_config": _valid_extra(save_chunk_meta=None)}, "save_chunk_meta"),
        ({}, {"extra_config": _valid_extra(save_chunk_meta="false")}, "JSON false"),
        ({}, {"extra_config": _valid_extra(transfer_timeout=None)}, "timeout=1"),
        ({}, {"extra_config": _valid_extra(transfer_timeout="x")}, "a number"),
        (
            {},
            {"extra_config": _valid_extra(mooncake_global_segment_size=None)},
            "global_segment_size=None",
        ),
        (
            {},
            {"extra_config": _valid_extra(mooncake_global_segment_size="4 GB")},
            "'4 GB'",
        ),
        ({}, {"extra_config": _valid_extra(mooncake_protocol=None)}, "tcp"),
    ],
)
def test_validate_refuses_settings_that_fail_silently(
    store_env, monkeypatch, env, cfg_overrides, message
):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        l2.validate_mooncake_store_config(_cfg(**cfg_overrides))


@pytest.mark.parametrize("key", sorted(l2._IGNORED_EXTRA_CONFIG_KEYS))
def test_validate_refuses_keys_mooncake_never_sees(store_env, key):
    with pytest.raises(ValueError, match=key):
        l2.validate_mooncake_store_config(_cfg(extra_config=_valid_extra(**{key: "v"})))


def test_prepare_pins_one_nic_and_installs_every_patch(store_env, monkeypatch):
    calls = []
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 2))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        l2, "requester_rdma_device", lambda index: calls.append(index) or "rdma2"
    )
    monkeypatch.setattr(l2, "install_thp_pinned_allocator", lambda: calls.append("thp"))
    monkeypatch.setattr(
        l2, "install_batch_is_exist_lookup", lambda: calls.append("lookup")
    )
    monkeypatch.setattr(
        l2, "install_non_blocking_l2_get_allocation", lambda: calls.append("get")
    )
    monkeypatch.setattr(l2, "require_thp_allowed", lambda: calls.append("prctl"))
    cfg = _cfg()
    l2.prepare_mooncake_store_l2(cfg)
    assert cfg.extra_config["mooncake_rdma_devices"] == "rdma2"
    assert calls == ["prctl", 2, "thp", "lookup", "get"]

    calls.clear()
    tcp = _cfg(extra_config=_valid_extra(mooncake_protocol="tcp"))
    l2.prepare_mooncake_store_l2(tcp)
    assert "mooncake_rdma_devices" not in tcp.extra_config
    assert calls == ["prctl", "thp", "lookup", "get"]


POOLS = {
    "rdma2": {"master": "10.0.0.1:50251", "metadata": "http://10.0.0.1:50280/metadata"},
    "rdma3": {"master": "10.0.0.1:50351", "metadata": "http://10.0.0.1:50380/metadata"},
}


def _prepare_on(monkeypatch, device):
    """prepare_mooncake_store_l2 for a worker whose NIC is ``device``."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0)),
    )
    monkeypatch.setattr(l2, "requester_rdma_device", lambda index: device)
    for patch in (
        "install_thp_pinned_allocator",
        "install_batch_is_exist_lookup",
        "install_non_blocking_l2_get_allocation",
        "require_thp_allowed",
    ):
        monkeypatch.setattr(l2, patch, lambda: None)
    cfg = _cfg()
    l2.prepare_mooncake_store_l2(cfg)
    return cfg


def test_parse_store_pools():
    assert l2.parse_store_pools(" ") == {}
    assert l2.parse_store_pools(json.dumps(POOLS)) == {
        "rdma2": l2.StorePool("10.0.0.1:50251", "http://10.0.0.1:50280/metadata"),
        "rdma3": l2.StorePool("10.0.0.1:50351", "http://10.0.0.1:50380/metadata"),
    }
    for bad, match in (
        ("rdma0=10.0.0.1:50051", "not JSON"),
        ("{}", "non-empty JSON object"),
        ('["rdma0"]', "non-empty JSON object"),
        ('{"rdma0": {"metadata": "http://h/metadata"}}', "'rdma0'"),
        ('{"rdma0": "10.0.0.1:50051"}', "'rdma0'"),
    ):
        with pytest.raises(ValueError, match=match):
            l2.parse_store_pools(bad)


def test_prepare_points_each_rank_at_the_pool_of_its_nic(store_env, monkeypatch):
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_POOLS", json.dumps(POOLS))
    cfg = _prepare_on(monkeypatch, "rdma3")
    assert cfg.extra_config["mooncake_rdma_devices"] == "rdma3"
    assert cfg.extra_config["mooncake_master_server_addr"] == "10.0.0.1:50351"
    assert (
        cfg.extra_config["mooncake_metadata_server"] == "http://10.0.0.1:50380/metadata"
    )
    assert cfg.remote_url == "mooncakestore://10.0.0.1:50351/"
    with pytest.raises(ValueError, match="no pool for RDMA device 'rdma0'"):
        _prepare_on(monkeypatch, "rdma0")


def test_prepare_without_pools_keeps_the_configured_master(store_env, monkeypatch):
    cfg = _prepare_on(monkeypatch, "rdma3")
    assert cfg.extra_config["mooncake_master_server_addr"] == "10.0.0.1:50051"
    assert cfg.remote_url == "mooncakestore://10.0.0.1:50051/"


def test_store_pool_of_needs_an_rdma_device(store_env, monkeypatch):
    assert l2.store_pool_of(None) is None
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_POOLS", json.dumps(POOLS))
    with pytest.raises(ValueError, match="tcp"):
        l2.store_pool_of(None)


def test_prepare_validates_before_it_patches(store_env, monkeypatch):
    monkeypatch.setenv("MC_NUM_QP_PER_EP", "2")
    monkeypatch.setattr(l2, "install_thp_pinned_allocator", pytest.fail)
    with pytest.raises(ValueError, match="MC_NUM_QP_PER_EP"):
        l2.prepare_mooncake_store_l2(_cfg())


# --- NIC per rank ------------------------------------------------------------


def test_rail_rdma_device_follows_the_pci_tree_not_the_gpu_ordinal(tmp_path):
    pci, ib = _make_node(tmp_path)
    picked = [
        l2.rail_rdma_device(f"0000:{bus}:00.0", ib_root=ib, pci_root=pci)
        for bus in GPU_BDF
    ]
    # HIP GPU 0 is bus 0x75, whose switch holds rdma3, not rdma0.
    assert picked == [f"rdma{i}" for i in (3, 0, 2, 1, 7, 4, 6, 5)]


def test_rail_rdma_device_skips_inactive_nics_and_refuses_guesses(tmp_path):
    pci, ib = _make_node(tmp_path, inactive={"rdma3"})
    # GPU 0's own NIC is down; no other ACTIVE NIC shares its host bridge.
    with pytest.raises(ValueError, match="cannot tell GPU 0000:75:00.0's NIC"):
        l2.rail_rdma_device("0000:75:00.0", ib_root=ib, pci_root=pci)
    with pytest.raises(ValueError, match="not in"):
        l2.rail_rdma_device("0000:aa:00.0", ib_root=ib, pci_root=pci)


def test_rail_rdma_device_refuses_two_equally_close_nics(tmp_path):
    pci, ib = _make_node(tmp_path)
    twin = ib / "rdma8"
    (twin / "ports/1").mkdir(parents=True)
    (twin / "ports/1/state").write_text("4: ACTIVE\n")
    (twin / "device").symlink_to((ib / "rdma3/device").resolve())
    with pytest.raises(ValueError, match=r"rdma3:\d+, rdma4:\d+.*rdma8:\d+"):
        l2.rail_rdma_device("0000:75:00.0", ib_root=ib, pci_root=pci)


@pytest.fixture
def node(tmp_path, monkeypatch, store_env):
    pci, ib = _make_node(tmp_path)
    monkeypatch.setattr(l2, "_IB_SYSFS_ROOT", ib)
    monkeypatch.setattr(l2, "_PCI_SYSFS_ROOT", pci)
    monkeypatch.setattr(l2, "gpu_pci_bdf", lambda index: f"0000:{GPU_BDF[index]}:00.0")
    # Keyword defaults were bound at import; rebind to the fake tree.
    real = l2.rail_rdma_device
    monkeypatch.setattr(
        l2,
        "rail_rdma_device",
        lambda bdf: real(bdf, ib_root=ib, pci_root=pci),
    )
    return ib


def test_requester_device_defaults_to_the_topology(node):
    assert [l2.requester_rdma_device(i) for i in range(4)] == [
        "rdma3",
        "rdma0",
        "rdma2",
        "rdma1",
    ]


def test_requester_device_override_is_indexed_by_gpu_ordinal(node, monkeypatch):
    table = "rdma0,rdma1,rdma2,rdma3,rdma0,rdma1,rdma2,rdma3"
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES", table)
    assert [l2.requester_rdma_device(i) for i in range(8)] == table.split(",")
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES", " rdma5 ")
    assert l2.requester_rdma_device(3) == "rdma5"
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES", "rdma0,rdma1")
    with pytest.raises(ValueError, match="has none for GPU 2"):
        l2.requester_rdma_device(2)
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_RDMA_DEVICES", "mlx5_9")
    with pytest.raises(ValueError, match="'mlx5_9'.*does not exist"):
        l2.requester_rdma_device(0)


def test_requester_device_never_shares_an_owner_nic(node, monkeypatch):
    monkeypatch.setenv(
        "ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES", "rdma4,rdma5,rdma6,rdma7"
    )
    assert l2.requester_rdma_device(0) == "rdma3"
    with pytest.raises(ValueError, match="'rdma7'.*Store owner"):
        l2.requester_rdma_device(4)


def test_requester_device_shares_the_owner_nic_of_its_own_pool(node, monkeypatch):
    monkeypatch.setenv(
        "ATOM_LMCACHE_MOONCAKE_OWNER_RDMA_DEVICES", "rdma0,rdma1,rdma2,rdma3"
    )
    monkeypatch.setenv("ATOM_LMCACHE_MOONCAKE_POOLS", json.dumps(POOLS))
    assert l2.requester_rdma_device(0) == "rdma3"


# --- THP pinned L1 -----------------------------------------------------------


def test_anon_huge_page_bytes_sums_the_vmas_inside_the_range(tmp_path):
    smaps = tmp_path / "smaps"
    smaps.write_text(
        "1000-2000 rw-p 00000000 00:00 0\n"
        "AnonHugePages:      999 kB\n"
        "200000-400000 rw-p 00000000 00:00 0\n"
        "Size:               2048 kB\n"
        "AnonHugePages:     2048 kB\n"
        "400000-800000 rw-p 00000000 00:00 0\n"
        "AnonHugePages:     2048 kB\n"
        "VmFlags: rd wr mr mw me ac hg\n"
        "800000-a00000 rw-p 00000000 00:00 0\n"
        "AnonHugePages:     2048 kB\n"
    )
    assert l2.anon_huge_page_bytes(0x200000, 0x800000, str(smaps)) == 4096 * 1024
    with pytest.raises(RuntimeError, match="straddles"):
        l2.anon_huge_page_bytes(0x300000, 0x800000, str(smaps))


def test_anon_huge_page_bytes_reads_mapping_names_that_are_not_utf8(tmp_path):
    smaps = tmp_path / "smaps"
    smaps.write_bytes(
        b"1000-2000 r--p 00000000 08:01 7 /tmp/caf\xe9.bin\n"
        b"AnonHugePages:        0 kB\n"
        b"200000-400000 rw-p 00000000 00:00 0\n"
        b"AnonHugePages:     2048 kB\n"
    )
    assert l2.anon_huge_page_bytes(0x200000, 0x400000, str(smaps)) == 2048 * 1024


def test_numa_node_meminfo_reads_kb_fields_as_bytes(tmp_path):
    (tmp_path / "node1").mkdir()
    (tmp_path / "node1/meminfo").write_text(
        "Node 1 MemTotal:       1585311824 kB\n"
        "Node 1 MemFree:         824911496 kB\n"
        "Node 1 FilePages:       745210764 kB\n"
        "Node 1 HugePages_Total:     0\n"
    )
    info = l2.numa_node_meminfo(1, root=tmp_path)
    assert info["MemFree"] == 824911496 * 1024
    assert info["FilePages"] == 745210764 * 1024
    assert info["HugePages_Total"] == 0
    assert l2.numa_node_meminfo(0, root=tmp_path) == {}


def test_current_gpu_numa_node_reads_the_gpus_pci_node(tmp_path, monkeypatch):
    available = [True]
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: available[0], current_device=lambda: 1
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(l2, "_PCI_SYSFS_ROOT", tmp_path)
    monkeypatch.setattr(l2, "gpu_pci_bdf", lambda index: f"0000:{GPU_BDF[index]}:00.0")
    gpu = tmp_path / "0000:05:00.0"
    gpu.mkdir()
    (gpu / "numa_node").write_text("1\n")
    assert l2._current_gpu_numa_node() == 1
    (gpu / "numa_node").write_text("-1\n")  # a node without NUMA information
    assert l2._current_gpu_numa_node() is None
    (gpu / "numa_node").unlink()
    assert l2._current_gpu_numa_node() is None
    available[0] = False
    assert l2._current_gpu_numa_node() is None


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """Real mmap/madvise/munmap; mbind, pinning, THP census and THP sysfs faked."""
    events = []
    monkeypatch.setattr(l2, "_thp_regions", {})
    monkeypatch.setattr(l2, "_reserved_l1", {})
    monkeypatch.setattr(l2, "require_thp_allowed", lambda: None)
    monkeypatch.setattr(l2, "_THP_SYSFS_ROOT", tmp_path / "thp")
    monkeypatch.setattr(l2, "_current_gpu_numa_node", lambda: None)
    monkeypatch.setattr(l2, "numa_node_meminfo", lambda node: {})
    monkeypatch.setattr(l2, "_mbind", lambda a, n, node: events.append(("mbind", node)))
    monkeypatch.setattr(
        l2, "_host_register", lambda a, n: events.append(("register", a, n))
    )
    monkeypatch.setattr(l2, "_host_unregister", lambda a: events.append(("unreg", a)))
    monkeypatch.setattr(l2, "anon_huge_page_bytes", lambda start, end: end - start)
    monkeypatch.setattr(
        l2, "_collapse_into_huge_pages", lambda a, n: events.append(("collapse", n))
    )
    return events


def test_alloc_rounds_to_huge_pages_aligns_and_frees(pinned):
    size = 3 * 1024 * 1024 + 5
    ptr = l2.alloc_thp_pinned_numa_ptr(size, 1)
    region = 4 * 1024 * 1024
    assert ptr % l2.HUGE_PAGE_BYTES == 0
    assert pinned == [("mbind", 1), ("register", ptr, region)]
    assert l2._thp_regions[ptr].node == 1
    # Each huge page was touched: one write per 2 MiB faults it in.
    for offset in range(0, region, l2.HUGE_PAGE_BYTES):
        assert ctypes.string_at(ptr + offset, 1) == b"\x01"
    ctypes.memset(ptr + region - 1, 7, 1)
    l2.free_thp_pinned_numa_ptr(ptr, size)
    assert pinned[-1] == ("unreg", ptr)
    assert l2._thp_regions == {}


def test_alloc_takes_the_region_reserved_before_the_weights_load(pinned):
    mib = 1024 * 1024
    l2.reserve_thp_l1(6 * mib, 1)
    address, region = l2._reserved_l1[1]
    assert region.region_bytes == 6 * mib
    assert pinned == [("mbind", 1)]  # faulted, not pinned yet
    # LMCache may ask for a little less (alignment): the reservation serves.
    ptr = l2.alloc_thp_pinned_numa_ptr(5 * mib + 1, 1)
    assert ptr == address
    assert pinned == [("mbind", 1), ("register", ptr, 6 * mib)]
    assert l2._reserved_l1 == {}
    l2.free_thp_pinned_numa_ptr(ptr)
    assert l2._thp_regions == {}


def test_alloc_beyond_the_reservation_frees_it_and_maps_anew(pinned, caplog):
    mib = 1024 * 1024
    l2.reserve_thp_l1(2 * mib, 1)
    with caplog.at_level(logging.WARNING, logger="atom"):
        ptr = l2.alloc_thp_pinned_numa_ptr(4 * mib, 1)
    assert "smaller than the" in caplog.text
    assert l2._thp_regions[ptr].region_bytes == 4 * mib
    assert pinned == [("mbind", 1), ("mbind", 1), ("register", ptr, 4 * mib)]
    l2.free_thp_pinned_numa_ptr(ptr)


def test_release_frees_a_reservation_no_pool_took(pinned, caplog):
    l2.reserve_thp_l1(2 * 1024 * 1024, 0)
    l2.reserve_thp_l1(4 * 1024 * 1024, 0)  # replaces the first
    assert l2._reserved_l1[0][1].region_bytes == 4 * 1024 * 1024
    with caplog.at_level(logging.WARNING, logger="atom"):
        l2.release_reserved_l1()
    assert l2._reserved_l1 == {}
    assert "no pool took the 0.00 GiB reserved on NUMA node 0" in caplog.text


def test_alloc_refuses_a_node_other_than_the_gpus(pinned, monkeypatch):
    monkeypatch.setattr(l2, "_current_gpu_numa_node", lambda: 0)
    with pytest.raises(ValueError, match="on NUMA node 1, but this worker's GPU is"):
        l2.alloc_thp_pinned_numa_ptr(4 * 1024 * 1024, 1)
    assert pinned == []  # refused before anything was mapped
    l2.free_thp_pinned_numa_ptr(l2.alloc_thp_pinned_numa_ptr(4 * 1024 * 1024, 0))


def test_alloc_on_a_short_node_warns_and_reports_progress(pinned, monkeypatch, caplog):
    monkeypatch.setattr(
        l2,
        "numa_node_meminfo",
        lambda node: {"MemFree": 2 * 1024 * 1024, "FilePages": 700 * 2**30},
    )
    monkeypatch.setattr(l2, "_TOUCH_SLICE_BYTES", l2.HUGE_PAGE_BYTES)
    monkeypatch.setattr(l2, "_TOUCH_PROGRESS_INTERVAL_S", 0.0)
    with caplog.at_level(logging.INFO, logger="atom"):
        ptr = l2.alloc_thp_pinned_numa_ptr(6 * 1024 * 1024, 0)
    l2.free_thp_pinned_numa_ptr(ptr)
    (warning,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "has 0.0 GiB free for a 0.0 GiB pool and 700.0 GiB of page cache" in (
        warning.getMessage()
    )
    progress = [r for r in caplog.records if "faulted in" in r.getMessage()]
    assert len(progress) == 3


def test_alloc_fails_and_unwinds_when_not_all_thp(pinned, monkeypatch):
    monkeypatch.setattr(
        l2, "anon_huge_page_bytes", lambda start, end: end - start - 4096
    )
    with pytest.raises(RuntimeError, match="has 4096 bytes outside transparent huge"):
        l2.alloc_thp_pinned_numa_ptr(8 * 1024 * 1024, 0)
    assert [event[0] for event in pinned] == ["mbind", "collapse", "register", "unreg"]
    assert l2._thp_regions == {}


def test_alloc_collapses_4k_stretches_before_pinning(pinned, monkeypatch, caplog):
    census = iter([-6 * 1024 * 1024, 0])
    monkeypatch.setattr(
        l2, "anon_huge_page_bytes", lambda start, end: end - start + next(census)
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        ptr = l2.alloc_thp_pinned_numa_ptr(8 * 1024 * 1024, 0)
    region = 8 * 1024 * 1024
    # Pinned pages cannot move, so the collapse comes before hipHostRegister.
    assert pinned == [("mbind", 0), ("collapse", region), ("register", ptr, region)]
    assert any("6291456 bytes" in r.getMessage() for r in caplog.records)
    l2.free_thp_pinned_numa_ptr(ptr)


def test_alloc_collapses_a_reservation_the_weight_load_split(pinned, monkeypatch):
    region = 6 * 1024 * 1024
    # Huge pages twice while reserving (after the touch, then for the log),
    # 4 KiB stretches when the allocation counts again, huge once pinned.
    census = iter([0, 0, -2 * 1024 * 1024, 0])
    monkeypatch.setattr(
        l2, "anon_huge_page_bytes", lambda start, end: end - start + next(census)
    )
    l2.reserve_thp_l1(region, 1)
    address, _ = l2._reserved_l1[1]
    ptr = l2.alloc_thp_pinned_numa_ptr(region, 1)
    assert ptr == address
    # Pinned pages cannot move, so the collapse comes before hipHostRegister.
    assert pinned == [("mbind", 1), ("collapse", region), ("register", ptr, region)]
    assert l2._reserved_l1 == {}
    l2.free_thp_pinned_numa_ptr(ptr)


@pytest.mark.parametrize("reserved", [False, True])
def test_alloc_unmaps_a_region_it_could_not_pin(pinned, monkeypatch, reserved):
    mib = 1024 * 1024
    real = l2._libc()
    unmapped = []

    class Libc:
        def __getattr__(self, name):
            return getattr(real, name)

        def munmap(self, address, length):
            unmapped.append(length)
            return real.munmap(address, length)

    def refuse(address, length):
        raise RuntimeError(f"hipHostRegister of {length} bytes failed: 1")

    monkeypatch.setattr(l2, "_libc", Libc)
    monkeypatch.setattr(l2, "_host_register", refuse)
    if reserved:
        l2.reserve_thp_l1(4 * mib, 0)
    with pytest.raises(RuntimeError, match="hipHostRegister of 4194304 bytes"):
        l2.alloc_thp_pinned_numa_ptr(4 * mib, 0)
    # The whole mapping (the region and its alignment page), and nothing to
    # unregister.
    assert unmapped == [6 * mib]
    assert [event[0] for event in pinned] == ["mbind"]
    assert l2._thp_regions == {}
    assert l2._reserved_l1 == {}


def test_mbind_builds_the_node_mask_the_kernel_reads(monkeypatch):
    calls = []

    def syscall(number, address, length, mode, mask, maxnode, flags):
        calls.append((number.value, list(mask), maxnode.value, mode.value, flags.value))
        return 0

    monkeypatch.setattr(l2.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(l2, "_libc", lambda: SimpleNamespace(syscall=syscall))
    for node in (0, 1, 64):
        l2._mbind(0x200000, 2 * 1024 * 1024, node)
    # One bit per node, and the kernel reads maxnode - 1 bits.
    assert [call[:3] for call in calls] == [
        (237, [1], 65),
        (237, [2], 65),
        (237, [0, 1], 129),
    ]
    assert {call[3:] for call in calls} == {(2, 3)}  # MPOL_BIND, STRICT | MOVE
    monkeypatch.setattr(l2, "_libc", lambda: SimpleNamespace(syscall=lambda *a: -1))
    with pytest.raises(OSError, match=r"mbind\(MPOL_BIND, node 1\) failed"):
        l2._mbind(0x200000, 4096, 1)


def test_alloc_and_reserve_refuse_a_gpu_without_a_numa_node(pinned):
    # LMCache's auto mode passes the GPU's sysfs numa_node as it reads it.
    with pytest.raises(ValueError, match="sysfs numa_node -1"):
        l2.alloc_thp_pinned_numa_ptr(4 * 1024 * 1024, -1)
    with pytest.raises(ValueError, match="LMCACHE_NUMA_MODE=manual"):
        l2.reserve_thp_l1(4 * 1024 * 1024, -1)
    assert pinned == []  # refused before anything was mapped
    assert l2._reserved_l1 == {}


def test_reserve_warns_when_the_kernel_may_split_the_huge_pages(
    pinned, tmp_path, caplog
):
    # 6.12+: shrink_underused splits a THP with more zero-filled 4 KiB pages
    # than max_ptes_none, and the first touch fills one of 512.
    thp = tmp_path / "thp"
    (thp / "khugepaged").mkdir(parents=True)
    for shrink_underused, max_ptes_none, warns in (
        ("1", "511", False),
        ("0", "255", False),
        ("1", "255", True),
    ):
        (thp / "shrink_underused").write_text(f"{shrink_underused}\n")
        (thp / "khugepaged/max_ptes_none").write_text(f"{max_ptes_none}\n")
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="atom"):
            l2.reserve_thp_l1(2 * 1024 * 1024, 0)
        assert ("may be split before they are pinned" in caplog.text) is warns
    assert "max_ptes_none=255 is below 511" in caplog.text
    l2.release_reserved_l1()


def _madvise_failing(errors, calls):
    """A madvise that fails with each of ``errors`` in turn, then succeeds."""
    errors = iter(errors)

    def madvise(address, length, advice):
        calls.append((address, length, advice))
        error = next(errors, 0)
        ctypes.set_errno(error)
        return -1 if error else 0

    return madvise


def test_collapse_into_huge_pages_retries_when_no_huge_page_is_free(
    monkeypatch, caplog
):
    calls, sleeps = [], []
    madvise = _madvise_failing([errno.ENOMEM, errno.EAGAIN], calls)
    monkeypatch.setattr(l2, "_libc", lambda: SimpleNamespace(madvise=madvise))
    monkeypatch.setattr(l2.time, "sleep", sleeps.append)
    with caplog.at_level(logging.INFO, logger="atom"):
        assert l2._collapse_into_huge_pages(4096, 2 * 1024 * 1024) is True
    assert calls == [(4096, 2 * 1024 * 1024, 25)] * 3
    assert sleeps == [1.0, 2.0]
    assert "succeeded on attempt 3" in caplog.records[-1].getMessage()


def test_collapse_into_huge_pages_gives_up(monkeypatch, caplog):
    calls, sleeps = [], []
    monkeypatch.setattr(l2.time, "sleep", sleeps.append)
    # Out of huge pages every time: one attempt per delay, then the last one.
    madvise = _madvise_failing([errno.ENOMEM] * 10, calls)
    monkeypatch.setattr(l2, "_libc", lambda: SimpleNamespace(madvise=madvise))
    with caplog.at_level(logging.WARNING, logger="atom"):
        assert l2._collapse_into_huge_pages(4096, 2 * 1024 * 1024) is False
    assert len(calls) == len(l2._COLLAPSE_RETRY_DELAYS_S) + 1
    assert sleeps == list(l2._COLLAPSE_RETRY_DELAYS_S)
    assert "MADV_COLLAPSE" in caplog.records[-1].getMessage()
    # A refusal that waiting cannot fix (an old kernel's EINVAL) is not retried.
    calls.clear()
    madvise = _madvise_failing([errno.EINVAL], calls)
    monkeypatch.setattr(l2, "_libc", lambda: SimpleNamespace(madvise=madvise))
    assert l2._collapse_into_huge_pages(4096, 2 * 1024 * 1024) is False
    assert len(calls) == 1


def test_alloc_refuses_when_thp_is_disabled_for_the_process(monkeypatch):
    fake_libc = SimpleNamespace(prctl=lambda *args: 1)
    monkeypatch.setattr(l2, "_libc", lambda: fake_libc)
    with pytest.raises(RuntimeError, match="PR_SET_THP_DISABLE"):
        l2.alloc_thp_pinned_numa_ptr(4096, 0)
    with pytest.raises(ValueError, match="positive"):
        l2.alloc_thp_pinned_numa_ptr(0, 0)


def test_alloc_on_this_kernel(monkeypatch):
    """mbind, madvise and the smaps census for real, when the host allows it."""
    thp = Path("/sys/kernel/mm/transparent_hugepage/enabled")
    if not thp.exists() or "[never]" in thp.read_text():
        pytest.skip("transparent huge pages are off on this host")
    monkeypatch.setattr(l2, "_thp_regions", {})
    monkeypatch.setattr(l2, "_current_gpu_numa_node", lambda: None)
    monkeypatch.setattr(l2, "_host_register", lambda a, n: None)
    monkeypatch.setattr(l2, "_host_unregister", lambda a: None)
    try:
        ptr = l2.alloc_thp_pinned_numa_ptr(4 * 1024 * 1024, 0)
    except OSError as exc:
        pytest.skip(f"cannot mbind here ({exc}); Docker needs --cap-add SYS_NICE")
    except RuntimeError as exc:
        pytest.skip(f"no free huge pages on node 0 right now: {exc}")
    try:
        assert l2.anon_huge_page_bytes(ptr, ptr + 4 * 1024 * 1024) == 4 * 1024 * 1024
    finally:
        l2.free_thp_pinned_numa_ptr(ptr)


def _fake_lmcache(monkeypatch, device_ops):
    module = types.ModuleType("lmcache")
    module.__path__ = []
    module.device_ops = device_ops
    monkeypatch.setitem(sys.modules, "lmcache", module)


def test_install_thp_allocator_patches_device_ops_once(monkeypatch):
    freed = []
    device_ops = SimpleNamespace(
        alloc_pinned_numa_ptr=lambda size, numa_id=0: 1,
        free_pinned_numa_ptr=lambda ptr, size=None: freed.append(ptr),
    )
    native_free = device_ops.free_pinned_numa_ptr
    _fake_lmcache(monkeypatch, device_ops)
    monkeypatch.setattr(l2, "_native_free_pinned_numa_ptr", None)
    monkeypatch.setattr(l2, "_thp_regions", {})
    l2.install_thp_pinned_allocator()
    l2.install_thp_pinned_allocator()
    assert device_ops.alloc_pinned_numa_ptr is l2.alloc_thp_pinned_numa_ptr
    assert device_ops.free_pinned_numa_ptr is l2.free_thp_pinned_numa_ptr
    assert l2._native_free_pinned_numa_ptr is native_free
    # A pool LMCache allocated before the patch goes back to its own free.
    device_ops.free_pinned_numa_ptr(0xABC000, 4096)
    assert freed == [0xABC000]


def _engine_with_l1(start, nbytes, allocator=None):
    buffer = SimpleNamespace(
        data_ptr=lambda: start, numel=lambda: nbytes, element_size=lambda: 1
    )
    if allocator is None:
        allocator = SimpleNamespace(pin_allocator=SimpleNamespace(buffer=buffer))
    local = SimpleNamespace(memory_allocator=allocator)
    return SimpleNamespace(
        storage_manager=SimpleNamespace(storage_backends={"LocalCPUBackend": local})
    )


def test_verify_thp_l1_accepts_only_a_region_of_the_thp_allocator(monkeypatch, caplog):
    region = 8 * l2.HUGE_PAGE_BYTES
    monkeypatch.setattr(
        l2,
        "_thp_regions",
        {0x40000000: l2._ThpRegion(0x3FF00000, region + l2.HUGE_PAGE_BYTES, region, 0)},
    )
    with caplog.at_level(logging.INFO, logger="atom"):
        l2.verify_thp_l1(_engine_with_l1(0x40000000, region))
    assert "bound to NUMA node 0" in caplog.text
    # LMCache's hipHostMalloc fallback (no NUMA mapping), or a pool that
    # outgrows the region, is not known to be huge pages on the GPU's node.
    for start, nbytes in ((0x80000000, region), (0x40000000, region + 1)):
        with pytest.raises(RuntimeError, match="did not come from ATOM's huge-page"):
            l2.verify_thp_l1(_engine_with_l1(start, nbytes))


def test_verify_thp_l1_needs_a_pinned_pool():
    paged = SimpleNamespace(cpu_buffer=object())  # e.g. LMCache's P2P allocator
    with pytest.raises(RuntimeError, match="no pinned CPU pool"):
        l2.verify_thp_l1(_engine_with_l1(0, 0, allocator=paged))
    with pytest.raises(RuntimeError, match="no pinned CPU pool"):
        l2.verify_thp_l1(SimpleNamespace())


# --- batched lookup ----------------------------------------------------------


class _Key:
    def __init__(self, name):
        self.name = name

    def to_string(self):
        return self.name


def test_count_stored_prefix_stops_at_the_first_absent_or_failed_key():
    asked = []

    class Store:
        def __init__(self, answer):
            self.answer = answer

        def batch_is_exist(self, names):
            asked.append(list(names))
            return self.answer

    keys = [_Key(f"k{i}") for i in range(4)]
    assert l2.count_stored_prefix(Store([1, 1, 1, 1]), keys) == 4
    assert l2.count_stored_prefix(Store([1, 0, 1, 1]), keys) == 1
    assert l2.count_stored_prefix(Store([1, 1, -704, 1]), keys) == 2
    assert asked[0] == ["k0", "k1", "k2", "k3"]
    assert l2.count_stored_prefix(Store([]), []) == 0
    assert len(asked) == 3


def _fake_connector_modules(monkeypatch, upstream_batched=False):
    class RemoteConnector:
        def support_batched_contains(self):
            return False

        def batched_contains(self, keys):
            raise NotImplementedError

    class MooncakestoreConnector(RemoteConnector):
        """The f22dec28 zero-copy get: one CPU-pool buffer per key, then a read."""

        def __init__(self, store, local_cpu_backend=None):
            self.store = store
            self.local_cpu_backend = local_cpu_backend

        async def batched_get(self, keys):
            memory_objs = [
                self.local_cpu_backend.allocate("shapes", "dtypes", "fmt") for _ in keys
            ]
            await asyncio.sleep(0)  # batch_get_into runs in a worker thread
            return memory_objs

    if upstream_batched:
        MooncakestoreConnector.support_batched_contains = lambda self: True
    base = types.ModuleType("lmcache.v1.storage_backend.connector.base_connector")
    base.RemoteConnector = RemoteConnector
    mooncake = types.ModuleType(
        "lmcache.v1.storage_backend.connector.mooncakestore_connector"
    )
    mooncake.MooncakestoreConnector = MooncakestoreConnector
    names = (
        "lmcache",
        "lmcache.v1",
        "lmcache.v1.storage_backend",
        "lmcache.v1.storage_backend.connector",
    )
    for name in names:
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, base.__name__, base)
    monkeypatch.setitem(sys.modules, mooncake.__name__, mooncake)
    return MooncakestoreConnector


def test_install_batch_is_exist_lookup(monkeypatch):
    connector_cls = _fake_connector_modules(monkeypatch)
    l2.install_batch_is_exist_lookup()
    patched = connector_cls.batched_contains
    l2.install_batch_is_exist_lookup()
    assert connector_cls.batched_contains is patched
    store = SimpleNamespace(batch_is_exist=lambda names: [1] * len(names))
    connector = connector_cls(store)
    assert connector.support_batched_contains()
    assert connector.batched_contains([_Key("a"), _Key("b")]) == 2


def test_install_batch_is_exist_lookup_keeps_an_upstream_implementation(monkeypatch):
    connector_cls = _fake_connector_modules(monkeypatch, upstream_batched=True)
    upstream = connector_cls.support_batched_contains
    l2.install_batch_is_exist_lookup()
    assert connector_cls.support_batched_contains is upstream
    assert "batched_contains" not in vars(connector_cls)


def test_parse_device_list():
    assert l2.parse_device_list(" rdma0, ,rdma1,") == ["rdma0", "rdma1"]
    assert l2.parse_device_list("") == []


# --- L2 gets without waiting for L1 room -------------------------------------


class _CpuPool:
    """LMCache's LocalCPUBackend.allocate: busy-waits for room unless told not to."""

    def __init__(self, free, give_up_after_s=2.0):
        self.free = free
        self.calls = []
        self.memory_allocator = object()
        self._give_up_after_s = give_up_after_s

    def allocate(self, shapes, dtypes, fmt=None, eviction=True, busy_loop=True):
        self.calls.append((eviction, busy_loop))
        deadline = time.monotonic() + self._give_up_after_s
        while self.free == 0:
            # LMCache sleeps without yielding the event loop it runs on; the
            # deadline only keeps a regression from hanging the test session.
            if not busy_loop or time.monotonic() > deadline:
                return None
            time.sleep(0.01)
        self.free -= 1
        return object()


def test_allocate_without_waiting_ends_the_batch_at_the_first_failure():
    pool = _CpuPool(free=2)
    view = l2._AllocateWithoutWaiting(pool)
    view.start_batch()
    allocated = [view.allocate("s", "d", "f") for _ in range(4)]
    assert [obj is not None for obj in allocated] == [True, True, False, False]
    # Eviction stays on; the third allocation found no room and ended the batch.
    assert pool.calls == [(True, False)] * 3
    pool.free = 1
    view.start_batch()
    assert view.allocate("s", "d", "f", busy_loop=True) is not None
    assert pool.calls[-1] == (True, False)
    assert view.memory_allocator is pool.memory_allocator


def test_install_non_blocking_l2_get_allocation(monkeypatch):
    connector_cls = _fake_connector_modules(monkeypatch)
    l2.install_non_blocking_l2_get_allocation()
    patched = connector_cls.batched_get
    l2.install_non_blocking_l2_get_allocation()
    assert connector_cls.batched_get is patched

    pool = _CpuPool(free=2)
    connector = connector_cls(store=None, local_cpu_backend=pool)
    got = asyncio.run(connector.batched_get([_Key(f"k{i}") for i in range(4)]))
    assert [obj is not None for obj in got] == [True, True, False, False]
    assert pool.calls == [(True, False)] * 3
    view = connector.local_cpu_backend
    assert isinstance(view, l2._AllocateWithoutWaiting)

    # The next get is a new batch, and the connector keeps one view.
    pool.free = 1
    got = asyncio.run(connector.batched_get([_Key("k0")]))
    assert got[0] is not None
    assert connector.local_cpu_backend is view


def test_l2_get_on_a_full_l1_leaves_the_loop_to_the_put_that_frees_it(monkeypatch):
    """The pit2-p03-g40 hang: the get waited on the loop the freeing put needs."""
    connector_cls = _fake_connector_modules(monkeypatch)
    l2.install_non_blocking_l2_get_allocation()
    pool = _CpuPool(free=0, give_up_after_s=10.0)
    connector = connector_cls(store=None, local_cpu_backend=pool)

    async def finish_put():
        pool.free += 1  # the put's on_complete drops its L1 reference

    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        get = asyncio.run_coroutine_threadsafe(
            connector.batched_get([_Key("k0")]), loop
        )
        put = asyncio.run_coroutine_threadsafe(finish_put(), loop)
        started = time.monotonic()
        assert get.result(timeout=5) == [None]
        put.result(timeout=5)
        assert time.monotonic() - started < 1.0
        assert pool.free == 1
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


# --- wiring into the engine build ------------------------------------------


@pytest.mark.parametrize(
    "remote_url, failing_check, expected",
    [
        (
            "mooncakestore://10.0.0.1:50051/",
            None,
            ["prepare", "engine", "post", "thp", "release", "verify"],
        ),
        ("lm://10.0.0.1:65432", None, ["engine", "post", "verify"]),
        (None, None, ["engine", "post"]),
        # A failed check tears the engine down so the worker can exit.
        ("lm://10.0.0.1:65432", "verify", ["engine", "post", "verify", "destroy e"]),
        (
            "mooncakestore://10.0.0.1:50051/",
            "thp",
            ["prepare", "engine", "post", "thp", "destroy e"],
        ),
    ],
)
def test_build_offload_engine_prepares_before_and_verifies_after(
    monkeypatch, remote_url, failing_check, expected
):
    from atom.kv_transfer.offload import _offload_common as common

    events = []

    class Engine:
        def post_init(self):
            events.append("post")

    class Builder:
        @staticmethod
        def get_or_create(*args):
            events.append("engine")
            return Engine()

        @staticmethod
        def destroy(engine_id):
            events.append(f"destroy {engine_id}")

    cache_engine = types.ModuleType("lmcache.v1.cache_engine")
    cache_engine.LMCacheEngineBuilder = Builder
    memory = types.ModuleType("lmcache.v1.memory_management")
    memory.MemoryFormat = SimpleNamespace(KV_2LTD="KV_2LTD")
    for name in ("lmcache", "lmcache.v1"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, cache_engine.__name__, cache_engine)
    monkeypatch.setitem(sys.modules, memory.__name__, memory)
    monkeypatch.setattr(common.offcfg, "scale_cpu_size_for_pp", lambda cfg, c: None)
    monkeypatch.setattr(
        common.offcfg,
        "build_lmcache_metadata",
        lambda *args: SimpleNamespace(chunk_size=16),
    )
    monkeypatch.setattr(
        common.mooncake_store_l2,
        "prepare_mooncake_store_l2",
        lambda cfg: events.append("prepare"),
    )

    def check(name):
        events.append(name)
        if failing_check == name:
            raise RuntimeError(f"{name} failed")

    monkeypatch.setattr(
        common.mooncake_store_l2, "verify_thp_l1", lambda engine: check("thp")
    )
    monkeypatch.setattr(
        common.mooncake_store_l2,
        "release_reserved_l1",
        lambda: events.append("release"),
    )
    monkeypatch.setattr(
        common, "verify_remote_backend", lambda engine, meta: check("verify")
    )
    build = functools.partial(
        common.build_offload_engine,
        SimpleNamespace(),
        engine_id="e",
        block_size=16,
        bytes_per_block=4,
        gpu_connector_factory=lambda cfg, meta: None,
        world=1,
        rank=0,
        cfg=SimpleNamespace(remote_url=remote_url),
    )
    if failing_check:
        with pytest.raises(RuntimeError, match=f"{failing_check} failed"):
            build()
    else:
        build()
    assert events == expected


def test_offload_transfer_config_finds_the_in_process_connector():
    from atom.kv_transfer.offload import _offload_common as common

    offload = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    assert common.offload_transfer_config(offload) is offload
    multi = {
        "kv_connector": "multi",
        "connectors": [{"kv_connector": "mooncake", "kv_role": "kv_producer"}, offload],
    }
    assert common.offload_transfer_config(multi) is offload
    # The factory's aliases and any case build the same connector.
    for alias in ("LMCacheConnectorV1", "LMCACHE_OFFLOAD"):
        aliased = {"kv_connector": alias, "kv_role": "offload"}
        assert common.offload_transfer_config(aliased) is aliased
    aliased = {"kv_connector": "LMCacheOffloadConnector", "kv_role": "offload"}
    upper = {
        "kv_connector": "MULTI",
        "connectors": [{"kv_connector": "mooncake"}, aliased],
    }
    assert common.offload_transfer_config(upper) is aliased
    for other in (
        None,
        {"kv_connector": "mooncake", "kv_role": "kv_consumer"},
        {"kv_connector": "lmcache_mp", "kv_role": "offload"},
        {"kv_connector": "multi", "connectors": [{"kv_connector": "mooncake"}]},
        {"kv_connector": "no_such_connector"},
        {"kv_role": "offload"},
    ):
        assert common.offload_transfer_config(other) is None


def test_reserve_l1_before_weights_load_sizes_it_as_the_engine_does(monkeypatch):
    from atom.kv_transfer.offload import _offload_common as common

    reserved = []
    built = []
    checked = []

    def build(sub):
        built.append(sub)
        return SimpleNamespace(remote_url=remote_url, max_local_cpu_size=48.0)

    def scale(cfg, config):
        cfg.max_local_cpu_size *= 1.5

    monkeypatch.setattr(common.offcfg, "build_lmcache_config", build)
    monkeypatch.setattr(common.offcfg, "scale_cpu_size_for_pp", scale)
    monkeypatch.setattr(common.mooncake_store_l2, "current_gpu_numa_node", lambda: 0)
    monkeypatch.setattr(
        common.mooncake_store_l2,
        "reserve_thp_l1",
        lambda size, node: reserved.append((size, node)),
    )
    monkeypatch.setattr(
        common.mooncake_store_l2,
        "validate_mooncake_store_config",
        lambda cfg: checked.append("recipe"),
    )
    monkeypatch.setattr(
        common.mooncake_store_l2,
        "require_thp_allowed",
        lambda: checked.append("thp"),
    )
    offload = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "multi", "connectors": [offload]}
    )
    remote_url = "mooncakestore://10.0.0.1:50051/"
    common.reserve_l1_before_weights_load(config)
    assert built == [offload]
    assert checked == ["recipe", "thp"]
    assert reserved == [(72 * 1024**3, 0)]
    # Any other L2 (or none) keeps the allocation at engine build.
    remote_url = "lm://10.0.0.1:65432"
    common.reserve_l1_before_weights_load(config)
    assert reserved == [(72 * 1024**3, 0)]
    assert checked == ["recipe", "thp"]
    common.reserve_l1_before_weights_load(SimpleNamespace(kv_transfer_config=None))
    assert len(built) == 2


def test_reserve_l1_before_weights_load_is_best_effort(monkeypatch, caplog):
    from atom.kv_transfer.offload import _offload_common as common

    def build(sub):
        raise ValueError("bad LMCache env")

    monkeypatch.setattr(common.offcfg, "build_lmcache_config", build)
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "offload"}
    )
    # The engine build raises that error again, with its context.
    with caplog.at_level(logging.WARNING, logger="atom"):
        common.reserve_l1_before_weights_load(config)
    assert "no reservation" not in caplog.text

    def refuse(size, node):
        raise OSError(errno.ENOMEM, "mmap failed")

    monkeypatch.setattr(
        common.offcfg,
        "build_lmcache_config",
        lambda sub: _cfg(max_local_cpu_size=8.0),
    )
    monkeypatch.setattr(common.offcfg, "scale_cpu_size_for_pp", lambda cfg, c: None)
    for name, stub in (
        ("validate_mooncake_store_config", lambda cfg: None),
        ("require_thp_allowed", lambda: None),
        ("current_gpu_numa_node", lambda: 0),
        ("reserve_thp_l1", refuse),
    ):
        monkeypatch.setattr(common.mooncake_store_l2, name, stub)
    with caplog.at_level(logging.WARNING, logger="atom"):
        common.reserve_l1_before_weights_load(config)
    assert "no reservation before the weights load" in caplog.text


def test_reserve_l1_before_weights_load_refuses_a_broken_recipe(store_env, monkeypatch):
    from atom.kv_transfer.offload import _offload_common as common

    monkeypatch.setattr(
        common.offcfg,
        "build_lmcache_config",
        lambda sub: _cfg(max_local_cpu_size=48.0),
    )
    monkeypatch.setattr(common.mooncake_store_l2, "reserve_thp_l1", pytest.fail)
    config = SimpleNamespace(
        kv_transfer_config={"kv_connector": "lmcache_offload", "kv_role": "offload"}
    )
    # Before the weights load, where the engine build would refuse it after.
    monkeypatch.delenv("MC_NUM_QP_PER_EP")
    with pytest.raises(ValueError, match="MC_NUM_QP_PER_EP=<unset, 2>"):
        common.reserve_l1_before_weights_load(config)
    monkeypatch.setenv("MC_NUM_QP_PER_EP", "1")

    def thp_disabled():
        raise RuntimeError("THP is disabled for this process")

    monkeypatch.setattr(common.mooncake_store_l2, "require_thp_allowed", thp_disabled)
    with pytest.raises(RuntimeError, match="THP is disabled"):
        common.reserve_l1_before_weights_load(config)


def test_release_failed_engine_stops_an_orphaned_storage_loop(monkeypatch):
    """A storage manager whose constructor raised leaves a non-daemon loop."""
    from atom.kv_transfer.offload import _offload_common as common

    cache_engine = types.ModuleType("lmcache.v1.cache_engine")
    destroyed = []
    cache_engine.LMCacheEngineBuilder = SimpleNamespace(destroy=destroyed.append)
    for name in ("lmcache", "lmcache.v1"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setitem(sys.modules, cache_engine.__name__, cache_engine)
    loops, threads = [], []

    def start_loop():
        loop = asyncio.new_event_loop()
        # As LMCache starts it: the loop rides in the thread's arguments.
        thread = threading.Thread(
            target=lambda lp: lp.run_forever(),
            args=(loop,),
            name="storage-manager-event-loop",
        )
        thread.start()
        loops.append(loop)
        threads.append(thread)
        return thread

    # A healthy engine's loop, running before the failed one was built.
    other = start_loop()
    other_loop_threads = common.storage_manager_loop_threads()
    assert other in other_loop_threads
    orphan = start_loop()
    try:
        common.release_failed_engine("e", other_loop_threads)
        orphan.join(timeout=5)
        assert not orphan.is_alive()
        assert other.is_alive()
        assert destroyed == ["e"]
    finally:
        for loop, thread in zip(loops, threads):
            if thread.is_alive():
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=5)
            loop.close()
