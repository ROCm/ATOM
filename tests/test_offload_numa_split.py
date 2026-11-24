# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""How the LMCache CPU pools are placed over NUMA nodes (`_normalize_numa_mapping`)."""

from types import SimpleNamespace

import torch

from atom.kv_transfer.offload import config as offcfg


def test_json_mapping_keys_become_ints():
    cfg = SimpleNamespace(
        numa_mode="manual",
        extra_config={"gpu_to_numa_mapping": {"0": 0, "1": "0", "2": 1, "3": 1}},
    )
    offcfg._normalize_numa_mapping(cfg)
    assert cfg.extra_config["gpu_to_numa_mapping"] == {0: 0, 1: 0, 2: 1, 3: 1}


def test_split_spreads_devices_over_nodes(monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_NUMA_SPLIT", "1")
    monkeypatch.setattr(offcfg, "_online_numa_nodes", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    cfg = SimpleNamespace(numa_mode=None, extra_config=None)
    offcfg._normalize_numa_mapping(cfg)
    assert cfg.numa_mode == "manual"
    assert cfg.extra_config["gpu_to_numa_mapping"] == {0: 0, 1: 0, 2: 1, 3: 1}


def test_split_leaves_an_explicit_mode_alone(monkeypatch):
    monkeypatch.setenv("ATOM_OFFLOAD_NUMA_SPLIT", "1")
    cfg = SimpleNamespace(numa_mode="auto", extra_config=None)
    offcfg._normalize_numa_mapping(cfg)
    assert cfg.numa_mode == "auto" and cfg.extra_config is None


def test_no_split_without_the_switch(monkeypatch):
    monkeypatch.delenv("ATOM_OFFLOAD_NUMA_SPLIT", raising=False)
    cfg = SimpleNamespace(numa_mode=None, extra_config=None)
    offcfg._normalize_numa_mapping(cfg)
    assert cfg.numa_mode is None


def test_online_numa_nodes_parses_ranges(monkeypatch, tmp_path):
    import builtins

    real_open = builtins.open
    f = tmp_path / "online"
    f.write_text("0-1,3\n")
    monkeypatch.setattr(
        builtins,
        "open",
        lambda p, *a, **k: real_open(f if p == "/sys/devices/system/node/online" else p, *a, **k),
    )
    assert offcfg._online_numa_nodes() == 3
