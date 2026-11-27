# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""The Mooncake P/D connector registers on its GPU's PCIe-local HCA.

sysfs is a temporary tree shaped like a rail-optimized MI355X node (pit2-p03-g40):
each GPU shares a PCIe switch with one NIC, and HIP numbers the GPUs in KFD order,
not PCI order, so ``rdma<HIP index>`` is another GPU's NIC for six of the eight.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiter_stub import atom_without_aiter

from atom.kv_transfer.offload.mooncake_store import nic

with atom_without_aiter() as load:
    mc = load("atom.kv_transfer.disaggregation.mooncake.mooncake_connector")

# HIP ordinal -> GPU bus; NIC N's bus. GPU bus t5 shares a switch with NIC bus t9.
GPU_BUS = ["75", "05", "65", "15", "f5", "85", "e5", "95"]
NIC_BUS = ["09", "19", "69", "79", "89", "99", "e9", "f9"]
# The PCIe-local NIC of HIP GPU 0..7.
LOCAL = [3, 0, 2, 1, 7, 4, 6, 5]


def _node(root: Path, prefix: str, inactive: tuple[str, ...]):
    """Per switch one host bridge, one GPU and one NIC."""
    pci, ib = root / "bus/pci/devices", root / "class/infiniband"
    pci.mkdir(parents=True)
    ib.mkdir(parents=True)
    for index, nic_bus in enumerate(NIC_BUS):
        top = nic_bus[0]
        switch = root / f"devices/pci0000:{top}0/0000:{top}0:01.1/0000:{top}1:00.0"
        gpu = switch / f"0000:{top}2:00.0/0000:{top}5:00.0"
        port = switch / f"0000:{top}2:01.0/0000:{nic_bus}:00.0"
        gpu.mkdir(parents=True)
        (gpu / "vendor").write_text("0x1002\n")
        (gpu / "class").write_text("0x120000\n")
        port.mkdir(parents=True)
        (pci / f"0000:{top}5:00.0").symlink_to(gpu)
        name = f"{prefix}{index}"
        (ib / name / "ports/1").mkdir(parents=True)
        state = "1: DOWN" if name in inactive else "4: ACTIVE"
        (ib / name / "ports/1/state").write_text(state + "\n")
        (ib / name / "device").symlink_to(port)
    return pci, ib


@pytest.fixture
def node(tmp_path, monkeypatch):
    """Builds the tree and points the connector's topology lookup at it."""

    def build(prefix="rdma", inactive=()):
        pci, ib = _node(tmp_path, prefix, inactive)
        monkeypatch.setattr(
            mc,
            "rail_rdma_device",
            lambda bdf: nic.rail_rdma_device(bdf, ib_root=ib, pci_root=pci),
        )

    return build


@pytest.mark.parametrize("prefix", ["rdma", "ionic_"])
def test_auto_select_follows_the_pci_tree_not_the_hip_index(node, prefix):
    node(prefix)
    picked = [mc._select_ib_devices("rdma", "", f"0000:{b}:00.0") for b in GPU_BUS]
    assert picked == [[f"{prefix}{n}"] for n in LOCAL]


def test_auto_select_refuses_a_guess_and_names_the_pd_override(node):
    node(inactive=("rdma3",))
    with pytest.raises(ValueError) as refused:
        mc._select_ib_devices("rdma", "", "0000:75:00.0")
    message = str(refused.value)
    assert message.startswith("cannot tell GPU 0000:75:00.0's NIC")
    assert message.endswith("; set ib_device or ATOM_MOONCAKE_IB_DEVICE")
    assert "mooncake_store" not in message


def connector(monkeypatch, role):
    """A connector built on visible device 1, with the engine and sockets faked."""
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setenv("ATOM_HOST_IP", "10.19.0.140")
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)
    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(mc, "get_tp_group", lambda: group)
    monkeypatch.setattr(mc, "get_dp_group", lambda: group)
    engine = MagicMock()
    engine.initialize.return_value = 0
    engine.get_rpc_port.return_value = 16578
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "TransferEngine", lambda: engine, raising=False)
    monkeypatch.setattr(mc, "get_open_port", lambda: 41000)
    monkeypatch.setattr(mc.zmq, "Context", MagicMock())
    monkeypatch.setattr(mc, "ThreadPoolExecutor", MagicMock())
    conn = mc.MooncakeConnector(
        SimpleNamespace(
            parallel_config=SimpleNamespace(pipeline_parallel_rank=0),
            pipeline_parallel_size=1,
            hf_config=SimpleNamespace(num_hidden_layers=2),
            kv_cache_block_size=16,
            decode_context_parallel_size=1,
            dcp_config=SimpleNamespace(interleave_size=1),
            kv_transfer_config={"kv_role": role, "protocol": "rdma"},
        )
    )
    return conn, engine


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer"])
@pytest.mark.parametrize(
    "masks",
    [
        {"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "1,2"},
        {"ROCR_VISIBLE_DEVICES": "GPU-e86b019ff114eec4,GPU-74a7fe3fe70c2cde"},
    ],
    ids=["ordinal-masks", "uuid-mask"],
)
def test_connector_registers_on_the_nic_of_the_gpu_it_drives(
    node, monkeypatch, role, masks
):
    node()
    # Visible device 1 here is bus 0xe5 (HIP GPU 6), whose switch holds rdma6.
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising=False)
    for name, value in masks.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(mc, "gpu_pci_bdf", {1: "0000:e5:00.0"}.__getitem__)
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE", raising=False)
    conn, engine = connector(monkeypatch, role)

    engine.initialize.assert_called_once_with(
        "10.19.0.140", "P2PHANDSHAKE", "rdma", "rdma6"
    )
    # A consumer advertises it as the destination HCA of its writes.
    assert conn.ib_devices == ["rdma6"] and conn.ib_device == "rdma6"


def test_an_override_registers_without_reading_the_topology(node, monkeypatch):
    node(inactive=("rdma6",))  # the topology would refuse bus 0xe5
    monkeypatch.setattr(mc, "gpu_pci_bdf", MagicMock(side_effect=AssertionError))
    monkeypatch.setenv("ATOM_MOONCAKE_IB_DEVICE", "rdma1")
    conn, engine = connector(monkeypatch, "kv_consumer")
    engine.initialize.assert_called_once_with(
        "10.19.0.140", "P2PHANDSHAKE", "rdma", "rdma1"
    )
    assert conn.ib_device == "rdma1"
