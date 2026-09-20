# SPDX-License-Identifier: MIT
"""Keep matched-rail control addresses independent of the selected HCA."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize(
    "netdev_ip",
    [None, "192.0.2.6", "192.0.2.2"],
    ids=["no-rail-ipv4", "separate-roce-address", "shared-address"],
)
@pytest.mark.parametrize("primary_device", ["ionic_2", "ionic_6"])
def test_matched_rail_rpc_uses_host_address_and_preserves_hca_selection(
    monkeypatch, netdev_ip, primary_device
):
    from atom.kv_transfer.disaggregation.mooncake import mooncake_connector as mc

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(mc, "get_tp_group", lambda: group)
    monkeypatch.setattr(mc, "get_dp_group", lambda: group)
    monkeypatch.setattr(mc, "get_ip", lambda: "192.0.2.2")
    monkeypatch.setattr(mc, "get_open_port", lambda: 17000)
    monkeypatch.setattr(mc, "_MOONCAKE_AVAILABLE", True)
    monkeypatch.setattr(mc, "_ib_device_exists", lambda device: True)
    monkeypatch.setattr(mc.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(mc, "ThreadPoolExecutor", Mock())
    monkeypatch.setattr(mc.zmq, "Context", Mock())
    monkeypatch.setenv("ATOM_MOONCAKE_MATCHED_RAILS", "ionic_2,ionic_6")
    monkeypatch.delenv("ATOM_MOONCAKE_IB_DEVICE", raising=False)
    monkeypatch.delenv("MC_FORCE_TCP", raising=False)

    listdir = mc.os.listdir

    def rail_netdevs(path):
        if path == "/sys/class/infiniband/ionic_2/device/net":
            return []
        if path == "/sys/class/infiniband/ionic_6/device/net":
            return ["rail6"]
        return listdir(path)

    monkeypatch.setattr(mc.os, "listdir", rail_netdevs)

    def read_address(args, **kwargs):
        assert args == [
            "ip",
            "-o",
            "-4",
            "addr",
            "show",
            "dev",
            "rail6",
            "scope",
            "global",
        ]
        if netdev_ip is None:
            raise subprocess.CalledProcessError(1, args)
        return f"6: rail6 inet {netdev_ip}/24 scope global rail6\n"

    address_lookup = Mock(side_effect=read_address)
    monkeypatch.setattr(subprocess, "check_output", address_lookup)
    engines = []

    def make_engine():
        engine = Mock()
        engine.initialize.return_value = 0
        engine.register_memory.return_value = 0
        engine.get_rpc_port.return_value = 17000
        engines.append(engine)
        return engine

    monkeypatch.setattr(mc, "TransferEngine", make_engine, raising=False)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_rank=0),
        pipeline_parallel_size=1,
        hf_config=SimpleNamespace(num_hidden_layers=1),
        kv_cache_block_size=16,
        decode_context_parallel_size=1,
        dcp_config=SimpleNamespace(interleave_size=1),
        kv_transfer_config={
            "kv_role": "kv_producer",
            "protocol": "rdma",
            "ib_device": primary_device,
        },
    )
    connector = mc.MooncakeConnector(config)
    pool = connector._rail_pool
    pool.set_regions([1024], [64])

    # Reuse the configured primary, even when it is not first in the rail list.
    assert len(engines) == 1
    engines[0].initialize.assert_called_once_with(
        "192.0.2.2", "P2PHANDSHAKE", "rdma", primary_device
    )
    assert pool.get(primary_device) is engines[0]

    # RoCE addresses (or their absence) must not change the RPC endpoint.
    # The RDMA device filter must still be the selected rail.
    extra_device = "ionic_6" if primary_device == "ionic_2" else "ionic_2"
    extra = pool.get(extra_device)
    extra.initialize.assert_called_once_with(
        "192.0.2.2", "P2PHANDSHAKE", "rdma", extra_device
    )
    extra.register_memory.assert_called_once_with(1024, 64)
    assert pool.get(extra_device) is extra
    assert len(engines) == 2
    assert extra is not engines[0]
    assert pool.get(primary_device) is engines[0]
    address_lookup.assert_not_called()
