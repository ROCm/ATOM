# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The LMCache MP server entry point with a Mooncake Store L2."""

from __future__ import annotations

import json

import pytest

from atom.kv_transfer.offload.mooncake_store_l2 import StorePool
from atom.kv_transfer.offload.mp import mooncake_l2_server as server


@pytest.fixture
def ran(monkeypatch):
    """Record the module the entry point runs and the argv it runs it with."""
    runs = []

    def run_module(name, run_name, alter_sys):
        runs.append((name, run_name, alter_sys, list(server.sys.argv)))

    monkeypatch.setattr(server.runpy, "run_module", run_module)
    monkeypatch.setattr(server, "requester_rdma_device", lambda gpu: f"rdma{3 - gpu}")
    return runs


def _adapter(argv):
    flag = argv.index("--l2-adapter")
    return json.loads(argv[flag + 1])


def test_server_gets_the_store_l2_of_its_gpus_nic_and_pool(monkeypatch, ran):
    pools = {"rdma2": StorePool("10.0.0.1:26151", "http://10.0.0.1:26180/metadata")}
    monkeypatch.setattr(server, "store_pool_of", lambda device: pools.get(device))

    server.main(["--gpu", "1", "--local-hostname", "10.0.0.1", "--", "--port", "25556"])

    name, run_name, alter_sys, argv = ran[0]
    assert (name, run_name, alter_sys) == (
        "lmcache.v1.multiprocess.server",
        "__main__",
        True,
    )
    assert argv[:3] == ["lmcache.v1.multiprocess.server", "--port", "25556"]
    assert _adapter(argv) == {
        "type": "mooncake_store",
        "num_workers": 8,
        "master_server_addr": "10.0.0.1:26151",
        "metadata_server": "http://10.0.0.1:26180/metadata",
        "local_hostname": "10.0.0.1",
        "protocol": "rdma",
        "rdma_devices": "rdma2",
        "global_segment_size": "0",
        "local_buffer_size": "0",
    }
    # One argv word, so a logged command line splits back into the same argv.
    assert " " not in argv[argv.index("--l2-adapter") + 1]


def test_shared_pool_comes_from_the_command_line(monkeypatch, ran):
    monkeypatch.setattr(server, "store_pool_of", lambda device: None)

    server.main(
        [
            "--gpu",
            "0",
            "--local-hostname",
            "10.0.0.1",
            "--master",
            "10.0.0.1:26051",
            "--metadata",
            "http://10.0.0.1:26080/metadata",
            "--",
        ]
    )

    adapter = _adapter(ran[0][3])
    assert adapter["rdma_devices"] == "rdma3"
    assert adapter["master_server_addr"] == "10.0.0.1:26051"


def test_no_pool_at_all_is_refused(monkeypatch, ran):
    monkeypatch.setattr(server, "store_pool_of", lambda device: None)

    with pytest.raises(SystemExit, match="--master and --metadata"):
        server.main(["--gpu", "0", "--local-hostname", "10.0.0.1", "--"])
    assert ran == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--gpu", "0", "--local-hostname", "h"],
        ["--gpu", "0", "--local-hostname", "h", "--", "--l2-adapter", "{}"],
        ["--gpu", "0", "--local-hostname", "h", "--", "--l2-adapter={}"],
    ],
)
def test_bad_command_lines_are_refused(argv, ran):
    with pytest.raises(SystemExit):
        server.main(argv)
    assert ran == []
