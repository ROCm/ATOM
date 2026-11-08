# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CoreManager with an external engine client (the atomesh Rust runtime): it
spawns the EngineCores and reaps them, but never connects to them -- the Rust
runtime binds their sockets from the endpoints it exports."""

import itertools
from types import SimpleNamespace

import pytest

from atom.model_engine import engine_core_mgr
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager

_pids = itertools.count(4242)


class FakeProcess:
    """An EngineCore process that exits when joined, as one does on Shutdown."""

    def __init__(self):
        self.pid = None
        self.alive = False
        self.closed = False

    def start(self):
        self.pid = next(_pids)
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.alive = False

    terminate = kill = join

    def close(self):
        self.closed = True


def _config(dp=2, pp=1, multinode=False):
    parallel = SimpleNamespace(
        is_multinode_dp=multinode,
        data_parallel_size=dp,
        data_parallel_size_local=dp,
        data_parallel_rank=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_master_port=0,
        data_parallel_base_port=0,
    )
    return SimpleNamespace(
        external_engine_client=True,
        pipeline_parallel_size=pp,
        parallel_config=parallel,
        enable_dp_attention=False,
        tensor_parallel_size=1,
        dp_load_balance="least_requests",
        keep_finished_outputs=False,
        asyncio_mode=False,
    )


@pytest.fixture
def launched(monkeypatch):
    processes = []

    def launch(config, dp_rank, local_dp_rank, addresses=None):
        process = FakeProcess()
        processes.append(process)
        socket_addresses = {
            name: f"ipc:///tmp/fake-{dp_rank}-{name}"
            for name in ("input_address", "output_address", "control_address")
        }
        return process, socket_addresses, local_dp_rank

    monkeypatch.setattr(engine_core_mgr, "launch_engine_core", launch)
    return processes


def test_spawns_the_cores_and_exports_their_endpoints_without_connecting(launched):
    mgr = CoreManager(_config(dp=2))
    assert [p.alive for p in launched] == [True, True]
    assert mgr.engine_endpoints == [
        {
            "dp_rank": rank,
            "pid": launched[rank].pid,
            "input": f"ipc:///tmp/fake-{rank}-input_address",
            "control": f"ipc:///tmp/fake-{rank}-control_address",
            "output": f"ipc:///tmp/fake-{rank}-output_address",
        }
        for rank in range(2)
    ]
    # Nothing bound, nothing read: those sockets belong to the Rust runtime.
    assert getattr(mgr, "input_sockets", []) == []
    assert mgr.output_threads == []
    assert mgr.is_healthy()
    mgr.close()


def test_close_reaps_the_cores_it_spawned(launched):
    mgr = CoreManager(_config(dp=2))
    mgr.close()
    assert [(p.alive, p.closed) for p in launched] == [(False, True), (False, True)]
    assert not mgr.is_healthy()


@pytest.mark.parametrize(
    "config",
    [_config(dp=1, pp=2), _config(dp=2, multinode=True)],
    ids=["pipeline_parallel", "multi_node_dp"],
)
def test_unsupported_topologies_are_refused_before_spawning(launched, config):
    with pytest.raises(ValueError, match="external engine client"):
        CoreManager(config)
    assert launched == []


def test_rapidserve_is_refused(launched):
    with pytest.raises(ValueError, match="external engine client"):
        DisaggCoreManager(_config(dp=1))
    assert launched == []
