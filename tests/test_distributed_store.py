# SPDX-License-Identifier: MIT
"""Rendezvous ports stay owned from allocation through worker shutdown."""

import copy
import errno
import multiprocessing
import socket
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from atom.config import ParallelConfig
from atom.model_engine.engine_core_mgr import CoreManager


@pytest.fixture
def manager(monkeypatch):
    for name in (
        "ATOM_DP_SIZE",
        "ATOM_DP_SIZE_LOCAL",
        "ATOM_DP_RANK",
        "ATOM_DP_RANK_LOCAL",
        "ATOM_DP_MASTER_IP",
        "ATOM_DP_MASTER_PORT",
        "ATOM_DP_BASE_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    manager = CoreManager.__new__(CoreManager)
    manager._init_shared_state(
        SimpleNamespace(dp_load_balance="round_robin"),
        label="test",
        local_engine_count=0,
    )
    try:
        yield manager
    finally:
        manager.close()
        manager.ctx.term()


def _assert_port_owned(port):
    with socket.socket() as contender:
        with pytest.raises(OSError) as error:
            contender.bind(("127.0.0.1", port))
        assert error.value.errno == errno.EADDRINUSE


def _spawn_workers(target, args_per_rank):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=target, args=args) for args in args_per_rank]
    try:
        for process in processes:
            process.start()
        deadline = time.monotonic() + 120
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        assert [p.exitcode for p in processes] == [0] * len(processes)
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            if process.pid is not None:
                process.join(timeout=5)
                process.close()


def _gloo_worker(pc, rank, world_size):
    store = dist.TCPStore(
        pc.data_parallel_master_ip,
        pc.data_parallel_base_port,
        is_master=False,
        timeout=timedelta(seconds=30),
    )
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        value = torch.tensor(rank + 1)
        dist.all_reduce(value)
        assert value.item() == world_size * (world_size + 1) // 2
    finally:
        dist.destroy_process_group()


def test_store_survives_workers_and_is_released_on_close(manager):
    config = SimpleNamespace(parallel_config=ParallelConfig())
    assert config.parallel_config.data_parallel_base_port == 0
    manager._start_distributed_store(config)
    pc = copy.deepcopy(config.parallel_config)
    assert pc._managed_distributed_store
    _assert_port_owned(pc.data_parallel_base_port)
    _spawn_workers(_gloo_worker, [(pc, rank, 2) for rank in range(2)])
    # Rank zero exiting must not destroy the coordinator's listener.
    _assert_port_owned(pc.data_parallel_base_port)
    manager.close()
    replacement = dist.TCPStore(
        pc.data_parallel_master_ip,
        pc.data_parallel_base_port,
        is_master=True,
        wait_for_workers=False,
    )
    assert replacement.port == pc.data_parallel_base_port


def test_separate_stores_have_distinct_ports_and_keys(manager):
    configs = [SimpleNamespace(parallel_config=ParallelConfig()) for _ in range(2)]
    for config in configs:
        manager._start_distributed_store(config)
    ports = [config.parallel_config.data_parallel_base_port for config in configs]
    assert ports[0] != ports[1]
    clients = [dist.TCPStore("127.0.0.1", port, is_master=False) for port in ports]
    for i, client in enumerate(clients):
        client.set("same-key", str(i))
    assert [client.get("same-key") for client in clients] == [b"0", b"1"]


def test_fixed_port_conflict_fails_without_changing_port(manager):
    owner = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    pc = ParallelConfig(data_parallel_base_port=owner.port)
    with pytest.raises(dist.DistNetworkError, match="EADDRINUSE"):
        manager._start_distributed_store(SimpleNamespace(parallel_config=pc))
    assert pc.data_parallel_base_port == owner.port
    assert not pc._managed_distributed_store
    assert not manager._distributed_stores


def test_remote_node_uses_coordinator_store(manager):
    owner = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
    pc = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=1,
        data_parallel_base_port=owner.port,
    )
    manager._start_distributed_store(
        SimpleNamespace(parallel_config=pc), multinode=True
    )
    assert pc._managed_distributed_store
    assert not manager._distributed_stores
    client = dist.TCPStore("127.0.0.1", pc.data_parallel_base_port, is_master=False)
    client.set("remote", "connected")
    assert owner.get("remote") == b"connected"


@pytest.mark.parametrize("rank", [0, 1])
def test_multinode_requires_shared_fixed_port(manager, rank):
    pc = ParallelConfig(
        data_parallel_size=2,
        data_parallel_size_local=1,
        data_parallel_rank=rank,
    )
    with pytest.raises(ValueError, match="same nonzero --data-parallel-base-port"):
        manager._start_distributed_store(
            SimpleNamespace(parallel_config=pc), multinode=True
        )
    assert not manager._distributed_stores


def _model_runner_worker(pc, layout, global_rank):
    from aiter import destroy_dist_env
    from aiter.dist.parallel_state import (
        get_dp_group,
        get_pcp_group,
        get_pp_group,
        get_tp_group,
    )

    from atom.model_engine.model_runner import ModelRunner

    tp, pp, dp, pcp, dcp = layout
    stage_span = tp * pcp
    pc.data_parallel_rank = global_rank // (pp * stage_span)
    pc.data_parallel_rank_local = pc.data_parallel_rank
    pc.pipeline_parallel_rank = global_rank // stage_span % pp
    config = SimpleNamespace(
        parallel_config=pc,
        tensor_parallel_size=tp,
        tp_world_size=tp,
        pipeline_parallel_size=pp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        master_addr="127.0.0.1",
        port=0,
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.config = config
    runner._setup_device_and_distributed(global_rank % stage_span, config)
    try:
        assert dist.get_rank() == global_rank
        assert dist.get_world_size() == tp * pp * dp * pcp
        assert get_tp_group().world_size == tp
        assert get_pp_group().world_size == pp
        assert get_dp_group().world_size == dp
        assert get_pcp_group().world_size == pcp
        value = torch.tensor(global_rank + 1, device=runner.device)
        dist.all_reduce(value)
        world_size = dist.get_world_size()
        assert value.item() == world_size * (world_size + 1) // 2
        torch.cuda.synchronize()
    finally:
        destroy_dist_env()


@pytest.mark.parametrize(
    "layout",
    [
        (2, 1, 1, 1, 1),
        (1, 1, 2, 1, 1),
        (2, 1, 2, 1, 1),
        (1, 2, 1, 1, 1),
        (1, 1, 1, 2, 1),
        (2, 1, 1, 1, 2),
        (1, 1, 8, 1, 1),
    ],
    ids=["tp", "dp", "tp_dp", "pp", "pcp", "dcp", "dp8"],
)
def test_model_runner_connects_to_managed_store(manager, layout):
    tp, pp, dp, pcp, _ = layout
    world_size = tp * pp * dp * pcp
    if not torch.version.hip or torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} ROCm GPUs and AITER")
    pytest.importorskip("aiter")
    config = SimpleNamespace(parallel_config=ParallelConfig(data_parallel_size=dp))
    manager._start_distributed_store(config)
    _spawn_workers(
        _model_runner_worker,
        [(config.parallel_config, layout, rank) for rank in range(world_size)],
    )
    _assert_port_owned(config.parallel_config.data_parallel_base_port)


def test_standalone_model_runner_keeps_rank_zero_rendezvous(manager):
    if not torch.version.hip or torch.cuda.device_count() < 2:
        pytest.skip("requires 2 ROCm GPUs and AITER")
    pytest.importorskip("aiter")
    from atom.utils import get_open_port

    # Standalone callers choose a shared endpoint without a CoreManager.
    pc = ParallelConfig(data_parallel_base_port=get_open_port())
    assert not pc._managed_distributed_store
    _spawn_workers(
        _model_runner_worker, [(pc, (2, 1, 1, 1, 1), rank) for rank in range(2)]
    )
