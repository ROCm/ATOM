# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One tensor-parallel group across nodes (atom/model_engine/tp_multinode.py)."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch.distributed as dist

from atom.config import Config, ParallelConfig
from atom.model_engine import arg_utils, tp_multinode
from atom.model_engine.arg_utils import EngineArgs


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    for name in (
        "ATOM_DP_SIZE",
        "ATOM_DP_SIZE_LOCAL",
        "ATOM_DP_RANK",
        "ATOM_DP_RANK_LOCAL",
        "ATOM_DP_MASTER_IP",
        "ATOM_DP_MASTER_PORT",
        "ATOM_DP_BASE_PORT",
        "ATOM_MULTINODE_TP_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)


def _worker_node_args():
    return EngineArgs(
        tensor_parallel_size=8, tensor_parallel_size_local=4, tensor_parallel_rank=4
    )


def _config(local, start, dp=1):
    """A Config holding only what the TP-slice check reads."""
    config = Config.__new__(Config)
    config.parallel_config = ParallelConfig(
        data_parallel_size=dp,
        data_parallel_master_ip="10.0.0.1",
        data_parallel_base_port=29600,
        tensor_parallel_size_local=local,
        tensor_parallel_rank=start,
    )
    config.tensor_parallel_size = 8
    config.fake_eplb = False
    config.enable_dp_attention = False
    config.pipeline_parallel_size = 1
    config.prefill_context_parallel_size = 1
    config.enable_rapidserve = False
    return config


@pytest.mark.parametrize(
    "local, start, dp, error",
    [
        (None, 0, 1, None),
        (8, 0, 2, None),  # one node, spelled out: no cross-node refusal
        (4, 0, 1, None),
        (4, 4, 1, None),
        (16, 0, 1, "must cut -tp 8"),
        (4, 2, 1, "must cut -tp 8"),
        (4, 0, 2, "does not support data parallelism"),
    ],
)
def test_tp_slices_are_checked_once_either_flag_is_set(local, start, dp, error):
    config = _config(local, start, dp)
    if error is None:
        config._check_multinode_tp()
    else:
        with pytest.raises(ValueError, match=error):
            config._check_multinode_tp()


def test_a_worker_node_reads_the_engine_address_the_way_the_engine_does(
    monkeypatch,
):
    from aiter_stub import stubbed_aiter

    monkeypatch.setenv("ATOM_DP_MASTER_IP", "10.0.0.1")
    monkeypatch.setenv("ATOM_DP_BASE_PORT", "29600")
    reached = []

    def store(ip, port, timeout_s):
        reached.append((ip, port))
        raise ConnectionError("stop before the bootstrap")

    monkeypatch.setattr(tp_multinode, "_store", store)
    with stubbed_aiter(), pytest.raises(ConnectionError):
        tp_multinode.serve_remote_tp_ranks(_worker_node_args())
    assert reached == [("10.0.0.1", 29600)]


def test_create_engine_on_a_worker_node_serves_its_ranks_instead(monkeypatch):
    served = []
    monkeypatch.setattr(
        tp_multinode, "serve_remote_tp_ranks", lambda args: served.append(args) or 0
    )
    monkeypatch.setattr(
        arg_utils, "LLMEngine", lambda *a, **k: pytest.fail("built an engine")
    )
    args = _worker_node_args()
    with pytest.raises(SystemExit) as stopped:
        args.create_engine()
    assert stopped.value.code == 0
    assert served == [args]


@pytest.mark.parametrize(
    "spans_nodes, env, timeout",
    [
        (True, None, timedelta(seconds=1800)),
        (True, "60", timedelta(seconds=60)),
        (False, "60", None),
    ],
)
def test_a_rank_store_client_waits_as_long_as_tp_setup(
    monkeypatch, spans_nodes, env, timeout
):
    # A cross-node TP rank fetches lazily created NCCL ids from this store; the
    # trays reach their first such collective as far apart as their loads differ.
    if env is not None:
        monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", env)
    clients = []
    monkeypatch.setattr(dist, "TCPStore", lambda *a, **k: clients.append((a, k)))
    config = SimpleNamespace(
        is_multinode_tp=spans_nodes,
        parallel_config=SimpleNamespace(
            data_parallel_master_ip="10.0.0.1", data_parallel_base_port=29600
        ),
    )
    tp_multinode.managed_store_client(config)
    expected = {"is_master": False}
    if timeout is not None:
        expected["timeout"] = timeout
    assert clients == [(("10.0.0.1", 29600), expected)]


@pytest.mark.parametrize("spans_nodes", [True, False])
def test_groups_without_a_timeout_wait_as_long_as_tp_setup(monkeypatch, spans_nodes):
    from torch.distributed import distributed_c10d

    monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", "3600")
    monkeypatch.setattr(distributed_c10d, "default_pg_timeout", timedelta(minutes=30))
    monkeypatch.setattr(
        distributed_c10d, "default_pg_nccl_timeout", timedelta(minutes=10)
    )
    config = SimpleNamespace(is_multinode_tp=spans_nodes)
    tp_multinode.raise_timeout_if_tp_spans_nodes(config)
    # What the default group and aiter's groups, created without one, get.
    gloo = distributed_c10d._get_default_timeout(dist.Backend.GLOO)
    nccl = distributed_c10d._get_default_timeout(dist.Backend.NCCL)
    if not spans_nodes:
        assert (gloo, nccl) == (timedelta(minutes=30), timedelta(minutes=10))
        return
    assert gloo == nccl == timedelta(hours=1)
    # A longer default is kept.
    monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", "60")
    tp_multinode.raise_timeout_if_tp_spans_nodes(config)
    assert distributed_c10d._get_default_timeout(dist.Backend.NCCL) == nccl


def test_a_remote_rank_replies_to_an_ipv6_host():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProcManager

    manager = SimpleNamespace(local_proc_num=4)
    reply = AsyncIOProcManager._reply_addr
    assert reply(manager, 4, "fd00::1").startswith("tcp://[fd00::1]:")
    assert reply(manager, 4, "10.0.0.1").startswith("tcp://10.0.0.1:")
