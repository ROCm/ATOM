# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One tensor-parallel group across nodes (atom/model_engine/tp_multinode.py)."""

import time
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
        "ATOM_MULTINODE_PG_TIMEOUT_S",
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
def test_a_tp_group_across_nodes_raises_the_group_timeout(monkeypatch, spans_nodes):
    from torch.distributed import distributed_c10d

    from atom.utils.distributed.utils import raise_timeout_if_tp_spans_nodes

    monkeypatch.setenv("ATOM_MULTINODE_PG_TIMEOUT_S", "3600")
    monkeypatch.setattr(distributed_c10d, "default_pg_timeout", timedelta(minutes=30))
    monkeypatch.setattr(
        distributed_c10d, "default_pg_nccl_timeout", timedelta(minutes=10)
    )
    raise_timeout_if_tp_spans_nodes(SimpleNamespace(is_multinode_tp=spans_nodes))
    # What the default group and aiter's groups, created without one, get.
    gloo = distributed_c10d._get_default_timeout(dist.Backend.GLOO)
    nccl = distributed_c10d._get_default_timeout(dist.Backend.NCCL)
    if spans_nodes:
        assert gloo == nccl == timedelta(hours=1)
    else:
        assert (gloo, nccl) == (timedelta(minutes=30), timedelta(minutes=10))


def test_a_remote_rank_replies_to_an_ipv6_host():
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProcManager

    manager = SimpleNamespace(local_proc_num=4)
    reply = AsyncIOProcManager._reply_addr
    assert reply(manager, 4, "fd00::1").startswith("tcp://[fd00::1]:")
    assert reply(manager, 4, "10.0.0.1").startswith("tcp://10.0.0.1:")


def test_config_checks_the_tp_slices(monkeypatch):
    class Reached(Exception):
        pass

    def reached(self):
        raise Reached

    monkeypatch.setattr(Config, "_check_multinode_tp", reached)
    with pytest.raises(Reached):
        Config(model="unused", tensor_parallel_size=8)


@pytest.fixture
def engine_store():
    from atom.utils import get_open_port

    port = get_open_port()
    store = dist.TCPStore("127.0.0.1", port, is_master=True)
    yield SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_master_ip="127.0.0.1", data_parallel_base_port=port
        )
    )
    del store


def _remote_ranks(config, monkeypatch, ps1="1"):
    monkeypatch.setenv("AITER_MLA_DECODE_PS1_FLYDSL", ps1)
    return tp_multinode.RemoteRanks(
        config,
        {"proc_num": 8, "local_proc_num": 4, "args": (None,), "handle": None},
    )


@pytest.mark.parametrize(
    "start, ps1, claimed, error",
    [
        (4, "0", False, "AITER_MLA_DECODE_PS1_FLYDSL must match"),
        (2, "1", False, "not remote ranks"),
        (4, "1", True, "TP rank 4 already runs on another node"),
    ],
)
def test_a_worker_node_refuses_a_slice_or_env_the_engine_does_not_expect(
    monkeypatch, engine_store, start, ps1, claimed, error
):
    from aiter_stub import stubbed_aiter

    import atom.config

    pc = engine_store.parallel_config
    remote = _remote_ranks(engine_store, monkeypatch)
    if claimed:  # another node holds rank 4
        claims = tp_multinode._store("127.0.0.1", pc.data_parallel_base_port, 5)
        claims.add(tp_multinode._engine_key(remote.engine_id, "rank/4"), 1)
    monkeypatch.setenv("AITER_MLA_DECODE_PS1_FLYDSL", ps1)
    monkeypatch.setattr(atom.config, "get_hf_config", lambda *a, **k: None)
    args = EngineArgs(
        tensor_parallel_size=8,
        tensor_parallel_size_local=2 if start == 2 else 4,
        tensor_parallel_rank=start,
        data_parallel_master_ip="10.0.0.1",
        data_parallel_base_port=pc.data_parallel_base_port,
    )
    # The engine's store answers on loopback; the address check needs a real one.
    real_store = tp_multinode._store
    monkeypatch.setattr(
        tp_multinode, "_store", lambda ip, port, t: real_store("127.0.0.1", port, t)
    )
    with stubbed_aiter(), pytest.raises(ValueError, match=error):
        tp_multinode.serve_remote_tp_ranks(args)


class _Proc:
    def __init__(self, name, exitcode=None):
        self.name, self.exitcode = name, exitcode

    def is_alive(self):
        return self.exitcode is None


def _supervise(config, remote, procs, monkeypatch):
    stopped = []
    monkeypatch.setattr(tp_multinode.signal, "signal", lambda *a: None)
    monkeypatch.setattr(tp_multinode.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        tp_multinode, "shutdown_all_processes", lambda p, **k: stopped.extend(p)
    )
    pc = config.parallel_config
    ip, port = pc.data_parallel_master_ip, pc.data_parallel_base_port
    store = tp_multinode._store(ip, port, 5)
    status = tp_multinode._supervise(procs, store, ip, port, remote.engine_id)
    assert stopped == procs
    return status


def test_a_failed_worker_is_reported_to_the_engine(monkeypatch, engine_store):
    remote = _remote_ranks(engine_store, monkeypatch)
    assert remote.failure() is None
    procs = [_Proc("ModelRunner4/8"), _Proc("ModelRunner5/8", exitcode=-9)]
    assert _supervise(engine_store, remote, procs, monkeypatch) == 1
    assert remote.failure() == "ModelRunner5/8 exited with -9"


def test_the_engine_stopping_stops_the_worker_node(monkeypatch, engine_store):
    remote = _remote_ranks(engine_store, monkeypatch)
    remote.close()
    assert _supervise(engine_store, remote, [_Proc("ModelRunner4/8")], monkeypatch) == 0
    assert remote.failure() is None


def test_the_engine_stops_when_a_worker_node_reports(monkeypatch):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.async_proc import AsyncIOProcManager

    manager = object.__new__(AsyncIOProcManager)
    exited, reported = [], []
    manager.procs, manager.label, manager.runner_label = [], "mgr", "runner"
    manager.still_running = True
    manager.on_worker_failure = lambda: reported.append(True)
    manager.exit = lambda: exited.append(True)
    manager.remote_ranks = SimpleNamespace(poll_s=0.01, failure=lambda: "lost")
    manager.monitor_procs()
    for _ in range(500):
        if exited:
            break
        time.sleep(0.01)
    assert exited and reported


class _Socket:
    """The broadcast queue's remote socket: replays *incoming*, records sends."""

    def __init__(self, *incoming):
        self.incoming, self.sent = list(incoming), []

    def poll(self, timeout):
        return bool(self.incoming)

    def recv(self):
        return self.incoming.pop(0)

    def send(self, msg):
        self.sent.append(msg)


def test_remote_readers_are_released_once_all_subscribe():
    sock = _Socket(b"\x01", b"\x01")
    tp_multinode.release_remote_readers(
        SimpleNamespace(remote_socket=sock), 2, lambda: True
    )
    assert sock.sent == [tp_multinode._READY]
    tp_multinode.await_release(SimpleNamespace(remote_socket=_Socket(*sock.sent)))


def test_remote_readers_that_never_subscribe_time_out(monkeypatch):
    monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", "0.01")
    sock = _Socket(b"\x01")
    with pytest.raises(TimeoutError, match="1 of 2 TP ranks"):
        tp_multinode.release_remote_readers(
            SimpleNamespace(remote_socket=sock), 2, lambda: True
        )
    assert sock.sent == []


@pytest.mark.parametrize(
    "incoming,error",
    [((b"\x01data",), RuntimeError), ((), TimeoutError)],
    ids=["wrong_first_message", "timeout"],
)
def test_a_remote_reader_refuses_anything_but_the_release(monkeypatch, incoming, error):
    monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", "0.01")
    with pytest.raises(error):
        tp_multinode.await_release(SimpleNamespace(remote_socket=_Socket(*incoming)))


@pytest.mark.parametrize(
    "start,rank,device", [(0, 0, 0), (0, 3, 3), (4, 4, 0), (4, 7, 3)]
)
def test_a_worker_nodes_ranks_use_its_own_gpus_from_zero(start, rank, device):
    from atom.config import local_device_index

    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_rank_local=0, pipeline_parallel_rank=0
        ),
        pipeline_parallel_size=1,
        tp_world_size=8,
        pcp_group_size=1,
        tp_rank_start=start,
    )
    assert local_device_index(config, rank) == device


@pytest.mark.parametrize(
    "pp,multinode_tp,reduced",
    [(1, False, False), (2, False, True), (1, True, True)],
)
def test_kv_blocks_reduce_to_the_minimum_when_ranks_can_differ(
    pp, multinode_tp, reduced
):
    from atom.utils.distributed.utils import kv_blocks_differ_across_ranks

    config = SimpleNamespace(pipeline_parallel_size=pp, is_multinode_tp=multinode_tp)
    assert kv_blocks_differ_across_ranks(config) is reduced
