# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One pipeline across nodes: what Config accepts, the TCP plan both nodes
derive for the engine and stage sockets, CoreManager on either node, and each
stage's GPU and global rank. GPU-free; the engines are stand-ins that speak
EngineCore's socket protocol."""

import argparse
import copy
import pickle
import random
import socket
import threading
import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import zmq

from atom.config import (
    Config,
    KVEventsConfig,
    ParallelConfig,
    local_device_index,
    worker_global_rank,
)
from atom.distributed.pp_transport import PPStageTransport
from atom.model_engine import engine_core_mgr, tp_multinode
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.engine_core_mgr import (
    CoreManager,
    build_internode_dp_socket_plan,
    build_internode_pp_channel_plan,
)
from atom.model_engine.engine_core_protocol import EngineCoreRequestType
from atom.utils import get_open_port, get_open_zmq_ipc_path
from atom.utils.distributed.utils import raise_timeout_if_model_spans_nodes

NODE_IPS = ["10.0.0.1", "10.0.0.2"]
MASTER_PORT = 29500


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
        "ATOM_DP_DISPATCH_DEPTH",
        "ATOM_DP_SESSION_AFFINITY",
    ):
        monkeypatch.delenv(name, raising=False)


def _config(
    local=4,
    start=0,
    *,
    pp=8,
    tp=1,
    node_ips=NODE_IPS,
    master_ip="10.0.0.1",
    master_port=MASTER_PORT,
    base_port=29501,
    dp=1,
    pcp=1,
    tp_local=None,
    dp_attention=False,
    rapidserve=False,
    fake_eplb=False,
    kv_transfer_config=None,
):
    """A node's Config holding what the slice checks, CoreManager and the
    placement helpers read, built without loading a model."""
    config = Config.__new__(Config)
    config.parallel_config = ParallelConfig(
        data_parallel_size=dp,
        data_parallel_master_ip=master_ip,
        data_parallel_master_port=master_port,
        data_parallel_base_port=base_port,
        tensor_parallel_size_local=tp_local,
        pipeline_parallel_size_local=local,
        pipeline_parallel_rank=start,
        pipeline_parallel_node_ips=list(node_ips),
    )
    config.pipeline_parallel_size = pp
    config.tensor_parallel_size = tp
    config.prefill_context_parallel_size = pcp
    config.sequence_parallel_size = 1
    config.fake_eplb = fake_eplb
    config.enable_dp_attention = dp_attention
    config.enable_rapidserve = rapidserve
    config.kv_transfer_config = kv_transfer_config or {}
    # What CoreManager's shared state reads.
    config.kv_events_config = KVEventsConfig()
    config.dp_load_balance = "least_requests"
    config.keep_finished_outputs = False
    config.asyncio_mode = False
    config.max_num_seqs = 4
    config.enable_prefix_caching = True
    return config


def _stage_config(node_config, stage):
    """What CoreManager hands a stage: its node's config, at that stage."""
    config = copy.deepcopy(node_config)
    config.parallel_config.pipeline_parallel_rank = stage
    config.parallel_config.data_parallel_rank_local = 0
    return config


# ---------------------------------------------------------------------------
# Config: which slices and companions a pipeline across nodes accepts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "local, start, node_ips, error",
    [
        (None, 0, [], None),  # single node, none of the new flags
        (8, 0, [], None),  # one node, spelled out: no node list needed
        (4, 0, NODE_IPS, None),
        (4, 4, NODE_IPS, None),
        (2, 6, ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"], None),
        (3, 0, NODE_IPS, "must cut -pp 8"),
        (4, 2, NODE_IPS, "must cut -pp 8"),
        (4, 8, NODE_IPS, "must cut -pp 8"),
        (16, 0, NODE_IPS, "must cut -pp 8"),
        (0, 0, NODE_IPS, "must cut -pp 8"),
        (None, 4, NODE_IPS, "must cut -pp 8"),
    ],
)
def test_pipeline_slices_are_checked_once_either_flag_is_set(
    local, start, node_ips, error
):
    config = _config(local, start, node_ips=node_ips)
    if error is None:
        config._check_multinode_pp()
    else:
        with pytest.raises(ValueError, match=error):
            config._check_multinode_pp()


def test_a_single_node_pipeline_is_unchanged():
    # No new flag, loopback master, OS-assigned store port: as before.
    config = _config(None, 0, node_ips=[], master_ip="127.0.0.1", base_port=0)
    config._check_multinode_pp()
    assert not config.is_multinode_pp
    assert config.pp_size_local == 8
    config.pipeline_parallel_size = 1
    config._check_multinode_pp()
    assert not config.is_multinode_pp


@pytest.mark.parametrize(
    "overrides, error",
    [
        ({"dp": 2}, "does not support data parallelism"),
        ({"pcp": 2}, "does not support prefill context parallelism"),
        ({"dp_attention": True}, "does not support --enable-dp-attention"),
        ({"rapidserve": True}, "does not support RapidServe"),
        ({"fake_eplb": True}, "does not support --fake-eplb"),
        ({"master_ip": "127.0.0.1"}, "--data-parallel-master-ip"),
        ({"master_ip": "localhost"}, "--data-parallel-master-ip"),
        ({"base_port": 0}, "--data-parallel-base-port"),
        ({"node_ips": []}, "--pipeline-parallel-node-ips"),
        ({"node_ips": ["10.0.0.1"]}, "--pipeline-parallel-node-ips"),
        ({"node_ips": [*NODE_IPS, "10.0.0.3"]}, "--pipeline-parallel-node-ips"),
        ({"node_ips": ["10.0.0.1", "127.0.0.1"]}, "--pipeline-parallel-node-ips"),
        ({"node_ips": ["10.0.0.1", "fd00::2"]}, "IPv4"),
    ],
)
@pytest.mark.parametrize("start", [0, 4])
def test_a_pipeline_across_nodes_refuses(start, overrides, error):
    config = _config(4, start, **overrides)
    with pytest.raises(ValueError, match=error):
        config._check_multinode_pp()


def test_a_tp_group_across_nodes_still_refuses_a_pipeline():
    # TP must stay inside a stage's node; the TP check says so first.
    config = _config(4, 0, tp=2, tp_local=1)
    with pytest.raises(ValueError, match="does not support pipeline parallelism"):
        config._check_multinode_tp()


def test_config_refuses_a_bad_slice_before_loading_anything(tmp_path):
    # Config.__post_init__ runs the check, ahead of the model and AITER.
    (tmp_path / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "model_type": "llama", '
        '"num_hidden_layers": 8}'
    )
    pc = ParallelConfig(
        data_parallel_master_ip="10.0.0.1",
        data_parallel_base_port=29501,
        pipeline_parallel_size_local=4,
        pipeline_parallel_rank=2,
        pipeline_parallel_node_ips=list(NODE_IPS),
    )
    with pytest.raises(ValueError, match="must cut -pp 8"):
        Config(model=str(tmp_path), pipeline_parallel_size=8, parallel_config=pc)


def test_a_plugin_run_is_one_whole_pipeline_on_one_node():
    # vLLM's own ParallelConfig has none of the cross-node PP fields; the host
    # framework places the stages, so a plugin run is the single-node case.
    config = _config(None, 0, node_ips=[])
    config.parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank=0,
        data_parallel_master_ip="127.0.0.1",
        data_parallel_base_port=0,
    )
    config.plugin_config = SimpleNamespace(is_vllm=True)
    config._check_multinode_pp()
    config._check_multinode_pp_offload()
    assert config.pp_size_local == 8
    assert not config.is_multinode_pp
    assert config.pp_stage_ips == []


_MOONCAKE_PRODUCER = {"kv_connector": "mooncake", "kv_role": "kv_producer"}


@pytest.mark.parametrize("tier", ["lmcache_offload", "mooncake_store", "lmcache_mp"])
@pytest.mark.parametrize("multi", [False, True])
def test_a_pipeline_across_nodes_refuses_a_kv_offload_tier(tier, multi):
    # An LMCache tier's lookups reach its workers over ipc:// paths, and a
    # store tier times its saves on the head's clock: neither crosses a node.
    offload = {"kv_connector": tier, "kv_role": "offload"}
    kv_transfer_config = (
        {"kv_connector": "multi", "connectors": [_MOONCAKE_PRODUCER, offload]}
        if multi
        else offload
    )
    for start in (0, 4):
        config = _config(4, start, kv_transfer_config=kv_transfer_config)
        with pytest.raises(ValueError, match=rf"KV offload tier \({tier}\)"):
            config._check_multinode_pp_offload()
    # On one node the tier stays.
    _config(
        None, 0, node_ips=[], kv_transfer_config=kv_transfer_config
    )._check_multinode_pp_offload()


def test_a_pipeline_across_nodes_keeps_its_pd_transport():
    for start in (0, 4):
        config = _config(4, start, kv_transfer_config=_MOONCAKE_PRODUCER)
        config._check_multinode_pp_offload()
        _config(4, start)._check_multinode_pp_offload()


def test_config_refuses_an_offload_tier_before_loading_anything(tmp_path):
    # Config.__post_init__ runs it once the CLI's JSON is parsed, still ahead
    # of the model and AITER.
    (tmp_path / "config.json").write_text(
        '{"architectures": ["LlamaForCausalLM"], "model_type": "llama", '
        '"num_hidden_layers": 8}'
    )
    pc = ParallelConfig(
        data_parallel_master_ip="10.0.0.1",
        data_parallel_base_port=29501,
        pipeline_parallel_size_local=4,
        pipeline_parallel_rank=4,
        pipeline_parallel_node_ips=list(NODE_IPS),
    )
    with pytest.raises(ValueError, match=r"KV offload tier \(lmcache_offload\)"):
        Config(
            model=str(tmp_path),
            pipeline_parallel_size=8,
            parallel_config=pc,
            kv_transfer_config='{"kv_connector": "lmcache_offload", '
            '"kv_role": "offload"}',
        )


# ---------------------------------------------------------------------------
# CLI: the flags reach this node's ParallelConfig
# ---------------------------------------------------------------------------


def _node_args(start):
    return [
        "-tp",
        "1",
        "-pp",
        "8",
        "--pipeline-parallel-size-local",
        "4",
        "--pipeline-parallel-rank",
        str(start),
        "--pipeline-parallel-node-ips",
        "10.0.0.1, 10.0.0.2",
        "--data-parallel-master-ip",
        "10.0.0.1",
        "--data-parallel-base-port",
        "29501",
    ]


@pytest.mark.parametrize("start", [0, 4])
def test_cli_flags_reach_the_parallel_config(start):
    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = EngineArgs.from_cli_args(parser.parse_args(_node_args(start)))
    kwargs = args._get_engine_kwargs()
    pc = kwargs["parallel_config"]
    assert pc.pipeline_parallel_size_local == 4
    assert pc.pipeline_parallel_rank == start
    assert pc.pipeline_parallel_node_ips == NODE_IPS
    assert kwargs["pipeline_parallel_size"] == 8
    # Node topology travels only inside the ParallelConfig.
    for name in (
        "pipeline_parallel_size_local",
        "pipeline_parallel_rank",
        "pipeline_parallel_node_ips",
    ):
        assert name not in kwargs


def test_cli_defaults_leave_a_single_node_parallel_config():
    pc = EngineArgs(pipeline_parallel_size=8).parallel_config()
    assert pc.pipeline_parallel_size_local is None
    assert pc.pipeline_parallel_rank == 0
    assert pc.pipeline_parallel_node_ips == []


# ---------------------------------------------------------------------------
# The address plan: the same on both nodes, every channel bound exactly once
# ---------------------------------------------------------------------------


def _channel_plan(config):
    """Built from a node's config exactly as CoreManager builds it."""
    pc = config.parallel_config
    return build_internode_pp_channel_plan(
        pp_size=config.pipeline_parallel_size,
        pp_size_local=config.pp_size_local,
        node_ips=pc.pipeline_parallel_node_ips,
        master_port=pc.data_parallel_master_port,
    )


def test_both_nodes_derive_one_plan_after_the_engine_sockets():
    node0, node1 = _channel_plan(_config(4, 0)), _channel_plan(_config(4, 4))
    assert node0 == node1
    engines = build_internode_dp_socket_plan(engine_count=8, master_port=MASTER_PORT)
    engine_ports = {
        port for p in engines for port in (p.input_port, p.output_port, p.control_port)
    }
    channel_ports = {*node0.meta_ports, node0.token_port, node0.kv_status_port}
    assert engine_ports == set(range(29600, 29624))
    assert channel_ports == set(range(29624, 29634))
    assert node0.stage_ips == ("10.0.0.1",) * 4 + ("10.0.0.2",) * 4


@pytest.mark.parametrize("local, start", [(4, 0), (4, 4), (2, 6)])
def test_each_node_names_every_stages_node_as_the_channel_plan_does(local, start):
    # What a P/D producer advertises for its stages (`remote_pp_hosts`).
    node_ips = NODE_IPS if local == 4 else [f"10.0.0.{k}" for k in range(1, 5)]
    config = _config(local, start, node_ips=node_ips)
    assert config.pp_stage_ips == list(_channel_plan(config).stage_ips)
    assert config.pp_stage_ips == [node_ips[s // local] for s in range(8)]
    # Each stage's copy answers as its node does.
    stage = _stage_config(config, start + 1)
    assert stage.pp_stage_ips == config.pp_stage_ips


def test_a_single_node_pipeline_names_no_stage_nodes():
    assert _config(None, 0, node_ips=[]).pp_stage_ips == []
    assert _config(8, 0, node_ips=[]).pp_stage_ips == []


def test_each_stage_binds_only_what_it_receives_and_connects_to_the_rest():
    plan = _channel_plan(_config(4, 0))
    binds = {}
    for stage in range(8):
        meta, token, kv_status = plan.stage_addresses(stage)
        if stage == 0:
            # The head pushes to every stage at its node and pulls both
            # return channels on every interface.
            assert meta[1:] == [
                f"tcp://{NODE_IPS[s // 4]}:{29624 + s}" for s in range(1, 8)
            ]
            assert (token, kv_status) == ("tcp://0.0.0.0:29632", "tcp://0.0.0.0:29633")
            binds[0] = [token, kv_status]
        else:
            assert meta[stage] == f"tcp://0.0.0.0:{29624 + stage}"
            assert (token, kv_status) == (
                "tcp://10.0.0.1:29632",
                "tcp://10.0.0.1:29633",
            )
            binds[stage] = [meta[stage]]
    # Every endpoint is bound once, on the node that receives on it.
    bound_ports = sorted(int(a.rsplit(":", 1)[1]) for b in binds.values() for a in b)
    assert bound_ports == list(range(29625, 29634))


def _free_master_port(span):
    """A master port whose derived ports (from master + 100, ``span`` of
    them) are all free here."""
    for _ in range(100):
        base = random.randint(20000, 50000)
        probes = []
        try:
            for port in range(base, base + span):
                probe = socket.socket()
                probes.append(probe)
                probe.bind(("0.0.0.0", port))
        except OSError:
            continue
        finally:
            for probe in probes:
                probe.close()
        return base - engine_core_mgr.INTERNODE_DP_SOCKET_PORT_OFFSET
    pytest.skip("no free port range")


def test_stage_channels_carry_batches_completions_and_status_over_tcp():
    # Both "nodes" on this host; each stage uses only its own view.
    master_port = _free_master_port(8 * 3 + 8 + 2)
    plan = build_internode_pp_channel_plan(
        pp_size=8,
        pp_size_local=4,
        node_ips=["127.0.0.1", "127.0.0.1"],
        master_port=master_port,
    )
    ctx = zmq.Context()
    stages = [
        PPStageTransport(s, 8, *plan.stage_addresses(s), ctx=ctx) for s in range(8)
    ]
    try:
        head, last = stages[0], stages[7]
        batch = {"req_ids": (3, 5), "positions": [7, 9]}
        head.send_metadata(batch)
        for stage in stages[1:]:
            assert stage.recv_metadata(timeout_ms=5000) == batch
        last.send_completion(batch["req_ids"], {"tokens": [1, 2]})
        assert head.recv_completion(timeout_ms=5000) == ((3, 5), {"tokens": [1, 2]})
        stages[5].send_kv_status("saved")
        assert head.recv_kv_status(timeout_ms=5000) == [(5, "saved")]
    finally:
        for stage in stages:
            stage.close()
        ctx.term()


# ---------------------------------------------------------------------------
# GPU and global rank of each stage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", range(8))
def test_pp8_tp1_stage_runs_on_its_nodes_gpu_with_its_global_rank(stage):
    config = _stage_config(_config(4, stage // 4 * 4), stage)
    assert local_device_index(config, 0) == stage % 4
    assert worker_global_rank(config, 0) == stage


@pytest.mark.parametrize("stage", range(4))
@pytest.mark.parametrize("rank", range(2))
def test_pp4_tp2_stage_workers_fill_their_nodes_gpus(stage, rank):
    # Two stages of TP2 per 4-GPU node.
    config = _stage_config(_config(2, stage // 2 * 2, pp=4, tp=2), stage)
    assert local_device_index(config, rank) == stage % 2 * 2 + rank
    assert worker_global_rank(config, rank) == stage * 2 + rank


@pytest.mark.parametrize("stage", range(8))
def test_a_single_node_stage_keeps_its_global_index_as_gpu(stage):
    config = _stage_config(_config(None, 0, node_ips=[]), stage)
    assert local_device_index(config, 0) == worker_global_rank(config, 0) == stage


@pytest.mark.parametrize("local_dp", range(2))
def test_single_node_dp_placement_is_unchanged(local_dp):
    config = _config(None, 0, pp=1, tp=2, dp=2, node_ips=[])
    config.parallel_config.data_parallel_rank = local_dp
    config.parallel_config.data_parallel_rank_local = local_dp
    for rank in range(2):
        assert local_device_index(config, rank) == local_dp * 2 + rank
        assert worker_global_rank(config, rank) == local_dp * 2 + rank


def test_a_stage_binds_its_groups_to_its_nodes_gpu(monkeypatch):
    from aiter_stub import stubbed_aiter

    with stubbed_aiter("aiter.ops", "aiter.ops.communication"):
        from atom.distributed import pp_comm
    world = {}
    monkeypatch.setattr(pp_comm, "set_custom_all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(
        pp_comm, "init_distributed_environment", lambda **k: world.update(k)
    )
    monkeypatch.setattr(
        pp_comm, "ensure_model_parallel_initialized", lambda *a, **k: None
    )
    pp_comm.init_pp_aware_dist_env(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=8,
        global_rank=5,
        world_size=8,
        distributed_init_method="tcp://10.0.0.1:29501",
        local_rank=1,
    )
    assert (world["rank"], world["local_rank"], world["world_size"]) == (5, 1, 8)


# ---------------------------------------------------------------------------
# Startup waits as long as a cross-node TP group's does
# ---------------------------------------------------------------------------


def test_a_stage_store_client_and_groups_wait_for_the_slower_node(monkeypatch):
    from torch.distributed import distributed_c10d

    monkeypatch.setenv("ATOM_MULTINODE_TP_TIMEOUT_S", "3600")
    monkeypatch.setenv("ATOM_MULTINODE_PG_TIMEOUT_S", "5400")
    clients = []
    monkeypatch.setattr(dist, "TCPStore", lambda *a, **k: clients.append((a, k)))
    monkeypatch.setattr(distributed_c10d, "default_pg_timeout", timedelta(minutes=30))
    monkeypatch.setattr(
        distributed_c10d, "default_pg_nccl_timeout", timedelta(minutes=10)
    )
    config = _stage_config(_config(4, 4), 6)
    assert config.is_multinode_pp and not config.is_multinode_tp
    tp_multinode.managed_store_client(config)
    assert clients == [
        (("10.0.0.1", 29501), {"is_master": False, "timeout": timedelta(hours=1)})
    ]
    # The groups take the process-group floor, as a cross-node TP group's do.
    raise_timeout_if_model_spans_nodes(config)
    assert distributed_c10d.default_pg_timeout == timedelta(minutes=90)
    assert distributed_c10d.default_pg_nccl_timeout == timedelta(minutes=90)


# ---------------------------------------------------------------------------
# CoreManager on either node, with stand-in engines
# ---------------------------------------------------------------------------


class _FakeEngine:
    """Speaks EngineCore's side of the engine sockets: connects them, reports
    READY, then records what arrives until SHUTDOWN. Also stands in for the
    engine's process, which CoreManager starts, polls, joins and closes."""

    def __init__(self, addresses, serve=True):
        self.addresses = addresses
        self.received = []
        self.shut_down = threading.Event()
        self._serves = serve
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        if self._serves:
            self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout=None):
        if self._thread.ident is not None:
            self._thread.join(timeout)

    def terminate(self):
        pass

    kill = terminate
    close = terminate

    def _serve(self):
        with zmq.Context() as ctx:
            sockets = {
                name: ctx.socket(kind)
                for name, kind in (
                    ("input_address", zmq.DEALER),
                    ("control_address", zmq.DEALER),
                    ("output_address", zmq.PUSH),
                )
            }
            for name, sock in sockets.items():
                sock.setsockopt(zmq.LINGER, 0)
                sock.connect(self.addresses[name])
            # What EngineCore sends before the ROUTER can address it.
            sockets["input_address"].send(b"")
            sockets["control_address"].send(b"")
            sockets["output_address"].send(
                pickle.dumps((EngineCoreRequestType.READY, {"max_pool_tokens": 64}))
            )
            poller = zmq.Poller()
            poller.register(sockets["input_address"], zmq.POLLIN)
            poller.register(sockets["control_address"], zmq.POLLIN)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and not self.shut_down.is_set():
                for sock, _ in poller.poll(100):
                    message = sock.recv()
                    if sock is sockets["control_address"]:
                        request_type, _ = pickle.loads(message)
                        if request_type == EngineCoreRequestType.SHUTDOWN:
                            self.shut_down.set()
                    else:
                        self.received.append(message)
            for sock in sockets.values():
                sock.close()


def _ipc_addresses():
    return {
        name: get_open_zmq_ipc_path()
        for name in ("input_address", "output_address", "control_address")
    }


@pytest.fixture
def fake_launch(monkeypatch):
    """Install stand-in engines for the stages a CoreManager launches on its
    node; returns the ``(stage config, engine)`` pairs as they launch."""
    # CoreManager would otherwise make spawn this process's default.
    monkeypatch.setattr(torch.multiprocessing, "set_start_method", lambda *a, **k: None)

    def install(serve=True):
        engines = []

        def launch(rank_config, dp_rank, local_dp_rank, addresses=None):
            engine = _FakeEngine(addresses or _ipc_addresses(), serve=serve)
            engines.append((rank_config, engine))
            return engine, engine.addresses, local_dp_rank

        monkeypatch.setattr(engine_core_mgr, "launch_engine_core", launch)
        return engines

    return install


def _close(manager):
    manager.close()
    for thread in manager.output_threads:
        thread.join(5)
    manager.ctx.destroy(linger=0)


def _build_manager(config, timeout=30):
    """CoreManager(config), off the test thread so a missing engine fails the
    test instead of hanging it."""
    built = {}

    def build():
        try:
            built["manager"] = CoreManager(config)
        except BaseException as e:  # noqa: BLE001 - reported below
            built["error"] = e

    thread = threading.Thread(target=build, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "CoreManager did not finish starting"
    if "error" in built:
        raise built["error"]
    return built["manager"]


def test_worker_node_launches_its_stages_against_the_coordinator(fake_launch):
    # Its engines are not served here: the worker node only waits for them.
    launched = fake_launch(serve=False)
    manager = _build_manager(_config(4, 4))
    try:
        assert not manager.is_coordinator
        assert manager.input_sockets == [] and manager.output_sockets == []
        # It serves no store; its runners use the coordinator's.
        assert manager._distributed_stores == []
        configs = [rank_config for rank_config, _ in launched]
        assert [c.parallel_config.pipeline_parallel_rank for c in configs] == [
            4,
            5,
            6,
            7,
        ]
        plan = _channel_plan(_config(4, 0))
        for config, engine in launched:
            stage = config.parallel_config.pipeline_parallel_rank
            pc = config.parallel_config
            assert pc._managed_distributed_store
            assert (pc.data_parallel_rank, pc.data_parallel_rank_local) == (0, 0)
            ports = (29600 + 3 * stage, 29601 + 3 * stage, 29602 + 3 * stage)
            assert engine.addresses == {
                "input_address": f"tcp://10.0.0.1:{ports[0]}",
                "output_address": f"tcp://10.0.0.1:{ports[1]}",
                "control_address": f"tcp://10.0.0.1:{ports[2]}",
            }
            assert (
                pc.pp_meta_addrs,
                pc.pp_token_addr,
                pc.pp_kv_status_addr,
            ) == plan.stage_addresses(stage)
            assert local_device_index(config, 0) == stage - 4
            assert worker_global_rank(config, 0) == stage
    finally:
        _close(manager)


def _remote_engines(master_port, stages):
    """What the other node's stages connect to the coordinator."""
    plan = build_internode_dp_socket_plan(engine_count=8, master_port=master_port)
    engines = {}
    for stage in stages:
        p = plan[stage]
        engines[stage] = _FakeEngine(
            {
                "input_address": f"tcp://127.0.0.1:{p.input_port}",
                "output_address": f"tcp://127.0.0.1:{p.output_port}",
                "control_address": f"tcp://127.0.0.1:{p.control_port}",
            }
        )
        engines[stage].start()
    return engines


def test_coordinator_serves_every_stage_and_shuts_down_the_remote_ones(fake_launch):
    launched = fake_launch()
    master_port = _free_master_port(8 * 3)
    config = _config(
        4,
        0,
        master_ip="127.0.0.1",
        master_port=master_port,
        base_port=get_open_port(),
        node_ips=["127.0.0.1", "127.0.0.1"],
    )
    remote = _remote_engines(master_port, range(4, 8))
    manager = _build_manager(config)
    try:
        assert manager.is_coordinator
        assert len(manager._distributed_stores) == 1
        # Sockets for all eight stages; processes only for this node's four.
        assert len(manager.input_sockets) == len(manager.output_sockets) == 8
        assert len(manager.engine_core_processes) == 4
        assert manager.global_engine_count == 8
        local = {c.parallel_config.pipeline_parallel_rank: e for c, e in launched}
        assert sorted(local) == [0, 1, 2, 3]
        head = next(
            c for c, _ in launched if c.parallel_config.pipeline_parallel_rank == 0
        )
        assert head.parallel_config.pp_token_addr.startswith("tcp://0.0.0.0:")
        assert head.parallel_config.pp_meta_addrs[4].startswith("tcp://127.0.0.1:")
        # Requests enter at the head only.
        manager._send_request(0, b"request")
        deadline = time.monotonic() + 10
        while not local[0].received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert local[0].received == [b"request"]
        assert not any(e.received for s, e in {**local, **remote}.items() if s)
    finally:
        _close(manager)
    # Every stage, local or on the other node, was told to shut down.
    for engine in [*local.values(), *remote.values()]:
        assert engine.shut_down.wait(10)


def test_single_node_pipeline_keeps_ipc_channels(fake_launch):
    launched = fake_launch()
    config = _config(None, 0, node_ips=[], master_ip="127.0.0.1", base_port=0, pp=4)
    manager = _build_manager(config)
    try:
        assert manager.is_coordinator
        assert len(manager.input_sockets) == len(manager.engine_core_processes) == 4
        views = {
            (
                tuple(c.parallel_config.pp_meta_addrs),
                c.parallel_config.pp_token_addr,
                c.parallel_config.pp_kv_status_addr,
            )
            for c, _ in launched
        }
        # One shared set of IPC endpoints, as before.
        assert len(views) == 1
        meta, token, kv_status = views.pop()
        assert all(a.startswith("ipc://") for a in (*meta, token, kv_status))
        assert all(
            e.addresses["input_address"].startswith("ipc://") for _, e in launched
        )
    finally:
        _close(manager)


def test_store_runs_only_on_the_node_of_stage_zero(monkeypatch):
    started = []
    monkeypatch.setattr(
        dist,
        "TCPStore",
        lambda *a, **k: started.append(a) or SimpleNamespace(port=a[1]),
    )
    for start in (0, 4):
        manager = CoreManager.__new__(CoreManager)
        manager.label = "test"
        manager._distributed_stores = []
        config = _config(4, start)
        manager._start_distributed_store(config, multinode=True)
        assert config.parallel_config._managed_distributed_store
    assert started == [("10.0.0.1", 29501)]
