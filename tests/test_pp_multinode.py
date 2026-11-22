# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One pipeline across nodes: what Config accepts, the TCP plan both nodes
derive for the engine and stage sockets, CoreManager on either node, and each
stage's GPU and global rank. GPU-free; the engines are stand-ins that speak
EngineCore's socket protocol."""

import argparse
import copy

import pytest

from atom.config import (
    Config,
    KVEventsConfig,
    ParallelConfig,
    local_device_index,
    worker_global_rank,
)
from atom.model_engine.arg_utils import EngineArgs
from atom.model_engine.engine_core_mgr import (
    build_internode_dp_socket_plan,
    build_internode_pp_channel_plan,
)

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


_MORIIO_PRODUCER = {"kv_connector": "moriio", "kv_role": "kv_producer"}
_MOONCAKE_PRODUCER = {"kv_connector": "mooncake", "kv_role": "kv_producer"}


@pytest.mark.parametrize("tier", ["lmcache_offload", "mooncake_store", "lmcache_mp"])
@pytest.mark.parametrize("multi", [False, True])
def test_a_pipeline_across_nodes_refuses_a_kv_offload_tier(tier, multi):
    # An LMCache tier's lookups reach its workers over ipc:// paths, and a
    # store tier times its saves on the head's clock: neither crosses a node.
    offload = {"kv_connector": tier, "kv_role": "offload"}
    kv_transfer_config = (
        {"kv_connector": "multi", "connectors": [_MORIIO_PRODUCER, offload]}
        if multi
        else offload
    )
    for start in (0, 4):
        config = _config(4, start, kv_transfer_config=kv_transfer_config)
        with pytest.raises(ValueError, match=rf"KV offload tier \({tier}\)"):
            config._check_multinode_pp_kv_connectors()
    # On one node the tier stays.
    _config(
        None, 0, node_ips=[], kv_transfer_config=kv_transfer_config
    )._check_multinode_pp_kv_connectors()


def test_a_pipeline_across_nodes_keeps_moriio_and_refuses_mooncake():
    # A Mooncake producer advertises one host for all of its stages.
    for start in (0, 4):
        _config(4, start)._check_multinode_pp_kv_connectors()
        _config(
            4, start, kv_transfer_config=_MORIIO_PRODUCER
        )._check_multinode_pp_kv_connectors()
        with pytest.raises(ValueError, match="mooncake connector"):
            _config(
                4, start, kv_transfer_config=_MOONCAKE_PRODUCER
            )._check_multinode_pp_kv_connectors()
    # On one node Mooncake stays.
    _config(
        None, 0, node_ips=[], kv_transfer_config=_MOONCAKE_PRODUCER
    )._check_multinode_pp_kv_connectors()


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


# ---------------------------------------------------------------------------
# Startup waits as long as a cross-node TP group's does
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CoreManager on either node, with stand-in engines
# ---------------------------------------------------------------------------
