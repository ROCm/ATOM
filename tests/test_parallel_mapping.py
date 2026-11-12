# SPDX-License-Identifier: MIT
"""Configuration-time topology, launch ranks, and runtime group binding."""

import copy
import sys
from dataclasses import FrozenInstanceError
from types import ModuleType, SimpleNamespace

import pytest
from conftest import atom_config_double

from atom.config import Config
from atom.distributed.parallel_mapping import bind_worker_parallel_mapping
from atom.model_engine.engine_core_mgr import iter_dp_rank_assignments
from atom.moe_config import MoEBackendConfig, get_moe_execution_plan
from atom.parallel_mapping import (
    apply_parallel_mapping,
    get_parallel_mapping,
    resolve_parallel_mapping,
)
from atom.plugin.config import (
    PluginConfig,
    _build_sglang_parallel_config,
    _normalize_sglang_parallel_config,
)


def _config(tp=1, dp=1, local_dp=None, **overrides):
    return atom_config_double(
        tensor_parallel_size=tp,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_size_local=local_dp or dp,
            data_parallel_rank=0,
            is_multinode_dp=False,
        ),
        moe_backend_config=MoEBackendConfig(mori_available=True, pcp_merge=True),
        **overrides,
    )


def test_native_dp_attention_mapping_precedes_launch_and_survives_spawn_copy():
    config = _config(tp=8, enable_dp_attention=True, enable_expert_parallel=True)
    Config._init_parallel_mapping(config)
    topology = config.parallel_mapping
    plan = config.moe_execution_plan
    assert (topology.tp_size, topology.dp_size) == (1, 8)
    assert config.tensor_parallel_size == 8
    assert plan.mapping.ep_size == 8 and plan.transport == "mori"
    assert topology.for_rank(7).moe.ep_rank == 7
    assert iter_dp_rank_assignments(config) == [(rank, rank) for rank in range(8)]

    apply_parallel_mapping(config)
    apply_parallel_mapping(config)
    assert config.tensor_parallel_size == 1
    assert config.parallel_config.data_parallel_size == 8
    assert get_parallel_mapping(config) is topology
    assert get_moe_execution_plan(config) is plan
    worker = copy.deepcopy(config)
    assert worker.parallel_mapping == topology
    assert worker.moe_execution_plan == plan
    assert worker.worker_moe_execution_plan is None
    with pytest.raises(FrozenInstanceError):
        topology.dp_size = 1


@pytest.mark.parametrize("world", [4, 8, 16])
def test_translated_sglang_dp_attention_is_not_folded_twice(world):
    tp, pcp, dp, _, _ = _normalize_sglang_parallel_config(
        tp_size=world, dp_size=2, tp_rank=world - 1, enable_dp_attention=True
    )
    config = _config(
        tp=tp,
        dp=dp,
        prefill_context_parallel_size=pcp,
        enable_dp_attention=True,
        enable_expert_parallel=True,
        plugin_config=PluginConfig(is_plugin_mode=True, is_sglang=True),
    )
    topology = get_parallel_mapping(config)
    assert (topology.tp_size, topology.dp_size, topology.world_size) == (
        1,
        world,
        world,
    )
    assert topology.for_rank(world - 1).moe.ep_rank == world - 1


@pytest.mark.parametrize("rank", range(8))
def test_sglang_worker_rank_is_not_a_native_node_slice_offset(rank):
    config = atom_config_double(
        parallel_config=_build_sglang_parallel_config(8, rank),
        plugin_config=PluginConfig(is_plugin_mode=True, is_sglang=True),
        enable_dp_attention=True,
        enable_expert_parallel=True,
    )
    topology = get_parallel_mapping(config)
    assert topology.dp_rank_offset == 0
    assert topology.for_rank(rank).moe.ep_rank == rank


def test_multinode_launch_assignments_share_the_normalized_mapping():
    config = _config(tp=2, dp=8, local_dp=4, enable_dp_attention=True)
    config.parallel_config.is_multinode_dp = True
    config.parallel_config.data_parallel_rank = 4
    topology = get_parallel_mapping(config)
    assert (topology.dp_size, topology.local_dp_size, topology.dp_rank_offset) == (
        16,
        8,
        8,
    )
    before = iter_dp_rank_assignments(config)
    apply_parallel_mapping(config)
    assert (
        before
        == iter_dp_rank_assignments(config)
        == [(8 + rank, rank) for rank in range(8)]
    )


def test_global_layout_projects_dp_pp_pcp_tp_and_expert_groups():
    config = _config(
        tp=4,
        dp=2,
        local_dp=1,
        pipeline_parallel_size=2,
        prefill_context_parallel_size=2,
        enable_expert_parallel=True,
        moe_ep_flatten_tp_across_dp=True,
    )
    topology = get_parallel_mapping(config)
    for dp in range(2):
        for pp in range(2):
            for pcp in range(2):
                for tp in range(4):
                    global_rank = ((dp * 2 + pp) * 2 + pcp) * 4 + tp
                    rank = topology.for_rank(global_rank)
                    assert (
                        rank.dp_rank,
                        rank.pp_rank,
                        rank.pcp_rank,
                        rank.tp_rank,
                    ) == (dp, pp, pcp, tp)
                    assert (
                        rank.moe.ep_size,
                        rank.moe.ep_rank,
                        rank.moe.local_ep_size,
                    ) == (16, (dp * 2 + pcp) * 4 + tp, 8)
                    assert (
                        topology.group_ranks(global_rank, "ep").index(global_rank)
                        == rank.moe.ep_rank
                    )
    assert topology.group_ranks(22, "tp") == (20, 21, 22, 23)
    assert topology.group_ranks(22, "dp") == (6, 22)
    assert topology.group_ranks(22, "pcp") == (18, 22)
    assert topology.group_ranks(22, "pp") == (22, 30)


def test_simulated_dp_resolves_physical_peers_and_logical_experts_at_config_time(
    monkeypatch,
):
    import torch

    config = _config(
        tp=16, enable_dp_attention=True, enable_expert_parallel=True, fake_eplb=True
    )
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    Config._init_parallel_mapping(config)
    topology = config.parallel_mapping
    plan = config.moe_execution_plan
    assert (topology.world_size, topology.dp_size, topology.dp_logical_size) == (
        2,
        2,
        16,
    )
    assert (plan.mapping.ep_size, plan.mapping.dp_logical_ratio) == (16, 8)
    assert plan.transport == "dp_gather_scatter"
    assert not plan.supports_tbo
    apply_parallel_mapping(config)
    assert config.parallel_config.data_parallel_size == 2
    assert config.dp_logical_size == 16
    assert iter_dp_rank_assignments(config) == [(0, 0), (1, 1)]


def test_simulated_tp_keeps_logical_shards_and_physical_group_members():
    topology = resolve_parallel_mapping(
        _config(tp=8, fake_eplb=True), visible_devices=4
    )
    assert (topology.tp_size, topology.tp_world_size, topology.world_size) == (8, 4, 4)
    assert topology.for_rank(3).moe.tp_size == 8
    assert topology.group_ranks(3, "tp") == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="Global rank"):
        topology.for_rank(4)


def _install_runtime_groups(
    monkeypatch, topology, global_rank, *, omit_singleton_pcp=False
):
    rank = topology.for_rank(global_rank)
    runtime = {}
    for dimension, size, index in (
        ("tp", topology.tp_size, rank.tp_rank),
        ("dp", topology.dp_size, rank.dp_rank),
        ("pp", topology.pp_size, rank.pp_rank),
        ("pcp", topology.pcp_size, rank.pcp_rank),
        ("ep", rank.moe.ep_size, rank.moe.ep_rank),
    ):
        runtime[dimension] = SimpleNamespace(
            world_size=size,
            rank_in_group=index,
            ranks=list(topology.group_ranks(global_rank, dimension)),
        )
    groups = ModuleType("aiter.dist.parallel_state")
    for dimension, group in runtime.items():
        setattr(groups, f"get_{dimension}_group", lambda group=group: group)
    groups.get_prefill_context_model_parallel_world_size = lambda: runtime[
        "pcp"
    ].world_size
    groups.get_prefill_context_model_parallel_rank = lambda: runtime[
        "pcp"
    ].rank_in_group
    if omit_singleton_pcp:

        def refuse_pcp():
            raise AssertionError("TP-only vLLM reuse has no PCP coordinator")

        groups.get_pcp_group = refuse_pcp
    aiter, dist = ModuleType("aiter"), ModuleType("aiter.dist")
    aiter.dist, dist.parallel_state = dist, groups
    for name, module in (
        ("aiter", aiter),
        ("aiter.dist", dist),
        ("aiter.dist.parallel_state", groups),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return runtime


def test_worker_binds_the_saved_plan_without_resolving_transport_again(monkeypatch):
    from atom import moe_config

    config = _config(
        tp=4, dp=2, enable_expert_parallel=True, moe_ep_flatten_tp_across_dp=True
    )
    topology = get_parallel_mapping(config)
    plan = get_moe_execution_plan(config)
    _install_runtime_groups(monkeypatch, topology, 6)
    monkeypatch.setattr(
        moe_config,
        "resolve_moe_execution_plan",
        lambda *_args, **_kwargs: pytest.fail("worker re-resolved transport"),
    )
    config.moe_all2all_backend = "none"
    bind_worker_parallel_mapping(config, global_rank=6)
    assert config.worker_parallel_mapping.tp_rank == 2
    assert config.worker_parallel_mapping.dp_rank == 1
    assert config.worker_moe_execution_plan.mapping.ep_rank == 6
    assert config.worker_moe_execution_plan.transport == plan.transport == "mori"
    assert get_moe_execution_plan(config) is config.worker_moe_execution_plan


@pytest.mark.parametrize(
    "dimension, mismatch", [("tp", "size"), ("dp", "rank"), ("ep", "order")]
)
def test_group_disagreement_fails_before_model_construction(
    monkeypatch, dimension, mismatch
):
    config = _config(
        tp=4, dp=2, enable_expert_parallel=True, moe_ep_flatten_tp_across_dp=True
    )
    topology = get_parallel_mapping(config)
    get_moe_execution_plan(config)
    groups = _install_runtime_groups(monkeypatch, topology, 6)
    if mismatch == "size":
        groups[dimension].world_size *= 2
    elif mismatch == "rank":
        groups[dimension].rank_in_group = 0
    else:
        groups[dimension].ranks.reverse()
    with pytest.raises(ValueError, match="must match"):
        bind_worker_parallel_mapping(config, global_rank=6)
    assert config.worker_moe_execution_plan is None


def test_vllm_tp_reuse_binds_without_a_singleton_pcp_coordinator(monkeypatch):
    config = _config(
        tp=4, plugin_config=PluginConfig(is_plugin_mode=True, is_vllm=True)
    )
    topology = get_parallel_mapping(config)
    _install_runtime_groups(monkeypatch, topology, 2, omit_singleton_pcp=True)
    bind_worker_parallel_mapping(config, global_rank=2)
    assert config.worker_moe_execution_plan.mapping.tp_rank == 2


def test_pcp_merge_setting_is_saved_before_worker_environment_changes(monkeypatch):
    config = atom_config_double(tensor_parallel_size=2, prefill_context_parallel_size=2)
    monkeypatch.setenv("ATOM_PCP_MOE_MERGE", "1")
    Config._init_moe_backend(config)
    topology = get_parallel_mapping(config)
    monkeypatch.setenv("ATOM_PCP_MOE_MERGE", "0")
    assert get_parallel_mapping(config) is topology
    assert topology.for_rank(3).moe.tp_size == 4


@pytest.mark.parametrize(
    "framework,reuse", [("sglang", False), ("vllm", True), ("vllm", False)]
)
def test_real_plugin_launcher_binds_after_group_creation_or_reuse(
    monkeypatch, framework, reuse
):
    from unittest.mock import Mock

    from atom.plugin.register import init_aiter_dist

    sglang = framework == "sglang"
    config = _config(
        tp=2,
        dp=1 if sglang else 2,
        prefill_context_parallel_size=2 if sglang else 1,
        enable_expert_parallel=not sglang and not reuse,
        moe_ep_flatten_tp_across_dp=not sglang and not reuse,
        plugin_config=PluginConfig(
            is_plugin_mode=True,
            is_sglang=sglang,
            is_vllm=not sglang,
            rank=3 if sglang else 1,
            sglang_aiter_rank_id=3,
            sglang_dist_init_addr="127.0.0.1:29500",
        ),
    )
    config.parallel_config.data_parallel_rank = 0 if sglang else 1
    config.parallel_config.data_parallel_master_ip = "127.0.0.1"
    config.parallel_config.data_parallel_master_port = 29500
    topology = get_parallel_mapping(config)
    get_moe_execution_plan(config)
    _install_runtime_groups(monkeypatch, topology, 3, omit_singleton_pcp=not sglang)
    initialize = Mock()
    sys.modules["aiter"].init_dist_env = initialize
    utilities = ModuleType("aiter.dist.utils")
    utilities.get_distributed_init_method = lambda ip, port: f"tcp://{ip}:{port}"
    monkeypatch.setitem(sys.modules, "aiter.dist.utils", utilities)
    reuse_module = ModuleType("atom.plugin.vllm.tp_group_reuse")
    reuse_module.init_aiter_dist_from_vllm = Mock(return_value=reuse)
    monkeypatch.setitem(sys.modules, reuse_module.__name__, reuse_module)

    init_aiter_dist(config)

    assert config.worker_parallel_mapping.global_rank == 3
    assert config.worker_moe_execution_plan is not None
    if reuse:
        initialize.assert_not_called()
        reuse_module.init_aiter_dist_from_vllm.assert_called_once_with(2)
    else:
        initialize.assert_called_once()
        assert (
            initialize.call_args.kwargs["tensor_model_parallel_size"]
            == topology.tp_size
        )
        assert initialize.call_args.kwargs["data_parallel_size"] == topology.dp_size
        assert (
            initialize.call_args.kwargs["prefill_context_model_parallel_size"]
            == topology.pcp_size
        )
