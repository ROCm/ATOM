# SPDX-License-Identifier: MIT
"""Entry points must finish topology normalization before mapping is cached."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import atom_config_double

from atom.config import Config, ParallelConfig
from atom.model_engine.engine_core_mgr import iter_dp_rank_assignments
from atom.moe_config import get_moe_execution_plan
from atom.parallel_mapping import get_parallel_mapping
from atom.plugin.config import PluginConfig, _generate_atom_config_from_vllm_config


@pytest.mark.parametrize("explicit", [False, True])
def test_legacy_dp_kwargs_are_normalized_before_real_config(monkeypatch, explicit):
    from atom import config as config_module
    from atom.model_engine import llm_engine

    monkeypatch.setattr(config_module, "QuantizationConfig", lambda *a, **k: None)
    monkeypatch.setattr(llm_engine, "InputOutputProcessor", Mock())
    monkeypatch.setattr(llm_engine, "CoreManager", Mock())
    monkeypatch.setattr(llm_engine, "WorkerGrammarPreflight", Mock())
    kwargs = {
        "data_parallel_size": 4 if explicit else 2,
        "data_parallel_master_port": 31000,
    }
    if explicit:
        kwargs["parallel_config"] = ParallelConfig(
            data_parallel_size=2, data_parallel_master_port=32000
        )
    engine = llm_engine.LLMEngine(
        str(Path(__file__).parent / "models/deepseek_v41/fixtures"),
        tokenizer=SimpleNamespace(bos_token_id=1, eos_token_id=2),
        kv_cache_dtype="bf16",
        index_cache_dtype="fp8",
        enforce_eager=True,
        enable_dp_attention=True,
        **kwargs,
    )
    config = engine.config
    assert config.parallel_mapping.dp_size == engine.data_parallel_size == 2
    assert iter_dp_rank_assignments(config) == [(0, 0), (1, 1)]
    assert config.parallel_mapping.global_rank(0, 1) == 1
    assert config.parallel_config.data_parallel_master_port == (
        32000 if explicit else 31000
    )


@pytest.mark.parametrize("rank", range(4))
def test_vllm_translation_carries_pipeline_stages(monkeypatch, rank):
    import sys
    from types import ModuleType

    from atom import config as config_module

    # The V4.1 cache bridge imports vLLM GPU layers; this non-V4.1 topology
    # translation does not use the bridge's model or cache implementation.
    bridge = ModuleType("atom.plugin.vllm.deepseek_v41_bridge")
    bridge.is_deepseek_v41_vllm_config = lambda config: False
    bridge.v41_kv_cache_dtype = Mock()
    monkeypatch.setitem(sys.modules, bridge.__name__, bridge)

    parallel = SimpleNamespace(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        rank=rank,
        data_parallel_size=1,
        data_parallel_size_local=1,
        data_parallel_rank=0,
        enable_expert_parallel=False,
    )
    external = SimpleNamespace(
        model_config=SimpleNamespace(
            model="test-model",
            max_model_len=4096,
            hf_config=SimpleNamespace(architectures=["LlamaForCausalLM"]),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4096, max_num_seqs=16),
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            gpu_memory_utilization=0.9,
            block_size=16,
            num_gpu_blocks=16,
            enable_prefix_caching=False,
        ),
        parallel_config=parallel,
        compilation_config=SimpleNamespace(mode=0),
        quant_config=None,
    )
    monkeypatch.setattr(config_module, "Config", atom_config_double)
    config = _generate_atom_config_from_vllm_config(external)
    mapping = get_parallel_mapping(config)
    pp, tp = divmod(rank, mapping.stage_size)
    projected = mapping.for_rank(mapping.global_rank(tp, 0, pp))
    assert config.pipeline_parallel_size == mapping.pp_size == 2
    assert (projected.pp_rank, projected.tp_rank) == divmod(rank, 2)


@pytest.mark.parametrize("dp,device_map", [(2, False), (1, True)])
def test_rlhf_isolated_worker_plan_is_saved_before_binding(monkeypatch, dp, device_map):
    import torch
    from aiter_stub import atom_under_stubs
    from test_parallel_mapping import _install_runtime_groups

    if device_map:
        monkeypatch.setenv("VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER", "0,1")
    else:
        monkeypatch.delenv("VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER", raising=False)
    config = atom_config_double(
        model="test-model",
        tensor_parallel_size=2,
        parallel_config=ParallelConfig(data_parallel_size=dp),
        runner_qualname="atom.rollout.model_runner_ext.RLHFModelRunner",
        enable_expert_parallel=True,
    )
    Config._init_parallel_mapping(config)
    launch = config.parallel_mapping
    plan = get_moe_execution_plan(config)
    assert launch.dp_size == dp
    assert launch.worker_topology().dp_size == plan.mapping.dp_size == 1
    assert plan.transport == "local"
    config.parallel_config.data_parallel_rank_local = dp - 1

    with atom_under_stubs() as load, monkeypatch.context() as patch:
        module = load("atom.rollout.model_runner_ext")
        _install_runtime_groups(patch, launch.worker_topology(), 1)
        initialize = Mock()
        patch.setattr(module, "init_dist_env", initialize)
        patch.setattr(torch.cuda, "device_count", lambda: 4)
        patch.setattr(torch.cuda, "set_device", Mock())
        patch.setattr(torch, "zeros", Mock())
        group = module.get_tp_group()
        group.device_communicator = SimpleNamespace(ca_comm=Mock())
        group.device_communicator.ca_comm._pool = {
            "input": SimpleNamespace(tensor=None)
        }
        patch.setattr(module, "get_tp_group", lambda: group)
        runner = object.__new__(module.RLHFModelRunner)
        runner._setup_device_and_distributed(1, config)

    assert initialize.call_args.kwargs["data_parallel_size"] == 1
    assert initialize.call_args.kwargs["local_rank"] == (dp - 1) * 2 + 1
    assert config.worker_parallel_mapping.global_rank == 1
    assert config.worker_moe_execution_plan.mapping.dp_size == 1
    assert config.worker_moe_execution_plan.mapping.ep_rank == 1
    assert config.worker_moe_execution_plan.transport == plan.transport


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("reuse", [True, False])
def test_vllm_pp_binds_the_second_pipeline_stage(monkeypatch, rank, reuse):
    import sys
    from types import ModuleType

    from test_parallel_mapping import _install_runtime_groups

    from atom.plugin.register import init_aiter_dist

    config = atom_config_double(
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
        plugin_config=PluginConfig(is_plugin_mode=True, is_vllm=True, rank=rank),
    )
    topology = get_parallel_mapping(config)
    _install_runtime_groups(monkeypatch, topology, rank, omit_singleton_pcp=True)
    reuse_module = ModuleType("atom.plugin.vllm.tp_group_reuse")
    reuse_module.init_aiter_dist_from_vllm = Mock(return_value=reuse)
    monkeypatch.setitem(sys.modules, reuse_module.__name__, reuse_module)
    sys.modules["aiter"].init_dist_env = Mock()
    utilities = ModuleType("aiter.dist.utils")
    utilities.get_distributed_init_method = lambda ip, port: f"tcp://{ip}:{port}"
    monkeypatch.setitem(sys.modules, utilities.__name__, utilities)
    pp = ModuleType("atom.distributed.pp_comm")
    pp.init_pp_aware_dist_env = Mock()
    monkeypatch.setitem(sys.modules, pp.__name__, pp)
    init_aiter_dist(config)
    assert config.worker_parallel_mapping.pp_rank == 1
    assert config.worker_parallel_mapping.tp_rank == rank % 2
    if reuse:
        pp.init_pp_aware_dist_env.assert_not_called()
    else:
        assert (
            pp.init_pp_aware_dist_env.call_args.kwargs["pipeline_model_parallel_size"]
            == 2
        )
        assert pp.init_pp_aware_dist_env.call_args.kwargs["global_rank"] == rank
