# SPDX-License-Identifier: MIT
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double

from atom.config import Config
from atom.distributed.ulysses_sp import check_sp_attention_layer
from atom.models.minimax_m3.image_processing import image_input_refusal


@pytest.mark.parametrize("world", [1, 2, 4, 8, 16, 32])
def test_gqa_kv_heads_can_replicate_to_query_owners(world):
    config = atom_config_double(
        sequence_parallel_size=world,
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        hf_config=SimpleNamespace(num_attention_heads=32, num_key_value_heads=4),
    )
    Config._init_sequence_parallel(config)
    assert config.prefill_context_parallel_size == world


@pytest.mark.parametrize(
    "query_heads,kv_heads,world", [(30, 4, 8), (32, 3, 8), (32, 6, 4)]
)
def test_gqa_rejects_partial_heads(query_heads, kv_heads, world):
    config = atom_config_double(
        sequence_parallel_size=world,
        tensor_parallel_size=1,
        prefill_context_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        hf_config=SimpleNamespace(
            num_attention_heads=query_heads, num_key_value_heads=kv_heads
        ),
    )
    with pytest.raises(ValueError, match="incompatible"):
        Config._init_sequence_parallel(config)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"parallel_config": SimpleNamespace(data_parallel_size=2)}, "DP MoE"),
        ({"enable_dp_attention": True}, "DP MoE"),
        ({"tensor_parallel_size": 2}, "requires --tensor-parallel-size 1"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
        ({"enable_expert_parallel": True}, "expert parallelism"),
        ({"prefill_context_parallel_size": 2}, "same rank dimension"),
        ({"enable_tbo": True}, "TBO token slicing"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": {"kv_connector": "mooncake"}}, "KV transfer"),
        ({"hf_config": SimpleNamespace(kv_lora_rank=512)}, "MHA models only"),
    ],
)
def test_unsharded_execution_paths_are_rejected(overrides, reason):
    config = atom_config_double(sequence_parallel_size=4, **overrides)
    with pytest.raises(ValueError, match=reason):
        Config._init_sequence_parallel(config)


def test_minimax_m3_builds_no_vision_tower_under_sp():
    config = atom_config_double(
        sequence_parallel_size=4,
        hf_config=SimpleNamespace(vision_config=object()),
        compilation_config=SimpleNamespace(cudagraph_mode=None),
    )
    assert image_input_refusal(config) == "Ulysses SP (--sequence-parallel-size > 1)"
    config.sequence_parallel_size = 1
    assert image_input_refusal(config) is None


def test_dp_isolated_rlhf_workers_refuse_sp(monkeypatch):
    # Importing the runner resolves the chip architecture through aiter.
    if not torch.cuda.is_available():
        pytest.skip("RLHFModelRunner needs a ROCm device to import")
    from atom.rollout.model_runner_ext import RLHFModelRunner

    monkeypatch.setenv(RLHFModelRunner.DP_DEVICE_MAP_ENV, "0,1,2,3")
    config = atom_config_double(
        prefill_context_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
    )
    runner = object.__new__(RLHFModelRunner)
    with pytest.raises(ValueError, match="Ulysses SP"):
        runner._setup_device_and_distributed(0, config)


def test_sp_attention_needs_rope_in_the_impl():
    # A model that ropes before attention sees only its token shard there.
    with pytest.raises(ValueError, match="RoPE inside the attention impl"):
        check_sp_attention_layer(4, "layers.0.attn", None, None, None, 0)


@pytest.mark.parametrize(
    "index_heads,world,ok", [(0, 4, True), (4, 8, True), (8, 4, True), (6, 4, False)]
)
def test_sp_indexer_heads_shard_or_replicate(index_heads, world, ok):
    expect = nullcontext() if ok else pytest.raises(ValueError, match="cannot shard")
    with expect:
        check_sp_attention_layer(
            world, "layers.0.attn", None, None, object(), index_heads
        )


def test_layouts_that_change_the_graph_do_not_share_compiled_artifacts():
    config = atom_config_double()
    keys = [Config.compute_hash(config)]
    for name, value in (
        ("sequence_parallel_size", 4),
        ("sp_mega_moe_enabled", True),
    ):
        setattr(config, name, value)
        key = Config.compute_hash(config)
        assert key == Config.compute_hash(config)
        keys.append(key)
    assert len(set(keys)) == len(keys)
