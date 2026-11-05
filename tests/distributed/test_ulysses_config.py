# SPDX-License-Identifier: MIT
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from aiter_stub import atom_without_aiter
from conftest import atom_config_double

from atom.config import Config, use_custom_atom_config
from atom.distributed.ulysses_sp import check_sp_attention_layer, sp_head_shards
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
    # SP keeps -pcp's value and widens only the process group it runs over.
    assert config.prefill_context_parallel_size == 1
    assert Config.pcp_group_size.fget(config) == world


@pytest.mark.parametrize("pcp,sp", [(4, 1), (1, 4)])
def test_sp_runs_over_the_pcp_group_without_enabling_pcp(monkeypatch, pcp, sp):
    with atom_without_aiter() as load:
        pcp_utils = load("atom.distributed.pcp_utils")
    config = atom_config_double(
        prefill_context_parallel_size=pcp, sequence_parallel_size=sp
    )
    group_size = Config.pcp_group_size.fget(config)
    monkeypatch.setattr(pcp_utils, "get_pcp_world_size", lambda: group_size)
    with use_custom_atom_config(config):
        assert group_size == 4 and pcp_utils.pcp_is_enabled() is (pcp > 1)


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
    with pytest.raises(ValueError, match="evenly"):
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
        ({"compilation_config": SimpleNamespace(level=1)}, "--level 1"),
        ({"speculative_config": object()}, "speculative decoding"),
        ({"kv_transfer_config": {"kv_connector": "mooncake"}}, "KV transfer"),
        ({"hf_config": SimpleNamespace(kv_lora_rank=512)}, "MHA models only"),
        ({"hf_config": SimpleNamespace(_multimodal_config=object())}, "multimodal"),
    ],
)
def test_unsharded_execution_paths_are_rejected(overrides, reason):
    config = atom_config_double(sequence_parallel_size=4, **overrides)
    with pytest.raises(ValueError, match=reason):
        Config._init_sequence_parallel(config)


@pytest.mark.parametrize(
    "sp,ep,error", [(4, False, None), (1, False, "expert"), (1, True, None)]
)
def test_mega_moe_backend_needs_expert_parallelism_only_without_sp(sp, ep, error):
    # Under SP it selects SP MegaMoE; _init_sequence_parallel refuses EP there.
    config = atom_config_double(
        moe_backend="mega", sequence_parallel_size=sp, enable_expert_parallel=ep
    )
    with pytest.raises(ValueError, match=error) if error else nullcontext():
        Config._init_moe_backend(config)


def test_minimax_m3_serves_text_only_under_sp():
    config = atom_config_double(
        sequence_parallel_size=4,
        hf_config=SimpleNamespace(
            vision_config=object(), num_attention_heads=32, num_key_value_heads=4
        ),
        compilation_config=SimpleNamespace(level=0, cudagraph_mode=None),
    )
    Config._init_sequence_parallel(config)
    assert image_input_refusal(config) == "Ulysses SP (--sequence-parallel-size > 1)"
    config.sequence_parallel_size = 1
    assert image_input_refusal(config) is None


def test_rlhf_rollout_refuses_sp_at_construction():
    with atom_without_aiter() as load:
        runner = load("atom.rollout.model_runner_ext").RLHFModelRunner
    with pytest.raises(ValueError, match="Ulysses SP"):
        runner(0, atom_config_double(sequence_parallel_size=4))


def test_sp_attention_needs_rope_in_the_impl():
    # A model that ropes before attention sees only its token shard there.
    with pytest.raises(ValueError, match="RoPE inside the attention impl"):
        check_sp_attention_layer("layers.0.attn", None, None, None)


@pytest.mark.parametrize(
    "heads,world,split", [(8, 4, (2, 4)), (4, 8, (1, 4)), (1, 4, (1, 1)), (6, 4, None)]
)
def test_heads_shard_or_replicate(heads, world, split):
    expect = (
        pytest.raises(ValueError, match="evenly") if split is None else nullcontext()
    )
    with expect:
        assert sp_head_shards(heads, world) == split


def test_sp_does_not_share_compiled_artifacts():
    config = atom_config_double()
    native = Config.compute_hash(config)
    config.sequence_parallel_size = 4
    assert Config.compute_hash(config) == Config.compute_hash(config) != native
