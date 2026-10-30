# SPDX-License-Identifier: MIT
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from conftest import atom_config_double

from atom.config import Config
from atom.distributed.ulysses_sp import check_sp_attention_layer


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
        ({"tensor_parallel_size": 2}, "requires --tensor-parallel-size 1"),
        ({"pipeline_parallel_size": 2}, "pipeline parallelism"),
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


def test_replicated_o_proj_rejects_speculative_decoding():
    # Its inline-recorded attention needs one (bs, q=1) request per token bucket.
    config = atom_config_double(
        tp_replicated_o_proj=True,
        tensor_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        speculative_config=object(),
    )
    with pytest.raises(ValueError, match="speculative decoding"):
        Config._validate_tp_replicated_o_proj(config)


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
        ("tp_replicated_o_proj", True),
        ("sp_mega_moe_enabled", True),
        ("enable_expert_parallel", True),
        ("moe_all2all_backend", "mori"),
    ):
        setattr(config, name, value)
        key = Config.compute_hash(config)
        assert key == Config.compute_hash(config)
        keys.append(key)
    assert len(set(keys)) == len(keys)
