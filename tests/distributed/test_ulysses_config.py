# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest
from conftest import atom_config_double

from atom.config import Config


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


def test_dp_sp_rejected_before_moe_can_skip_sp_communication():
    config = SimpleNamespace(
        sequence_parallel_size=4,
        parallel_config=SimpleNamespace(data_parallel_size=2),
    )
    with pytest.raises(ValueError, match="DP MoE forward path"):
        Config._init_sequence_parallel(config)


@pytest.mark.parametrize("world", [0, -1])
def test_sequence_parallel_width_must_be_positive(world):
    with pytest.raises(ValueError, match="sequence_parallel_size must be at least 1"):
        Config(model="unused", sequence_parallel_size=world)


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"tensor_parallel_size": 2}, "requires --tensor-parallel-size 1"),
        ({"prefill_context_parallel_size": 2}, "same rank dimension"),
        ({"enable_tbo": True}, "TBO token slicing"),
        ({"enable_tbo_decode": True}, "TBO token slicing"),
        ({"speculative_config": object()}, "speculative decoding"),
        (
            {"hf_config": SimpleNamespace(kv_lora_rank=512)},
            "MHA models only",
        ),
        (
            {"hf_config": SimpleNamespace(model_type="qwen3_next")},
            "MHA models only",
        ),
    ],
)
def test_unsharded_execution_paths_are_rejected(overrides, reason):
    config = atom_config_double(sequence_parallel_size=4, **overrides)
    with pytest.raises(ValueError, match=reason):
        Config._init_sequence_parallel(config)


def test_moe_layouts_cannot_reuse_incompatible_sp_compilation_artifacts():
    config = SimpleNamespace(
        quant_config=None,
        compilation_config=None,
        parallel_config=None,
        tensor_parallel_size=1,
        prefill_context_parallel_size=8,
        sequence_parallel_size=8,
        enable_expert_parallel=False,
        moe_all2all_backend="none",
        moe_backend="standard",
        dcp_config=None,
        enable_dp_attention=False,
        index_cache_dtype="bf16",
        hf_config=SimpleNamespace(),
    )
    keys = [Config.compute_hash(config)]
    config.enable_expert_parallel = True
    for backend in ("none", "rccl", "mori"):
        config.moe_all2all_backend = backend
        key = Config.compute_hash(config)
        assert key == Config.compute_hash(config)
        keys.append(key)
    assert len(set(keys)) == len(keys)
