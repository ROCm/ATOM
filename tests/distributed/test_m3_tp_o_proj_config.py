# SPDX-License-Identifier: MIT
"""CPU validation of the experimental M3 topology and compilation cache key."""

from types import SimpleNamespace

import pytest
import torch
from conftest import atom_config_double

from atom.config import Config


def _config(**overrides):
    values = {
        "m3_tp_replicated_o_proj": True,
        "tensor_parallel_size": 4,
        "torch_dtype": torch.bfloat16,
        "hf_config": SimpleNamespace(
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            hidden_size=6144,
            num_attention_heads=64,
            num_key_value_heads=4,
            head_dim=128,
        ),
    }
    values.update(overrides)
    return atom_config_double(**values)


def test_supported_layout_keeps_tp_and_sp_dimensions():
    config = _config()
    Config._validate_m3_tp_replicated_o_proj(config)
    assert config.tensor_parallel_size == 4
    assert config.sequence_parallel_size == config.prefill_context_parallel_size == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"tensor_parallel_size": 1},
        {"sequence_parallel_size": 4},
        {"prefill_context_parallel_size": 4},
        {"decode_context_parallel_size": 2},
        {"pipeline_parallel_size": 2},
        {"parallel_config": SimpleNamespace(data_parallel_size=2)},
        {"enable_dp_attention": True},
        {"enable_expert_parallel": True},
        {"moe_all2all_backend": "rccl"},
        {"moe_backend": "mega"},
        {"fake_eplb": True},
        {"enable_tbo": True},
        {"enable_tbo_decode": True},
        {"speculative_config": object()},
        {"dcp_config": SimpleNamespace(indexer_dcp_only=True)},
        {"torch_dtype": torch.float16},
        {"hf_config": SimpleNamespace(architectures=["LlamaForCausalLM"])},
    ],
)
def test_unsupported_combinations_fail_before_model_construction(overrides):
    with pytest.raises(ValueError, match="ATOM_M3_TP_REPLICATED_O_PROJ"):
        Config._validate_m3_tp_replicated_o_proj(_config(**overrides))


def test_disabled_experiment_does_not_constrain_existing_sp():
    config = _config(m3_tp_replicated_o_proj=False, sequence_parallel_size=4)
    Config._validate_m3_tp_replicated_o_proj(config)


def test_compilation_cache_separates_layouts_and_uses_snapshot(monkeypatch):
    config = _config()
    key = Config.compute_hash(config)
    monkeypatch.setenv("ATOM_M3_TP_REPLICATED_O_PROJ", "0")
    assert Config.compute_hash(config) == key
    config.m3_tp_replicated_o_proj = False
    assert Config.compute_hash(config) != key
