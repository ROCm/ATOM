# SPDX-License-Identifier: MIT
"""Admission gates, empty ranks and exact production owner accounting."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry
from atom.models.deepseek_v41.config import normalize_hf_config, validate_runtime_config
from atom.models.deepseek_v41.runtime import DeepseekV41RuntimeModel


def test_production_geometry_has_only_four_global_owners():
    fixture = Path(__file__).parents[2] / "models/deepseek_v41/fixtures/config.json"
    config = normalize_hf_config(json.loads(fixture.read_text()))
    geo = V41PoolGeometry(
        config.num_hidden_layers,
        tuple(
            (owner, config.compress_ratios[owner])
            for owner in config.kv_source_layer_ids
        ),
        16,
        config.sliding_window,
        config.head_dim,
        config.index_head_dim,
    )
    assert geo.owners == ((2, 2), (8, 2), (14, 2), (20, 1))
    assert geo.tail_owners == (2, 8, 14)
    assert len(geo.page_fields) == 8
    assert geo.page_bytes == 16 * (3 / 2 + 1) * (512 + 128) * 2
    assert sum(f.bytes_per_entry for f in geo.state_fields) <= geo.state_bytes
    assert geo.state_fields[0].layers == 40
    assert all(field.in_checkpoint for field in geo.state_fields)


def runtime_config(**overrides):
    fields = {
        "enforce_eager": True,
        "compilation_config": SimpleNamespace(level=0),
        "speculative_config": None,
        "pipeline_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "parallel_config": SimpleNamespace(data_parallel_size=1),
        "enable_dp_attention": False,
        "enable_tbo": False,
        "enable_tbo_decode": False,
        "kv_transfer_config": None,
        "enable_rapidserve": False,
        "plugin_config": None,
        "online_quant_config": None,
        "eplb_enable": False,
        "kv_cache_dtype": "bf16",
        "index_cache_dtype": "bf16",
        "kv_cache_block_size": 16,
        "tensor_parallel_size": 4,
        "enable_expert_parallel": True,
        "hf_config": SimpleNamespace(index_topk_tie_break="small_position"),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "override",
    [
        {"enforce_eager": False},
        {"compilation_config": SimpleNamespace(level=3)},
        {"speculative_config": object()},
        {"pipeline_parallel_size": 2},
        {"prefill_context_parallel_size": 2},
        {"decode_context_parallel_size": 2},
        {"parallel_config": SimpleNamespace(data_parallel_size=2)},
        {"enable_dp_attention": True},
        {"enable_tbo": True},
        {"enable_tbo_decode": True},
        {"kv_transfer_config": {"connector": "moriio"}},
        {"enable_rapidserve": True},
        {"plugin_config": object()},
        {"online_quant_config": {}},
        {"eplb_enable": True},
        {"kv_cache_dtype": "fp8"},
        {"index_cache_dtype": "fp4"},
        {"kv_cache_block_size": 3},
        {"enable_expert_parallel": False},
    ],
)
def test_unimplemented_modes_fail_before_loading(override):
    validate_runtime_config(runtime_config())
    with pytest.raises(ValueError):
        validate_runtime_config(runtime_config(**override))


def test_empty_rank_padding_has_no_cache_writes(monkeypatch):
    from atom.models.deepseek_v41 import runtime

    geo = V41PoolGeometry(2, ((1, 2),), 4, 4, 512, 32)
    cache = PagedAttentionCache(geo, 4, 2, "cpu")
    cache.backing.fill_(57)
    before = cache.backing.clone()
    step = cache.begin_step([])
    metadata = SimpleNamespace(
        step=step, cache=cache, next_histories=np.empty((0, 3), dtype=np.int64)
    )
    monkeypatch.setattr(
        runtime, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata)
    )
    model = DeepseekV41RuntimeModel.__new__(DeepseekV41RuntimeModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=64)
    model.embed = torch.nn.Embedding(16, 64)
    # No layers are constructed: any attempted execution of padded rows fails.
    output = model(torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32))
    assert output.shape == (8, 64) and output.count_nonzero() == 0
    torch.testing.assert_close(cache.backing, before, rtol=0, atol=0)
