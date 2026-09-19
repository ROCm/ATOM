# SPDX-License-Identifier: MIT

from types import SimpleNamespace

from atom.config import (
    _normalize_minimax_m3_text_config,
    _resolve_minimax_m3_kv_cache_block_size,
    _resolve_minimax_m3_sparse_attention_config,
)


def _flat_m3_config():
    text_config = SimpleNamespace(
        hidden_act="swigluoai",
        swiglu_beta=None,
        layer_types=[
            "full_attention",
            "full_attention",
            "full_attention",
            "minimax_m3_sparse",
        ],
        index_n_heads=4,
        index_head_dim=128,
        index_block_size=128,
        index_topk_blocks=16,
        index_local_blocks=1,
    )
    return SimpleNamespace(
        architectures=["MiniMaxM3SparseForConditionalGeneration"],
        text_config=text_config,
    )


def test_flat_transformers_schema_recovers_sparse_attention_config():
    config = _flat_m3_config()

    _normalize_minimax_m3_text_config(config)

    sparse = config.text_config.sparse_attention_config
    assert sparse["sparse_num_index_heads"] == 4
    assert sparse["sparse_index_dim"] == 128
    assert sparse["sparse_block_size"] == 128
    assert sparse["sparse_topk_blocks"] == 16
    assert sparse["sparse_local_block"] == 1
    assert sparse["sparse_attention_freq"] == [0, 0, 0, 1]
    assert sparse["use_sparse_attention"] is True
    assert config.text_config.index_block_size == 128


def test_existing_nested_sparse_fields_are_not_overwritten():
    config = _flat_m3_config()
    config.text_config.sparse_attention_config = {
        "sparse_block_size": 256,
        "use_sparse_attention": False,
        "checkpoint_only_field": "keep",
    }

    sparse = _resolve_minimax_m3_sparse_attention_config(config)

    assert sparse["sparse_block_size"] == 256
    assert sparse["use_sparse_attention"] is False
    assert sparse["checkpoint_only_field"] == "keep"
    assert sparse["sparse_index_dim"] == 128


def test_default_scheduler_block_is_promoted_to_sparse_page_size():
    config = _flat_m3_config()

    assert _resolve_minimax_m3_kv_cache_block_size(config, 16) == 128
    assert _resolve_minimax_m3_kv_cache_block_size(config, 256) == 256


def test_text_only_config_is_normalized_without_a_wrapper():
    config = _flat_m3_config().text_config
    config.architectures = ["MiniMaxM3SparseForCausalLM"]
    config.text_config = None

    _normalize_minimax_m3_text_config(config)

    assert config.sparse_attention_config["sparse_index_dim"] == 128
    assert config.swiglu_beta == 1.0


def test_non_minimax_config_is_unchanged():
    config = SimpleNamespace(
        architectures=["LlamaForCausalLM"],
        index_block_size=128,
    )

    _normalize_minimax_m3_text_config(config)

    assert not hasattr(config, "sparse_attention_config")
    assert _resolve_minimax_m3_kv_cache_block_size(config, 16) == 16
