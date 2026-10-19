# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Muse Spark serves text-only: the engine must see the text sub-config.

CPU-only. The checkpoint's root config is ``muse_spark`` with a nested
``muse_spark_text`` config, and this image's transformers knows neither type.
"""

import pytest
from transformers import AutoConfig, PretrainedConfig

from atom import config as atom_config
from atom.model_loader.weight_names import CheckpointNameRewriter


def _muse_config_dict() -> dict:
    layer_types = ["sliding_attention", "full_attention"] + ["sliding_attention"] * 2
    return {
        "model_type": "muse_spark",
        "architectures": ["MuseSparkForConditionalGeneration"],
        "text_config": {
            "model_type": "muse_spark_text",
            "num_hidden_layers": 62,
            "hidden_size": 8192,
            "head_dim": 64,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "sliding_window": 2048,
            "layer_types": layer_types * 15 + layer_types[:2],
            "eos_token_id": 200001,
        },
        "vision_config": {"model_type": "muse_spark_vision", "hidden_size": 1536},
        "quantization_config": {
            "quant_method": "quark",
            "exclude": ["lm_head"],
            "global_quant_config": {},
        },
    }


@pytest.fixture
def muse_checkpoint(monkeypatch):
    raw = _muse_config_dict()
    monkeypatch.setattr(
        PretrainedConfig, "get_config_dict", lambda *_args, **_kwargs: (raw, {})
    )
    return raw


def test_muse_spark_root_type_is_unknown_to_transformers(muse_checkpoint):
    # The reason the config needs registering at all: without it the generic
    # AutoConfig path raises for this checkpoint.
    with pytest.raises(ValueError):
        AutoConfig.for_model("muse_spark_text")


def test_get_hf_config_returns_the_text_sub_config(muse_checkpoint):
    hf = atom_config.get_hf_config("unused")

    assert hf.model_type == "muse_spark_text"
    assert hf.num_hidden_layers == 62
    assert hf.num_key_value_heads == 16
    assert len(hf.layer_types) == 62
    assert hf.architectures == ["MuseSparkForConditionalGeneration"]
    assert hf.quantization_config["quant_method"] == "quark"
    assert hf.text_config is hf


def _rewriter(num_hidden_layers: int) -> CheckpointNameRewriter:
    return CheckpointNameRewriter(num_hidden_layers=num_hidden_layers)


def test_layers_past_a_truncated_depth_are_dropped_under_language_model():
    rewriter = _rewriter(4)
    kept = "model.language_model.layers.3.self_attn.q_proj.weight"
    dropped = "model.language_model.layers.7.self_attn.q_proj.weight"

    assert rewriter.rewrite(kept) == kept
    assert rewriter.rewrite(dropped) is None


def test_plain_model_layers_filter_is_unchanged():
    rewriter = _rewriter(4)

    assert rewriter.rewrite("model.layers.3.mlp.weight") == "model.layers.3.mlp.weight"
    assert rewriter.rewrite("model.layers.4.mlp.weight") is None
