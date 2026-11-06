# SPDX-License-Identifier: MIT
"""Reject unsupported FP4 KV before a backend allocates its cache."""

from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import Config

MODELS = [
    ("qwen3", "Qwen3ForCausalLM", None),
    ("qwen3_next", "Qwen3NextForCausalLM", None),
    ("deepseek_v3", "DeepseekV3ForCausalLM", 512),
    ("deepseek_v3", "DeepseekV4ForCausalLM", 512),
    ("kimi_linear", "KimiK3ForConditionalGeneration", 512),
    ("qwen4_exp_text", "Qwen4ExpForCausalLM", None),
]


@pytest.mark.parametrize("model_type,arch,latent", MODELS)
@pytest.mark.parametrize("dtype", ["bf16", "fp8", "fp4"])
def test_only_fp4_is_refused_for_other_attention_families(
    model_type, arch, latent, dtype
):
    config = SimpleNamespace(
        kv_cache_dtype=dtype,
        hf_config=SimpleNamespace(
            model_type=model_type, architectures=[arch], kv_lora_rank=latent
        ),
    )
    if dtype == "fp4":
        with pytest.raises(ValueError, match="fp4.*DeepSeek-V4.1"):
            Config._check_kv_cache_dtype(config)
    else:
        Config._check_kv_cache_dtype(config)


@pytest.mark.parametrize(
    "model_type,arch",
    [
        ("deepseek_v41", "DeepseekV41ForCausalLM"),
        ("deepseek_v41_text", "DeepseekV41ForCausalLM"),
        ("deepseek_v41_dspark", "DeepseekV41DSparkModel"),
        ("deepseek_v3", "DeepseekV41ForCausalLM"),
    ],
)
def test_csa2_keeps_model_specific_fp4_handling(model_type, arch):
    config = SimpleNamespace(
        kv_cache_dtype="fp4",
        hf_config=SimpleNamespace(model_type=model_type, architectures=[arch]),
    )
    Config._check_kv_cache_dtype(config)


@pytest.mark.parametrize("model_type,arch,latent", [MODELS[0], MODELS[2]])
def test_config_construction_rejects_fp4_before_loading_weights(
    monkeypatch, model_type, arch, latent
):
    hf = PretrainedConfig(
        architectures=[arch], num_hidden_layers=2, kv_lora_rank=latent
    )
    hf.model_type = model_type
    monkeypatch.setattr(config_module, "get_hf_config", lambda *a, **kw: hf)
    with pytest.raises(ValueError, match="fp4.*DeepSeek-V4.1"):
        Config(model="unused-local-config", kv_cache_dtype="fp4")
