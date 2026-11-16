# SPDX-License-Identifier: MIT
"""Reject unsupported FP4 KV before a backend allocates its cache."""

import pickle
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import CompilationConfig, Config, _check_fp4_kv
from atom.models.deepseek_v41.pool_geometry import build_v41_pool_geometry
from tests.conftest import gpu_stack_import

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
    if dtype == "fp4":
        with pytest.raises(ValueError, match="fp4.*DeepSeek-V4.1"):
            _check_fp4_kv(dtype, is_m3=False, is_v41=False)
    else:
        _check_fp4_kv(dtype, is_m3=False, is_v41=False)


@pytest.mark.parametrize("is_m3,is_v41", [(True, False), (False, True)])
def test_supported_models_keep_fp4(is_m3, is_v41):
    _check_fp4_kv("fp4", is_m3=is_m3, is_v41=is_v41)


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


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def chip(monkeypatch):
    chip = ModuleType("aiter.jit.utils.chip_info")
    chip.get_gfx = Mock(side_effect=AssertionError("build target queried"))
    chip.get_gfx_runtime = Mock(return_value="gfx950")
    monkeypatch.setitem(sys.modules, chip.__name__, chip)
    monkeypatch.setenv("GPU_ARCHS", "gfx950;gfx1250")
    monkeypatch.setattr(config_module, "get_generation_config", lambda _: None)
    monkeypatch.setattr(config_module, "QuantizationConfig", lambda *a, **kw: None)
    return chip


def config(heads, tp, dp, dtype, **kwargs):
    return Config(
        model=str(ROOT / "tests/models/deepseek_v41/fixtures"),
        hf_overrides={"num_attention_heads": heads},
        tensor_parallel_size=tp,
        enable_dp_attention=dp,
        enable_expert_parallel=True,
        kv_cache_dtype=dtype,
        index_cache_dtype="fp8",
        enforce_eager=True,
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **kwargs,
    )


def geometry(cfg):
    return build_v41_pool_geometry(
        cfg.hf_config,
        cfg.kv_cache_block_size,
        kv_dtype=cfg.kv_cache_dtype,
        speculative_tokens=0,
        index_fp4=cfg.index_cache_dtype == "fp4",
        index_layout=cfg.index_cache_layout,
    )


@pytest.mark.parametrize(
    "heads,tp,dp", [(64, 1, False), (64, 4, False), (128, 4, True)]
)
@pytest.mark.parametrize("a8w4_mode", ["auto", "off"])
def test_config_preserves_fp4_sizing_without_probing_architecture(
    monkeypatch, chip, heads, tp, dp, a8w4_mode
):
    chip.get_gfx_runtime.side_effect = AssertionError("configuration probed hardware")
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", a8w4_mode)
    cfg = config(heads, tp, dp, "fp4")
    assert cfg.kv_cache_dtype == "fp4"
    assert cfg.index_cache_dtype == "fp8"
    assert cfg.dspark.draft_attention_dtype == "auto"
    fp8 = config(heads, tp, dp, "fp8")
    assert geometry(cfg).paged_bytes < geometry(fp8).paged_bytes
    assert cfg.compute_hash() != fp8.compute_hash()
    chip.get_gfx.assert_not_called()
    chip.get_gfx_runtime.assert_not_called()
    # Worker startup must not reinterpret the already sized allocation.
    worker = pickle.loads(pickle.dumps(cfg))
    assert worker.kv_cache_dtype == "fp4"
    assert geometry(worker) == geometry(cfg)
    assert worker.compute_hash() == cfg.compute_hash()


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_explicit_non_fp4_does_not_probe_architecture(chip, dtype):
    chip.get_gfx_runtime.side_effect = AssertionError("unneeded device probe")
    cfg = config(128, 4, True, dtype)
    assert cfg.kv_cache_dtype == dtype


def test_vllm_proxy_and_native_preserve_the_same_fp4_pool(chip, monkeypatch):
    pytest.importorskip("vllm")
    tp = 2
    native = config(128, tp, False, "fp4")
    api = gpu_stack_import("atom.plugin.vllm.deepseek_v41_bridge")
    monkeypatch.setattr(api, "v41_text_config", lambda _: native.hf_config)
    monkeypatch.setattr(api, "v41_speculative_tokens", lambda _: 0)
    plugin = SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype="nvfp4"),
        parallel_config=SimpleNamespace(tensor_parallel_size=tp, data_parallel_size=4),
    )
    chip.get_gfx_runtime.side_effect = AssertionError("proxy dtype probed the GPU")
    assert api.v41_kv_cache_dtype(plugin) == "fp4"
    # Compare real layouts to detect a wider allocation than the user requested.
    native.kv_cache_block_size = 256
    actual = api.v41_proxy_geometry(plugin)
    assert actual == geometry(native)
    assert api.v41_kv_cache_dtype(plugin) == "fp4"
    assert api.v41_proxy_geometry(plugin) is actual


def test_invalid_a8w4_mode_fails_at_startup_before_loading(monkeypatch):
    monkeypatch.setenv("ATOM_DSV41_A8W4_ATTN", "unknown")
    loader = Mock(side_effect=AssertionError("model loading started"))
    monkeypatch.setattr(config_module, "get_hf_config", loader)
    with pytest.raises(ValueError, match="ATOM_DSV41_A8W4_ATTN must be auto or off"):
        Config(model="unused-local-config")
    loader.assert_not_called()
