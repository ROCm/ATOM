# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The KV dtype of a speculative draft: ``--spec-decode-kv-cache-dtype`` (what was
asked, ``SpeculativeConfig.kv_cache_dtype``) and what ``Config`` settles it to
(``draft_kv_dtype``, the one every reader uses). GPU-free: a real ``Config`` over a
stubbed HF config."""

import types

import pytest
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import (
    CompilationConfig,
    Config,
    SpeculativeConfig,
    resolve_draft_kv_dtype,
)
from tests.test_v4_indexer_defaults import _GfxProbe, _install_chip_info_stub

OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}


def _config(monkeypatch, *, architectures, **overrides):
    hf_config = PretrainedConfig(
        architectures=architectures, max_position_embeddings=8192
    )
    monkeypatch.setattr(
        config_module, "get_hf_config", lambda _m, trust_remote_code=False: hf_config
    )
    monkeypatch.setattr(config_module, "get_generation_config", lambda _m: None)
    monkeypatch.setattr(
        config_module,
        "QuantizationConfig",
        lambda *_a, **_k: types.SimpleNamespace(
            exclude_layers=[], compute_hash=lambda: ""
        ),
    )
    monkeypatch.setattr(config_module, "is_plugin_mode", lambda: False)
    _install_chip_info_stub(monkeypatch, _GfxProbe("gfx950"))
    return Config(
        model="test-model",
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **{"kv_cache_dtype": "bf16", **overrides},
    )


def _llama(monkeypatch, **overrides):
    return _config(monkeypatch, architectures=["LlamaForCausalLM"], **overrides)


@pytest.fixture(autouse=True)
def no_synthetic_acceptance_import(monkeypatch):
    """`SpeculativeConfig` pulls in triton (the rejection sampler) to resolve a
    forced-acceptance knob these tests never set."""
    monkeypatch.setattr(SpeculativeConfig, "_resolve_synthetic_acceptance", lambda s: 0)


def _draft(method="dspark", asked=None):
    """A speculative config as `Config` holds it: what `resolve_draft_kv_dtype` reads."""
    return types.SimpleNamespace(method=method, kv_cache_dtype=asked)


@pytest.mark.parametrize("target", ["bf16", "fp8"])
@pytest.mark.parametrize("asked", [None, "bf16", "fp8"])
def test_an_own_pool_draft_keeps_the_targets_dtype_unless_asked(target, asked):
    assert resolve_draft_kv_dtype(_draft(asked=asked), target, own_pool=True) == (
        asked or target
    )


def test_a_shared_pool_draft_has_the_targets_dtype():
    assert resolve_draft_kv_dtype(_draft("mtp"), "fp8", own_pool=False) == "fp8"
    assert resolve_draft_kv_dtype(_draft("mtp", "bf16"), "bf16", own_pool=False) == (
        "bf16"
    )
    with pytest.raises(ValueError, match="caches in the target's KV pool"):
        resolve_draft_kv_dtype(_draft("mtp", "bf16"), "fp8", own_pool=False)


def _dspark_spec(**kw):
    """A standalone MHA DSpark draft (MiniMax-M3.1's), built the way the CLI builds it."""
    hf = PretrainedConfig(
        architectures=["DSparkMiniMaxDraftModel"],
        model_type="minimax_m3",
        dspark_target_layer_ids=[1, 2],
        dspark_noise_token_id=0,
    )
    return SpeculativeConfig(
        method="dspark",
        model="draft",
        num_speculative_tokens=3,
        draft_model_hf_config=hf,
        **kw,
    )


def _eagle3_spec(**kw):
    """An EAGLE3 draft that is an MHA model (llama): it has a pool of its own."""
    hf = PretrainedConfig(architectures=["LlamaForCausalLM"], model_type="llama")
    return SpeculativeConfig(
        method="eagle3",
        model="draft",
        num_speculative_tokens=3,
        draft_model_hf_config=hf,
        **kw,
    )


@pytest.mark.parametrize("target", ["bf16", "fp8"])
@pytest.mark.parametrize("asked", [None, "bf16", "fp8"])
def test_config_settles_the_draft_dtype_and_keeps_what_was_asked(
    monkeypatch, target, asked
):
    spec = _dspark_spec(kv_cache_dtype=asked)
    config = _llama(monkeypatch, kv_cache_dtype=target, speculative_config=spec)

    assert spec.kv_cache_dtype == asked
    assert spec.draft_kv_dtype == (asked or target)
    assert config.kv_cache_dtype == target


def test_the_drafts_dtype_keys_the_compile_cache(monkeypatch):
    hashes = {
        asked: _llama(
            monkeypatch,
            speculative_config=_dspark_spec(kv_cache_dtype=asked),
        ).compute_hash()
        for asked in ("bf16", "fp8")
    }
    assert hashes["bf16"] != hashes["fp8"]


@pytest.mark.parametrize("asked", [None, "bf16", "fp8"])
def test_an_mha_eagle3_draft_has_a_pool_of_its_own_and_takes_its_own_dtype(
    monkeypatch, asked
):
    spec = _eagle3_spec(kv_cache_dtype=asked)
    _llama(monkeypatch, kv_cache_dtype="fp8", speculative_config=spec)
    assert spec.draft_kv_dtype == (asked or "fp8")


def test_an_mtp_draft_shares_the_targets_pool(monkeypatch):
    spec = _eagle3_spec(kv_cache_dtype="bf16")
    spec.method = "mtp"
    with pytest.raises(ValueError, match="caches in the target's KV pool"):
        _llama(monkeypatch, kv_cache_dtype="fp8", speculative_config=spec)


def test_a_plugin_config_with_a_v41_dspark_draft_never_asks_for_a_backend(
    monkeypatch,
):
    """V4.1 DSpark is a supported plugin deployment and vLLM / SGLang have no CSA2
    backend: whether the draft owns a pool is a native runner's question, so `Config`
    must not route it through a backend lookup (which raises for CSA2), nor import
    one (aiter) into the API process."""
    from atom.plugin import prepare
    from atom.utils import selector

    def refuse(*_a):
        raise AssertionError("a backend was resolved")

    hf = PretrainedConfig(
        architectures=["DeepseekV41DSparkModel"], model_type="deepseek_v41_dspark"
    )
    monkeypatch.setattr(selector, "get_attn_backend", refuse)
    for framework in ("vllm", "atom"):
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", framework)
        spec = SpeculativeConfig(
            method="dspark",
            model="draft",
            num_speculative_tokens=3,
            draft_model_hf_config=hf,
        )
        _llama(monkeypatch, speculative_config=spec)
        assert spec.draft_kv_dtype == "bf16"


def test_a_draft_dtype_beside_the_targets_cannot_share_the_offload_namespace(
    monkeypatch,
):
    # the offload namespace keys the target's KV dtype alone
    with pytest.raises(
        ValueError,
        match="draft KV dtype fp8 differs.*--spec-decode-kv-cache-dtype bf16",
    ):
        _llama(
            monkeypatch,
            kv_transfer_config=OFFLOAD,
            speculative_config=_dspark_spec(kv_cache_dtype="fp8"),
        )
    # the remedy it names works
    spec = _dspark_spec(kv_cache_dtype="bf16")
    _llama(monkeypatch, kv_transfer_config=OFFLOAD, speculative_config=spec)
    assert spec.draft_kv_dtype == "bf16"
