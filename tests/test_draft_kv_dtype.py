# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The KV dtype of a speculative draft: ``--spec-decode-kv-cache-dtype`` (what was
asked, ``SpeculativeConfig.kv_cache_dtype``) and what ``Config`` settles it to
(``draft_kv_dtype``, the one every reader uses). GPU-free: a real ``Config`` over a
stubbed HF config."""

import sys
import types
from pathlib import Path

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


def _config(monkeypatch, *, architectures=None, hf_config=None, **overrides):
    if hf_config is None:
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
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **{"model": "test-model", "kv_cache_dtype": "bf16", **overrides},
    )


def _llama(monkeypatch, **overrides):
    return _config(monkeypatch, architectures=["LlamaForCausalLM"], **overrides)


@pytest.fixture(autouse=True)
def no_synthetic_acceptance_import(monkeypatch):
    """`SpeculativeConfig` pulls in triton (the rejection sampler) to resolve a
    forced-acceptance knob these tests never set."""
    monkeypatch.setattr(SpeculativeConfig, "_resolve_synthetic_acceptance", lambda s: 0)


@pytest.mark.parametrize("target", ["bf16", "fp8"])
@pytest.mark.parametrize("asked", [None, "bf16", "fp8"])
def test_an_own_pool_draft_keeps_the_targets_dtype_unless_asked(target, asked):
    assert resolve_draft_kv_dtype(target, own_pool=True, requested=asked) == (
        asked or target
    )


def test_an_fp4_target_has_no_fp4_draft_layout_so_an_own_pool_draft_is_bf16():
    # the clause #2962 added: kept by the one resolver, with and without a request
    assert resolve_draft_kv_dtype("fp4", own_pool=True) == "bf16"
    assert resolve_draft_kv_dtype("fp4", own_pool=True, requested="bf16") == "bf16"
    assert resolve_draft_kv_dtype("fp4", own_pool=True, requested="fp8") == "fp8"


@pytest.mark.parametrize("own_pool", [True, False])
@pytest.mark.parametrize("asked", ["int4", "fp4", "auto", ""])
def test_an_unsupported_draft_dtype_is_refused_whoever_asks(own_pool, asked):
    with pytest.raises(ValueError, match="must be 'bf16' or 'fp8'"):
        resolve_draft_kv_dtype("bf16", own_pool=own_pool, requested=asked)


def test_a_shared_pool_draft_has_the_targets_dtype():
    assert resolve_draft_kv_dtype("fp8", own_pool=False) == "fp8"
    assert resolve_draft_kv_dtype("bf16", own_pool=False, requested="bf16") == "bf16"
    assert resolve_draft_kv_dtype("fp4", own_pool=False) == "fp4"  # MTP: refused later
    with pytest.raises(ValueError, match="caches in the target's KV pool"):
        resolve_draft_kv_dtype("fp8", own_pool=False, requested="bf16")


def test_config_refuses_an_unsupported_draft_dtype_at_configuration(monkeypatch):
    # a Python caller bypasses argparse's choices
    with pytest.raises(ValueError, match="must be 'bf16' or 'fp8'"):
        _llama(monkeypatch, speculative_config=_dspark_spec(kv_cache_dtype="int4"))


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


def test_a_draft_dtype_beside_the_targets_is_refused_with_offload(monkeypatch):
    # unverified with a KV connector or offload, so refused at startup
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


V41_FIXTURES = Path(__file__).parent / "models/deepseek_v41/fixtures"


def _v41_dspark(monkeypatch, *, target, asked=None, knob="auto"):
    """A real `Config` for a native DeepSeek-V4.1 DSpark run (the target checkpoint is
    the draft), built the way the CLI builds it; only the fp8 capability probe, which
    needs a GPU-side aiter symbol, is stubbed."""
    hf = PretrainedConfig(
        architectures=["DeepseekV41DSparkModel"], model_type="deepseek_v41_dspark"
    )
    spec = SpeculativeConfig(
        method="dspark",
        model=str(V41_FIXTURES),
        num_speculative_tokens=5,
        draft_model_hf_config=hf,
        kv_cache_dtype=asked,
    )
    probe = types.ModuleType("atom.models.deepseek_v41.draft_kv")
    probe.validate_fp8_draft_support = lambda _config: None
    monkeypatch.setitem(sys.modules, probe.__name__, probe)
    config = _config(
        monkeypatch,
        hf_config=config_module.get_hf_config(str(V41_FIXTURES)),
        model=str(V41_FIXTURES),
        kv_cache_dtype=target,
        enforce_eager=True,
        speculative_config=spec,
        dspark=config_module.DSparkConfig(draft_kv_dtype=knob),
    )
    return config


@pytest.mark.parametrize("knob", ["auto", "fp8"])
@pytest.mark.parametrize("asked", ["bf16", "fp8"])
@pytest.mark.parametrize("target", ["bf16", "fp8", "fp4"])
def test_the_flag_is_refused_for_a_v41_dspark_draft_and_names_its_own_knob(
    monkeypatch, target, asked, knob
):
    # V4.1's draft reads the target's pool; its dtype is DSparkConfig.draft_kv_dtype.
    # (A bf16 target with knob fp8 is the knob's own refusal, raised first.)
    refused = (
        "requires native DeepSeek-V4.1"
        if (target, knob) == ("bf16", "fp8")
        else r"does not apply to a DeepSeek-V4\.1.*draft_kv_dtype"
    )
    with pytest.raises(ValueError, match=refused):
        _v41_dspark(monkeypatch, target=target, asked=asked, knob=knob)


@pytest.mark.parametrize(
    ("target", "knob"),
    [
        ("bf16", "auto"),
        ("fp8", "auto"),
        ("fp8", "fp8"),
        ("fp4", "auto"),
        ("fp4", "fp8"),
    ],
)
def test_a_v41_dspark_draft_is_set_by_dspark_config_alone(monkeypatch, target, knob):
    config = _v41_dspark(monkeypatch, target=target, knob=knob)
    assert config.dspark.draft_kv_dtype == knob
    assert config.speculative_config.draft_kv_dtype == target  # the target's pool
