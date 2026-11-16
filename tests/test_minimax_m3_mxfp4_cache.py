# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's MXFP4 KV pool, its startup refusals, and the draft's KV dtype."""

import sys
import types
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import (
    CompilationConfig,
    Config,
    SpeculativeConfig,
    _check_fp4_kv,
    resolve_draft_kv_dtype,
)
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, Mxfp4KvLayout
from atom.models.minimax_m3.config import validate_runtime_config
from tests.test_v4_indexer_defaults import _GfxProbe, _install_chip_info_stub

M3 = ["MiniMaxM3SparseForConditionalGeneration"]


def sparse_hf(layers=4, freq=None, **sparse):
    """The text-config fields of an all-sparse MiniMax-M3.1."""
    cfg = {"use_sparse_attention": True, "sparse_local_block": 1, **sparse}
    cfg["sparse_attention_freq"] = [1] * layers if freq is None else freq
    cfg["sparse_block_size"] = 128
    return {
        "num_hidden_layers": layers,
        "moe_layer_freq": [0, 1] * layers,
        "sparse_attention_config": cfg,
    }


def _m3(monkeypatch, *, hf=None, **overrides):
    """A real `Config` for a MiniMax-M3 checkpoint, HF config, quantization and chip
    probe (gfx950) stubbed."""
    hf_config = PretrainedConfig(
        architectures=M3, max_position_embeddings=8192, **(hf or {})
    )
    patch = monkeypatch.setattr
    patch(config_module, "get_hf_config", lambda _m, trust_remote_code=False: hf_config)
    patch(config_module, "get_generation_config", lambda _m: None)
    patch(config_module, "is_plugin_mode", lambda: False)
    patch(
        config_module,
        "QuantizationConfig",
        lambda *_a, **_k: SimpleNamespace(exclude_layers=[], compute_hash=lambda: ""),
    )
    probe = _GfxProbe("gfx950")
    _install_chip_info_stub(monkeypatch, probe)
    chip = sys.modules["aiter.jit.utils.chip_info"]
    patch(chip, "get_gfx_runtime", probe, raising=False)
    return Config(
        model="test-model",
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **{"kv_cache_dtype": "fp8", "kv_cache_block_size": 128, **overrides},
    )


@pytest.mark.parametrize("heads", [1, 2])
def test_pool_bytes_and_views_alias_the_pool(heads):
    def pool(layers):
        return MhaKvPool(
            layers=layers,
            block_size=128,
            num_kv_heads=heads,
            head_dim=128,
            kv_dtype=torch.float8_e4m3fnuz,
            layout=Mxfp4KvLayout(),
        )

    # 60 layers x (K + V) x 8704 B a 128-token block: 136 B a token (one head)
    assert pool(60).entry_bytes == 60 * heads * 2 * PAGE_BYTES
    p = pool(2)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, heads, PAGE_BYTES) == v.shape and k.dtype == torch.uint8
    k.fill_(7)
    assert not (v == 7).any() and not (p.kv_views(0)[0] == 7).any()


def check(*, hf=None, arch="gfx950", plugin_mode=False, **kw):
    """`validate_runtime_config` on an M3.1-shaped config, a field changed."""
    text = PretrainedConfig(**(sparse_hf() if hf is None else hf))
    fields = {
        "kv_cache_dtype": "fp4",
        "index_cache_dtype": "fp4",
        "kv_cache_block_size": 128,
        "torch_dtype": torch.bfloat16,
        "kv_transfer_config": {},
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "sequence_parallel_size": 1,
        "speculative_config": None,
        "hf_config": text,
    }
    chip = types.ModuleType("aiter.jit.utils.chip_info")  # CI has no aiter
    chip.get_gfx_runtime = lambda: arch
    with (
        mock.patch.dict(sys.modules, {"aiter.jit.utils.chip_info": chip}),
        mock.patch.object(config_module, "is_plugin_mode", lambda: plugin_mode),
    ):
        validate_runtime_config(SimpleNamespace(**{**fields, **kw}))


INDEX_ONLY = {"kv_cache_dtype": "fp8", "index_cache_dtype": "fp4"}
NOT_SPARSE = "indexed sparse"
REFUSALS = [  # (config change, message): one row per refusal reason
    ({"plugin_mode": True}, "bridges"),
    ({"kv_cache_block_size": 16}, "128 tokens"),
    ({"arch": "gfx942"}, "gfx950, got gfx942"),
    ({"decode_context_parallel_size": 2}, "decode-context-parallel-size 2"),
    (
        {"decode_context_parallel_size": 2, **INDEX_ONLY},
        "decode-context-parallel-size 2",
    ),
    ({"prefill_context_parallel_size": 2}, "prefill-context-parallel-size 2"),
    ({"prefill_context_parallel_size": 2, **INDEX_ONLY}, "prefill-context-parallel"),
    ({"plugin_mode": True, **INDEX_ONLY}, "bridges"),  # the index alone is held to it
    ({"arch": "gfx942", **INDEX_ONLY}, "gfx950, got gfx942"),
    ({"index_cache_dtype": "fp8"}, "needs --index-cache-dtype fp4, got fp8"),
    ({"sequence_parallel_size": 2}, "sequence-parallel-size > 1"),
    ({"torch_dtype": torch.float16}, "needs a bfloat16 model"),
    ({"kv_transfer_config": {"kv_connector": "lmcache_offload"}}, "KV offload"),
    # EAGLE3 / MTP share the target's pool: an fp4 draft has no layout
    ({"speculative_config": SimpleNamespace(draft_kv_dtype="fp4")}, "draft would"),
    ({"hf": sparse_hf(freq=[1, 1, 0, 1])}, NOT_SPARSE),  # one dense layer
    ({"hf": sparse_hf(freq=[1, 1, 1])}, NOT_SPARSE),  # one entry short
    ({"hf": sparse_hf(freq=[1, 1, 2, 1])}, NOT_SPARSE),  # non-zero, not the model's 1
    ({"hf": sparse_hf(freq=[1, 1, True, 1])}, NOT_SPARSE),  # truthy, not an int 1
    ({"hf": sparse_hf(use_sparse_attention=False)}, NOT_SPARSE),
    ({"hf": {"num_hidden_layers": 4}}, NOT_SPARSE),  # M3's original: no sparse config
]


@pytest.mark.parametrize(("kw", "match"), REFUSALS)
def test_startup_refusals(kw, match):
    with pytest.raises(ValueError, match=match):
        check(**kw)


def test_fp4_kv_is_m3_and_v41_only():
    check()
    check(hf={}, torch_dtype=torch.float16, **INDEX_ONLY)  # index alone: no pa_decode
    _check_fp4_kv("fp4", is_m3=False, is_v41=True)
    with pytest.raises(ValueError, match="MiniMax-M3.1 and DeepSeek-V4.1 only"):
        _check_fp4_kv("fp4", is_m3=False, is_v41=False)  # every other model


@pytest.mark.parametrize(
    ("kv", "index", "want"),
    [
        ("bf16", "fp4", ("bf16", "fp4")),  # index-only: KV stays as asked
        ("fp4", "fp4", ("fp4", "fp4")),
        ("fp4", None, ("fp4", "fp4")),  # an omitted index follows the fp4 KV
        ("fp8", None, ("fp8", "fp8")),
    ],
)
def test_m3_index_default_follows_the_kv(monkeypatch, kv, index, want):
    config = _m3(
        monkeypatch, kv_cache_dtype=kv, index_cache_dtype=index, hf=sparse_hf()
    )
    assert (config.kv_cache_dtype, config.index_cache_dtype) == want
    if kv == "fp4":
        with pytest.raises(ValueError, match="indexed sparse"):
            _m3(
                monkeypatch,
                kv_cache_dtype=kv,
                index_cache_dtype=index,
                hf=sparse_hf(freq=[1, 1, 0, 1]),
            )


# ────────────────────────────────────────────── the draft's own KV dtype ──


@pytest.fixture(autouse=True)
def no_synthetic_acceptance_import(monkeypatch):
    """`SpeculativeConfig` would import triton for a knob these tests never set."""
    monkeypatch.setattr(SpeculativeConfig, "_resolve_synthetic_acceptance", lambda s: 0)


def _spec(method, arch, model_type, **kw):
    hf = PretrainedConfig(architectures=[arch], model_type=model_type, **kw)
    return SpeculativeConfig(
        method=method, model="draft", num_speculative_tokens=3, draft_model_hf_config=hf
    )


def _dspark_spec():  # the standalone M3.1 DSpark draft, as the CLI builds it
    return _spec(
        "dspark",
        "DSparkMiniMaxDraftModel",
        "minimax_m3",
        dspark_target_layer_ids=[1, 2],
        dspark_noise_token_id=0,
    )


def _eagle3_spec():  # an MHA EAGLE3 draft: a pool of its own, like a DSpark one
    return _spec("eagle3", "LlamaForCausalLM", "llama")


OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}
FP4 = {"index_cache_dtype": "fp4", "hf": sparse_hf()}


@pytest.mark.parametrize("spec", [_dspark_spec, _eagle3_spec])
@pytest.mark.parametrize(
    ("target", "want", "extra"),
    [
        ("bf16", "bf16", {}),
        ("fp8", "fp8", {"kv_transfer_config": OFFLOAD}),
        ("fp4", "bf16", FP4),  # fp4 has no draft layout
    ],
)
def test_an_own_pool_draft_has_the_targets_dtype_bf16_under_fp4(
    monkeypatch, spec, target, want, extra
):
    config = _m3(monkeypatch, kv_cache_dtype=target, speculative_config=spec(), **extra)
    assert config.speculative_config.draft_kv_dtype == want


def test_a_shared_pool_draft_has_the_targets_dtype_and_fp4_beside_offload_refuses(
    monkeypatch,
):
    for target in ("bf16", "fp8", "fp4"):
        assert resolve_draft_kv_dtype(target, own_pool=False) == target
    with pytest.raises(ValueError, match="KV offload"):  # not for its bf16 draft
        _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            kv_transfer_config=OFFLOAD,
            speculative_config=_dspark_spec(),
            **FP4,
        )


def test_a_plugin_config_with_a_v41_dspark_draft_never_asks_for_a_backend(
    monkeypatch,
):
    """Config must not resolve a backend to decide a draft's pool (it raises for CSA2
    under vLLM / SGLang, and would import aiter into the API process natively)."""
    from atom.plugin import prepare
    from atom.utils import selector

    def refuse(*_a):
        raise AssertionError("a backend was resolved")

    monkeypatch.setattr(selector, "get_attn_backend", refuse)
    spec = _spec("dspark", "DeepseekV41DSparkModel", "deepseek_v41_dspark")
    for framework in ("vllm", "atom"):
        monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", framework)
        config = _m3(
            monkeypatch,
            kv_cache_dtype="bf16",
            index_cache_dtype="bf16",
            speculative_config=spec,
        )
        assert config.speculative_config.draft_kv_dtype == "bf16"
