"""MiniMax-M3's MXFP4 index cache page views and its startup checks.

The index cache is one 8704 B uint8 page a block -- [128 x 64 B e2m1][128 x 4 B
e8m0] -- which is what the aiter writer produces and `score_flydsl_fp4` reads.
Pinned here: the views the scorer is handed, the combinations that must fail at
startup and not in a forward. `--kv-cache-dtype fp4` (the MXFP4 KV) is
MiniMax-M3.1's: its gate takes an all-indexed-sparse model and refuses -- with no
FP8 fallback -- what it has no MXFP4 path for (dense attention layers, KV offload,
an fp16 model, a skip-index layer). The index dtype stays its own choice.
"""

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
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.models.minimax_m3.config import validate_runtime_config
from tests.test_v4_indexer_defaults import _GfxProbe, _install_chip_info_stub

M3 = ["MiniMaxM3SparseForConditionalGeneration"]


def sparse_hf(layers=4, freq=None, **sparse):
    """The text-config fields of an all-sparse MiniMax-M3.1: one indexed sparse
    attention layer each, whatever the MLPs are (`moe_layer_freq` is not read)."""
    return {
        "num_hidden_layers": layers,
        "moe_layer_freq": [0, 1] * layers,
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_attention_freq": [1] * layers if freq is None else freq,
            "sparse_local_block": 1,
            "sparse_block_size": 128,
            **sparse,
        },
    }


def _make_config(monkeypatch, *, architectures=M3, hf=None, **overrides):
    """A real `Config` for a MiniMax-M3 checkpoint, with the HF config, quantization
    and chip probe stubbed (gfx950 unless `gfx` is overridden)."""
    gfx = overrides.pop("gfx", "gfx950")
    hf_config = PretrainedConfig(
        architectures=architectures,
        max_position_embeddings=8192,
        **{
            "sparse_attention_config": {
                "sparse_local_block": 1,
                "sparse_block_size": 128,
            },
            **(hf or {}),
        },
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
    probe = _GfxProbe(gfx)
    _install_chip_info_stub(monkeypatch, probe)
    monkeypatch.setattr(
        sys.modules["aiter.jit.utils.chip_info"],
        "get_gfx_runtime",
        probe,
        raising=False,
    )
    return Config(
        model="test-model",
        torch_profiler_dir=None,
        compilation_config=CompilationConfig(level=0, use_cudagraph=False),
        **{"kv_cache_dtype": "fp8", **overrides},
    )


def _m3(monkeypatch, **overrides):
    return _make_config(monkeypatch, kv_cache_block_size=128, **overrides)


def test_page_views_split_data_from_scale():
    pages = torch.arange(3 * 2 * PAGE_BYTES, dtype=torch.int64).remainder(251)
    pages = pages.to(torch.uint8).view(3, 2, PAGE_BYTES)
    data, scale = page_views(pages)
    assert data.shape == (3, 2, 128, 64) and scale.shape == (3, 2, 128, 4)
    assert data.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 64, 1)
    assert scale.stride() == (2 * PAGE_BYTES, PAGE_BYTES, 4, 1)
    assert torch.equal(data[2, 1].flatten(), pages[2, 1, :8192])
    assert torch.equal(scale[2, 1].flatten(), pages[2, 1, 8192:])
    # an index cache is [pages, 8704] with no head axis
    idata, iscale = page_views(pages.view(6, PAGE_BYTES)[1:5])
    assert idata.shape == (4, 128, 64) and iscale.shape == (4, 128, 4)
    assert torch.equal(iscale[0].flatten(), pages.view(6, PAGE_BYTES)[1, 8192:])


@pytest.fixture(autouse=True)
def chip_info(monkeypatch):
    """The architecture query: CI has no aiter, and a GPU host's answer is its own."""
    mod = types.ModuleType("aiter.jit.utils.chip_info")
    mod.get_gfx_runtime = lambda: "gfx950"
    monkeypatch.setitem(sys.modules, "aiter.jit.utils.chip_info", mod)
    return mod


def check(**kw):
    a = {
        "index_cache_dtype": "fp4",
        "block_size": 128,
        "plugin_mode": False,
        **kw,
    }
    config = types.SimpleNamespace(
        kv_cache_dtype="fp8",
        index_cache_dtype=a["index_cache_dtype"],
        kv_cache_block_size=a["block_size"],
    )
    with mock.patch.object(config_module, "is_plugin_mode", lambda: a["plugin_mode"]):
        validate_runtime_config(config)


def test_refusals_name_what_is_wrong():
    with pytest.raises(ValueError, match="bridges"):
        check(plugin_mode=True)
    with pytest.raises(ValueError, match="128 tokens"):
        check(block_size=16)


def test_only_gfx950_is_served(chip_info):
    chip_info.get_gfx_runtime = lambda: "gfx942"
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        check()


def test_the_baselines_are_never_refused():
    # The MXFP4 index off: nothing to check, whatever else is set.
    check(index_cache_dtype="fp8", block_size=16, plugin_mode=True)


def test_a_fp4_kv_is_m3s_and_v41s_and_refused_elsewhere():
    _check_fp4_kv("fp8", is_m3=False, is_v41=False)
    _check_fp4_kv("fp4", is_m3=True, is_v41=False)
    _check_fp4_kv("fp4", is_m3=False, is_v41=True)
    with pytest.raises(ValueError, match="MiniMax-M3 and DeepSeek-V4.1 only"):
        _check_fp4_kv("fp4", is_m3=False, is_v41=False)


@pytest.mark.parametrize("index", [None, "fp8", "bf16", "fp4"])
def test_m3_takes_a_fp4_kv_and_leaves_the_index_to_its_own_flag(monkeypatch, index):
    config = _m3(
        monkeypatch, kv_cache_dtype="fp4", index_cache_dtype=index, hf=sparse_hf()
    )
    # None resolves to fp8 (never the KV's fp4); anything explicit is kept
    assert config.kv_cache_dtype == "fp4"
    assert config.index_cache_dtype == (index or "fp8")


def test_an_m3_export_named_only_by_model_type_takes_the_fp4_kv_gate(monkeypatch):
    kw = {
        "architectures": ["SomeOtherForCausalLM"],
        "kv_cache_block_size": 128,
        "kv_cache_dtype": "fp4",
    }
    hf = {"model_type": "minimax_m3"}
    config = _make_config(monkeypatch, hf={**hf, **sparse_hf()}, **kw)
    assert config.kv_cache_dtype == "fp4"
    # and its dense checkpoint is refused by the same gate, not by the arch name
    with pytest.raises(ValueError, match="indexed sparse"):
        _make_config(monkeypatch, hf=hf, **kw)


def test_a_non_m3_model_still_refuses_a_fp4_kv(monkeypatch):
    with pytest.raises(ValueError, match="MiniMax-M3 and DeepSeek-V4.1 only"):
        _make_config(
            monkeypatch,
            architectures=["LlamaForCausalLM"],
            kv_cache_block_size=128,
            kv_cache_dtype="fp4",
        )


def test_the_m3_fp4_kv_hashes_apart_from_fp8(monkeypatch):
    fp8 = _m3(monkeypatch, kv_cache_dtype="fp8", hf=sparse_hf())
    fp4 = _m3(monkeypatch, kv_cache_dtype="fp4", hf=sparse_hf())
    assert fp8.compute_hash() != fp4.compute_hash()


# ─────────────────────────────────────────────── the fp4 KV's own refusals ──


def kv_check(monkeypatch, *, hf=None, text_config=False, offload="", **kw):
    """`validate_runtime_config` on the one M3.1-shaped config, a field changed."""
    monkeypatch.setenv("ATOM_KV_OFFLOAD", offload)
    text = PretrainedConfig(**(sparse_hf() if hf is None else hf))
    fields = {
        "kv_cache_dtype": "fp4",
        "index_cache_dtype": "fp8",
        "kv_cache_block_size": 128,
        "torch_dtype": torch.bfloat16,
        "kv_transfer_config": {},
        "speculative_config": None,
        "hf_config": SimpleNamespace(text_config=text) if text_config else text,
    }
    with mock.patch.object(config_module, "is_plugin_mode", lambda: False):
        validate_runtime_config(SimpleNamespace(**{**fields, **kw}))


@pytest.mark.parametrize("text_config", [False, True])
def test_an_all_sparse_m3_passes_the_kv_gate(monkeypatch, text_config):
    kv_check(monkeypatch, text_config=text_config)


@pytest.mark.parametrize(
    "hf",
    [
        sparse_hf(freq=[1, 1, 0, 1]),  # one dense layer
        sparse_hf(freq=[0] * 4),  # no sparse layer
        sparse_hf(freq=[1, 1, 1]),  # one entry short of num_hidden_layers
        sparse_hf(freq=[1] * 5),  # one entry past it
        sparse_hf(freq=[1, 1, 2, 1]),  # non-zero, but not the model's `1`
        sparse_hf(freq=[1, 1, True, 1]),  # truthy, but not an int 1
        sparse_hf(sparse_attention_freq=None),  # none declared
        sparse_hf(use_sparse_attention=False),
        {"num_hidden_layers": 4},  # M3's original: no sparse config at all
        {k: v for k, v in sparse_hf().items() if k != "num_hidden_layers"},
    ],
)
def test_the_fp4_kv_refuses_a_model_that_is_not_all_indexed_sparse(monkeypatch, hf):
    with pytest.raises(ValueError, match="indexed sparse"):
        kv_check(monkeypatch, hf=hf)


def test_the_fp4_kv_gate_reads_the_attention_freq_not_the_mlp_layout(monkeypatch):
    dense_attention = sparse_hf(freq=[1, 0, 1, 1])
    dense_attention["moe_layer_freq"] = [1] * 4  # an all-MoE MLP changes nothing
    with pytest.raises(ValueError, match="indexed sparse"):
        kv_check(monkeypatch, hf=dense_attention)


@pytest.mark.parametrize("spec_dtype", ["fp8", "bf16", None])
def test_the_fp4_kv_takes_a_draft_that_is_not_fp4(monkeypatch, spec_dtype):
    # FP4 target + an own-pool DSpark draft resolved to fp8: the supported pair
    kv_check(
        monkeypatch,
        speculative_config=SimpleNamespace(method="dspark", kv_cache_dtype=spec_dtype),
    )


def test_the_fp4_kv_refuses_a_draft_resolved_to_fp4(monkeypatch):
    # EAGLE3 / MTP share the target's pool: fp4 target, fp4 draft, no layout for it
    with pytest.raises(ValueError, match="draft would have FP4 KV"):
        kv_check(
            monkeypatch,
            speculative_config=SimpleNamespace(method="eagle3", kv_cache_dtype="fp4"),
        )


@pytest.mark.parametrize("mode", ["lmcache", "lmcache_mp", "mooncake_store"])
def test_the_fp4_kv_refuses_kv_offload_by_env(monkeypatch, mode):
    with pytest.raises(ValueError, match="KV offload"):
        kv_check(monkeypatch, offload=mode)


def test_the_fp4_kv_refuses_a_kv_transfer_config(monkeypatch):
    # what EngineArgs folds ATOM_KV_OFFLOAD into, or a P/D connector
    with pytest.raises(ValueError, match="KV offload"):
        kv_check(monkeypatch, kv_transfer_config={"kv_connector": "lmcache_offload"})


@pytest.mark.parametrize("off", ["", "0", "off", "None"])
def test_a_switched_off_offload_env_does_not_refuse(monkeypatch, off):
    kv_check(monkeypatch, offload=off)


def test_the_fp4_kv_needs_a_bf16_model(monkeypatch):
    with pytest.raises(ValueError, match="bfloat16"):
        kv_check(monkeypatch, torch_dtype=torch.float16)


def test_the_fp4_kv_keeps_the_shared_mxfp4_refusals(monkeypatch, chip_info):
    with pytest.raises(ValueError, match="128 tokens"):
        kv_check(monkeypatch, kv_cache_block_size=16)
    chip_info.get_gfx_runtime = lambda: "gfx942"
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        kv_check(monkeypatch)


def test_the_fp4_kv_is_refused_under_the_bridges():
    with (
        mock.patch.object(config_module, "is_plugin_mode", lambda: True),
        pytest.raises(ValueError, match="bridges"),
    ):
        validate_runtime_config(
            SimpleNamespace(kv_cache_dtype="fp4", index_cache_dtype="fp8")
        )


# ─────────────────────────────────────── the draft's own KV dtype ──


class _Backend:
    """What `get_attn_backend` answers for an MHA draft (aiter is not here)."""

    DRAFT_OWNS_KV_POOL = True


@pytest.fixture(autouse=True)
def mha_draft_backend(monkeypatch):
    from atom.utils import selector

    monkeypatch.setattr(selector, "get_attn_backend", lambda family: _Backend)


def test_the_own_pool_predicate_is_the_draft_kv_builders(monkeypatch):
    """`resolve_draft_kv_dtype` asks the backend, as `draft_kv_builder` does: a
    backend with no pool of its own for drafts makes the draft share the target's."""

    class Shared:
        DRAFT_OWNS_KV_POOL = False

    from atom.utils import selector

    monkeypatch.setattr(selector, "get_attn_backend", lambda family: Shared)
    assert resolve_draft_kv_dtype(draft(), "fp4", is_m3=True) == "fp4"
    with pytest.raises(ValueError, match="target's KV pool"):
        resolve_draft_kv_dtype(draft(asked="fp8"), "fp4", is_m3=True)


def draft(method="dspark", **hf):
    """A speculative config as `Config` holds it once built: just what
    `resolve_draft_kv_dtype` reads."""
    return SimpleNamespace(
        method=method,
        kv_cache_dtype=hf.pop("asked", None),
        draft_model_hf_config=PretrainedConfig(model_type="minimax_m3", **hf),
    )


@pytest.mark.parametrize("target", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize(
    "asked, want", [(None, "fp8"), ("fp8", "fp8"), ("bf16", "bf16")]
)
def test_an_own_pool_draft_defaults_to_fp8_whatever_the_target(target, asked, want):
    spec = draft(asked=asked)
    assert resolve_draft_kv_dtype(spec, target, is_m3=True) == want


def test_an_own_pool_draft_refuses_fp4_rather_than_map_it():
    with pytest.raises(ValueError, match="no FP4 KV layout"):
        resolve_draft_kv_dtype(draft(asked="fp4"), "fp4", is_m3=True)


@pytest.mark.parametrize("method", ["mtp", "eagle3"])
def test_a_shared_pool_draft_has_the_targets_dtype(method):
    assert resolve_draft_kv_dtype(draft(method), "fp4", is_m3=True) == "fp4"
    assert (
        resolve_draft_kv_dtype(draft(method, asked="fp8"), "fp8", is_m3=True) == "fp8"
    )
    with pytest.raises(ValueError, match="caches in the target's KV pool"):
        resolve_draft_kv_dtype(draft(method, asked="bf16"), "fp8", is_m3=True)


def test_a_dspark_draft_of_a_non_m3_target_is_shared_too():
    # not an M3 target: no pool of its own, so the target's dtype and no second one
    assert resolve_draft_kv_dtype(draft(), "bf16", is_m3=False) == "bf16"
    with pytest.raises(ValueError, match="target's KV pool"):
        resolve_draft_kv_dtype(draft(asked="fp8"), "bf16", is_m3=False)


@pytest.fixture(autouse=True)
def no_synthetic_acceptance_import(monkeypatch):
    """`SpeculativeConfig` pulls in triton (the rejection sampler) to resolve a
    forced-acceptance knob these tests never set."""
    monkeypatch.setattr(SpeculativeConfig, "_resolve_synthetic_acceptance", lambda s: 0)


def _dspark_spec(**kw):
    """The standalone M3.1 DSpark draft, built the way the CLI builds it."""
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


@pytest.mark.parametrize("asked", [None, "bf16", "fp8"])
def test_fp4_target_with_a_dspark_draft_resolves_the_draft_in_config(
    monkeypatch, asked
):
    config = _m3(
        monkeypatch,
        kv_cache_dtype="fp4",
        hf=sparse_hf(),
        speculative_config=_dspark_spec(kv_cache_dtype=asked),
    )
    assert config.kv_cache_dtype == "fp4"
    assert config.speculative_config.kv_cache_dtype == (asked or "fp8")


def test_the_draft_dtype_keys_the_compile_cache(monkeypatch):
    hashes = {
        asked: _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            hf=sparse_hf(),
            speculative_config=_dspark_spec(kv_cache_dtype=asked),
        ).compute_hash()
        for asked in ("fp8", "bf16")
    }
    assert hashes["fp8"] != hashes["bf16"]


def test_a_draft_dtype_beside_the_targets_cannot_share_the_offload_namespace(
    monkeypatch,
):
    # the offload namespace keys the target's KV dtype alone
    monkeypatch.setenv("ATOM_KV_OFFLOAD", "lmcache")
    with pytest.raises(
        ValueError,
        match="draft KV dtype fp8 differs.*--spec-decode-kv-cache-dtype bf16",
    ):
        _m3(
            monkeypatch,
            kv_cache_dtype="bf16",
            speculative_config=_dspark_spec(),
        )
    # the remedy it names works
    config = _m3(
        monkeypatch,
        kv_cache_dtype="bf16",
        speculative_config=_dspark_spec(kv_cache_dtype="bf16"),
    )
    assert config.speculative_config.kv_cache_dtype == "bf16"


def test_an_fp4_target_with_a_draft_pool_in_it_is_refused_in_config(monkeypatch):
    with pytest.raises(ValueError, match="draft would have FP4 KV"):
        _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            hf=sparse_hf(),
            speculative_config=SpeculativeConfig(
                method="eagle3",
                model="draft",
                num_speculative_tokens=2,
                draft_model_hf_config=PretrainedConfig(
                    architectures=["LlamaForCausalLMEagle3"], model_type="llama"
                ),
            ),
        )


def _skip_hf(**kw):
    return {**sparse_hf(layers=6), **kw}


def test_use_index_cache_alone_is_not_a_skip_index_layer(monkeypatch):
    kv_check(monkeypatch, hf=_skip_hf(use_index_cache=True, index_topk_freq=1))
    # and neither is a freq with the cache off: the model skips nothing then
    kv_check(monkeypatch, hf=_skip_hf(use_index_cache=False, index_topk_freq=2))


@pytest.mark.parametrize(
    "extra",
    [
        {"index_topk_freq": 2},  # every second sparse layer reuses the last top-k
        {"index_topk_pattern": ["F", "S", "F", "S", "F", "S"]},
    ],
)
def test_a_layer_that_reuses_the_top_k_has_no_mxfp4_writer(monkeypatch, extra):
    with pytest.raises(ValueError, match="reuses another's index top-k"):
        kv_check(monkeypatch, hf=_skip_hf(use_index_cache=True, **extra))


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"index_topk_freq": 1},
        {"index_topk_freq": 2},
        {"index_topk_freq": 3, "index_skip_topk_offset": 2},
        {"index_topk_pattern": ["F", "S", "F", "S", "F", "S"]},
        {"index_topk_pattern": ["F", "F", "F", "F", "F", "F"]},
        {"index_topk_pattern": ["F", "F"]},  # shorter than the model
    ],
)
@pytest.mark.parametrize("use_cache", [True, False])
def test_the_startup_skip_rule_is_the_models(extra, use_cache):
    """The gate keeps a GPU-free copy of the model's rule; hold them together."""
    from atom.models.minimax_m3.config import _reuses_index_topk

    model = pytest.importorskip("atom.models.minimax_m3.model")
    text = PretrainedConfig(**_skip_hf(use_index_cache=use_cache, **extra))
    want = any(model._should_skip_minimax_m3_index_topk(text, i)[0] for i in range(6))
    assert _reuses_index_topk(text, 6) is want


def test_the_startup_skip_rule_needs_no_model_import():
    from atom.models.minimax_m3.config import _reuses_index_topk

    text = PretrainedConfig(use_index_cache=True, index_topk_freq=2)
    assert _reuses_index_topk(text, 6) and not _reuses_index_topk(text, 1)


def test_an_all_full_index_pattern_is_not_a_skip(monkeypatch):
    kv_check(
        monkeypatch,
        hf=_skip_hf(use_index_cache=True, index_topk_pattern=["F"] * 6),
    )


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_m3_keeps_the_kv_and_takes_an_mxfp4_index_beside_it(monkeypatch, kv):
    for dtype in (torch.bfloat16, torch.float16):
        config = _m3(
            monkeypatch,
            kv_cache_dtype=kv,
            index_cache_dtype="fp4",
            hf={"dtype": dtype},
        )
        assert (config.kv_cache_dtype, config.index_cache_dtype) == (kv, "fp4")


def test_the_m3_index_defaults_to_the_kv_dtype(monkeypatch):
    config = _m3(monkeypatch, kv_cache_dtype="fp8")
    assert (config.kv_cache_dtype, config.index_cache_dtype) == ("fp8", "fp8")


def test_m3_mxfp4_index_keeps_the_m3_startup_refusals(monkeypatch):
    with pytest.raises(ValueError, match="128 tokens"):
        _make_config(monkeypatch, index_cache_dtype="fp4", kv_cache_block_size=16)


def test_m3_mxfp4_index_hashes_apart_from_fp8(monkeypatch):
    fp8 = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp8")
    fp4 = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp4")

    assert fp8.compute_hash() != fp4.compute_hash()
