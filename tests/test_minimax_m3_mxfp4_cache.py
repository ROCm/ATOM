"""MiniMax-M3's MXFP4 index cache page views and its startup checks.

The index cache is one 8704 B uint8 page a block -- [128 x 64 B e2m1][128 x 4 B
e8m0] -- which is what the aiter writer produces and `score_flydsl_fp4` reads.
Pinned here: the views the scorer is handed, the combinations that must fail at
startup and not in a forward, and that `--kv-cache-dtype fp4` is refused (the
index cache is the only MXFP4 cache M3 has).
"""

import sys
import types
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import CompilationConfig, Config, _check_fp4_kv
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.models.minimax_m3.config import validate_runtime_config
from tests.test_v4_indexer_defaults import _GfxProbe, _install_chip_info_stub

M3 = ["MiniMaxM3SparseForConditionalGeneration"]


def _make_config(monkeypatch, *, architectures=M3, hf=None, **overrides):
    """A real `Config` for a MiniMax-M3 checkpoint, with the HF config, quantization
    and chip probe stubbed (gfx950 unless `gfx` is overridden)."""
    gfx = overrides.pop("gfx", "gfx950")
    hf_config = PretrainedConfig(
        architectures=architectures,
        max_position_embeddings=8192,
        sparse_attention_config={"sparse_local_block": 1, "sparse_block_size": 128},
        **(hf or {}),
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


def test_a_fp4_kv_is_v41s_and_refused_elsewhere():
    _check_fp4_kv("fp8", is_v41=False)
    _check_fp4_kv("fp4", is_v41=True)
    with pytest.raises(ValueError, match="DeepSeek-V4.1 only"):
        _check_fp4_kv("fp4", is_v41=False)


@pytest.mark.parametrize("index", [None, "fp8", "fp4"])
def test_m3_refuses_a_fp4_kv_at_startup(monkeypatch, index):
    with pytest.raises(ValueError, match="DeepSeek-V4.1 only"):
        _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype=index)


def test_an_m3_export_named_only_by_model_type_refuses_a_fp4_kv(monkeypatch):
    with pytest.raises(ValueError, match="DeepSeek-V4.1 only"):
        _make_config(
            monkeypatch,
            architectures=["SomeOtherForCausalLM"],
            hf={"model_type": "minimax_m3"},
            kv_cache_block_size=128,
            kv_cache_dtype="fp4",
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
