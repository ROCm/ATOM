# SPDX-License-Identifier: MIT
"""MiniMax-M3's MXFP4 sparse KV pool and its startup refusals.

The pool lays one (page, kv head) of K, of V and of the index key out as one
8704 B field -- [128 x 64 B e2m1][128 x 4 B e8m0] -- which is what the aiter
writer produces and `pa_decode` (MXFP4 K/V) / `score_flydsl_fp4` read. Pinned here:
the bytes per token the capacity arithmetic rests on, the views the kernels are
handed, and the combinations that must fail at startup and not in a forward.
"""

import sys
import types
from unittest import mock

import pytest
import torch
from transformers import PretrainedConfig

from atom import config as config_module
from atom.config import CompilationConfig, Config, _check_fp4_kv
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.mxfp4_pages import (
    PAGE_BYTES,
    Mxfp4KvLayout,
    page_views,
)
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


def pool(layers=3, heads=1, **kw):
    return MhaKvPool(
        layers=layers,
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.float8_e4m3fnuz,
        layout=Mxfp4KvLayout(),
        **kw,
    )


def test_a_token_costs_136_bytes_of_k_and_v():
    p = pool(layers=60, heads=1)
    # 60 layers x (K + V) x 8704 B a 128-token block, 256 B field alignment holds.
    assert p.entry_bytes == 60 * 2 * PAGE_BYTES
    assert p.entry_bytes / (60 * 128) == 136
    assert p.scale_fields == []


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_views_are_pages_of_heads_and_alias_the_pool(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, heads, PAGE_BYTES) == v.shape
    assert k.dtype == torch.uint8
    assert k.stride() == (heads * PAGE_BYTES, PAGE_BYTES, 1)
    k.fill_(7)
    assert not (v == 7).any()
    assert not (p.kv_views(0)[0] == 7).any()


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


@pytest.mark.parametrize(
    "kw", [{"blocks_per_entry": 2}, {"num_kv_heads": 1, "head_dim": 64}]
)
def test_only_the_128_page_of_head_dim_128_is_a_pool(kw):
    base = {
        "layers": 1,
        "block_size": 128,
        "num_kv_heads": 1,
        "head_dim": 128,
        "kv_dtype": torch.float8_e4m3fnuz,
        "layout": Mxfp4KvLayout(),
    }
    with pytest.raises(ValueError):
        MhaKvPool(**{**base, **kw})


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
        "kv_fp4": True,
        "block_size": 128,
        "plugin_mode": False,
        "torch_dtype": torch.bfloat16,
        **kw,
    }
    config = types.SimpleNamespace(
        kv_cache_dtype="fp4" if a["kv_fp4"] else "fp8",
        index_cache_dtype=a["index_cache_dtype"],
        kv_cache_block_size=a["block_size"],
        torch_dtype=a["torch_dtype"],
        hf_config=types.SimpleNamespace(),
    )
    with mock.patch.object(config_module, "is_plugin_mode", lambda: a["plugin_mode"]):
        validate_runtime_config(config)


def test_refusals_name_what_is_wrong():
    with pytest.raises(ValueError, match="bridges"):
        check(plugin_mode=True)
    with pytest.raises(ValueError, match="128 tokens"):
        check(block_size=16)
    # the MXFP4 index alone, MiniMax-M3's own layout, is held to the same
    with pytest.raises(ValueError, match="128 tokens"):
        check(kv_fp4=False, block_size=16)


def test_the_mxfp4_kv_needs_a_bf16_model():
    # aiter's MXFP4 pa_decode writes bf16 only; the output takes the model dtype
    with pytest.raises(ValueError, match="needs a bfloat16 model, got torch.float16"):
        check(torch_dtype=torch.float16)
    # the MXFP4 index alone never reaches that kernel
    check(kv_fp4=False, torch_dtype=torch.float16)


def test_only_gfx950_is_served(chip_info):
    chip_info.get_gfx_runtime = lambda: "gfx942"
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        check()


def test_the_baselines_are_never_refused():
    # Neither MXFP4 switch on: nothing to check, whatever else is set.
    check(index_cache_dtype="fp8", kv_fp4=False, block_size=16, plugin_mode=True)


def test_a_fp4_kv_is_m3s_or_v41s_and_refused_elsewhere():
    _check_fp4_kv("fp8", is_m3=False, is_v41=False)
    _check_fp4_kv("fp4", is_m3=True, is_v41=False)
    _check_fp4_kv("fp4", is_m3=False, is_v41=True)
    with pytest.raises(ValueError, match="MiniMax-M3 and DeepSeek-V4.1 only"):
        _check_fp4_kv("fp4", is_m3=False, is_v41=False)


@pytest.mark.parametrize(
    ("kv", "index", "want"),
    [
        # KV-only: the index default follows the fp8 storage dtype.
        ("fp4", None, ("fp4", "fp8")),
        # index-only: KV stays as asked.
        ("bf16", "fp4", ("bf16", "fp4")),
        ("fp4", "fp4", ("fp4", "fp4")),
        ("fp4", "fp8", ("fp4", "fp8")),
        ("fp8", None, ("fp8", "fp8")),
    ],
)
def test_m3_fp4_keeps_the_kv_and_defaults_the_index_to_its_storage(
    monkeypatch, kv, index, want
):
    config = _m3(monkeypatch, kv_cache_dtype=kv, index_cache_dtype=index)

    assert (config.kv_cache_dtype, config.index_cache_dtype) == want


def test_the_storage_dtype_is_fp8_only_for_an_m3_fp4_kv(monkeypatch):
    assert _m3(monkeypatch, kv_cache_dtype="fp4").kv_storage_dtype == "fp8"
    assert _m3(monkeypatch, kv_cache_dtype="bf16").kv_storage_dtype == "bf16"
    assert _m3(monkeypatch, kv_cache_dtype="fp8").kv_storage_dtype == "fp8"


def test_m3_fp4_keeps_the_m3_startup_refusals(monkeypatch):
    with pytest.raises(ValueError, match="MXFP4 pages are 128 tokens"):
        _make_config(monkeypatch, kv_cache_dtype="fp4", kv_cache_block_size=16)


def test_m3_fp4_hashes_apart_from_fp8(monkeypatch):
    fp8 = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp8")
    fp4 = _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype="fp4")
    kv_only = _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype="fp8")

    assert len({fp8.compute_hash(), fp4.compute_hash(), kv_only.compute_hash()}) == 3


def test_an_fp16_m3_refuses_mxfp4_kv_at_startup(monkeypatch):
    # aiter's MXFP4 pa_decode writes bf16 only, so the KV flavor needs a bf16 model
    with pytest.raises(ValueError, match="needs a bfloat16 model"):
        _m3(monkeypatch, kv_cache_dtype="fp4", hf={"dtype": torch.float16})
    # bf16 passes; the MXFP4 index alone (and fp8 KV) never reach that kernel
    assert _m3(monkeypatch, kv_cache_dtype="fp4").kv_cache_dtype == "fp4"
    for dtype in (torch.bfloat16, torch.float16):
        config = _m3(
            monkeypatch,
            kv_cache_dtype="bf16",
            index_cache_dtype="fp4",
            hf={"dtype": dtype},
        )
        assert (config.kv_cache_dtype, config.index_cache_dtype) == ("bf16", "fp4")


def test_an_m3_export_named_only_by_model_type_gets_the_m3_mapping(monkeypatch):
    # `_is_minimax_m3_config` (the builder's and the offload layout's test) also
    # reads `model_type`; the MXFP4 gate must agree with it.
    config = _make_config(
        monkeypatch,
        architectures=["SomeOtherForCausalLM"],
        hf={"model_type": "minimax_m3"},
        kv_cache_block_size=128,
        kv_cache_dtype="fp4",
        index_cache_dtype="fp4",
    )

    assert (config.kv_cache_dtype, config.index_cache_dtype) == ("fp4", "fp4")


@pytest.mark.parametrize(
    "draft_arch", ["LlamaForCausalLMEagle3", "DSparkMiniMaxDraftModel"]
)
def test_a_draft_of_an_mxfp4_target_gets_the_fp8_storage_dtype(monkeypatch, draft_arch):
    """The draft has no MXFP4 layers and its own architecture, so a dtype derived
    from its hf_config could not tell: `draft_config` fixes it on the target."""
    target = _m3(monkeypatch, kv_cache_dtype="fp4")
    target.compilation_config.level = config_module.CompilationLevel.PIECEWISE
    level = target.compilation_config.level
    draft = target.draft_config(PretrainedConfig(architectures=[draft_arch]))

    assert (draft.kv_cache_dtype, draft.kv_storage_dtype) == ("fp8", "fp8")
    assert (
        draft.compilation_config.level == config_module.CompilationLevel.NO_COMPILATION
    )
    # the target is untouched
    assert target.kv_cache_dtype == "fp4" and target.compilation_config.level == level
    assert target.hf_config is not draft.hf_config
