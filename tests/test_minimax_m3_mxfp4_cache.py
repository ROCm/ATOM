# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's MXFP4 KV pool, its startup refusals, and the draft's own KV dtype.

The pool lays one (page, kv head) of K, of V and of the index key out as one
8704 B field -- [128 x 64 B e2m1][128 x 4 B e8m0] -- which is what the aiter
writer produces and `pa_decode` (MXFP4 K/V) / `score_flydsl_fp4` read. Pinned here:
the bytes per token the capacity arithmetic rests on, the views the kernels are
handed, and the combinations that must fail at startup and not in a forward.
`--kv-cache-dtype fp4` takes an all-indexed-sparse model (MiniMax-M3.1) and refuses,
with no FP8 fallback, what it has no MXFP4 path for: dense attention layers, KV
offload, an fp16 model, an FP4 draft. A DSpark draft with a pool of its own keeps
the target's dtype unless asked (fp8 under an fp4 target, which has no draft layout);
the index dtype stays its own choice.
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
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.mxfp4_pages import (
    PAGE_BYTES,
    Mxfp4KvLayout,
    page_views,
)
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


@pytest.mark.parametrize("heads", [1, 2])
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


def check(monkeypatch, *, hf=None, text_config=False, **kw):
    """`validate_runtime_config` on the one M3.1-shaped config, a field changed."""
    text = PretrainedConfig(**(sparse_hf() if hf is None else hf))
    fields = {
        "kv_cache_dtype": "fp4",
        "index_cache_dtype": "fp4",
        "kv_cache_block_size": 128,
        "torch_dtype": torch.bfloat16,
        "kv_transfer_config": {},
        "speculative_config": None,
        "hf_config": SimpleNamespace(text_config=text) if text_config else text,
    }
    plugin = kw.pop("plugin_mode", False)
    with mock.patch.object(config_module, "is_plugin_mode", lambda: plugin):
        validate_runtime_config(SimpleNamespace(**{**fields, **kw}))


@pytest.mark.parametrize("text_config", [False, True])
def test_an_all_sparse_m3_passes_the_kv_gate(monkeypatch, text_config):
    check(monkeypatch, text_config=text_config)


def test_an_fp4_kv_with_another_index_cache_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="needs --index-cache-dtype fp4, got fp8"):
        check(monkeypatch, index_cache_dtype="fp8")


@pytest.mark.parametrize("kv", ["fp4", "fp8"])  # the MXFP4 index alone is held to it
def test_the_shared_mxfp4_refusals(monkeypatch, chip_info, kv):
    kw = {"kv_cache_dtype": kv, "index_cache_dtype": "fp4"}
    with pytest.raises(ValueError, match="bridges"):
        check(monkeypatch, plugin_mode=True, **kw)
    with pytest.raises(ValueError, match="128 tokens"):
        check(monkeypatch, kv_cache_block_size=16, **kw)
    chip_info.get_gfx_runtime = lambda: "gfx942"
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        check(monkeypatch, **kw)


@pytest.mark.parametrize(
    "hf",
    [
        sparse_hf(freq=[1, 1, 0, 1]),  # one dense layer
        sparse_hf(freq=[1, 1, 1]),  # one entry short of num_hidden_layers
        sparse_hf(freq=[1, 1, 2, 1]),  # non-zero, but not the model's `1`
        sparse_hf(freq=[1, 1, True, 1]),  # truthy, but not an int 1
        sparse_hf(use_sparse_attention=False),
        {**sparse_hf(freq=[1, 0, 1, 1]), "moe_layer_freq": [1] * 4},  # not the MLPs
        {"num_hidden_layers": 4},  # M3's original: no sparse config at all
        {k: v for k, v in sparse_hf().items() if k != "num_hidden_layers"},
    ],
)
def test_the_fp4_kv_refuses_a_model_that_is_not_all_indexed_sparse(monkeypatch, hf):
    with pytest.raises(ValueError, match="indexed sparse"):
        check(monkeypatch, hf=hf)


def test_the_fp4_kv_refuses_a_draft_resolved_to_fp4(monkeypatch):
    # EAGLE3 / MTP share the target's pool: fp4 target, fp4 draft, no layout for it
    with pytest.raises(ValueError, match="draft would have FP4 KV"):
        check(
            monkeypatch,
            speculative_config=SimpleNamespace(method="eagle3", kv_cache_dtype="fp4"),
        )


def test_the_fp4_kv_refuses_kv_offload_and_kv_transfer_config(monkeypatch):
    # what EngineArgs folds ATOM_KV_OFFLOAD into, or a P/D connector
    with pytest.raises(ValueError, match="KV offload"):
        check(monkeypatch, kv_transfer_config={"kv_connector": "lmcache_offload"})
    check(monkeypatch, kv_transfer_config={})  # nothing configured: no refusal


def test_the_fp4_kv_needs_a_bf16_model(monkeypatch):
    with pytest.raises(ValueError, match="needs a bfloat16 model, got torch.float16"):
        check(monkeypatch, torch_dtype=torch.float16)
    # the MXFP4 index alone never reaches the bf16-only pa_decode kernel
    check(
        monkeypatch,
        hf={},
        kv_cache_dtype="fp8",
        index_cache_dtype="fp4",
        torch_dtype=torch.float16,
    )


def test_v41_still_takes_a_fp4_kv():  # the refusal elsewhere: test_v4_indexer_defaults
    _check_fp4_kv("fp4", is_m3=False, is_v41=True)


@pytest.mark.parametrize(
    ("kv", "index", "want"),
    [
        # KV fp4: the index follows it (the MXFP4 attention needs the MXFP4 scorer).
        ("fp4", None, ("fp4", "fp4")),
        # index-only: KV stays as asked.
        ("bf16", "fp4", ("bf16", "fp4")),
        ("fp4", "fp4", ("fp4", "fp4")),
        ("fp8", None, ("fp8", "fp8")),
    ],
)
def test_m3_fp4_keeps_the_kv_and_leaves_the_index_to_its_own_flag(
    monkeypatch, kv, index, want
):
    config = _m3(
        monkeypatch, kv_cache_dtype=kv, index_cache_dtype=index, hf=sparse_hf()
    )

    assert (config.kv_cache_dtype, config.index_cache_dtype) == want


def test_m3_fp4_hashes_apart_from_fp8(monkeypatch):
    hf = sparse_hf()
    fp8 = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp8", hf=hf)
    fp4 = _m3(monkeypatch, kv_cache_dtype="fp4", index_cache_dtype="fp4", hf=hf)
    index_only = _m3(monkeypatch, kv_cache_dtype="fp8", index_cache_dtype="fp4", hf=hf)

    assert len({fp8.compute_hash(), fp4.compute_hash(), index_only.compute_hash()}) == 3


# ────────────────────────────────────────────── the draft's own KV dtype ──


class _Backend:
    """What `get_attn_backend` answers for an MHA draft (aiter is not here)."""

    DRAFT_OWNS_KV_POOL = True


@pytest.fixture(autouse=True)
def mha_draft_backend(monkeypatch):
    from atom.utils import selector

    monkeypatch.setattr(selector, "get_attn_backend", lambda family: _Backend)


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
    "asked, want",
    [(None, None), ("fp8", "fp8"), ("bf16", "bf16")],  # None: the target's, fp8 for fp4
)
def test_an_own_pool_draft_keeps_the_targets_dtype_unless_asked(target, asked, want):
    default = "fp8" if target == "fp4" else target  # fp4 has no draft layout
    assert resolve_draft_kv_dtype(draft(asked=asked), target, own_pool=True) == (
        want or default
    )


def test_an_own_pool_draft_refuses_fp4_rather_than_map_it():
    with pytest.raises(ValueError, match="no FP4 KV layout"):
        resolve_draft_kv_dtype(draft(asked="fp4"), "fp4", own_pool=True)


def test_a_shared_pool_draft_has_the_targets_dtype():
    assert resolve_draft_kv_dtype(draft("eagle3"), "fp4", own_pool=False) == "fp4"
    with pytest.raises(ValueError, match="caches in the target's KV pool"):
        resolve_draft_kv_dtype(draft("eagle3", asked="bf16"), "fp8", own_pool=False)
    assert resolve_draft_kv_dtype(draft(), "bf16", own_pool=False) == "bf16"
    with pytest.raises(ValueError, match="target's KV pool"):
        resolve_draft_kv_dtype(draft(asked="fp8"), "bf16", own_pool=False)


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


def test_fp4_target_with_a_dspark_draft_resolves_and_hashes_the_draft_dtype(
    monkeypatch,
):
    hashes = {}
    for asked in (None, "bf16", "fp8"):
        config = _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            hf=sparse_hf(),
            speculative_config=_dspark_spec(kv_cache_dtype=asked),
        )
        assert config.kv_cache_dtype == "fp4"
        assert config.speculative_config.kv_cache_dtype == (asked or "fp8")
        hashes[asked or "fp8"] = config.compute_hash()
    # the draft dtype keys the compile cache
    assert hashes["fp8"] != hashes["bf16"]


OFFLOAD = {"kv_connector": "lmcache_offload", "kv_role": "offload"}


@pytest.mark.parametrize("target", ["bf16", "fp8"])
def test_an_own_pool_draft_without_a_flag_keeps_the_targets_dtype(monkeypatch, target):
    """No flag changed, no draft change: a bf16 or fp8 target's own-pool DSpark draft
    is bf16 / fp8 as it was before it could be asked, offload or not."""
    for extra in ({}, {"kv_transfer_config": OFFLOAD}):
        config = _m3(
            monkeypatch,
            kv_cache_dtype=target,
            speculative_config=_dspark_spec(),
            **extra,
        )
        assert config.speculative_config.kv_cache_dtype == target


def test_a_draft_dtype_beside_the_targets_cannot_share_the_offload_namespace(
    monkeypatch,
):
    # the offload namespace keys the target's KV dtype alone
    with pytest.raises(
        ValueError,
        match="draft KV dtype fp8 differs.*--spec-decode-kv-cache-dtype bf16",
    ):
        _m3(
            monkeypatch,
            kv_cache_dtype="bf16",
            kv_transfer_config=OFFLOAD,
            speculative_config=_dspark_spec(kv_cache_dtype="fp8"),
        )
    # the remedy it names works
    config = _m3(
        monkeypatch,
        kv_cache_dtype="bf16",
        kv_transfer_config=OFFLOAD,
        speculative_config=_dspark_spec(kv_cache_dtype="bf16"),
    )
    assert config.speculative_config.kv_cache_dtype == "bf16"


def test_an_fp4_target_beside_offload_is_refused_for_that_not_for_its_draft(
    monkeypatch,
):
    """The draft (fp8 under an fp4 target) can never match it, so the draft-dtype
    refusal would send the user to `--spec-decode-kv-cache-dtype fp4`, which is
    refused in turn: the real reason, the fp4 KV beside offload, comes first."""
    with pytest.raises(ValueError, match="KV offload") as err:
        _m3(
            monkeypatch,
            kv_cache_dtype="fp4",
            hf=sparse_hf(),
            kv_transfer_config=OFFLOAD,
            speculative_config=_dspark_spec(),
        )
    assert "spec-decode-kv-cache-dtype" not in str(err.value)


def test_a_dspark_draft_with_a_pool_of_its_own_takes_its_dtype_on_any_model(
    monkeypatch,
):
    """Not an M3 matter: the backend says whether the draft has its own pool."""
    config = _make_config(
        monkeypatch,
        architectures=["LlamaForCausalLM"],
        kv_cache_dtype="fp8",
        speculative_config=_dspark_spec(kv_cache_dtype="bf16"),
    )
    assert config.speculative_config.kv_cache_dtype == "bf16"
