# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's NVFP4 KV pool and index pages, and their startup refusals.

An NVFP4 KV page is one buffer [K side | V side] of ``2 * Hkv * 9216`` B (e2m1 data and
e4m3 scales per 16, no per-tensor scale in the page); an index page is
[128 x 64 B data][128 x 8 B scales] = 9216 B. Pinned here: the bytes per token the
capacity arithmetic rests on, the views the kernels are handed, the combinations that
must fail at startup and not in a forward.
"""

import types

import pytest
import torch

from atom.config import _check_fp4_kv, resolve_draft_kv_dtype
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.model_ops.minimax_m3.nvfp4_pages import (
    INDEX_PAGE_BYTES,
    Nvfp4KvLayout,
    head_regions,
    index_page_views,
    kv_page_bytes,
)
from tests.test_minimax_m3_mxfp4_cache import _m3, check, sparse_hf


def pool(layers=3, heads=1, **kw):
    return MhaKvPool(
        layers=layers,
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.uint8,
        layout=Nvfp4KvLayout(),
        **kw,
    )


def test_page_geometry():
    assert INDEX_PAGE_BYTES == 9216 == 128 * (64 + 8)
    assert kv_page_bytes(1) == 18432 and kv_page_bytes(4) == 4 * 18432
    # side regions of 2 heads: K data, K scales, V data, V scales, in this order
    assert head_regions(0, 0, 2) == (0, 2 * 8192)
    assert head_regions(0, 1, 2) == (8192, 2 * 8192 + 1024)
    assert head_regions(1, 1, 2) == (2 * 9216 + 8192, 2 * 9216 + 2 * 8192 + 1024)


def test_a_token_costs_144_bytes_of_k_and_v():
    p = pool(layers=60, heads=1)
    assert p.entry_bytes == 60 * kv_page_bytes(1)
    assert p.entry_bytes / (60 * 128) == 144
    assert p.scale_fields == []


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_the_view_is_one_page_of_both_sides_and_aliases_the_pool(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, kv_page_bytes(heads)) and k.dtype == torch.uint8
    assert k.stride() == (kv_page_bytes(heads), 1) and v.data_ptr() == k.data_ptr()
    k.fill_(7)
    assert not (p.kv_views(0)[0] == 7).any()


def test_index_page_views_split_data_from_scale():
    pages = (torch.arange(6 * INDEX_PAGE_BYTES) % 251).to(torch.uint8)
    pages = pages.view(6, INDEX_PAGE_BYTES)
    data, scale = index_page_views(pages[1:5])
    assert data.shape == (4, 128, 64) and scale.shape == (4, 128, 8)
    assert data.stride() == (INDEX_PAGE_BYTES, 64, 1)
    assert scale.stride() == (INDEX_PAGE_BYTES, 8, 1)
    assert torch.equal(data[0].flatten(), pages[1, :8192])
    assert torch.equal(scale[3].flatten(), pages[4, 8192:])
    with pytest.raises(ValueError):
        index_page_views(pages[:, :8704])


@pytest.mark.parametrize(
    "kw", [{"blocks_per_entry": 2}, {"num_kv_heads": 1, "head_dim": 64}]
)
def test_only_the_128_page_of_head_dim_128_is_a_pool(kw):
    base = {
        "layers": 1,
        "block_size": 128,
        "num_kv_heads": 1,
        "head_dim": 128,
        "kv_dtype": torch.uint8,
        "layout": Nvfp4KvLayout(),
    }
    with pytest.raises(ValueError):
        MhaKvPool(**{**base, **kw})


NV = {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "nvfp4"}


def test_nvfp4_kv_and_index_pass():
    check(**NV)


@pytest.mark.parametrize("kv", ["fp8", "bf16"])
def test_nvfp4_index_needs_nvfp4_kv(kv):
    with pytest.raises(ValueError, match="needs --kv-cache-dtype nvfp4"):
        check(kv_cache_dtype=kv, index_cache_dtype="nvfp4")


@pytest.mark.parametrize(
    "kw, msg",
    [
        (
            {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "bf16"},
            "needs --index-cache-dtype nvfp4, got bf16",
        ),
        (
            {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "fp4"},
            "needs --index-cache-dtype nvfp4, got fp4",
        ),
        (
            {"kv_cache_dtype": "fp4", "index_cache_dtype": "nvfp4"},
            "needs --index-cache-dtype fp4, got nvfp4",
        ),
    ],
)
def test_nvfp4_and_mxfp4_do_not_mix(kw, msg):
    with pytest.raises(ValueError, match=msg):
        check(**kw)


def test_the_shared_refusals_hold_for_nvfp4():
    with pytest.raises(ValueError, match="bridges"):
        check(plugin_mode=True, **NV)
    with pytest.raises(ValueError, match="128 tokens"):
        check(kv_cache_block_size=16, **NV)
    with pytest.raises(ValueError, match="indexed sparse"):
        check(hf=sparse_hf(freq=[1, 1, 0, 1]), **NV)
    with pytest.raises(ValueError, match="KV offload"):
        check(kv_transfer_config={"kv_connector": "x"}, **NV)
    with pytest.raises(ValueError, match="draft would have FP4 KV"):
        check(
            speculative_config=types.SimpleNamespace(
                method="eagle3", draft_kv_dtype="nvfp4"
            ),
            **NV,
        )
    with pytest.raises(ValueError, match="gfx950, got gfx942"):
        check(arch="gfx942", **NV)


def test_nvfp4_kv_is_for_minimax_m3_only():
    _check_fp4_kv("nvfp4", is_m3=True, is_v41=False)
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _check_fp4_kv("nvfp4", is_m3=False, is_v41=True)


def test_a_config_with_nvfp4_kv_defaults_the_index_to_nvfp4(monkeypatch):
    cfg = _m3(monkeypatch, kv_cache_dtype="nvfp4", hf=sparse_hf())
    assert cfg.index_cache_dtype == "nvfp4"


def test_nvfp4_index_is_refused_off_minimax_m3(monkeypatch):
    with pytest.raises(ValueError, match="MiniMax-M3 only"):
        _m3(
            monkeypatch,
            architectures=["LlamaForCausalLM"],
            kv_cache_dtype="bf16",
            index_cache_dtype="nvfp4",
        )


def test_the_own_pool_draft_of_an_nvfp4_target_is_bf16_a_shared_one_keeps_nvfp4():
    assert resolve_draft_kv_dtype("nvfp4", own_pool=True) == "bf16"
    assert resolve_draft_kv_dtype("nvfp4", own_pool=False) == "nvfp4"


def test_nvfp4_index_cache_is_one_9216_byte_uint8_page():
    mod = pytest.importorskip("atom.model_ops.attentions.aiter_attention")

    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size = 128
    b._has_sparse_attention = True
    b.row_counts = lambda: {mod.INDEX_ROWS: 3}
    cfg = types.SimpleNamespace(
        index_cache_dtype="nvfp4",
        hf_config=types.SimpleNamespace(
            sparse_attention_config={"sparse_index_dim": 128}
        ),
    )
    b.model_runner = types.SimpleNamespace(config=cfg)
    assert b._index_cache_spec(128) == (
        3,
        (INDEX_PAGE_BYTES,),
        torch.uint8,
        (-1, INDEX_PAGE_BYTES),
    )
    with pytest.raises(ValueError, match="NVFP4 index cache is 128-wide"):
        b._index_cache_spec(256)
