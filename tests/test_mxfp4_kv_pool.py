# SPDX-License-Identifier: MIT
"""MiniMax-M3.1's MXFP4 sparse KV pool (``--kv-cache-dtype fp4``): bytes and regions.

One (page, kv head) of K and of V is one 8704 B uint8 field -- [128 x 64 B e2m1]
[128 x 4 B e8m0] -- with no fp32 scale planes. Pinned here: what an entry
costs, the views a binder hands the sparse reader, the regions a transfer would
publish, and that bf16 / fp8 pools are the same bytes with and without the
layout argument. The pool half is pure torch; the builder half needs aiter.
"""

import itertools
import types

import pytest
import torch

from atom.model_ops.attentions.mha_kv_pool import MhaKvPool, ShuffleLayout
from atom.model_ops.attentions.pool_layout.entry_arena import EntryField
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.model_ops.minimax_m3.mxfp4_pages import (
    PAGE_BYTES,
    PAGE_TOKENS,
    Mxfp4KvLayout,
    page_views,
)

FP8 = torch.float8_e4m3fnuz


def pool(layers=3, heads=1, extra=(), **kw):
    return MhaKvPool(
        layers=layers,
        block_size=128,
        num_kv_heads=heads,
        head_dim=128,
        kv_dtype=torch.uint8,
        layout=Mxfp4KvLayout(),
        extra_fields=extra,
        **kw,
    )


def test_the_page_is_8704_bytes_of_128_tokens():
    assert (PAGE_TOKENS, PAGE_BYTES) == (128, 8704)


def test_a_token_costs_136_bytes_of_k_and_v():
    p = pool(layers=60, heads=1)
    # 60 layers x (K + V) x 8704 B a 128-token block; 256 B field alignment holds.
    assert p.entry_bytes == 60 * 2 * PAGE_BYTES
    assert p.entry_bytes / (60 * 128) == 136
    assert p.scale_fields == []
    assert [f.name for f in p.cache_fields] == ["k", "v"]


@pytest.mark.parametrize("heads", [1, 2, 4])
def test_views_are_pages_of_heads_and_alias_the_pool(heads):
    p = pool(layers=2, heads=heads)
    p.allocate(5, "cpu")
    k, v = p.kv_views(1)
    assert k.shape == (5, heads, PAGE_BYTES) == v.shape
    assert k.dtype == v.dtype == torch.uint8
    assert k.stride() == (heads * PAGE_BYTES, PAGE_BYTES, 1)
    k.fill_(7)
    assert not (v == 7).any()
    assert not (p.kv_views(0)[0] == 7).any()
    # the reader splits a page into its data and scale, per (page, head)
    data, scale = page_views(k)
    assert data.shape == (5, heads, PAGE_TOKENS, 64)
    assert scale.shape == (5, heads, PAGE_TOKENS, 4)


def test_only_the_128_page_of_head_dim_128_is_a_pool():
    base = {
        "layers": 1,
        "block_size": 128,
        "num_kv_heads": 1,
        "head_dim": 128,
        "kv_dtype": torch.uint8,
        "layout": Mxfp4KvLayout(),
    }
    for kw in ({"blocks_per_entry": 2}, {"head_dim": 64}, {"block_size": 16}):
        with pytest.raises(ValueError):
            MhaKvPool(**{**base, **kw})


def test_regions_publish_k_v_and_index_whole_with_no_scale_region():
    """What a transfer registers is exactly the declared bytes: one region per
    (field, layer), K then V then the index key, each a whole page-head unit."""
    heads, layers, entries = 2, 3, 4
    index = EntryField("index", layers, (PAGE_BYTES,), torch.uint8)
    p = pool(layers=layers, heads=heads, extra=[index])
    p.allocate(entries, "cpu")

    regions = p.region_tensors()
    roles = [name.split(".")[0] for name, _ in regions]
    assert roles == ["k"] * layers + ["v"] * layers + ["index"] * layers
    for name, t in regions:
        unit = t.stride(0) * t.element_size()
        want = PAGE_BYTES if name.startswith("index") else heads * PAGE_BYTES
        assert (t.dtype, t.shape[0], unit) == (torch.uint8, entries, want), name
        assert t.is_contiguous()
    published = sum(f.bytes_per_entry for g in p.field_groups for f in g)
    assert published == layers * (2 * heads + 1) * PAGE_BYTES
    assert p.entry_bytes >= published  # only alignment padding is unpublished
    # K, V and index stay disjoint bytes of the one buffer
    spans = sorted((t.data_ptr(), t.data_ptr() + t.numel()) for _, t in regions)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(spans))


def test_the_index_field_rides_beside_the_kv_unchanged():
    index = EntryField("index", 3, (PAGE_BYTES,), torch.uint8)
    plain, with_index = pool(), pool(extra=[index])
    assert with_index.entry_bytes - plain.entry_bytes == 3 * PAGE_BYTES
    with_index.allocate(2, "cpu")
    assert with_index.field_view("index", 1, torch.uint8, (-1, PAGE_BYTES)).shape == (
        2,
        PAGE_BYTES,
    )


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, FP8])
def test_the_dtype_wide_pools_are_the_same_with_and_without_a_layout(kv_dtype):
    kw = {"layers": 4, "block_size": 128, "num_kv_heads": 2, "head_dim": 128}
    a = MhaKvPool(**kw, kv_dtype=kv_dtype)
    b = MhaKvPool(**kw, kv_dtype=kv_dtype, layout=ShuffleLayout(kv_dtype))
    assert a.entry_bytes == b.entry_bytes
    assert [(f.name, f.shape) for g in a.field_groups for f in g] == [
        (f.name, f.shape) for g in b.field_groups for f in g
    ]
    # bf16 buys no scale planes; fp8 does
    assert bool(a.scale_fields) == (kv_dtype.itemsize == 1)
    a.allocate(3, "cpu")
    k, v = a.kv_views(0)
    x = shuffle_pack_of(kv_dtype)
    assert k.shape == (3, 2, 128 // x, 128, x) and v.shape == (3, 2, 128 // x, 128, x)


def shuffle_pack_of(dtype):
    return 16 // dtype.itemsize


def test_a_geometry_row_space_names_itself_by_its_format():
    assert str(KvGeometry(2, 256)) == "h2d256"
    assert f"mha.{KvGeometry(8, 128, Mxfp4KvLayout())}.k.layer_0" == (
        "mha.h8d128fp4.k.layer_0"
    )
    # the layout is part of the key: an MXFP4 pool and a dense one never merge
    assert KvGeometry(1, 128, Mxfp4KvLayout()) != KvGeometry(1, 128)
    assert hash(KvGeometry(1, 128, Mxfp4KvLayout())) == hash(
        KvGeometry(1, 128, Mxfp4KvLayout())
    )


# --- the builder: sizing and binding read one declaration (needs aiter) ----------

aiter = pytest.importorskip("aiter")

from atom.model_ops.attentions import aiter_attention as mod


class _Impl:
    is_indexed_sparse_attention = True


def _sparse(heads=1):
    return types.SimpleNamespace(
        impl=_Impl(),
        base_attention=True,
        use_mla=False,
        num_kv_heads=heads,
        head_dim=128,
        layer_num=0,
        k_scale=None,
        v_scale=None,
    )


def _dense(heads=1):
    m = _sparse(heads)
    m.impl = types.SimpleNamespace()
    return m


def _builder(kv_dtype, layers=3, module=_sparse):
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size, b.block_ratio = 128, 1
    b._has_sparse_attention = True
    mods = [module() for _ in range(layers)]
    hf = types.SimpleNamespace(sparse_attention_config={"sparse_index_dim": 128})
    config = types.SimpleNamespace(
        kv_cache_dtype=kv_dtype, index_cache_dtype="fp4", hf_config=hf, max_model_len=8
    )
    b.model_runner = types.SimpleNamespace(
        config=config, device="cpu", draft_shares_kv_pool=lambda: False
    )
    b.model_runner.model = types.SimpleNamespace(modules=lambda: iter(mods))
    return b, mods


def test_an_fp4_sparse_layer_is_an_mxfp4_row_space_and_bf16_fp8_are_not():
    fp4, (m,) = _builder("fp4", layers=1)
    assert fp4._module_kinds(m)[1] == KvGeometry(1, 128, Mxfp4KvLayout())
    for dtype in ("bf16", "fp8"):
        b, (m,) = _builder(dtype, layers=1)
        assert b._module_kinds(m)[1] == KvGeometry(1, 128)
    # a module that is not an indexed sparse layer keeps the dtype-wide cache
    b, (m,) = _builder("fp4", layers=1, module=_dense)
    assert b._module_kinds(m) == (KvGeometry(1, 128),)


def test_sizing_binding_and_regions_agree_for_fp4():
    layers = 3
    b, mods = _builder("fp4", layers=layers)
    index_bytes = layers * PAGE_BYTES
    # sizing: K + V of every layer, plus the index key, 256 B aligned fields
    want = layers * 2 * PAGE_BYTES + index_bytes
    assert b._paged_entry_bytes() == want
    assert b.paged_pool_bytes(5) == 5 * want

    blocks = 5
    buf = torch.zeros(b.paged_pool_bytes(blocks), dtype=torch.uint8)
    b.allocate_kv_cache_tensors(blocks=blocks, buf=buf)
    ((geometry, p),) = b.kv_pools.items()
    assert str(geometry) == "h1d128fp4" and p.scale_fields == []

    bound = [b.build_kv_cache_tensor(m) for m in mods]
    for row, (m, t) in enumerate(zip(mods, bound)):
        assert t.k_cache.shape == t.v_cache.shape == (blocks, 1, PAGE_BYTES)
        assert t.k_cache.dtype == torch.uint8
        assert t.k_cache.data_ptr() == p.kv_views(row)[0].data_ptr()
        assert m.k_scale is None and m.v_scale is None
        assert m.impl.index_cache.shape == (blocks, PAGE_BYTES)

    # the transfer sees the same three fields, nothing else
    pages = b.get_kv_transfer_tensors().pages
    assert len(pages) == layers * 3
    assert {r.region.unit_bytes for r in pages} == {PAGE_BYTES}
    assert sum(r.region.total_bytes for r in pages) == blocks * want


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_bf16_fp8_sparse_layers_are_not_changed(dtype):
    b, _ = _builder(dtype, layers=2)
    from aiter import dtypes

    kv_dtype = dtypes.d_dtypes[dtype]
    ref = MhaKvPool(
        layers=2,
        block_size=128,
        num_kv_heads=1,
        head_dim=128,
        kv_dtype=kv_dtype,
        extra_fields=b._index_cache_fields(128),
    )
    assert b._paged_entry_bytes() == ref.entry_bytes
    (p,) = b._declare_kv_pools().values()
    assert p.layout == ShuffleLayout(kv_dtype)
