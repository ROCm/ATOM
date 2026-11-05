# SPDX-License-Identifier: MIT
"""The pools MiniMax-M3 ``--kv-cache-dtype fp4`` declares, target and draft.

Dense, sliding and indexed layers all key one MXFP4 row space per head geometry:
a 128-token [pages, Hkv, 8704] uint8 K/V pool with no scale planes, built without
a dtype-table lookup. The draft's pool is MXFP4 at the draft's own geometry. fp8
keeps the dtype-wide cache. Needs aiter like the other `aiter_attention` builder
tests, so the CPU CI job skips it.
"""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
from aiter import dtypes

from atom.model_ops.attentions import aiter_attention as mod
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, Mxfp4KvLayout
from atom.spec_decode import draft_kv

M3_HF = types.SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"])


class _Attn(nn.Module):
    """One paged, non-MLA layer; `indexed` is the M3 sparse one."""

    def __init__(self, heads, head_dim=128, indexed=False):
        super().__init__()
        self.base_attention = True
        self.use_mla = False
        self.num_kv_heads, self.head_dim = heads, head_dim
        self.impl = types.SimpleNamespace(is_indexed_sparse_attention=indexed)


def config(kv):
    return types.SimpleNamespace(
        kv_cache_dtype=kv, index_cache_dtype="fp8", hf_config=M3_HF
    )


def builder(layers, kv="fp4"):
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size, b.block_ratio, b._has_sparse_attention = 128, 1, False
    b.model_runner = types.SimpleNamespace(
        config=config(kv),
        model=nn.Sequential(*layers),
        draft_shares_kv_pool=lambda: False,
    )
    return b


def geometries(b):
    """The KV row spaces; the indexer's keys are a numbering of their own."""
    return {k: n for k, n in b.row_counts().items() if isinstance(k, KvGeometry)}


# dense, indexed, sliding (a second head count in the same block-id space)
LAYERS = [_Attn(2), _Attn(2, indexed=True), _Attn(2), _Attn(1)]


def test_dense_sliding_and_indexed_layers_share_the_mxfp4_geometry():
    layout = Mxfp4KvLayout()
    assert geometries(builder(LAYERS)) == {
        KvGeometry(2, 128, layout): 3,
        KvGeometry(1, 128, layout): 1,
    }


def test_fp8_geometry_has_no_layout():
    assert {g.layout for g in geometries(builder(LAYERS, kv="fp8"))} == {None}


def test_the_mxfp4_pools_are_uint8_pages_and_never_read_the_dtype_table(monkeypatch):
    # An empty table: a lookup for the layout path would raise KeyError.
    monkeypatch.setattr(dtypes, "d_dtypes", {})
    b = builder(LAYERS)
    pools = b._declare_kv_pools()
    assert len(pools) == 2
    for geometry, pool in pools.items():
        assert pool.scale_fields == [] and pool.block_size == 128
        rows = b.row_counts()[geometry]
        assert pool.entry_bytes == rows * 2 * geometry.num_kv_heads * PAGE_BYTES
    wide = next(p for g, p in pools.items() if g.num_kv_heads == 2)
    wide.allocate(4, "cpu")
    k, v = wide.kv_views(2)
    assert k.shape == v.shape == (4, 2, 8704)
    assert k.dtype == v.dtype == torch.uint8


def test_an_fp8_pool_still_reads_the_dtype_table(monkeypatch):
    monkeypatch.setattr(dtypes, "d_dtypes", {"fp8": torch.uint8})
    pools = builder(LAYERS, kv="fp8")._declare_kv_pools()
    assert all(p.scale_fields for p in pools.values())


def _draft_builder(monkeypatch, draft_hf, kv="fp4"):
    monkeypatch.setattr(dtypes, "d_dtypes", {"fp8": torch.uint8})
    monkeypatch.setattr("atom.utils.selector.attn_family", lambda _hf: "mha")
    monkeypatch.setattr(
        "atom.utils.selector.get_attn_backend", lambda _family: mod.AiterBackend
    )
    runner = types.SimpleNamespace(
        config=config(kv),
        world_size=1,
        drafter=types.SimpleNamespace(model=nn.Sequential(_Attn(2), _Attn(2))),
        attn_metadata_builder=types.SimpleNamespace(block_size=128),
    )
    return draft_kv.draft_kv_builder(runner, draft_hf)


def test_the_draft_pool_is_mxfp4_at_the_drafts_own_geometry(monkeypatch):
    hf = types.SimpleNamespace(num_key_value_heads=2, head_dim=128)
    pool = _draft_builder(monkeypatch, hf).kv_pool
    assert pool.layout == Mxfp4KvLayout() and pool.scale_fields == []
    assert (pool.num_kv_heads, pool.head_dim) == (2, 128)
    assert pool.entry_bytes == 2 * 2 * 2 * PAGE_BYTES


def test_a_draft_whose_heads_are_not_128_wide_is_refused(monkeypatch):
    hf = types.SimpleNamespace(num_key_value_heads=2, head_dim=64)
    with pytest.raises(ValueError, match="MXFP4 KV page"):
        _ = _draft_builder(monkeypatch, hf).kv_pool


def test_an_fp8_draft_pool_keeps_its_scales(monkeypatch):
    hf = types.SimpleNamespace(num_key_value_heads=2, head_dim=128)
    pool = _draft_builder(monkeypatch, hf, kv="fp8").kv_pool
    assert pool.scale_fields


def _prefill(monkeypatch, *, mxfp4):
    """`prepare_prefill` over a no-prefix dense batch; what it publishes."""
    md = types.SimpleNamespace(block_tables=None, has_cached=False)
    monkeypatch.setattr(
        mod.CommonAttentionBuilder,
        "prepare_prefill",
        lambda self, batch, bs, tokens: (md, None),
    )
    monkeypatch.setattr(
        mod,
        "block_table_state",
        lambda _buf: types.SimpleNamespace(publish=lambda bs: ("table", bs)),
    )
    monkeypatch.setattr(mod.envs, "ATOM_USE_UNIFIED_ATTN", False, raising=False)
    b = builder(LAYERS, kv="fp4" if mxfp4 else "fp8")
    b._mxfp4_kv = mxfp4
    b._tbo_token_split = False
    b.model_runner.forward_vars = {"block_tables": None}
    b._attach_tbo_prefill_cpu_lens = lambda *a: None
    batch = types.SimpleNamespace(block_tables=[[0, 1]], total_seqs_num_prefill=1)
    b.prepare_prefill(batch, 1, 8)
    return md.block_tables


def test_a_no_prefix_dense_prefill_over_mxfp4_pages_gets_its_block_table(monkeypatch):
    # The dense prefill gathers K/V back out of the pages, so it reads the table
    # with no prefix hit too; fp8 keeps publishing it only for a hit.
    assert _prefill(monkeypatch, mxfp4=True) == ("table", 1)
    assert _prefill(monkeypatch, mxfp4=False) is None
