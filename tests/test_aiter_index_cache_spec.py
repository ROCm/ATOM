# SPDX-License-Identifier: MIT
"""MiniMax-M3's index cache spec: the entry field's bytes and the view the impl reads.

The ``fp4`` cache is one 8704 B uint8 page a block, whatever dtype aiter's registry
gives fp8 (uint8 too where torch has no float4 / no fp8 on the gfx); fp8 and bf16
stay priced as keys. Needs aiter, like the other `aiter_attention` builder tests,
so the CPU CI job skips it.
"""

import types

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes

from atom.model_ops.attentions import aiter_attention as mod
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES
from atom.model_ops.minimax_m3.nvfp4_pages import INDEX_PAGE_BYTES


def builder(monkeypatch, index):
    reg = dict(dtypes.d_dtypes, fp8=torch.uint8)
    monkeypatch.setattr(dtypes, "d_dtypes", reg)
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size = 128
    b._has_sparse_attention = True
    b.row_counts = lambda: {mod.INDEX_ROWS: 3}
    hf = types.SimpleNamespace(sparse_attention_config={"sparse_index_dim": 128})
    config = types.SimpleNamespace(index_cache_dtype=index, hf_config=hf)
    b.model_runner = types.SimpleNamespace(config=config)
    return b


def test_fp4_is_one_uint8_page_even_when_fp8_is_uint8(monkeypatch):
    spec = builder(monkeypatch, "fp4")._index_cache_spec(128)
    assert spec == (3, (PAGE_BYTES,), torch.uint8, (-1, PAGE_BYTES))
    (f,) = builder(monkeypatch, "fp4")._index_cache_fields(128)
    assert (f.shape, f.dtype, f.layers) == ((PAGE_BYTES,), torch.uint8, 3)


def test_nvfp4_is_one_9216_byte_uint8_page(monkeypatch):
    spec = builder(monkeypatch, "nvfp4")._index_cache_spec(128)
    assert spec == (3, (INDEX_PAGE_BYTES,), torch.uint8, (-1, INDEX_PAGE_BYTES))
    with pytest.raises(ValueError, match="NVFP4 index cache is 128-wide"):
        builder(monkeypatch, "nvfp4")._index_cache_spec(256)


def test_fp8_stays_byte_priced(monkeypatch):
    spec = builder(monkeypatch, "fp8")._index_cache_spec(128)
    assert spec == (3, (128 * 128,), torch.uint8, (-1, 128, 128))


def test_bf16_is_two_bytes_a_key(monkeypatch):
    spec = builder(monkeypatch, "bf16")._index_cache_spec(256)
    assert spec.field_shape == (256 * 128 * 2,) and spec.view_dtype is torch.bfloat16


@pytest.mark.parametrize(
    "index, tokens, match",
    [("fp6", 128, "index_cache_dtype"), ("fp4", 256, "128-wide keys")],
)
def test_an_index_cache_the_page_cannot_hold_is_refused(
    monkeypatch, index, tokens, match
):
    with pytest.raises(ValueError, match=match):
        builder(monkeypatch, index)._index_cache_spec(tokens)


def test_no_sparse_attention_no_index_cache(monkeypatch):
    b = builder(monkeypatch, "fp8")
    b._has_sparse_attention = False
    assert b._index_cache_spec(128) is None and b._index_cache_fields(128) == ()
