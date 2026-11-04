# SPDX-License-Identifier: MIT
"""MiniMax-M3's ``mxfp4`` index cache: logical dtype fp4x2, physical uint8 page.

The resolver reads aiter's registry (``d_dtypes["fp4x2"]``); the entry's shape follows
the config string, because where torch has no float4 the registry's fp8 and fp4x2 are
both uint8 and the dtype cannot say which cache it is. Needs aiter, like the other
`aiter_attention` builder tests, so the CPU CI job skips it.
"""

import types

import pytest
import torch

pytest.importorskip("aiter")
from aiter import dtypes

from atom.model_ops.attentions import aiter_attention as mod
from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES


@pytest.mark.parametrize(
    "index,key", [("mxfp4", "fp4x2"), ("fp8", "fp8"), ("bf16", "bf16")]
)
def test_resolves_through_the_aiter_registry(index, key):
    config = types.SimpleNamespace(index_cache_dtype=index)
    assert mod._resolve_index_cache_dtype(config) is dtypes.d_dtypes[key]


def builder(monkeypatch, index):
    # the collision: fp8 and fp4x2 are the same uint8 (no float4 torch / no fp8 gfx)
    reg = dict(dtypes.d_dtypes, fp8=torch.uint8, fp4x2=torch.uint8)
    monkeypatch.setattr(dtypes, "d_dtypes", reg)
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size = 128
    b._has_sparse_attention = True
    b.row_counts = lambda: {mod.INDEX_ROWS: 3}
    hf = types.SimpleNamespace(sparse_attention_config={"sparse_index_dim": 128})
    config = types.SimpleNamespace(index_cache_dtype=index, hf_config=hf)
    b.model_runner = types.SimpleNamespace(config=config)
    return b


def test_mxfp4_is_one_uint8_page_even_when_fp8_is_uint8(monkeypatch):
    (f,) = builder(monkeypatch, "mxfp4")._index_cache_fields(128)
    assert (f.shape, f.dtype) == ((PAGE_BYTES,), torch.uint8)


def test_fp8_stays_byte_priced_even_when_it_equals_fp4x2(monkeypatch):
    (f,) = builder(monkeypatch, "fp8")._index_cache_fields(128)
    assert (f.shape, f.dtype) == ((128 * 128,), torch.uint8)
