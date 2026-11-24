# SPDX-License-Identifier: Apache-2.0
"""``SparseMHAPagedAttentionImpl._fused_index_select``: when a decode step lets the NVFP4
attention select its own top-16 (ATOM_M3_FUSED_INDEX_SELECT) and when it keeps scorer + selector."""

import sys
import types
from types import SimpleNamespace

import pytest

from atom.model_ops import attention_mha
from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
from atom.utils import envs


def _impl(**kw):
    base = dict(
        kv_cache_dtype="nvfp4",
        indexer_cp_group=None,
        index_cache_dtype="nvfp4",
        topk=16,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _md(nvfp4=True, scorer=True):
    return SimpleNamespace(index_scorer=SimpleNamespace(nvfp4=nvfp4) if scorer else None)


@pytest.fixture
def aiter_with_select(monkeypatch):
    mod = types.ModuleType("aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode")
    mod.decode_select_supported = lambda rows, topk=16: 1 <= rows <= 256
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    monkeypatch.setattr(attention_mha, "FP4_CACHE_DTYPES", ("fp4", "nvfp4"))
    monkeypatch.setattr(envs, "ATOM_M3_FUSED_INDEX_SELECT", True, raising=False)


def _call(impl, rows, md):
    return SparseMHAPagedAttentionImpl._fused_index_select(impl, rows, md)


def test_taken(aiter_with_select):
    assert _call(_impl(), 64, _md())


@pytest.mark.parametrize(
    "impl_kw, md_kw, rows",
    [
        ({"kv_cache_dtype": "fp4"}, {}, 64),
        ({"indexer_cp_group": object()}, {}, 64),
        ({"index_cache_dtype": "bf16"}, {}, 64),
        ({"topk": 8}, {}, 64),
        ({}, {"nvfp4": False}, 64),
        ({}, {"scorer": False}, 64),
        ({}, {}, 512),
    ],
)
def test_refused(aiter_with_select, impl_kw, md_kw, rows):
    assert not _call(_impl(**impl_kw), rows, _md(**md_kw))


def test_flag_off(aiter_with_select, monkeypatch):
    monkeypatch.setattr(envs, "ATOM_M3_FUSED_INDEX_SELECT", False, raising=False)
    assert not _call(_impl(), 64, _md())


def test_old_aiter_keeps_chain(monkeypatch):
    monkeypatch.setattr(attention_mha, "FP4_CACHE_DTYPES", ("fp4", "nvfp4"))
    monkeypatch.setattr(envs, "ATOM_M3_FUSED_INDEX_SELECT", True, raising=False)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode", None)
    assert not _call(_impl(), 64, _md())
