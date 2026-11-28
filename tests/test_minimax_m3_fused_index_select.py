# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""``SparseMHAPagedAttentionImpl._fused_index_select``: when a decode step lets the NVFP4
attention select its own top-16 (ATOM_M3_FUSED_INDEX_SELECT) and when it keeps scorer + selector.
"""

import sys
import types
from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

try:
    from atom.model_ops import attention_mha
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
    from atom.utils import envs
except ImportError as exc:  # no aiter on a CPU-only runner
    skip_if_dependency_missing(exc, "atom.model_ops.attention_mha")


def _impl(**kw):
    base = {
        "_decode_select_supported": lambda rows, topk=16: 1 <= rows <= 256,
        "kv_cache_dtype": "nvfp4",
        "indexer_cp_group": None,
        "index_cache_dtype": "nvfp4",
        "topk": 16,
    }
    base.update(kw)
    return SimpleNamespace(**base)


def _md(nvfp4=True, scorer=True):
    return SimpleNamespace(
        index_scorer=SimpleNamespace(nvfp4=nvfp4) if scorer else None
    )


def _aiter_module(monkeypatch, supported=True):
    name = "aiter.ops.flydsl.kernels.sparse_pa_nvfp4_decode"
    if supported:
        mod = types.ModuleType(name)
        mod.decode_select_supported = lambda rows, topk=16: 1 <= rows <= 256
        monkeypatch.setitem(sys.modules, name, mod)
    else:  # an aiter without the fused selection
        monkeypatch.setitem(sys.modules, name, None)


@pytest.fixture
def aiter_with_select(monkeypatch):
    monkeypatch.setattr(attention_mha, "FP4_CACHE_DTYPES", ("fp4", "nvfp4"))


def _call(impl, rows, md):
    return SparseMHAPagedAttentionImpl._fused_index_select(impl, rows, md)


def _probe(monkeypatch, flag=True, supported=True, kv="nvfp4"):
    _aiter_module(monkeypatch, supported)
    monkeypatch.setattr(envs, "ATOM_M3_FUSED_INDEX_SELECT", flag, raising=False)
    return SparseMHAPagedAttentionImpl._probe_decode_select(
        SimpleNamespace(kv_cache_dtype=kv)
    )


def test_taken(aiter_with_select):
    assert _call(_impl(), 64, _md())


@pytest.mark.parametrize(
    "impl_kw, md_kw, rows",
    [
        ({"_decode_select_supported": None}, {}, 64),
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


def test_probe_binds_aiter_check(monkeypatch):
    assert _probe(monkeypatch)(64) is True


@pytest.mark.parametrize(
    "kw",
    [{"flag": False}, {"supported": False}, {"kv": "fp4"}, {"kv": "bf16"}],
)
def test_probe_off(monkeypatch, kw):
    assert _probe(monkeypatch, **kw) is None
