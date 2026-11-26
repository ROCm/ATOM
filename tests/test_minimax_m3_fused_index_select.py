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
import torch
from import_guard import skip_if_dependency_missing

try:
    from atom.model_ops import attention_mha
    from atom.model_ops.attention_mha import SparseMHAPagedAttentionImpl
    from atom.model_ops.minimax_m3 import index_topk
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


def _sparse_decode_layer(*, skip_index_topk):
    """A minimal ``SparseMHAPagedAttentionImpl`` for ``_sparse_decode``, NVFP4 / fused
    index select configured, no GPU op bound beyond what each test monkeypatches."""
    layer = object.__new__(SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype = layer.index_cache_dtype = "nvfp4"
    layer.indexer_cp_group = None
    layer.topk, layer.init_blocks, layer.local_blocks = 16, 1, 0
    layer.num_kv_heads, layer.scale = 1, 1.0
    layer.index_cache, layer.layer_num, layer.sparse_layer_ordinal = None, 0, 0
    layer.nvfp4_k_scale = layer.nvfp4_v_scale = None
    layer._decode_select_supported = lambda rows, topk=16: True
    layer.skip_index_topk = skip_index_topk
    layer.index_topk_cache_state = {}
    return layer


def _sparse_decode_ctx(rows):
    seq_lens = torch.zeros(rows, dtype=torch.int32)
    block_table = torch.zeros(rows, 4, dtype=torch.int32)
    md = SimpleNamespace(
        max_seq_len=100,
        n_valid_column_per_row=None,
        index_scorer=SimpleNamespace(nvfp4=True),
        decode=SimpleNamespace(
            seq_lens=seq_lens, block_table=block_table, max_query_len=1
        ),
    )
    return SimpleNamespace(attn_metadata=SimpleNamespace(sparse_attention_metadata=md))


def test_fused_decode_select_reaches_attention_and_is_replayed_by_a_skip_layer(
    monkeypatch,
):
    """``_sparse_decode`` end to end with fusion taken: the Triton/aiter selector
    (``minimax_m3_index_topk_decode``) never runs, the scorer's output reaches
    attention as ``index_score``, and attention's own written-back indices are what a
    later skip-index layer (same step, ``skip_index_topk=True``) replays without
    rescoring -- the gap Copilot found (no test ran the fused branch end to end,
    github.com/ROCm/ATOM/pull/3785#discussion_r4703422658)."""
    rows, heads = 4, 1
    calls = {"score": 0, "select": 0, "attend": []}

    def score_decode(
        index_q, cache, block_table, seq_lens, max_seq_len, nkv, scale, mql, scorer
    ):
        calls["score"] += 1
        assert index_q.shape == (rows, heads, 8)
        return torch.full((nkv, rows), 3.0)

    def select_decode(*a, **kw):
        calls["select"] += 1
        raise AssertionError("the selector must not run when fusion is taken")

    def attend(q, k_cache, topk_idx, block_table, seq_lens, scale, *, out, **kw):
        calls["attend"].append(kw.get("index_score"))
        topk_idx.fill_(5)  # the kernel's own selection, written back for skip layers
        out.fill_(1)

    monkeypatch.setattr(index_topk, "minimax_m3_index_score_decode", score_decode)
    monkeypatch.setattr(index_topk, "minimax_m3_index_topk_decode", select_decode)

    ctx = _sparse_decode_ctx(rows)
    q = torch.zeros(rows, heads, 8)

    producer = _sparse_decode_layer(skip_index_topk=False)
    producer._sparse_pa_nvfp4_decode = attend
    producer._index_q = torch.zeros(rows, heads, 8)
    # _sparse_decode clears `_index_q` to None at the end (the producer's own cue that
    # the index is consumed), so the shape/dtype/device a skip layer needs has to be
    # read off before the call, not after.
    producer_index_q_shape = tuple(producer._index_q.shape)
    producer_index_q_dtype = producer._index_q.dtype
    producer_index_q_device = producer._index_q.device
    producer._index_q_cache_key_info = None
    out1 = producer._sparse_decode(q, None, None, None, None, None, None, ctx)

    assert calls["score"] == 1 and calls["select"] == 0
    assert calls["attend"][0] is not None and bool((calls["attend"][0] == 3.0).all())
    assert bool((out1 == 1).all())
    state = ctx.attn_metadata.sparse_attention_metadata._index_topk_cache_state
    cached_topk_idx = state["topk"]["value"][0]
    assert bool((cached_topk_idx == 5).all())  # attention's write-back, not garbage

    consumer = _sparse_decode_layer(skip_index_topk=True)

    def attend_consumer(
        q, k_cache, topk_idx, block_table, seq_lens, scale, *, out, **kw
    ):
        calls["attend"].append(kw.get("index_score"))
        assert (
            topk_idx.data_ptr() == cached_topk_idx.data_ptr()
        )  # replayed, not rebuilt
        out.fill_(1)

    consumer._sparse_pa_nvfp4_decode = attend_consumer
    consumer._index_q = None
    consumer._index_q_cache_key_info = (
        producer_index_q_shape,
        producer_index_q_dtype,
        producer_index_q_device,
    )
    out2 = consumer._sparse_decode(q, None, None, None, None, None, None, ctx)

    # no rescoring and no selector call on the replay
    assert calls["score"] == 1 and calls["select"] == 0
    assert calls["attend"][1] is None  # a plain replay: no index_score kwarg this time
    assert bool((out2 == 1).all())
