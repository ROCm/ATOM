# SPDX-License-Identifier: MIT
"""MiniMax-M3 decode indexer through aiter's FlyDSL scorer + selector
(``ATOM_M3_DECODE_INDEX_TOPK_FLYDSL``; gfx1250, NVFP4 index cache).

The gate in plain Python, against a stand-in aiter module: off it never imports
aiter's op, on it hands aiter the bf16 query, the cache's two views, a score
plane and the outputs, and aiter declining keeps today's path.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m
from atom.model_ops.minimax_m3.nvfp4_pages import INDEX_PAGE_BYTES

D, PAGE, TOPK = 128, 128, 16
FLAG = "ATOM_M3_DECODE_INDEX_TOPK_FLYDSL"
AITER_OP = "aiter.ops.flydsl.index_score_decode"


def _fake_aiter(monkeypatch, serves=True):
    calls = types.SimpleNamespace(asked=[], ran=[])

    def supported(*a):
        calls.asked.append(a)
        return serves

    def run(*a, **kw):
        calls.ran.append((a, kw))

    mod = types.ModuleType(AITER_OP)
    mod.index_topk_decode_nvfp4_supported = supported
    mod.index_topk_decode_nvfp4 = run
    mod.index_topk_decode_score_plane = lambda h, r, b, dev: torch.empty(h, r, 128)[
        ..., :b
    ]
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    return calls


def _inputs(batch=3, ql=8, heads=4, blocks=8):
    q = torch.zeros(batch * ql, heads, D, dtype=torch.float8_e4m3fn)
    cache = torch.zeros(batch * blocks, INDEX_PAGE_BYTES, dtype=torch.uint8)
    bt = torch.arange(batch * blocks, dtype=torch.int32).view(batch, blocks)
    seq = torch.tensor([blocks * PAGE, 5, 700][:batch], dtype=torch.int32)
    return q, cache, bt, seq


def _scorer(nvfp4=True):
    return types.SimpleNamespace(nvfp4=nvfp4, work_map=None, max_block=8)


def test_off_by_default_and_not_imported(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setitem(sys.modules, AITER_OP, None)
    q, cache, *_ = _inputs()
    assert m._flydsl_decode_index_topk(q, cache, _scorer(), 8, 8, TOPK) is None


def test_only_the_nvfp4_index(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setitem(sys.modules, AITER_OP, None)
    q, cache, *_ = _inputs()
    assert m._flydsl_decode_index_topk(q, cache, None, 8, 8, TOPK) is None
    assert m._flydsl_decode_index_topk(q, cache, _scorer(False), 8, 8, TOPK) is None


def test_the_flag_needs_an_aiter_with_the_op(monkeypatch):
    """No probing: on with an aiter without the op, the import fails."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setitem(sys.modules, AITER_OP, None)
    q, cache, *_ = _inputs()
    with pytest.raises(ImportError):
        m._flydsl_decode_index_topk(q, cache, _scorer(), 8, 8, TOPK)


def test_aiter_declines(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    calls = _fake_aiter(monkeypatch, serves=False)
    q, cache, *_ = _inputs()
    assert m._flydsl_decode_index_topk(q, cache, _scorer(), 8, 8, TOPK) is None
    assert len(calls.asked) == 1


@pytest.mark.parametrize("emit", [True, False])
def test_aiter_runs_the_whole_indexer(monkeypatch, emit):
    """One call gets the bf16 query, the cache's views, aiter's score plane, the
    outputs and the emit geometry; no ATOM scorer or selector runs."""
    monkeypatch.setenv(FLAG, "1")
    calls = _fake_aiter(monkeypatch)
    launched = []
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: launched.append(1))
    monkeypatch.setattr(m, "_fp4_index_ops", lambda *a: launched.append(2))
    q, cache, bt, seq = _inputs()
    kw = {"emit_sparse_block_table": True} if emit else {}
    out = m.minimax_m3_index_topk_decode(
        q, cache, bt, seq, 8 * PAGE, TOPK, 1, 2, 4, D**-0.5,
        max_query_len=8, index_scorer=_scorer(), **kw,
    )  # fmt: skip
    assert not launched
    ((args, akw),) = calls.ran
    q16, payload, scale, score = args[:4]
    assert q16.dtype == torch.bfloat16 and q16.shape == q.shape
    assert payload.shape == (24, PAGE, D // 2) and scale.shape == (24, PAGE, D // 16)
    assert scale.storage_offset() == PAGE * D // 2
    assert score.shape == (4, 24, 8) and score.stride(1) == 128
    topk_idx = out[0] if emit else out
    assert args[4] is bt and args[5] is seq
    assert args[6:] == (D**-0.5, 8, topk_idx, 1, 2)
    if emit:
        assert akw["sparse_bt"] is out[1] and akw["sparse_ctx"] is out[2]
        assert akw["pages_per_block"] == m.PAGES_PER_SPARSE_BLOCK
    else:
        assert not akw
