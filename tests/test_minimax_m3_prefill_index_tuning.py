# SPDX-License-Identifier: MIT
"""MiniMax-M3 prefill indexer launch tuning by env: the keywords handed to
aiter's FlyDSL scorer (``ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL_OPTS``) and the
Triton selector's tile cap (``ATOM_M3_PREFILL_TOPK_BLOCK_SIZE_K``). Both leave
the scores and the selection as they are; unset, the launches are today's.
Plain Python: the kernels are stand-ins."""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m

D, PAGE, TOPK = 128, 128, 16
OPTS = "ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL_OPTS"
TILE = "ATOM_M3_PREFILL_TOPK_BLOCK_SIZE_K"
AITER_OP = "aiter.ops.flydsl.index_score_prefill"


@pytest.fixture
def aiter_scorer(monkeypatch):
    """Stand-in for aiter's prefill scorer module: records each launch."""
    calls = []
    mod = types.ModuleType(AITER_OP)
    mod.index_score_prefill_nvfp4_supported = lambda *a: True
    mod.index_score_prefill_nvfp4 = lambda *a, **kw: calls.append((a, kw))
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL", "1")
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_MIN_ROWS", "1")
    return calls


def _prefill(monkeypatch):
    """One prefill through `minimax_m3_index_topk` with the Triton kernels
    replaced; returns the selector launch's (block_size_k, num_warps)."""
    selected = []
    monkeypatch.setattr(m, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(m, "_compute_units", lambda device_index: 256)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: selected.append((a, kw)))
    rows, heads, cached = 300, 4, 200000
    blocks = -(-(cached + rows) // PAGE)
    q = torch.zeros(rows, heads, D, dtype=torch.bfloat16)
    payload = torch.zeros(blocks, PAGE, D // 2, dtype=torch.uint8)
    scale = torch.zeros(blocks, PAGE, D // 16, dtype=torch.uint8)
    out = m.minimax_m3_index_topk(
        q,
        payload,
        torch.arange(blocks, dtype=torch.int32).view(1, blocks),
        torch.tensor([0, rows], dtype=torch.int32),
        torch.tensor([cached + rows], dtype=torch.int32),
        torch.tensor([cached], dtype=torch.int32),
        rows,
        cached + rows,
        TOPK,
        0,
        1,
        heads,
        D**-0.5,
        index_kv_scale=scale,
    )
    assert out.shape == (heads, rows, TOPK)
    ((_, kw),) = selected
    return kw["block_size_k"], kw["num_warps"]


def test_unset_launches_as_before(monkeypatch, aiter_scorer):
    monkeypatch.delenv(OPTS, raising=False)
    monkeypatch.delenv(TILE, raising=False)
    assert _prefill(monkeypatch) == (
        m.PREFILL_TOPK_MAX_BLOCK_SIZE_K,
        m.PREFILL_TOPK_NUM_WARPS,
    )
    ((args, kw),) = aiter_scorer
    assert kw == {} and len(args) == 11


def test_the_scorer_gets_the_tuned_tile(monkeypatch, aiter_scorer):
    monkeypatch.setenv(OPTS, "block_q=128, waves=2,ring=2")
    monkeypatch.delenv(TILE, raising=False)
    _prefill(monkeypatch)
    ((args, kw),) = aiter_scorer
    assert kw == {"block_q": 128, "waves": 2, "ring": 2} and len(args) == 11


def test_the_selector_gets_the_tuned_tile(monkeypatch, aiter_scorer):
    monkeypatch.delenv(OPTS, raising=False)
    monkeypatch.setenv(TILE, "256")
    assert _prefill(monkeypatch) == (256, m.PREFILL_TOPK_NUM_WARPS)
    assert aiter_scorer[0][1] == {}


@pytest.mark.parametrize(
    "cap,max_block,want",
    [
        (None, 1, 128),
        (None, 600, 1024),
        (None, 5000, 1024),
        ("256", 1, 128),
        ("256", 200, 256),
        ("256", 5000, 256),
        ("128", 5000, 128),
        ("4096", 5000, 4096),
    ],
)
def test_tile_cap(monkeypatch, cap, max_block, want):
    if cap is None:
        monkeypatch.delenv(TILE, raising=False)
    else:
        monkeypatch.setenv(TILE, cap)
    assert m._prefill_topk_block_size_k(max_block) == want


@pytest.mark.parametrize("cap", ["0", "64", "300", "-256"])
def test_a_tile_cap_the_kernel_cannot_sort_raises(monkeypatch, cap):
    monkeypatch.setenv(TILE, cap)
    with pytest.raises(ValueError):
        m._prefill_topk_block_size_k(5000)


def test_score_opts_parse():
    assert m._flydsl_prefill_score_opts("") == ()
    assert m._flydsl_prefill_score_opts(" , ") == ()
    assert m._flydsl_prefill_score_opts("block_q=128,waves=2,ring=2,lds_scores=1") == (
        ("block_q", 128),
        ("waves", 2),
        ("ring", 2),
        ("lds_scores", 1),
    )
    for bad in ("block_q", "block_q=wide", "128"):
        with pytest.raises(ValueError):
            m._flydsl_prefill_score_opts(bad)


def test_the_tuned_tile_is_one_aiter_builds():
    """The P setting against the real aiter scorer: keywords of its wrapper,
    and a tile its compile entry accepts (needs FlyDSL; nothing is launched)."""
    pytest.importorskip("flydsl")
    import inspect

    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4
    from aiter.ops.flydsl.kernels.index_score_prefill_nvfp4_gfx1250 import (
        CHUNK_MAX,
        compile_index_score_prefill_nvfp4,
    )

    params = inspect.signature(index_score_prefill_nvfp4).parameters
    opts = dict(m._flydsl_prefill_score_opts("block_q=128,waves=2,ring=2"))
    assert all(params[k].kind is inspect.Parameter.KEYWORD_ONLY for k in opts)
    compiled = compile_index_score_prefill_nvfp4(**opts, chunk_max=CHUNK_MAX)
    assert compiled["launch"] is not None
