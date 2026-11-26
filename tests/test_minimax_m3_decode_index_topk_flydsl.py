# SPDX-License-Identifier: MIT
"""MiniMax-M3 decode indexer through aiter's FlyDSL scorer + selector
(``ATOM_M3_DECODE_INDEX_TOPK_FLYDSL=1``, gfx1250, FP4 index cache).

The gate is plain Python (a stand-in aiter module). The kernel half needs a
gfx1250 GPU: the decode entry with the flag against the Triton scorer +
``_topk_index_packed_kernel`` -- the scores are the same bits (the FlyDSL
scorer issues the Triton scorer's WMMA pair; its flat plan only moves blocks
between workgroups), so the top-k ids, their order, the page table and the
context lengths must be equal, eagerly and from a captured graph.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from test_minimax_m3_decode_score_flydsl import CASES, DEV, _cache, gfx1250

from atom.model_ops.minimax_m3 import index_topk as m
from atom.utils import envs

D, PAGE, TOPK = 128, 128, 16
FLAG = "ATOM_M3_DECODE_INDEX_TOPK_FLYDSL"
SCORE_FLAG = "ATOM_M3_DECODE_SCORE_FLYDSL"
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
    mod.index_score_decode_nvfp4_supported = lambda *a: False
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    return calls


def _inputs(batch=3, ql=8, heads=4, blocks=8):
    q = torch.zeros(batch * ql, heads, D, dtype=torch.bfloat16)
    pages = batch * blocks
    payload = torch.zeros(pages, PAGE, D // 2, dtype=torch.uint8)
    scale = torch.zeros(pages, PAGE, D // 16, dtype=torch.uint8)
    bt = torch.arange(pages, dtype=torch.int32).view(batch, blocks)
    seq = torch.tensor([blocks * PAGE, 5, 700][:batch], dtype=torch.int32)
    return q, payload, scale, bt, seq


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert not envs.ATOM_M3_DECODE_INDEX_TOPK_FLYDSL
    monkeypatch.setitem(sys.modules, AITER_OP, None)  # not even imported
    q, payload, scale, *_ = _inputs()
    assert m._flydsl_decode_index_topk(q, payload, scale, 8, 8, TOPK) is None


def test_the_flag_needs_an_aiter_with_the_op(monkeypatch):
    """No probing: asked for on an aiter without the op, the import fails."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setitem(sys.modules, AITER_OP, None)
    q, payload, scale, *_ = _inputs()
    with pytest.raises(ImportError):
        m._flydsl_decode_index_topk(q, payload, scale, 8, 8, TOPK)


@pytest.mark.parametrize("serves", [True, False])
@pytest.mark.parametrize("emit", [True, False])
def test_aiter_runs_the_whole_indexer(monkeypatch, serves, emit):
    """Flag on and aiter serving: one call gets the scorer's inputs, aiter's
    score plane, the outputs and the emit geometry, and no Triton kernel or
    selector runs; aiter declining leaves the Triton path as it was (on a
    plane of ATOM's own)."""
    monkeypatch.setenv(FLAG, "1")
    calls = _fake_aiter(monkeypatch, serves=serves)
    launched = []

    class Launch:  # a Triton kernel stand-in: records its launches
        def __getitem__(self, grid):
            return lambda *a, **k: launched.append(grid)

    monkeypatch.setattr(m, "_decode_index_score_transposed_kernel", Launch())
    monkeypatch.setattr(m, "_decode_index_score_tiled_kernel", Launch())
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: launched.append("select"))
    monkeypatch.setattr(m, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(m, "get_gfx", lambda: "gfx1250")
    q, payload, scale, bt, seq = _inputs()
    kw = {}
    if emit:
        kw = {
            "emit_sparse_block_table": True,
            "emit_pages_per_block": m.NVFP4_PAGES_PER_BLOCK,
            "block_page_stride": m.NVFP4_PAGES_PER_BLOCK,
        }
    out = m.minimax_m3_index_topk_decode(
        q, payload, bt, seq, 8 * PAGE, TOPK, 1, 2, 4, D**-0.5,
        max_query_len=8, index_kv_scale=scale, **kw,
    )  # fmt: skip
    (asked,) = calls.asked
    assert asked[0] is q and asked[1] is payload and asked[2] is scale
    assert asked[3].shape == (4, 24, 8) and asked[3].stride(1) == 128
    assert asked[4:] == (8, TOPK)
    if not serves:
        assert not calls.ran and launched[-1] == "select"
        return
    assert not launched
    ((args, akw),) = calls.ran
    topk_idx = out[0] if emit else out
    assert args[0] is q and args[1] is payload and args[2] is scale
    assert args[3] is asked[3] and args[4] is bt and args[5] is seq
    assert args[6:] == (D**-0.5, 8, topk_idx, 1, 2)
    if emit:
        assert akw["sparse_bt"] is out[1] and akw["sparse_ctx"] is out[2]
        assert akw["pages_per_block"] == m.NVFP4_PAGES_PER_BLOCK
        assert akw["block_page_stride"] == m.NVFP4_PAGES_PER_BLOCK
    else:
        assert not akw


# ---------------------------------------------------------------------------
# Kernels. Needs a gfx1250 GPU, triton and an aiter with the op.
# ---------------------------------------------------------------------------
def _decode(monkeypatch, flag, q, payload, scale, bt, seq, bound, ql, heads, **kw):
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv(SCORE_FLAG, "0")
    if flag == "1":
        served = m._flydsl_decode_index_topk(
            q, payload, scale, -(-bound // PAGE), ql, TOPK
        )
        assert served is not None, "the comparison would be Triton against Triton"
    return m.minimax_m3_index_topk_decode(
        q, payload, bt, seq, bound, TOPK, kw.pop("init", 0), kw.pop("local", 1),
        heads, D**-0.5, max_query_len=ql, index_kv_scale=scale, **kw,
    )  # fmt: skip


EMITS = {
    "none": {},
    "page128": {
        "emit_sparse_block_table": True,
        "emit_pages_per_block": m.NVFP4_PAGES_PER_BLOCK,
        "block_page_stride": m.NVFP4_PAGES_PER_BLOCK,
    },
    "page16": {"emit_sparse_block_table": True, "block_page_stride": 16},
}


# The decode-score cases, and a batch past one wave of requests (the flat
# plan's per-32-request loops).
CASES_X = [*CASES, ([4000 + 997 * i for i in range(44)], 65536, 8, 4, "rand")]


@gfx1250
@pytest.mark.parametrize("case", range(len(CASES_X)))
@pytest.mark.parametrize("init,local", [(0, 1), (1, 2)])
@pytest.mark.parametrize("emit", list(EMITS))
def test_flydsl_indexer_selects_what_triton_selects(
    case, init, local, emit, monkeypatch
):
    lens, bound, ql, heads, mode = CASES_X[case]
    payload, scale, bt, seq = _cache(lens, -(-bound // PAGE), case, mode)
    g = torch.Generator(device=DEV).manual_seed(case + 1)
    q = torch.randn(len(lens) * ql, heads, D, device=DEV, generator=g).to(
        torch.bfloat16
    )
    if mode == "zero":
        q = torch.ones_like(q)
    kw = dict(EMITS[emit], init=init, local=local)
    args = (q, payload, scale, bt, seq, bound, ql, heads)
    ref = _decode(monkeypatch, "0", *args, **dict(kw))
    got = _decode(monkeypatch, "1", *args, **dict(kw))
    torch.cuda.synchronize()
    ref, got = (ref, got) if emit != "none" else ((ref,), (got,))
    for name, x, y in zip(("topk_idx", "sparse_bt", "sparse_ctx"), ref, got):
        assert torch.equal(x, y), name


FULL = [256 * PAGE] * 4  # budget 1: S = 128, every split a full window
RAGGED = [128 * PAGE * 2, 128 * PAGE * 2 - 1, 33 * PAGE + 5, 7, 0]
MANY = [PAGE * (5 + 37 * i % 300) for i in range(44)]  # past one wave of requests


@gfx1250
@pytest.mark.parametrize(
    "workgroups,ring,lens",
    [
        (1, 4, FULL),
        (1, 8, FULL),
        (1, 4, RAGGED),
        (96, 6, RAGGED),
        (4096, 8, RAGGED),
        (None, 2, RAGGED),
        (1, 4, MANY),
        (None, 4, MANY),
    ],
)
def test_flydsl_flat_scores_do_not_depend_on_the_launch_knobs(
    workgroups, ring, lens, monkeypatch
):
    """The flat plan's budget and the ring move blocks, not bits: the scores
    equal the per-request plan's (itself bitwise Triton's) on every live slot;
    dead slots are not written. A budget of 1 still launches batch *
    (ceil(blocks / 128) + 1) workgroups: on FULL every split is a whole
    128-block window."""
    from aiter.ops.flydsl.index_score_decode import (
        index_score_decode_nvfp4,
        index_topk_decode_score_plane,
    )

    ql, heads = 8, 4
    bound = max(lens)
    blocks = bound // PAGE
    payload, scale, bt, seq = _cache(lens, blocks, 21, "rand")
    q = torch.randn(len(lens) * ql, heads, D, device=DEV).to(torch.bfloat16)
    ref = index_topk_decode_score_plane(heads, len(lens) * ql, blocks, DEV)
    ref.fill_(float("nan"))
    index_score_decode_nvfp4(q, payload, scale, ref, bt, seq, D**-0.5, ql)
    got = index_topk_decode_score_plane(heads, len(lens) * ql, blocks, DEV)
    got.fill_(float("nan"))
    index_score_decode_nvfp4(
        q, payload, scale, got, bt, seq, D**-0.5, ql,
        workgroups=workgroups, ring=ring, flat=True,
    )  # fmt: skip
    torch.cuda.synchronize()
    nb = torch.clamp((seq.long() + PAGE - 1) // PAGE, max=blocks)
    live = (torch.arange(blocks, device=DEV)[None] < nb[:, None]).repeat_interleave(
        ql, 0
    )
    live = live[None].expand(heads, -1, -1)
    assert torch.equal(ref.view(torch.int32)[live], got.view(torch.int32)[live])
    assert torch.isnan(got[~live]).all(), "a dead block was written"


@gfx1250
def test_flydsl_indexer_replays_from_a_graph(monkeypatch):
    """Captured once, replayed with other (ragged) lengths and contents."""
    bs, bound, ql, heads = 16, 600000, 8, 4
    payload, scale, bt, seq = _cache([590000] * bs, -(-bound // PAGE), 7, "rand")
    q = torch.randn(bs * ql, heads, D, device=DEV).to(torch.bfloat16)
    args = (q, payload, scale, bt, seq, bound, ql, heads)
    kw = EMITS["page128"]
    _decode(monkeypatch, "1", *args, **dict(kw))  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _decode(monkeypatch, "1", *args, **dict(kw))
    ragged = [1, 129, 70000, 590000, 4096, 5, 130000, 999] * 2
    for lens in (ragged, [88000] * bs, [0] * (bs - 1) + [300000], [599000] * bs):
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        q.normal_()
        graph.replay()
        torch.cuda.synchronize()
        replayed = [t.clone() for t in got]
        ref = _decode(monkeypatch, "0", *args, **dict(kw))
        for name, x, y in zip(("topk_idx", "sparse_bt", "sparse_ctx"), ref, replayed):
            assert torch.equal(x, y), name
