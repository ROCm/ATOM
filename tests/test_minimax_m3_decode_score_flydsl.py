# SPDX-License-Identifier: MIT
"""MiniMax-M3 decode index scores through aiter's FlyDSL scorer
(``ATOM_M3_DECODE_SCORE_FLYDSL=1``, gfx1250, FP4 index cache).

Two halves. The gate (``_flydsl_decode_scorer`` and the launch it replaces) is
plain Python and runs wherever triton is installed. The kernel half needs a
gfx1250 GPU: the FlyDSL scorer against the Triton
``_decode_index_score_transposed_kernel`` it replaces -- the same scores bit for
bit on every live slot (both issue the same scaled WMMA pair), so the same
top-k ids, page table and bounds, eagerly and from a captured graph.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m
from atom.utils import envs

D, PAGE, TOPK = 128, 128, 16
FLAG = "ATOM_M3_DECODE_SCORE_FLYDSL"
AITER_OP = "aiter.ops.flydsl.index_score_decode"


def _fake_aiter(monkeypatch, serves=True):
    """Stand-in for aiter's op module: records the predicate's and the scorer's
    arguments."""
    calls = types.SimpleNamespace(asked=[], scored=[])

    def supported(*a):
        calls.asked.append(a)
        return serves

    def score(*a, **kw):
        calls.scored.append((a, kw))

    mod = types.ModuleType(AITER_OP)
    mod.index_score_decode_nvfp4_supported = supported
    mod.index_score_decode_nvfp4 = score
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    return calls, score


@pytest.fixture(autouse=True)
def _scorer_under_the_triton_selector(monkeypatch):
    """These tests are about the scorer alone: the whole FlyDSL indexer (on by
    default, see test_minimax_m3_decode_index_topk_flydsl.py) is off."""
    monkeypatch.setenv("ATOM_M3_DECODE_INDEX_TOPK_FLYDSL", "0")


class TestGate:
    q = torch.zeros(16, 4, D, dtype=torch.bfloat16)
    payload = torch.zeros(3, PAGE, D // 2, dtype=torch.uint8)
    scale = torch.zeros(3, PAGE, D // 16, dtype=torch.uint8)

    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        assert not envs.ATOM_M3_DECODE_SCORE_FLYDSL
        # Not even imported: an aiter without the op keeps working.
        monkeypatch.setitem(sys.modules, AITER_OP, None)
        assert m._flydsl_decode_scorer(self.q, self.payload, self.scale, 8) is None

    def test_needs_the_fp4_index_cache(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        monkeypatch.setitem(sys.modules, AITER_OP, None)
        assert m._flydsl_decode_scorer(self.q, self.payload, None, 8) is None

    def test_aiter_decides_whether_it_serves(self, monkeypatch):
        monkeypatch.setenv(FLAG, "1")
        calls, score = _fake_aiter(monkeypatch, serves=True)
        assert m._flydsl_decode_scorer(self.q, self.payload, self.scale, 8) is score
        (asked,) = calls.asked
        assert asked[0] is self.q and asked[1] is self.payload
        assert asked[2] is self.scale and asked[3] == 8
        _fake_aiter(monkeypatch, serves=False)
        assert m._flydsl_decode_scorer(self.q, self.payload, self.scale, 8) is None

    def test_the_flag_needs_an_aiter_with_the_scorer(self, monkeypatch):
        """No probing: asked for on an aiter without the op, the import fails."""
        monkeypatch.setenv(FLAG, "1")
        monkeypatch.setitem(sys.modules, AITER_OP, None)
        with pytest.raises(ImportError):
            m._flydsl_decode_scorer(self.q, self.payload, self.scale, 8)


class _Launch:
    """A Triton kernel stand-in: records its launches."""

    def __init__(self):
        self.launches = []

    def __getitem__(self, grid):
        return lambda *a, **kw: self.launches.append((grid, a, kw))


@pytest.mark.parametrize("flag", ["0", "1"])
@pytest.mark.parametrize("serves", [True, False])
def test_the_flydsl_scorer_replaces_the_triton_launch(monkeypatch, flag, serves):
    """With the flag on and aiter serving, the decode entry hands aiter the
    scorer's inputs and the score plane the selector then reads, and launches
    no Triton scorer; otherwise the Triton scorer runs as before."""
    monkeypatch.setenv(FLAG, flag)
    calls, _ = _fake_aiter(monkeypatch, serves=serves)
    transposed, tiled, selected = _Launch(), _Launch(), []
    monkeypatch.setattr(m, "_decode_index_score_transposed_kernel", transposed)
    monkeypatch.setattr(m, "_decode_index_score_tiled_kernel", tiled)
    monkeypatch.setattr(m, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(m, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: selected.append((a, kw)))
    batch, ql, heads, ctx = 3, 8, 4, 1000
    q = torch.zeros(batch * ql, heads, D, dtype=torch.bfloat16)
    pages = batch * 8
    payload = torch.zeros(pages, PAGE, D // 2, dtype=torch.uint8)
    scale = torch.zeros(pages, PAGE, D // 16, dtype=torch.uint8)
    bt = torch.arange(pages, dtype=torch.int32).view(batch, 8)
    seq = torch.tensor([ctx, 5, 700], dtype=torch.int32)
    out = m.minimax_m3_index_topk_decode(
        q, payload, bt, seq, ctx, TOPK, 0, 1, heads, D**-0.5,
        max_query_len=ql, index_kv_scale=scale,
    )  # fmt: skip
    assert out.shape == (heads, batch * ql, TOPK)
    ((sel_args, _),) = selected
    score = sel_args[0]
    assert score.shape == (heads, batch * ql, 8) and score.dtype == torch.float32
    if flag == "1" and serves:
        assert not transposed.launches and not tiled.launches
        ((args, kw),) = calls.scored
        assert not kw
        assert args[0] is q and args[1] is payload and args[2] is scale
        assert args[3] is score and args[4] is bt and args[5] is seq
        assert args[6:] == (D**-0.5, ql)
    else:
        assert not calls.scored and not tiled.launches
        assert len(transposed.launches) == 1
        assert bool(calls.asked) == (flag == "1")


# ---------------------------------------------------------------------------
# Kernels. Needs a gfx1250 GPU, triton and an aiter with the scorer.
# ---------------------------------------------------------------------------
def _gfx() -> str:
    if not torch.cuda.is_available():
        return ""
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]


gfx1250 = pytest.mark.skipif(_gfx() != "gfx1250", reason="needs a gfx1250 GPU")
DEV = "cuda"


def _cache(lens, bound_blocks, seed, mode):
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = [-(-n // PAGE) for n in lens]
    pages = sum(nb) + 1
    payload = torch.empty(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.empty(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    if mode == "ties":
        base = (torch.randn(4 * PAGE, D, device=DEV, generator=g) * 3).to(
            torch.bfloat16
        )
        keys = base.repeat(-(-pages // 4), 1)[: pages * PAGE]
    elif mode == "zero":
        keys = torch.zeros(pages * PAGE, D, device=DEV, dtype=torch.bfloat16)
    else:
        keys = (torch.randn(pages * PAGE, D, device=DEV, generator=g) * 3).to(
            torch.bfloat16
        )
    index_fp4_store(keys, torch.arange(pages * PAGE, device=DEV), payload, scale)
    if mode == "nan":  # NaN E4M3 scales: single rows and whole pages
        at = torch.randint(0, scale.numel(), (64,), device=DEV, generator=g)
        scale.view(-1)[at] = 0x7F
        scale[
            torch.randint(0, pages, (max(1, pages // 8),), device=DEV, generator=g)
        ] = 0xFF
    if mode == "raw":  # every byte pattern, NaN scales included
        payload.random_(0, 256, generator=g)
        scale.random_(0, 256, generator=g)
    perm = torch.randperm(pages, device=DEV, generator=g).to(torch.int32)
    bt = torch.zeros(len(lens), bound_blocks, dtype=torch.int32, device=DEV)
    off = 0
    for b, n in enumerate(nb):
        bt[b, :n] = perm[off : off + n]
        off += n
    return payload, scale, bt, torch.tensor(lens, dtype=torch.int32, device=DEV)


_RAGGED24 = [
    int(x)
    for x in torch.randint(1, 300000, (24,), generator=torch.Generator().manual_seed(1))
]
CASES = [
    # lengths, context bound, query rows a request, index heads, mode
    ([150000] * 14, 300000, 8, 4, "rand"),  # the TP1 decode shape
    ([300001, 1, 5, 129], 400000, 8, 4, "rand"),
    ([1, 2, 7, 8, 9, 127, 128, 129], 4096, 8, 4, "rand"),
    ([1, 2, 7, 8, 9, 127, 128, 129], 4096, 1, 4, "rand"),
    ([1000, 999, 4097, 70000, 3], 131072, 4, 4, "rand"),
    (_RAGGED24, 307200, 8, 4, "rand"),
    ([17000 + 997 * i for i in range(12)], 65536, 8, 4, "ties"),
    ([17000, 3000, 129, 40000, 640, 100000], 131072, 8, 4, "zero"),
    ([17000, 3000, 1290, 40000, 6400, 100000], 131072, 8, 4, "nan"),
    ([17000, 3000, 1290, 40000, 6400, 100000], 131072, 8, 4, "raw"),
    ([20000, 140000, 5, 130, 7000, 99999, 128, 256], 262144, 8, 1, "rand"),
    ([70000, 5000, 129], 131072, 1, 1, "rand"),
    ([70000, 5000, 129, 8191], 131072, 3, 5, "rand"),
    ([64 * PAGE * 16, 64 * PAGE * 16 + 1], 1 << 20, 16, 2, "rand"),
]


def _decode(monkeypatch, flag, q, payload, scale, bt, seq, bound, ql, heads, **kw):
    """The decode entry with the scorer chosen by ``flag``; returns (what it
    returns, the score plane it handed the selector)."""
    monkeypatch.setenv(FLAG, flag)
    sm_scale = kw.pop("sm_scale", D**-0.5)
    if flag == "1":
        # Otherwise the comparison below would be Triton against Triton.
        assert m._flydsl_decode_scorer(q, payload, scale, ql) is not None
    seen = []
    select = m._launch_select
    monkeypatch.setattr(
        m, "_launch_select", lambda score, *a, **k: (seen.append(score), select(score, *a, **k))[1]
    )  # fmt: skip
    out = m.minimax_m3_index_topk_decode(
        q, payload, bt, seq, bound, TOPK, kw.pop("init", 0), kw.pop("local", 1),
        heads, sm_scale, max_query_len=ql, index_kv_scale=scale, **kw,
    )  # fmt: skip
    monkeypatch.setattr(m, "_launch_select", select)
    (score,) = seen
    return out, score


def _live(seq, ql, heads, max_block):
    blocks = torch.clamp((seq.long() + PAGE - 1) // PAGE, max=max_block)
    live = torch.arange(max_block, device=DEV)[None, :] < blocks[:, None]
    return live.repeat_interleave(ql, 0)[None].expand(heads, -1, -1)


@gfx1250
@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("init,local", [(0, 1), (1, 2)])
def test_flydsl_scores_and_selection_are_the_triton_scorers(
    case, init, local, monkeypatch
):
    lens, bound, ql, heads, mode = CASES[case]
    bound_blocks = -(-bound // PAGE)
    payload, scale, bt, seq = _cache(lens, bound_blocks, case, mode)
    g = torch.Generator(device=DEV).manual_seed(case + 1)
    q = torch.randn(len(lens) * ql, heads, D, device=DEV, generator=g).to(
        torch.bfloat16
    )
    if mode == "zero":
        q = torch.ones_like(q)
    emit = {
        "emit_sparse_block_table": True,
        "emit_pages_per_block": m.NVFP4_PAGES_PER_BLOCK,
        "block_page_stride": m.NVFP4_PAGES_PER_BLOCK,
        "init": init,
        "local": local,
    }
    args = (q, payload, scale, bt, seq, bound, ql, heads)
    ref, ref_score = _decode(monkeypatch, "0", *args, **emit)
    got, got_score = _decode(monkeypatch, "1", *args, **emit)
    torch.cuda.synchronize()
    live = _live(seq, ql, heads, bound_blocks)
    a, b = ref_score.view(torch.int32)[live], got_score.view(torch.int32)[live]
    err_ratio = (a != b).float().mean().item() if a.numel() else 0.0
    assert err_ratio == 0.0, f"score errRatio {err_ratio:.3e} over {a.numel()} slots"
    for name, x, y in zip(("topk_idx", "sparse_bt", "sparse_ctx"), ref, got):
        assert torch.equal(x, y), name


@gfx1250
@pytest.mark.parametrize("sm_scale", [192**-0.5, 1 / 7])
@pytest.mark.parametrize("case", [1, 4, 12])
def test_flydsl_scores_round_the_scale_as_triton_does(case, sm_scale, monkeypatch):
    """sm_scale * log2(e) is an fp32 product of two fp32 values in the Triton
    kernel; at 128**-0.5 that equals the double product rounded once, at these
    it does not (192**-0.5: 0x3dd53b94 against 0x3dd53b95)."""
    lens, bound, ql, heads, mode = CASES[case]
    payload, scale, bt, seq = _cache(lens, -(-bound // PAGE), case, mode)
    q = torch.randn(len(lens) * ql, heads, D, device=DEV).to(torch.bfloat16)
    args = (q, payload, scale, bt, seq, bound, ql, heads)
    _, ref = _decode(monkeypatch, "0", *args, sm_scale=sm_scale)
    _, got = _decode(monkeypatch, "1", *args, sm_scale=sm_scale)
    torch.cuda.synchronize()
    live = _live(seq, ql, heads, -(-bound // PAGE))
    assert torch.equal(ref.view(torch.int32)[live], got.view(torch.int32)[live])


@gfx1250
@pytest.mark.parametrize(
    "workgroups,ring", [(1, 4), (1, 2), (1, 8), (96, 6), (4096, 3), (None, 5)]
)
def test_flydsl_scores_do_not_depend_on_the_launch_knobs(workgroups, ring, monkeypatch):
    """The workgroup budget and the LDS ring move work, not bits. One
    workgroup budget gives every workgroup a full 64-block window (the widest
    chunk; the default budget keeps them far shorter at these batch sizes)."""
    from aiter.ops.flydsl.index_score_decode import index_score_decode_nvfp4

    lens, ql, heads = [64 * PAGE * 2, 64 * PAGE * 2 - 1, 33 * PAGE + 5, 7], 8, 4
    bound = max(lens)
    blocks = bound // PAGE
    payload, scale, bt, seq = _cache(lens, blocks, 21, "rand")
    q = torch.randn(len(lens) * ql, heads, D, device=DEV).to(torch.bfloat16)
    _, ref = _decode(monkeypatch, "0", q, payload, scale, bt, seq, bound, ql, heads)
    got = torch.full_like(ref, float("nan"))
    index_score_decode_nvfp4(
        q, payload, scale, got, bt, seq, D**-0.5, ql, workgroups=workgroups, ring=ring
    )
    torch.cuda.synchronize()
    live = _live(seq, ql, heads, blocks)
    assert torch.equal(ref.view(torch.int32)[live], got.view(torch.int32)[live])
    assert torch.isnan(got[~live]).all(), "a dead block was written"


@gfx1250
@pytest.mark.parametrize("case", [0, 1, 2, 5, 12])
def test_flydsl_scorer_leaves_dead_slots_alone(case):
    """Blocks past a request's live ones, and rows past the batch's, are not
    written (the selector never reads them; a write there would be out of the
    Triton scorer's contract)."""
    from aiter.ops.flydsl.index_score_decode import index_score_decode_nvfp4

    lens, bound, ql, heads, mode = CASES[case]
    bound_blocks = -(-bound // PAGE)
    payload, scale, bt, seq = _cache(lens, bound_blocks, case, mode)
    rows = len(lens) * ql
    q = torch.randn(rows + ql, heads, D, device=DEV).to(torch.bfloat16)
    score = torch.full((heads, rows + ql, bound_blocks), float("nan"), device=DEV)
    index_score_decode_nvfp4(q, payload, scale, score, bt, seq, D**-0.5, ql)
    torch.cuda.synchronize()
    live = _live(seq, ql, heads, bound_blocks)
    assert torch.isnan(score[:, rows:]).all(), "a row past the batch was written"
    assert torch.isnan(score[:, :rows][~live]).all(), "a dead block was written"
    if mode == "rand":
        assert not torch.isnan(score[:, :rows][live]).any(), "a live block was not"


@gfx1250
def test_flydsl_scores_replay_from_a_graph(monkeypatch):
    """Captured once, replayed with other lengths and contents: the launch
    depends on the batch and the context bound only."""
    bs, bound, ql, heads = 8, 300000, 8, 4
    bb = -(-bound // PAGE)
    payload, scale, bt, seq = _cache([200000] * bs, bb, 7, "rand")
    q = torch.randn(bs * ql, heads, D, device=DEV).to(torch.bfloat16)
    emit = {
        "emit_sparse_block_table": True,
        "emit_pages_per_block": m.NVFP4_PAGES_PER_BLOCK,
        "block_page_stride": m.NVFP4_PAGES_PER_BLOCK,
    }
    args = (q, payload, scale, bt, seq, bound, ql, heads)
    _decode(monkeypatch, "1", *args, **emit)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got, _ = _decode(monkeypatch, "1", *args, **emit)
    for lens in ([1, 129, 70000, 200000, 4096, 5, 130000, 999], [199999] * 8):
        seq.copy_(torch.tensor(lens, dtype=torch.int32))
        q.normal_()
        graph.replay()
        torch.cuda.synchronize()
        replayed = [t.clone() for t in got]
        ref, _ = _decode(monkeypatch, "0", *args, **emit)
        for name, x, y in zip(("topk_idx", "sparse_bt", "sparse_ctx"), ref, replayed):
            assert torch.equal(x, y), name
