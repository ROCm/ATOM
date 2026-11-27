# SPDX-License-Identifier: MIT
"""MiniMax-M3 prefill indexer through aiter's fused scorer + top-16
(``ATOM_M3_INDEX_TOPK_FLYDSL_PREFILL``): the scores never reach memory, each of
the op's workgroups hands on a row's 16 best blocks of its chunk as
``_pack_score_key`` keys and ``_prefill_merge_topk_kernel`` picks the row's 16
and emits.

- the gate and the launches, plain Python (the kernels are stand-ins);
- the merge over chunk lists against the packed selector, on a GPU or on the
  CPU under ``TRITON_INTERPRET=1``: the lists a row does not reach are never
  read, and a chunk may hold fewer than its own 16 best;
- on a gfx1250 GPU with the aiter op: the whole call against the FlyDSL scorer
  + Triton selector on real FP4 planes (exact ties and +-0 scores among them),
  every output equal, also replayed from a graph.
"""

from __future__ import annotations

import os
import sys
import types

import pytest
import torch

triton = pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m

D, PAGE, TOPK = 128, 128, 16
FLAG = "ATOM_M3_INDEX_TOPK_FLYDSL_PREFILL"
OPTS = "ATOM_M3_INDEX_TOPK_FLYDSL_PREFILL_OPTS"
AITER_OP = "aiter.ops.flydsl.index_score_prefill"


def _gfx() -> str:
    if not torch.cuda.is_available():
        return ""
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]


GFX = _gfx()
DEV = "cuda" if GFX else "cpu"
table_kernels = pytest.mark.skipif(
    not GFX and os.environ.get("TRITON_INTERPRET") != "1",
    reason="needs a GPU, or TRITON_INTERPRET=1 to run on the CPU",
)


@pytest.fixture
def aiter_op(monkeypatch):
    """Stand-in for aiter's prefill index module: records each launch."""
    calls = types.SimpleNamespace(fused=[], scorer=[], serves=True, preferred=True)
    mod = types.ModuleType(AITER_OP)
    mod.index_topk_prefill_chunk_blocks = lambda *a, **kw: 256
    mod.index_topk_prefill_nvfp4_preferred = lambda *a: calls.preferred
    mod.index_score_prefill_nvfp4_supported = lambda *a: True
    mod.index_score_prefill_nvfp4 = lambda *a, **kw: calls.scorer.append((a, kw))
    mod.index_topk_prefill_nvfp4_supported = lambda *a: calls.serves

    def fused(*a, **kw):
        calls.fused.append((a, kw))
        return "keys"

    mod.index_topk_prefill_nvfp4 = fused
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL", "1")
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_MIN_ROWS", "256")
    monkeypatch.setattr(m, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(m, "_compute_units", lambda device_index: 256)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    calls.select, calls.merge = [], []
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: calls.select.append(a))
    monkeypatch.setattr(m, "_launch_prefill_merge", lambda *a: calls.merge.append(a))
    return calls


def _prefill(rows=300, fp4=True, emit=True, pages_per_block=1, score_table=None):
    heads, cached = 4, 200000
    blocks = -(-(cached + rows) // PAGE)
    payload = torch.zeros(blocks, PAGE, D // 2, dtype=torch.uint8)
    out = m.minimax_m3_index_topk(
        torch.zeros(rows, heads, D, dtype=torch.bfloat16),
        payload if fp4 else torch.zeros(blocks, PAGE, D, dtype=torch.bfloat16),
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
        emit_sparse_block_table=emit,
        index_kv_scale=(
            torch.zeros(blocks, PAGE, D // 16, dtype=torch.uint8) if fp4 else None
        ),
        emit_pages_per_block=pages_per_block,
        score_block_table=score_table,
    )
    return out, blocks


def test_on_by_default_with_the_tuned_launch(monkeypatch, aiter_op):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(OPTS, raising=False)
    _prefill()
    assert not aiter_op.scorer and not aiter_op.select
    ((_, kw),) = aiter_op.fused
    assert kw == {
        "fold_max": 1,
        "pad_lds": 1,
        "ring": 3,
        "sel32": 1,
        "chunk_blocks": 256,  # aiter's, from the grid
    }
    assert len(aiter_op.merge) == 1


def test_off_keeps_the_scorer_and_the_selector(monkeypatch, aiter_op):
    monkeypatch.setenv(FLAG, "0")
    _prefill()
    assert not aiter_op.fused and not aiter_op.merge
    assert len(aiter_op.scorer) == 1 and len(aiter_op.select) == 1


@pytest.mark.parametrize("pages_per_block", [1, m.PAGES_PER_SPARSE_BLOCK])
def test_the_fused_op_feeds_the_merge(monkeypatch, aiter_op, pages_per_block):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(OPTS, "")  # aiter's defaults
    made = []
    monkeypatch.setattr(
        m, "prefill_score_buffer", lambda *a: made.append(a)
    )  # no score plane
    gathered = torch.zeros(1, 1600, dtype=torch.int32)  # a sharded index cache's
    (topk_idx, table, ctx), blocks = _prefill(
        pages_per_block=pages_per_block, score_table=gathered
    )
    assert not made and not aiter_op.scorer and not aiter_op.select
    ((args, kw),) = aiter_op.fused
    assert kw == {"chunk_blocks": 256} and len(args) == 12
    assert args[3] is gathered  # scored through it, emitted from block_table
    # ..., sm_scale, max_query_len, max_block, init_blocks, local_blocks
    assert args[7:] == (D**-0.5, 300, blocks, 0, 1)
    (merge,) = aiter_op.merge
    assert merge[0] == "keys" and merge[1] is topk_idx and merge[4] is not gathered
    assert merge[5][0] is table and merge[5][1] is ctx
    # rows a request, topk, block_page_stride, pages a block, chunk
    assert merge[6:] == (300, 16, m.PAGES_PER_SPARSE_BLOCK, pages_per_block, 256)
    assert topk_idx.shape == (4, 300, TOPK)
    assert table.shape == (4 * 300, TOPK * pages_per_block) and ctx.shape == (1200,)


def test_the_op_gets_the_tuned_launch(monkeypatch, aiter_op):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(OPTS, "chunk_blocks=128, block_q=128,waves=2")
    _prefill()
    assert aiter_op.fused[0][1] == {"chunk_blocks": 128, "block_q": 128, "waves": 2}
    assert aiter_op.merge[0][-1] == 128


@pytest.mark.parametrize(
    "case",
    [
        "bf16 index",
        "no emission",
        "short prefill",
        "aiter declines",
        "aiter prefers two",
    ],
)
def test_other_calls_keep_the_selector(monkeypatch, aiter_op, case):
    monkeypatch.setenv(FLAG, "1")
    aiter_op.serves = case != "aiter declines"
    aiter_op.preferred = case != "aiter prefers two"
    if case in ("bf16 index", "short prefill"):
        # These go to the Triton scorer (BF16 index; FP4 under the FlyDSL row
        # bound), a stand-in too: the inputs are CPU tensors.
        launch = types.SimpleNamespace(
            __getitem__=lambda self, grid: lambda *a, **kw: 0
        )
        monkeypatch.setattr(
            m, "_index_block_score_kernel", type("K", (), vars(launch))()
        )
    _prefill(
        rows=200 if case == "short prefill" else 300,
        fp4=case != "bf16 index",
        emit=case != "no emission",
    )
    assert len(aiter_op.select) == 1
    assert not aiter_op.fused and not aiter_op.merge


def _block_table(nreq, maxb, seed):
    g = torch.Generator().manual_seed(seed)
    bt = torch.stack(
        [torch.randperm(4 * maxb, generator=g)[:maxb] for _ in range(nreq)]
    )
    return bt.to(torch.int32).to(DEV)


@table_kernels
@pytest.mark.parametrize("dist", ["gauss", "ties", "inf_nan"])
@pytest.mark.parametrize("pins", [(0, 1), (2, 3), (0, 0)])
def test_merge_of_chunk_lists_is_the_packed_selection(dist, pins):
    """Chunk lists as the fused op writes them -- a row's 16 best keys of every
    chunk it reaches, here from the partial selector at a 64-block tile; the
    lists past a row's chunks never written (poison), and entries under the
    row's 16th dropped from some chunks (what a shared floor does) -- merged
    with CHUNK_BLOCKS: the packed selector's top-k, table and lengths."""
    init, local = pins
    chunk, heads, PPB = 64, 2, m.PAGES_PER_SPARSE_BLOCK
    qlens = torch.tensor([3, 0, 9, 6])
    prefix = torch.tensor([0, 77, 40 * PAGE - 4, 300 * PAGE + 50], dtype=torch.int32)
    rows, batch = int(qlens.sum()), len(qlens)
    maxb = triton.cdiv(int((prefix + qlens).max()), PAGE)
    starts = torch.cat([torch.zeros(1), qlens.cumsum(0)]).to(torch.int32).to(DEV)
    prefix = prefix.to(DEV)
    g = torch.Generator().manual_seed(11)
    score = torch.randn(heads, rows, maxb, generator=g)
    if dist == "ties":
        score = (score * 2).round() / 2
    if dist == "inf_nan":
        score[torch.rand(score.shape, generator=g) < 0.3] = float("-inf")
        score[torch.rand(score.shape, generator=g) < 0.1] = float("nan")
    score = score.to(DEV)
    bt = _block_table(batch, maxb, 5)

    def outputs():
        idx = torch.full((heads, rows, TOPK), -7, dtype=torch.int32, device=DEV)
        out, args = m._alloc_emit(rows, heads, TOPK, bt, True, DEV, PPB)
        return idx, out, args

    want, want_out, args = outputs()
    m._launch_select(
        score, want, starts, prefix, batch, TOPK, init, local, bt, args, heads,
        rows_per_req=int(qlens.max()), decode_max_q=0, block_size_k=128,
        num_warps=1, emit=True, block_page_stride=PPB, pages_per_block=PPB,
    )  # fmt: skip

    chunks = triton.cdiv(maxb, chunk)
    keys = torch.empty(heads, rows, chunks * TOPK, dtype=torch.int64, device=DEV)
    m._prefill_partial_topk_kernel[(int(qlens.max()), batch * heads, chunks)](
        score, keys, starts, prefix, heads, init, local,
        *score.stride(), *keys.stride(),
        SPARSE_BLOCK=PAGE, BLOCK_SIZE_K=chunk, BLOCK_SIZE_T=TOPK, num_warps=1,
    )  # fmt: skip
    row_req = torch.repeat_interleave(torch.arange(batch), qlens)
    pos = prefix.cpu()[row_req] + torch.arange(rows) - starts.cpu()[row_req].long()
    reached = (pos // PAGE + chunk) // chunk * TOPK  # keys of the chunks a row reaches
    cols = torch.arange(chunks * TOPK)
    keys = keys.cpu()
    # Under the row's 16th key nothing is needed: drop it from the odd chunks.
    kth = keys.masked_fill(cols >= reached[None, :, None], 0).sort(-1, True)[0]
    kth = kth[..., TOPK - 1 : TOPK]
    odd = (cols // TOPK % 2 == 1)[None, None]
    keys = keys.masked_fill(odd & (keys < kth), 0)
    keys = keys.masked_fill(cols >= reached[None, :, None], 0x7EAD_BEEF_7EAD_BEEF)
    got, got_out, args = outputs()
    m._launch_prefill_merge(
        keys.to(DEV), got, starts, prefix, bt, args, int(qlens.max()), TOPK, PPB, PPB,
        chunk,
    )  # fmt: skip
    assert torch.equal(got, want)
    assert torch.equal(got_out[0], want_out[0]) and torch.equal(got_out[1], want_out[1])


gfx1250 = pytest.mark.skipif(GFX != "gfx1250", reason="needs a gfx1250 GPU")
HEADS = 4


def _fp4_prefill(reqs, seed, heads=HEADS):
    """Real FP4 index planes for a batch of (new tokens, cached tokens). Every
    third block of a request is its neighbour's page again -- two blocks with
    the same keys score the same wherever the causal mask leaves both whole,
    so the tie rule decides -- and every seventh query row is zero (+-0
    scores)."""
    pytest.importorskip("aiter.ops.flydsl.index_score_prefill")
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    g = torch.Generator(device=DEV).manual_seed(seed)
    nbs = [triton.cdiv(c + n, PAGE) for n, c in reqs]
    pages = sum(nbs) + 3
    index_k = (torch.randn(pages * PAGE, D, device=DEV, generator=g) * 3).bfloat16()
    payload = torch.zeros(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    index_fp4_store(index_k, torch.arange(pages * PAGE, device=DEV), payload, scale)
    total = sum(n for n, _ in reqs)
    q = (torch.randn(total, heads, D, device=DEV, generator=g) * 3).bfloat16()
    q[::7] = 0
    perm = torch.randperm(pages, device=DEV, generator=g).int()
    bt = torch.zeros(len(reqs), max(nbs), dtype=torch.int32, device=DEV)
    at = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[at : at + nb]
        bt[i, 1:nb:3] = bt[i, 0 : nb - 1 : 3]
        at += nb
    lens = torch.tensor([n for n, _ in reqs], dtype=torch.int32)
    cu = torch.cat([torch.zeros(1, dtype=torch.int32), lens.cumsum(0).int()]).to(DEV)
    prefix = torch.tensor([c for _, c in reqs], dtype=torch.int32, device=DEV)
    return q, payload, scale, bt, cu, prefix + lens.to(DEV), prefix, int(lens.max())


def _index_topk(monkeypatch, fused, data, pins, opts, ppb, max_seq=None):
    """The whole call, fused or by the FlyDSL scorer + Triton selector; opts:
    the fused op's OPTS (None: ATOM's default). aiter's shape preference is
    bypassed (these shapes are short), and the path taken is checked."""
    from aiter.ops.flydsl import index_score_prefill as op

    q, payload, scale, bt, cu, seq, prefix, max_q = data
    monkeypatch.setenv(FLAG, "1" if fused else "0")
    if opts is None:
        monkeypatch.delenv(OPTS, raising=False)
    else:
        monkeypatch.setenv(OPTS, opts)
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_PREFILL", "1")
    monkeypatch.setenv("ATOM_M3_INDEX_SCORE_FLYDSL_MIN_ROWS", "1")
    monkeypatch.setattr(op, "index_topk_prefill_nvfp4_preferred", lambda *a: True)
    merged, merge = [], m._launch_prefill_merge
    monkeypatch.setattr(
        m, "_launch_prefill_merge", lambda *a: (merged.append(1), merge(*a))[1]
    )
    out = m.minimax_m3_index_topk(
        q, payload, bt, cu, seq, prefix, max_q, max_seq or int(seq.max()), TOPK,
        pins[0], pins[1], q.shape[1], D**-0.5, emit_sparse_block_table=True,
        block_page_stride=ppb, index_kv_scale=scale, emit_pages_per_block=ppb,
    )  # fmt: skip
    assert bool(merged) == fused
    return out


@gfx1250
@pytest.mark.parametrize(
    "reqs",
    [
        [(2048, 0)],
        [(2304, 70000)],
        [(700, 300), (2100, 150000), (0, 9), (1500, 40000)],
    ],
)
@pytest.mark.parametrize("pins", [(0, 1), (2, 3)])
@pytest.mark.parametrize(
    "opts,heads",
    [
        (None, HEADS),  # ATOM's default launch
        (None, 1),  # one index head a rank (TP4)
        ("", HEADS),  # aiter's
        ("chunk_blocks=64,block_q=128,waves=2,ring=2", HEADS),
    ],
)
@pytest.mark.parametrize("ppb", [1, m.PAGES_PER_SPARSE_BLOCK])
def test_fused_prefill_is_the_scorer_and_selector(
    monkeypatch, reqs, pins, opts, heads, ppb
):
    """Through `minimax_m3_index_topk`: the fused op + merge against aiter's
    FlyDSL scorer + the Triton selector (the same WMMA for every dot, so the
    same scores): top-k in order, page table and lengths."""
    data = _fp4_prefill(reqs, len(reqs), heads)
    want = _index_topk(monkeypatch, False, data, pins, opts, ppb)
    # Freed blocks of these sizes poisoned: a row the fused path left alone
    # must not pass on what an earlier case left in the allocator.
    poison = [torch.full_like(t, -77) for t in want]
    del poison
    got = _index_topk(monkeypatch, True, data, pins, opts, ppb)
    torch.cuda.synchronize()
    for a, b in zip(want, got):
        assert torch.equal(a, b)


@gfx1250
def test_fused_prefill_replays_from_a_graph(monkeypatch):
    """Captured once, replayed with other queries and lengths under the same
    bounds: the key buffer and its floor column are filled inside the graph."""
    reqs = [(2048, 100000), (2048, 30000)]
    data = _fp4_prefill(reqs, 9)
    q, seq, prefix = data[0], data[5], data[6]
    bound = int(seq.max())
    args = (data, (0, 1), None, 1, bound)
    _index_topk(monkeypatch, True, *args)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = _index_topk(monkeypatch, True, *args)
    for cached in ([100000, 30000], [5000, 99000], [0, 64]):
        prefix.copy_(torch.tensor(cached, dtype=torch.int32))
        seq.copy_(prefix + 2048)
        q.normal_()
        graph.replay()
        torch.cuda.synchronize()
        replayed = [t.clone() for t in got]
        want = _index_topk(monkeypatch, False, *args)
        for a, b in zip(want, replayed):
            assert torch.equal(a, b)
