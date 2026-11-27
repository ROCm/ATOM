# SPDX-License-Identifier: MIT
"""MiniMax-M3 prefill indexer through aiter's fused scorer + top-16
(``ATOM_M3_INDEX_TOPK_FLYDSL_PREFILL``): the scores never reach memory, each of
the op's workgroups hands on a row's 16 best blocks of its chunk as
``_pack_score_key`` keys and ``_prefill_merge_topk_kernel`` picks the row's 16
and emits.

- the gate and the launches, plain Python (the kernels are stand-ins);
- the merge over chunk lists against the packed selector, on a GPU or on the
  CPU under ``TRITON_INTERPRET=1``;
- on a gfx1250 GPU with the aiter op: the whole call against the FlyDSL scorer
  + Triton selector on real FP4 planes (exact ties and +-0 scores among them).
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
    """Stand-in for aiter's fused op: records each launch."""
    calls = types.SimpleNamespace(fused=[], serves=True, preferred=True)
    mod = types.ModuleType(AITER_OP)
    mod.index_topk_prefill_chunk_blocks = lambda *a, **kw: 256
    mod.index_topk_prefill_nvfp4_preferred = lambda *a: calls.preferred
    mod.index_topk_prefill_nvfp4_supported = lambda *a: calls.serves

    def fused(*a, **kw):
        calls.fused.append((a, kw))
        return "keys"

    mod.index_topk_prefill_nvfp4 = fused
    monkeypatch.setitem(sys.modules, AITER_OP, mod)
    calls.select, calls.merge, calls.scored = [], [], []
    monkeypatch.setattr(m, "_launch_select", lambda *a, **kw: calls.select.append(a))
    monkeypatch.setattr(m, "_launch_prefill_merge", lambda *a: calls.merge.append(a))
    monkeypatch.setattr(m, "_score_chunk_blocks", lambda *a: 64)
    monkeypatch.setattr(
        m, "_flydsl_prefill_scorer", lambda *a: lambda *b: calls.scored.append(b)
    )
    return calls


def _prefill(rows=2048, fp4=True, emit=True, pages_per_block=1):
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
    )
    return out, blocks


def test_off_keeps_the_scorer_and_the_selector(monkeypatch, aiter_op):
    monkeypatch.setenv(FLAG, "0")
    monkeypatch.setitem(sys.modules, AITER_OP, None)  # never imported
    _prefill()
    assert not aiter_op.fused and not aiter_op.merge
    assert len(aiter_op.scored) == 1 and len(aiter_op.select) == 1


@pytest.mark.parametrize("pages_per_block", [1, m.PAGES_PER_SPARSE_BLOCK])
def test_the_fused_op_feeds_the_merge(monkeypatch, aiter_op, pages_per_block):
    monkeypatch.setenv(FLAG, "1")
    (topk_idx, table, ctx), blocks = _prefill(pages_per_block=pages_per_block)
    assert not aiter_op.scored and not aiter_op.select
    ((args, kw),) = aiter_op.fused
    assert kw == {"chunk_blocks": 256} and len(args) == 12
    # ..., sm_scale, max_query_len, max_block, init_blocks, local_blocks
    assert args[7:] == (D**-0.5, 2048, blocks, 0, 1)
    (merge,) = aiter_op.merge
    assert merge[0] == "keys" and merge[1] is topk_idx
    assert merge[5][0] is table and merge[5][1] is ctx
    # rows a request, topk, block_page_stride, pages a block, chunk
    assert merge[6:] == (2048, 16, m.PAGES_PER_SPARSE_BLOCK, pages_per_block, 256)
    assert topk_idx.shape == (4, 2048, TOPK)
    assert table.shape == (4 * 2048, TOPK * pages_per_block) and ctx.shape == (8192,)


@pytest.mark.parametrize(
    "case",
    [
        "bf16 index",
        "no emission",
        "aiter declines",
        "aiter prefers two",
    ],
)
def test_other_calls_keep_the_selector(monkeypatch, aiter_op, case):
    monkeypatch.setenv(FLAG, "1")
    aiter_op.serves = case != "aiter declines"
    aiter_op.preferred = case != "aiter prefers two"
    if case == "bf16 index":
        monkeypatch.setattr(m, "_flydsl_prefill_scorer", lambda *a: None)
        launch = types.SimpleNamespace(
            __getitem__=lambda self, grid: lambda *a, **kw: 0
        )
        monkeypatch.setattr(
            m, "_index_block_score_kernel", type("K", (), vars(launch))()
        )
    _prefill(
        fp4=case != "bf16 index",
        emit=case != "no emission",
    )
    assert len(aiter_op.select) == 1
    assert not aiter_op.fused and not aiter_op.merge


def _chunk_lists(score, starts, prefix, init, local, chunk):
    """The fused op's lists in torch: a row's 16 best `_pack_score_key` keys of
    every chunk it reaches (pins included), 0 for no block."""
    heads, rows, maxb = score.shape
    batch = starts.shape[0] - 1
    row_req = torch.repeat_interleave(torch.arange(batch), starts.diff().long())
    pos = prefix[row_req].long() + torch.arange(rows) - starts[row_req].long()
    vb = (pos // PAGE + 1)[None, :, None]
    col = torch.arange(maxb)[None, None]
    valid = col < vb
    s = torch.where(valid & (col < init), 1e30, score)
    s = torch.where(valid & (col >= vb - local), 1e29, s)
    bits = s.view(torch.int32).long() & 0xFFFFFFFF
    bits = torch.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ torch.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    ordered = torch.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = torch.where(valid, (1 << 48) | (ordered << 16) | (col + 1), 0)
    chunks = triton.cdiv(maxb, chunk)
    key = torch.nn.functional.pad(key, (0, chunks * chunk - maxb))
    lists = key.view(heads, rows, chunks, chunk).sort(-1, True)[0][..., :TOPK]
    return lists.reshape(heads, rows, chunks * TOPK)


@table_kernels
@pytest.mark.parametrize("dist", ["gauss", "ties", "inf_nan"])
@pytest.mark.parametrize("pins", [(0, 1), (2, 3), (0, 0)])
def test_merge_of_chunk_lists_is_the_packed_selection(dist, pins):
    """Chunk lists merged with CHUNK_BLOCKS: the packed selector's top-k, table
    and lengths (lists past a row's chunks are never read: poisoned here)."""
    init, local = pins
    chunk, heads, PPB = 64, 2, m.PAGES_PER_SPARSE_BLOCK
    qlens = torch.tensor([3, 0, 9, 6])
    prefix = torch.tensor([0, 77, 40 * PAGE - 4, 300 * PAGE + 50], dtype=torch.int32)
    rows, batch = int(qlens.sum()), len(qlens)
    maxb = triton.cdiv(int((prefix + qlens).max()), PAGE)
    starts = torch.cat([torch.zeros(1), qlens.cumsum(0)]).to(torch.int32)
    g = torch.Generator().manual_seed(11)
    score = torch.randn(heads, rows, maxb, generator=g)
    if dist == "ties":
        score = (score * 2).round() / 2
    if dist == "inf_nan":
        score[torch.rand(score.shape, generator=g) < 0.3] = float("-inf")
        score[torch.rand(score.shape, generator=g) < 0.1] = float("nan")
    bt = torch.stack(
        [torch.randperm(4 * maxb, generator=g)[:maxb] for _ in range(batch)]
    ).to(torch.int32)
    keys = _chunk_lists(score, starts, prefix, init, local, chunk)
    row_req = torch.repeat_interleave(torch.arange(batch), qlens)
    pos = prefix[row_req] + torch.arange(rows) - starts[row_req].long()
    reached = (pos // PAGE + chunk) // chunk * TOPK
    cols = torch.arange(keys.shape[-1])
    keys = keys.masked_fill(cols >= reached[None, :, None], 0x7EAD_BEEF_7EAD_BEEF)
    score, starts, prefix, bt, keys = (
        t.to(DEV) for t in (score, starts, prefix, bt, keys)
    )

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
    got, got_out, args = outputs()
    m._launch_prefill_merge(
        keys, got, starts, prefix, bt, args, int(qlens.max()), TOPK, PPB, PPB, chunk
    )
    assert torch.equal(got, want)
    assert torch.equal(got_out[0], want_out[0]) and torch.equal(got_out[1], want_out[1])


gfx1250 = pytest.mark.skipif(GFX != "gfx1250", reason="needs a gfx1250 GPU")
HEADS = 4


def _fp4_prefill(reqs, seed):
    """Real FP4 index planes for a batch of (new tokens, cached tokens); every
    third block is its neighbour's page again (ties) and every seventh query
    row is zero (+-0 scores)."""
    from atom.model_ops.minimax_m3.index_fp4 import index_fp4_store

    g = torch.Generator(device=DEV).manual_seed(seed)
    nbs = [triton.cdiv(c + n, PAGE) for n, c in reqs]
    pages = sum(nbs) + 3
    index_k = (torch.randn(pages * PAGE, D, device=DEV, generator=g) * 3).bfloat16()
    payload = torch.zeros(pages, PAGE, D // 2, dtype=torch.uint8, device=DEV)
    scale = torch.zeros(pages, PAGE, D // 16, dtype=torch.uint8, device=DEV)
    index_fp4_store(index_k, torch.arange(pages * PAGE, device=DEV), payload, scale)
    total = sum(n for n, _ in reqs)
    q = (torch.randn(total, HEADS, D, device=DEV, generator=g) * 3).bfloat16()
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


@gfx1250
@pytest.mark.parametrize(
    "reqs",
    [[(2048, 0)], [(2304, 70000)], [(700, 300), (2100, 150000), (0, 9), (1500, 40000)]],
)
@pytest.mark.parametrize("pins", [(0, 1), (2, 3)])
@pytest.mark.parametrize("ppb", [1, m.PAGES_PER_SPARSE_BLOCK])
def test_fused_prefill_is_the_scorer_and_selector(monkeypatch, reqs, pins, ppb):
    """Through `minimax_m3_index_topk`, aiter's shape preference bypassed: the
    fused op + merge against the FlyDSL scorer + Triton selector -- top-k in
    order, page table and lengths."""
    aiter_mod = pytest.importorskip(AITER_OP)
    monkeypatch.setattr(
        aiter_mod, "index_topk_prefill_nvfp4_preferred", lambda *a: True
    )
    q, payload, scale, bt, cu, seq, prefix, max_q = _fp4_prefill(reqs, len(reqs))

    def run(flag):
        monkeypatch.setenv(FLAG, flag)
        return m.minimax_m3_index_topk(
            q, payload, bt, cu, seq, prefix, max_q, int(seq.max()), TOPK,
            pins[0], pins[1], HEADS, D**-0.5, emit_sparse_block_table=True,
            block_page_stride=ppb, index_kv_scale=scale, emit_pages_per_block=ppb,
        )  # fmt: skip

    want, got = run("0"), run("1")
    torch.cuda.synchronize()
    for a, b in zip(want, got):
        assert torch.equal(a, b)
