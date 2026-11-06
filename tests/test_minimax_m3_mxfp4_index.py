# SPDX-License-Identifier: MIT
"""MiniMax-M3 index selection over the MXFP4 index cache (gfx950).

Scores come from aiter's FlyDSL scorers; what is pinned here is the ATOM side:
the views of the pool page, the work map built once per step into a persistent
buffer (and still right when a captured graph replays it with new lengths), and
the page-128 block table the selection emits for the MXFP4 KV (one page a block,
exact context with the tail block last). The oracle is the op's definition on the
dequantized cache, not another kernel.
"""

import pytest
import torch

from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, page_views
from atom.mono.weights.formats import dequantize_mxfp4

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

TOPK, HEADS, DIM = 16, 1, 128
SM_SCALE = DIM**-0.5


def make_case(batch, S, seed, max_len=128 * 70, fp8_q=False, heads=HEADS):
    g = torch.Generator(device="cuda").manual_seed(seed)
    seq = torch.randint(S + 1, max_len, (batch,), device="cuda", generator=g)
    seq[0] = max_len  # one full-length request
    blocks = -(-max_len // 128)
    pages = batch * blocks + 3
    cache = torch.randint(
        0, 256, (pages, PAGE_BYTES), device="cuda", generator=g, dtype=torch.uint8
    )
    # e8m0 bytes within 2^-2..2^2 of unity, as a quantized activation's are (a wider spread
    # leaves e4m3 range, which the prefill scorer stages through)
    d = cache.view(pages, 1, PAGE_BYTES)[:, :, 8192:]
    d.copy_(
        torch.randint(125, 130, d.shape, device="cuda", generator=g).to(torch.uint8)
    )
    table = torch.stack(
        [
            torch.randperm(pages, device="cuda", generator=g)[:blocks]
            for _ in range(batch)
        ]
    ).to(torch.int32)
    q = torch.randn(batch * S, heads, DIM, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    return cache, table, seq.to(torch.int32), q


def reference(cache, table, seq, q, S):
    """scores [H, rows, blocks] in the natural-log domain (the scorers return them
    times log2(e); selection is unaffected), nan past the causal block, and the expected
    selection: set of block ids, and ctx, per row."""
    deq = dequantize_mxfp4(*page_views(cache))  # [pages, 128, 128]
    batch, blocks = table.shape
    rows = batch * S
    score = torch.full((HEADS, rows, blocks), float("nan"), device="cuda")
    picks, ctx = [], []
    for b in range(batch):
        for t in range(S):
            r = b * S + t
            causal = int(seq[b]) - S + t + 1
            last = (causal - 1) // 128
            qr = q[r].float()  # [H, 128]
            for p in range(last + 1):
                k = deq[table[b, p].long()]  # [128, 128]
                n = min(128, causal - p * 128)
                score[:, r, p] = (k[:n] @ qr.T).amax(0) * SM_SCALE
            # tail block pinned (local_blocks=1); the rest by score
            others = score[0, r, :last]
            take = min(TOPK - 1, last)
            chosen = set(others.topk(take).indices.tolist()) if take else set()
            picks.append(chosen | {last})
            ctx.append(len(chosen) * 128 + (causal - last * 128))
    return score, picks, ctx


def work_map_buf(max_block, S, max_batch=32):
    """The persistent dispatch-map buffer, capacity for `max_batch` requests."""
    from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

    rows = minimax_m3_index_score_work_map_capacity(max_batch, max_block, S, HEADS)
    return torch.empty((rows, 2), dtype=torch.int32, device="cuda")


def work_map(seq, max_block, S, out=None):
    from aiter.ops.flydsl import minimax_m3_index_score_work_map

    return minimax_m3_index_score_work_map(seq, max_block, S, HEADS, out=out)


# `ppb` 16-pages a block: 1 for the MXFP4 KV, 8 for the page-16 SHUFFLE KV that
# an index-only MXFP4 run keeps (stride 8: its blocks are contiguous)
@pytest.mark.parametrize(
    "batch,S,ppb,fp8_q",
    [(5, 8, 1, False), (17, 8, 1, True), (6, 1, 1, False), (5, 8, 8, True)],
)
def test_decode_selection_and_block_table(batch, S, ppb, fp8_q):
    from atom.model_ops.minimax_m3.index_mxfp4 import IndexScoreStep
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    cache, table, seq, q = make_case(batch, S, seed=batch * 10 + S, fp8_q=fp8_q)
    max_block = table.shape[1]
    wmap = work_map(seq, max_block, S, work_map_buf(max_block, S))
    topk_idx, sbt, sctx = minimax_m3_index_topk_decode(
        q, cache, table, seq, int(seq.max()), TOPK, 0, 1, HEADS, SM_SCALE,
        emit_sparse_block_table=True, max_query_len=S, block_page_stride=ppb,
        index_mxfp4=True, pages_per_block=ppb,
        index_score=IndexScoreStep(wmap, max_block, None),
    )  # fmt: skip
    _, picks, ctx = reference(cache, table, seq, q, S)
    assert sbt.shape == (batch * S, TOPK * ppb) and sctx.shape == (batch * S,)
    for r in range(batch * S):
        got = {int(x) for x in topk_idx[0, r].tolist() if x >= 0}
        assert got == picks[r], f"row {r}"
        assert int(sctx[r]) == ctx[r], f"row {r} ctx"
        n = len(got)
        row = table[r // S].tolist()
        want_pages = {row[blk] * ppb + j for blk in picks[r] for j in range(ppb)}
        assert {int(x) for x in sbt[r, : n * ppb].tolist()} == want_pages
        # the tail block is last: it is the only one that can be partial
        tail = (int(seq[r // S]) - S + r % S) // 128
        want_tail = [row[tail] * ppb + j for j in range(ppb)]
        assert sbt[r, (n - 1) * ppb : n * ppb].tolist() == want_tail


def test_prefill_selection_matches_the_decode_definition():
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

    batch, S = 3, 5
    cache, table, seq, q = make_case(batch, S, seed=7)
    # a ragged prefill of S new tokens on top of seq - S cached ones, per request
    cu = torch.arange(0, (batch + 1) * S, S, device="cuda", dtype=torch.int32)
    prefix = (seq - S).to(torch.int32)
    topk_idx, _, sctx = minimax_m3_index_topk(
        q, cache, table, cu, seq, prefix, S, int(seq.max()), TOPK, 0, 1, HEADS,
        SM_SCALE, emit_sparse_block_table=True, block_page_stride=1,
        index_mxfp4=True, pages_per_block=1,
    )  # fmt: skip
    score, picks, ctx = reference(cache, table, seq, q, S)
    for r in range(batch * S):
        got = {int(x) for x in topk_idx[0, r].tolist() if x >= 0}
        # The prefill scorer rounds K and Q to e4m3 (the dequantized pages are
        # staged as fp8, as an fp8 index cache is), so near-ties may flip against the
        # exact oracle (up to ~0.2 std of the row on this data): what
        # holds is that every pick scores within a rounding step of the best.
        last = max(picks[r])
        assert last in got and len(got) == len(picks[r]) and int(sctx[r]) == ctx[r]
        row = score[0, r, :last]
        if last > TOPK - 1:
            tol = 0.25 * row.std().item()
            cut = row.topk(TOPK - 1).values[-1].item()
            assert all(row[b].item() >= cut - tol for b in got - {last})
            assert all(row[b].item() <= cut + tol for b in set(range(last)) - got)
        else:
            assert got == picks[r]


def _score_steps(batch, S, seed):
    """Three decode steps over one cache: long lengths; the same requests permuted
    onto other rows and shortened (so a row's pages shrink); a smaller batch."""
    cache, table, seq, q = make_case(batch, S, seed=seed)
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    perm = torch.randperm(batch, device="cuda", generator=g)
    short = (seq[perm] // 3).clamp(min=S + 1).to(torch.int32)
    qs = q.view(batch, S, HEADS, DIM)[perm].reshape(batch * S, HEADS, DIM)
    n = max(2, batch - 2)
    seq3 = torch.randint(S + 1, int(seq.max()), (n,), device="cuda", generator=g)
    return cache, [
        (table, seq, q),
        (table[perm], short, qs),
        (table[:n], seq3.to(torch.int32), q[: n * S]),
    ]


def test_a_persistent_score_buffer_changes_nothing():
    """The scorer leaves a page no row reaches unstored, so the buffer holds -inf there
    only if refilled: one refill a step must give the very scores (and selection)
    a fresh -inf buffer per call does, across permuted, shorter and smaller steps."""
    from aiter.ops.flydsl import minimax_m3_index_score_fp4_flydsl

    from atom.model_ops.minimax_m3.index_mxfp4 import (
        IndexScoreStep,
        fresh_index_score,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    batch, S = 6, 8
    cache, steps = _score_steps(batch, S, seed=21)
    max_block = steps[0][0].shape[1]
    wbuf = work_map_buf(max_block, S)
    sbuf = torch.empty(HEADS * 32 * S * max_block, dtype=torch.float32, device="cuda")
    k_data, k_scale = page_views(cache)

    def run(table, seq, q, out):
        wmap = work_map(seq, max_block, S, wbuf)
        score = minimax_m3_index_score_fp4_flydsl(
            q, k_data, k_scale, table, seq, S, HEADS, SM_SCALE, max_block,
            out=out, work_map=wmap,
        )  # fmt: skip
        sel = minimax_m3_index_topk_decode(
            q, cache, table, seq, max_block * 128, TOPK, 0, 1, HEADS, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            index_mxfp4=True, pages_per_block=1,
            index_score=IndexScoreStep(wmap, max_block, out),
        )  # fmt: skip
        return score.clone(), sel

    for i, (table, seq, q) in enumerate(steps):
        rows = table.shape[0] * S
        if (
            i == 1
        ):  # control: without the refill, the shorter step reads step 0's scores
            stale = run(
                table,
                seq,
                q,
                sbuf[: HEADS * rows * max_block].view(HEADS, rows, max_block),
            )[0]
            fresh = run(table, seq, q, None)[0]
            assert not torch.equal(stale, fresh), "the control must see stale scores"  # fmt: skip
        out = fresh_index_score(sbuf, rows, HEADS, max_block, S)
        got_score, got_sel = run(table, seq, q, out)
        want_score, want_sel = run(table, seq, q, None)
        assert torch.equal(got_score, want_score), f"step {i} scores"
        for g, w in zip(got_sel, want_sel):
            assert torch.equal(g, w), f"step {i} selection"


def _pad(table, seq, q, S, running, fill):
    """The scheduled rows of a step, then `running - scheduled` pad rows the way the
    builder publishes them: length 0, a filled block-table row, zero activations.
    The builder zero-fills the row; 0x7FFFFFFF is a page id no cache has, so a run
    that passes with it never dereferenced a pad row's table."""
    n = seq.shape[0]
    t = torch.full((running, table.shape[1]), fill, dtype=table.dtype, device="cuda")
    s = torch.zeros(running, dtype=seq.dtype, device="cuda")
    x = torch.zeros((running * S, *q.shape[1:]), dtype=q.dtype, device="cuda")
    t[:n], s[:n], x[: n * S] = table, seq, q
    return t, s, x


def _assert_pad_rows_are_empty(got, live, heads):
    """What the sparse attention is handed for a pad row: nothing. A context of
    exactly 0 (never negative: decode kernels take 0 <= context) and page id 0,
    whatever the pad row's block table holds. The emitted rows are token-major,
    head-minor, so `live` request tokens are `live * heads` rows."""
    _, sbt, sctx = got
    assert int(sctx[live * heads :].abs().max()) == 0, "a pad row has a context"
    assert int(sbt[live * heads :].abs().max()) == 0, "a pad row names a page"


def _assert_live_rows_match(got, want, live, heads):
    assert torch.equal(got[0][:, :live], want[0])
    assert torch.equal(got[1][: live * heads], want[1])
    assert torch.equal(got[2][: live * heads], want[2])


def _metadata(table, seq, S, buffers, nvalid, heads=HEADS):
    """What the builder refills each step, outside any graph."""
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    bound = table.shape[1]
    return make_sparse_decode_metadata(
        seq_lens=seq,
        block_table=table,
        slot_mapping=torch.zeros(seq.shape[0] * S, dtype=torch.int64, device="cuda"),
        max_seq_len=bound * 128,
        num_idx_heads=heads,
        max_query_len=S,
        n_valid_column_per_row_out=nvalid,
        index_score_buffers=buffers,
    )


def _select(q, cache, table, seq, S, md, heads=HEADS):
    """The decode selection as the attention layer calls it."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    bound = table.shape[1]
    return minimax_m3_index_topk_decode(
        q, cache, table, seq, bound * 128, TOPK, 0, 1, heads, SM_SCALE,
        emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
        index_mxfp4=True, pages_per_block=1,
        n_valid_column_per_row=md.n_valid_column_per_row,
        index_score=md.index_score,
    )  # fmt: skip


def _buffers(running, S, bound, heads=HEADS):
    from atom.model_ops.minimax_m3.index_mxfp4 import IndexScoreBuffers

    buffers = IndexScoreBuffers.alloc(running, S, heads, bound, "cuda")
    nvalid = torch.empty(heads * running * S, dtype=torch.int32, device="cuda")
    return buffers, nvalid


@pytest.mark.parametrize(
    "S,heads,scheduled,fill",
    [
        (8, 1, 3, 0),
        (1, 2, 3, 0x7FFFFFFF),
        (8, 2, 1, 0x7FFFFFFF),
        (1, 1, 0, 0),  # an idle DP rank still runs the group's rows
    ],
)
def test_a_padded_eager_step_selects_like_the_scheduled_batch(
    S, heads, scheduled, fill
):
    """An eager step padded for DP runs `running_bs` rows of idx_q: the scorer takes
    its batch off them, so lengths and block table hold pad rows too, and the rows
    that are real select what they do unpadded."""
    running = 8
    cache, table, seq, q = make_case(max(scheduled, 1), S, seed=70 + S, heads=heads)
    table, seq, q = table[:scheduled], seq[:scheduled], q[: scheduled * S]
    bound = table.shape[1]
    pt, ps, pq = _pad(table, seq, q, S, running, fill)
    bufs = _buffers(running, S, bound, heads)
    got = _select(pq, cache, pt, ps, S, _metadata(pt, ps, S, *bufs, heads), heads)
    if scheduled:
        bufs = _buffers(scheduled, S, bound, heads)
        md = _metadata(table, seq, S, *bufs, heads)
        want = _select(q, cache, table, seq, S, md, heads)
        _assert_live_rows_match(got, want, scheduled * S, heads)
    _assert_pad_rows_are_empty(got, scheduled * S, heads)


@pytest.mark.parametrize(
    "S,heads,fill", [(8, 1, 0), (1, 2, 0x7FFFFFFF), (8, 2, 0x7FFFFFFF)]
)
def test_a_graph_replays_a_padded_step_like_the_scheduled_batch(S, heads, fill):
    """Captured at 8 requests, replayed with 3 scheduled and 5 pad rows: the
    metadata refilled outside the graph is the padded batch's."""
    running, scheduled = 8, 3
    cache, table, seq, q = make_case(running, S, seed=80 + S, heads=heads)
    bound = table.shape[1]
    buffers, nvalid = _buffers(running, S, bound, heads)
    static = [x.clone() for x in (table, seq, q)]

    def refill():
        return _metadata(static[0], static[1], S, buffers, nvalid, heads)

    def run(md):
        return _select(static[2], cache, static[0], static[1], S, md, heads)

    md = refill()
    run(md)  # JIT outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run(md)
    t, s, x = _pad(
        table[:scheduled], seq[:scheduled], q[: scheduled * S], S, running, fill
    )
    for dst, src in zip(static, (t, s, x)):
        dst.copy_(src)
    refill()  # the builder's per-step refill, outside the graph
    graph.replay()
    torch.cuda.synchronize()
    bufs = _buffers(scheduled, S, bound, heads)
    md = _metadata(table[:scheduled], seq[:scheduled], S, *bufs, heads)
    want = _select(q[: scheduled * S], cache, table[:scheduled], seq[:scheduled], S, md, heads)  # fmt: skip
    _assert_live_rows_match(got, want, scheduled * S, heads)
    _assert_pad_rows_are_empty(got, scheduled * S, heads)


@pytest.mark.parametrize("S,heads", [(8, 1), (1, 2)])
def test_an_eager_step_at_the_batch_block_count_selects_like_the_model_width(S, heads):
    """The step's own block count (`eager`) must not change what is selected: the
    scorer's grid, the score width and the selector's inputs shrink, the selection
    does not. The batch is much shorter than the buffers' model-length bound."""
    from atom.model_ops.minimax_m3.index_mxfp4 import IndexScoreBuffers
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    batch = 5
    cache, table, seq, q = make_case(
        batch, S, seed=90 + S, max_len=128 * 12, heads=heads
    )
    model_blocks = 64  # the buffers' bound: far past this batch's 12 blocks
    # the block table is as wide as the model, as the builder's is
    table = torch.nn.functional.pad(table, (0, model_blocks - table.shape[1]))
    max_seq = int(seq.max())
    buffers = IndexScoreBuffers.alloc(batch, S, heads, model_blocks, "cuda")
    nvalid = torch.empty(heads * batch * S, dtype=torch.int32, device="cuda")
    outs = []
    for eager in (False, True):
        md = make_sparse_decode_metadata(
            seq_lens=seq,
            block_table=table,
            slot_mapping=torch.zeros(batch * S, dtype=torch.int64, device="cuda"),
            max_seq_len=max_seq,
            num_idx_heads=heads,
            max_query_len=S,
            n_valid_column_per_row_out=nvalid,
            index_score_buffers=buffers,
            index_score_eager=eager,
        )
        assert md.index_score.max_block == (-(-max_seq // 128) if eager else 64)
        out = minimax_m3_index_topk_decode(
            q, cache, table, seq, max_seq, TOPK, 0, 1, heads, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            index_mxfp4=True, pages_per_block=1,
            n_valid_column_per_row=md.n_valid_column_per_row,
            index_score=md.index_score,
        )  # fmt: skip
        outs.append(out)
    for graph_width, eager_width in zip(*outs):
        assert torch.equal(graph_width, eager_width)
