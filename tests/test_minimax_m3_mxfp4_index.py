# SPDX-License-Identifier: MIT
"""MiniMax-M3 index selection over the MXFP4 index cache (gfx950).

Scores come from aiter's FlyDSL scorers; what is pinned here is the ATOM side:
the views of the pool page, the work map built once per step into a persistent
buffer (and still right when a captured graph replays it with new lengths), and
the page-128 block table the selection emits for the MXFP4 KV (one page a block,
exact context with the tail block last). The oracle is the op's definition on the
dequantized cache, not another kernel.
"""

import contextlib

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


def _record(running, S, bound, heads=HEADS, prefill_rows=0):
    """The builder's scorer record: the capacity buffer of the dispatch map for
    `running` requests, and the score workspace every layer shares, at the map's
    width and tall enough for `running * S` decode rows or `prefill_rows` -- never
    filled, here with values no selector may read (NaN, +-inf, a huge finite)."""
    from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity

    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

    rows = minimax_m3_index_score_work_map_capacity(running, bound, S, heads)
    n = heads * max(running * S, prefill_rows) * bound
    scores = LogitsWorkspace(n, "cuda")
    junk = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 3e38], device="cuda"
    )
    flat = scores.logits(1, n)
    flat.copy_(junk[torch.randint(0, 4, flat.shape, device="cuda")])
    return IndexScoreBuffers(
        scores, torch.empty((rows, 2), dtype=torch.int32, device="cuda"), bound
    )


@contextlib.contextmanager
def _aiter_allocates_its_scores():
    """Run the scorers as before the workspace: given no `out`, aiter allocates the
    scores and fills them with -inf itself (transposed at S * H >= 16, which the
    selector no longer takes: the values are made contiguous, the same scores)."""
    from atom.model_ops.minimax_m3 import index_topk  # binds the scorers

    names = (
        "minimax_m3_index_score_fp4_flydsl",
        "minimax_m3_index_score_prefill_fp4_flydsl",
    )
    real = {n: getattr(index_topk, n) for n in names}
    for n in names:
        setattr(
            index_topk,
            n,
            lambda *a, _f=real[n], **k: _f(*a, **{**k, "out": None}).contiguous(),
        )
    try:
        yield
    finally:
        for n in names:
            setattr(index_topk, n, real[n])


# `ppb` 16-pages a block: 1 for the MXFP4 KV, 8 for the page-16 SHUFFLE KV that
# an index-only MXFP4 run keeps (stride 8: its blocks are contiguous)
@pytest.mark.parametrize(
    "batch,S,ppb,fp8_q",
    [(5, 8, 1, False), (17, 8, 1, True), (6, 1, 1, False), (5, 8, 8, True)],
)
def test_decode_selection_and_block_table(batch, S, ppb, fp8_q):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    cache, table, seq, q = make_case(batch, S, seed=batch * 10 + S, fp8_q=fp8_q)
    max_block = table.shape[1]
    record, nvalid = _buffers(batch, S, max_block)
    md = _metadata(table, seq, S, record, nvalid)
    topk_idx, sbt, sctx = minimax_m3_index_topk_decode(
        q, cache, table, seq, int(seq.max()), TOPK, 0, 1, HEADS, SM_SCALE,
        emit_sparse_block_table=True, max_query_len=S, block_page_stride=ppb,
        pages_per_block=ppb, index_scorer=md.index_scorer,
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
    record = _record(batch, S, table.shape[1], prefill_rows=batch * S)
    topk_idx, _, sctx = minimax_m3_index_topk(
        q, cache, table, cu, seq, prefix, S, int(seq.max()), TOPK, 0, 1, HEADS,
        SM_SCALE, emit_sparse_block_table=True, block_page_stride=1,
        pages_per_block=1, index_scorer=record,
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


def _metadata(table, seq, S, record, nvalid, heads=HEADS):
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
        index_scorer=record,
    )


def _select(q, cache, table, seq, S, md, heads=HEADS, allocated=False):
    """The decode selection as the attention layer calls it; `allocated` runs it
    with aiter allocating (and -inf filling) its own scores, as before the workspace."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    bound = table.shape[1]
    with _aiter_allocates_its_scores() if allocated else contextlib.nullcontext():
        return minimax_m3_index_topk_decode(
            q, cache, table, seq, bound * 128, TOPK, 0, 1, heads, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            pages_per_block=1,
            n_valid_column_per_row=md.n_valid_column_per_row,
            index_scorer=md.index_scorer,
        )  # fmt: skip


def _buffers(running, S, bound, heads=HEADS):
    """The record (see `_record`) and the buffer of live columns per row."""
    nvalid = torch.empty(heads * running * S, dtype=torch.int32, device="cuda")
    return _record(running, S, bound, heads), nvalid


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


# S * heads >= 16 is where aiter's scorer lays the scores out transposed when it
# allocates them itself, and with 2048+ blocks a row is wide enough for the aiter
# selector, which only a contiguous layout can use there
@pytest.mark.parametrize(
    "S,heads,blocks",
    [(1, 1, 70), (8, 1, 70), (4, 4, 70), (8, 2, 70), (1, 2, 2100), (8, 2, 2100)],
)
def test_the_unfilled_shared_workspace_selects_like_aiters_own_filled_scores(
    S, heads, blocks
):
    """The scorer writing into the shared workspace (contiguous, filled with junk, reused
    by a second step with other lengths) leaves the selection, the block table and
    the contexts of every row bit for bit what aiter's own allocated, -inf-filled
    scores give: ragged lengths, pad rows past the scheduled ones, e8m0-bounded
    random pages."""
    running, scheduled = 8, 5
    record = nvalid = None
    for step, seed in enumerate((90, 91)):
        cache, table, seq, q = make_case(
            scheduled, S, seed=seed + S, max_len=128 * blocks, heads=heads
        )
        if step:  # the second step is shorter than the one that wrote the workspace
            seq = (seq // 3).clamp(min=S + 1).to(torch.int32)
        pt, ps, pq = _pad(table, seq, q, S, running, 0x7FFFFFFF)
        bound = table.shape[1]
        record, nvalid = (
            _buffers(running, S, bound, heads) if record is None else (record, nvalid)
        )
        md = _metadata(pt, ps, S, record, nvalid, heads)
        got = _select(pq, cache, pt, ps, S, md, heads)
        want = _select(pq, cache, pt, ps, S, md, heads, allocated=True)
        for g, w in zip(got, want):
            assert torch.equal(g, w), f"step {step}"
        _assert_pad_rows_are_empty(got, scheduled * S, heads)


def _prefill_case(batch, heads, seed, blocks, max_query):
    """Ragged chunked prefill: each request brings 1..max_query new tokens behind a
    prefix of its own (some with none), and a length that is not a multiple of 128."""
    cache, table, seq, _ = make_case(
        batch, max_query, seed=seed, max_len=128 * blocks, heads=heads
    )
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    qlen = torch.randint(1, max_query + 1, (batch,), device="cuda", generator=g)
    seq = torch.maximum(seq - 17, qlen + 1).to(torch.int32)  # off the block boundary
    seq[0] = 128 * blocks - 5
    prefix = (seq - qlen).to(torch.int32)
    cu = torch.cat([qlen.new_zeros(1), qlen.cumsum(0)]).to(torch.int32)
    q = torch.randn(int(cu[-1]), heads, DIM, device="cuda", generator=g) * 0.7
    return cache, table, seq, prefix, cu, q.to(torch.bfloat16), int(qlen.max())


def _prefill_select(case, heads, record, allocated):
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_prefill_metadata

    cache, table, seq, prefix, cu, q, max_q = case
    batch = seq.shape[0]
    md = make_sparse_prefill_metadata(
        cu_seqlens_q=cu,
        seq_lens=seq,
        block_table=table,
        slot_mapping=torch.zeros(q.shape[0], dtype=torch.int64, device="cuda"),
        max_query_len=max_q,
        max_seq_len=int(seq.max()),
        num_prefills=batch,
        num_prefill_tokens=q.shape[0],
        num_idx_heads=heads,
        index_scorer=record,
    )
    with _aiter_allocates_its_scores() if allocated else contextlib.nullcontext():
        return minimax_m3_index_topk(
            q, cache, table, cu, seq, prefix, max_q, int(seq.max()), TOPK, 0, 1,
            heads, SM_SCALE, emit_sparse_block_table=True, block_page_stride=1,
            pages_per_block=1,
            n_valid_column_per_row=md.n_valid_column_per_row,
            index_scorer=md.index_scorer,
        )  # fmt: skip


@pytest.mark.parametrize(
    "heads,batch,blocks,max_query",
    [
        (1, 4, 70, 40),
        (2, 7, 70, 130),
        (4, 3, 70, 300),
        (1, 3, 2100, 64),
        (2, 3, 2100, 200),
    ],
)
def test_the_prefill_scorer_into_the_unfilled_workspace_selects_like_aiters_own(
    heads, batch, blocks, max_query
):
    """The prefill scorer (ragged chunks, prefixes, lengths off a block boundary)
    writing into the junk-filled workspace, reused by a second chunk of other
    rows, selects bit for bit what aiter's own allocated -inf-filled scores give."""
    rows = batch * max_query
    record = _record(1, 1, blocks, heads, prefill_rows=rows)
    for seed in (30, 31):
        case = _prefill_case(batch, heads, seed, blocks, max_query)
        got = _prefill_select(case, heads, record, allocated=False)
        want = _prefill_select(case, heads, record, allocated=True)
        for g, w in zip(got, want):
            assert torch.equal(g, w), f"chunk {seed}"
        # the workspace is wide enough for any chunk, so a shorter second one reuses it
        batch, max_query = max(1, batch - 1), max(8, max_query // 2)


@pytest.mark.parametrize("S,heads", [(8, 1), (4, 4)])
def test_a_graph_replays_into_the_shared_workspace_like_aiters_own_scores(S, heads):
    """The workspace's pointer is baked into the graph: captured once with 8 requests,
    replayed after the builder's refill with 3 scheduled and 5 pad rows, the selection
    is what aiter's own scores give for that batch, and the workspace is the same
    storage before and after."""
    running, scheduled = 8, 3
    cache, table, seq, q = make_case(running, S, seed=95 + S, heads=heads)
    bound = table.shape[1]
    record, nvalid = _buffers(running, S, bound, heads)
    ptr = record.scores.logits(1, 1).data_ptr()
    static = [x.clone() for x in (table, seq, q)]

    def refill():
        return _metadata(static[0], static[1], S, record, nvalid, heads)

    def run(md):
        return _select(static[2], cache, static[0], static[1], S, md, heads)

    md = refill()
    run(md)  # JIT outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run(md)
    t, s, x = _pad(
        table[:scheduled], seq[:scheduled], q[: scheduled * S], S, running, 0x7FFFFFFF
    )
    for dst, src in zip(static, (t, s, x)):
        dst.copy_(src)
    refill()
    graph.replay()
    torch.cuda.synchronize()
    assert record.scores.logits(1, 1).data_ptr() == ptr
    rec, nv = _buffers(scheduled, S, bound, heads)
    md = _metadata(table[:scheduled], seq[:scheduled], S, rec, nv, heads)
    want = _select(q[: scheduled * S], cache, table[:scheduled], seq[:scheduled], S, md, heads, allocated=True)  # fmt: skip
    _assert_live_rows_match(got, want, scheduled * S, heads)
    _assert_pad_rows_are_empty(got, scheduled * S, heads)
