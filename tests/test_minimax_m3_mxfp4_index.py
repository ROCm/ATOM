# SPDX-License-Identifier: MIT
"""MiniMax-M3 index selection over the MXFP4 index cache (gfx950). The scorers are
aiter's; pinned here is the ATOM side: selection and the page-128 block table against
the op's definition, and the persistent work map / shared score workspace under
padded eager steps, graph replay and prefill chunks."""

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
TOPK, DIM, POISON = 16, 128, 0x7FFFFFFF  # POISON: a page id no cache has
KW = {"device": "cuda"}


def make_case(batch, S, seed, blocks=70, fp8_q=False, heads=1):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def ri(lo, hi, *shape):
        return torch.randint(lo, hi, shape, generator=g, **KW)

    seq, pages = ri(S + 1, 128 * blocks, batch), batch * blocks + 3
    seq[0] = 128 * blocks
    cache = ri(0, 256, pages, PAGE_BYTES).to(torch.uint8)
    cache[:, 8192:] = ri(125, 130, pages, 512).to(torch.uint8)  # scales near unity
    table = [torch.randperm(pages, generator=g, **KW)[:blocks] for _ in range(batch)]
    q = torch.randn(batch * S, heads, DIM, generator=g, **KW) * 0.7
    q = q.to(torch.float8_e4m3fn if fp8_q else torch.bfloat16)
    return cache, torch.stack(table).int(), seq.int(), q


def reference(cache, table, seq, q, S):
    """Expected selection (block ids) and context per row: top-k by max q.k, tail pinned."""
    deq, picks, ctx = dequantize_mxfp4(*page_views(cache)), [], []
    for r in range(table.shape[0] * S):
        b, causal = r // S, int(seq[r // S]) - S + r % S + 1
        last = (causal - 1) // 128
        sc = [
            (deq[table[b, p]][: causal - p * 128] @ q[r].float().T).amax()
            for p in range(last)
        ]
        top = torch.stack(sc).topk(min(TOPK - 1, last)).indices.tolist() if last else []
        picks.append(set(top) | {last})
        ctx.append(len(top) * 128 + causal - last * 128)
    return picks, ctx


def _buffers(running, S, bound, heads=1, prefill_rows=0):
    """The builder's scorer record (work-map capacity + the shared score workspace,
    filled with values no selector may read) and the live-column buffer."""
    from aiter.ops.flydsl import minimax_m3_index_score_work_map_capacity as cap

    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

    n = heads * max(running * S, prefill_rows) * bound
    ws = LogitsWorkspace(n, "cuda")
    junk = torch.tensor([float("nan"), float("inf"), -float("inf"), 3e38], **KW)
    ws.logits(1, n).copy_(junk[torch.arange(n, **KW) % 4])
    wmap = torch.empty((cap(running, bound, S, heads), 2), dtype=torch.int32, **KW)
    return IndexScoreBuffers(ws, wmap, bound), torch.empty(
        heads * running * S, dtype=torch.int32, **KW
    )


@contextlib.contextmanager
def _aiter_allocates_its_scores(on=True):
    """Scorers as before the workspace: no `out`, aiter allocates and -inf fills."""
    from atom.model_ops.minimax_m3 import index_topk

    ops = index_topk.flydsl_ops()
    names = (
        "minimax_m3_index_score_fp4_flydsl",
        "minimax_m3_index_score_prefill_fp4_flydsl",
    )
    real = {n: getattr(ops, n) for n in names}
    try:
        for n in names if on else ():
            setattr(ops, n, _without_out(real[n]))
        yield
    finally:
        for n, f in real.items():
            setattr(ops, n, f)


def _without_out(f):
    return lambda *a, **k: f(*a, **{**k, "out": None}).contiguous()


def _metadata(table, seq, S, record, nvalid, heads):
    """What the builder refills each step, outside any graph."""
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    return make_sparse_decode_metadata(
        seq_lens=seq, block_table=table, max_seq_len=table.shape[1] * 128,
        slot_mapping=torch.zeros(len(seq) * S, dtype=torch.int64, **KW), num_idx_heads=heads,
        max_query_len=S, n_valid_column_per_row_out=nvalid, index_scorer=record,
    )  # fmt: skip


def _run(
    q, cache, table, seq, S, heads=1, buffers=None, allocated=False, ppb=1, md=None
):
    """One decode step as the attention layer runs it (buffers sized to it by default)."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    if md is None:  # a graph capture passes its own: nothing may be allocated inside it
        buffers = buffers or _buffers(len(seq), S, table.shape[1], heads)
        md = _metadata(table, seq, S, *buffers, heads)
    with _aiter_allocates_its_scores(allocated):
        return minimax_m3_index_topk_decode(
            q, cache, table, seq, table.shape[1] * 128, TOPK, 0, 1, heads, DIM**-0.5,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=ppb,
            pages_per_block=ppb, n_valid_column_per_row=md.n_valid_column_per_row,
            index_scorer=md.index_scorer,
        )  # fmt: skip


def _check(got, live, heads, want=None):
    """Live rows equal `want`; a pad row has context 0 and page 0, whatever its table
    holds. Rows are token-major, head-minor."""
    live_rows = (live * heads,) * 2
    if want is not None:
        assert torch.equal(got[0][:, :live], want[0][:, :live])
        assert all(
            torch.equal(g[:n], w[:n]) for g, w, n in zip(got[1:], want[1:], live_rows)
        )
    assert all(int(g[n:].abs().max()) == 0 for g, n in zip(got[1:], live_rows))


# ppb 16-pages a block: 1 for the MXFP4 KV, 8 for the SHUFFLE KV of an index-only run
@pytest.mark.parametrize("S,ppb,fp8_q", [(8, 1, False), (8, 1, True), (8, 8, True)])
def test_decode_selection_and_block_table(S, ppb, fp8_q):
    cache, table, seq, q = make_case(5, S, seed=S + ppb, fp8_q=fp8_q)
    ids, sbt, sctx = _run(q, cache, table, seq, S, ppb=ppb)
    picks, ctx = reference(cache, table, seq, q, S)
    for r in range(5 * S):
        got = {int(x) for x in ids[0, r].tolist() if x >= 0}
        assert got == picks[r] and int(sctx[r]) == ctx[r], f"row {r}"
        row, n = table[r // S].tolist(), len(got)
        assert {int(x) for x in sbt[r, : n * ppb]} == {
            row[b] * ppb + j for b in picks[r] for j in range(ppb)
        }
        tail = (
            int(seq[r // S]) - S + r % S
        ) // 128  # the only block that can be partial
        assert sbt[r, (n - 1) * ppb : n * ppb].tolist() == [
            row[tail] * ppb + j for j in range(ppb)
        ]


# S*heads >= 16 is where aiter's own scorer lays scores out transposed; 2100 blocks is
# wide enough for aiter's selector, which needs them contiguous
@pytest.mark.parametrize(
    "graph,S,heads,scheduled,blocks",
    [
        (False, 8, 1, 3, 70),
        (False, 4, 4, 3, 70),
        (False, 8, 2, 3, 2100),
        (False, 1, 1, 0, 70),
        (True, 8, 2, 3, 70),
    ],
)
def test_a_padded_step_into_the_unfilled_workspace_selects_like_the_scheduled_batch(
    graph, S, heads, scheduled, blocks
):
    """8 requests run, `scheduled` real (0: an idle DP rank), eager or a graph captured
    and replayed after the builder's refill: the real rows select bit for bit what a
    batch of their own gives with aiter's own -inf-filled scores, pad rows are empty
    whatever their table holds, and the junk-filled workspace is the same storage."""
    cache, table, seq, q = make_case(8, S, seed=70 + S, blocks=blocks, heads=heads)
    t, s, x = table[:scheduled], seq[:scheduled], q[: scheduled * S]
    # pad rows: length 0, a block-table row of POISON, zero activations
    pad = [torch.full((8, blocks), POISON, dtype=torch.int32, **KW), torch.zeros(8, dtype=torch.int32, **KW),
           torch.zeros((8 * S, heads, DIM), dtype=q.dtype, **KW)]  # fmt: skip
    for dst, src in zip(pad, (t, s, x)):
        dst[: len(src)] = src
    buffers = _buffers(8, S, blocks, heads)
    ptr = buffers[0].scores.logits(1, 1).data_ptr()
    if not graph:
        got = _run(pad[2], cache, pad[0], pad[1], S, heads, buffers)
    else:
        st = [v.clone() for v in (table, seq, q)]
        md = _metadata(st[0], st[1], S, *buffers, heads)
        _run(st[2], cache, st[0], st[1], S, heads, md=md)  # JIT outside the capture
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            got = _run(st[2], cache, st[0], st[1], S, heads, md=md)
        for dst, src in zip(st, pad):
            dst.copy_(src)
        _metadata(st[0], st[1], S, *buffers, heads)  # the builder's refill
        g.replay()
        torch.cuda.synchronize()
    assert buffers[0].scores.logits(1, 1).data_ptr() == ptr
    want = _run(x, cache, t, s, S, heads, allocated=True) if scheduled else None
    _check(got, scheduled * S, heads, want)


@pytest.mark.parametrize("heads,blocks,max_query", [(1, 70, 40), (2, 2100, 200)])
def test_the_prefill_scorer_into_the_unfilled_workspace_selects_like_aiters_own(
    heads, blocks, max_query
):
    """Ragged chunked prefill (a prefix per request, lengths off a block boundary): the
    scorer writing into the junk-filled workspace, reused by a second smaller chunk,
    selects bit for bit what aiter's own allocated scores give."""
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_prefill_metadata

    batch, record = 3, _buffers(1, 1, blocks, heads, prefill_rows=3 * max_query)[0]
    for seed in (30, 31):
        cache, table, seq, _ = make_case(batch, max_query, seed, blocks, heads=heads)
        g = torch.Generator(device="cuda").manual_seed(seed)
        qlen = torch.randint(1, max_query + 1, (batch,), generator=g, **KW)
        seq = torch.maximum(seq - 17, qlen + 1).int()
        cu = torch.cat([qlen.new_zeros(1), qlen.cumsum(0)]).int()
        q = (torch.randn(int(cu[-1]), heads, DIM, generator=g, **KW) * 0.7).bfloat16()
        md = make_sparse_prefill_metadata(
            cu_seqlens_q=cu, seq_lens=seq, block_table=table, max_query_len=int(qlen.max()),
            slot_mapping=torch.zeros(len(q), dtype=torch.int64, **KW), max_seq_len=int(seq.max()),
            num_prefills=batch, num_prefill_tokens=len(q), num_idx_heads=heads, index_scorer=record,
        )  # fmt: skip
        out = []
        for allocated in (False, True):
            with _aiter_allocates_its_scores(allocated):
                out.append(minimax_m3_index_topk(
                    q, cache, table, cu, seq, (seq - qlen).int(), int(qlen.max()), int(seq.max()),
                    TOPK, 0, 1, heads, DIM**-0.5, emit_sparse_block_table=True,
                    block_page_stride=1, pages_per_block=1,
                    n_valid_column_per_row=md.n_valid_column_per_row, index_scorer=md.index_scorer,
                ))  # fmt: skip
        assert all(torch.equal(a, b) for a, b in zip(*out)), f"chunk {seed}"
        max_query //= 2
