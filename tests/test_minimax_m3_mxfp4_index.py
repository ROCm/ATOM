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

from atom.model_ops.minimax_m3.kv_mxfp4 import PAGE_BYTES, dequantize_pages

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter.ops, "flydsl"),
    reason="needs gfx950 and aiter's FlyDSL",
)

TOPK, HEADS, DIM = 16, 1, 128
SM_SCALE = DIM**-0.5


def make_case(batch, S, seed, max_len=128 * 70, fp8_q=False):
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
    q = torch.randn(batch * S, HEADS, DIM, device="cuda", generator=g) * 0.7
    q = q.to(torch.float8_e4m3fn) if fp8_q else q.to(torch.bfloat16)
    return cache, table, seq.to(torch.int32), q


def reference(cache, table, seq, q, S):
    """scores [H, rows, blocks] in the natural-log domain (the scorers return them
    times log2(e); selection is unaffected), nan past the causal block, and the expected
    selection: set of block ids, and ctx, per row."""
    deq = dequantize_pages(cache)  # [pages, 128, 128]
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


@pytest.mark.parametrize("batch,S", [(1, 8), (5, 8), (17, 8), (6, 1)])
@pytest.mark.parametrize("fp8_q", [False, True])
def test_decode_selection_and_page128_block_table(batch, S, fp8_q):
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    cache, table, seq, q = make_case(batch, S, seed=batch * 10 + S, fp8_q=fp8_q)
    max_block = table.shape[1]
    buf = torch.empty(
        (index_work_map_capacity(32, max_block, S, HEADS), 2),
        dtype=torch.int32,
        device="cuda",
    )
    wmap = build_index_work_map(
        seq, max_block=max_block, max_query_len=S, num_idx_heads=HEADS, out=buf
    )
    topk_idx, sbt, sctx = minimax_m3_index_topk_decode(
        q, cache, table, seq, int(seq.max()), TOPK, 0, 1, HEADS, SM_SCALE,
        emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
        index_mxfp4=True, pages_per_block=1,
        index_score_work_map=wmap, index_score_max_block=max_block,
    )  # fmt: skip
    _, picks, ctx = reference(cache, table, seq, q, S)
    assert sbt.shape == (batch * S, TOPK) and sctx.shape == (batch * S,)
    for r in range(batch * S):
        got = {int(x) for x in topk_idx[0, r].tolist() if x >= 0}
        assert got == picks[r], f"row {r}"
        assert int(sctx[r]) == ctx[r], f"row {r} ctx"
        n = len(got)
        want_pages = {int(table[r // S, blk]) for blk in picks[r]}
        assert {int(x) for x in sbt[r, :n].tolist()} == want_pages
        # the tail block is last: it is the only one that can be partial
        tail = (int(seq[r // S]) - S + r % S) // 128
        assert int(sbt[r, n - 1]) == int(table[r // S, tail])


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


def test_a_captured_decode_follows_the_refilled_work_map():
    """The graph bakes the map's pointer and grid; a replay refills it in place."""
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    batch, S = 5, 8
    cache, table, seq, q = make_case(batch, S, seed=3)
    max_block = table.shape[1]
    buf = torch.empty(
        (index_work_map_capacity(32, max_block, S, HEADS), 2),
        dtype=torch.int32,
        device="cuda",
    )
    static_seq = seq.clone()
    wmap = build_index_work_map(
        static_seq, max_block=max_block, max_query_len=S, num_idx_heads=HEADS, out=buf
    )

    def run():
        return minimax_m3_index_topk_decode(
            q, cache, table, static_seq, max_block * 128, TOPK, 0, 1, HEADS, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            index_mxfp4=True, pages_per_block=1,
            index_score_work_map=wmap, index_score_max_block=max_block,
        )  # fmt: skip

    run()  # warm up: JIT compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    for new_seq in (seq, (seq // 2).clamp(min=S + 1).to(torch.int32)):
        static_seq.copy_(new_seq)
        build_index_work_map(
            static_seq, max_block=max_block, max_query_len=S, num_idx_heads=HEADS,
            out=buf,
        )  # fmt: skip
        graph.replay()
        torch.cuda.synchronize()
        _, picks, ctx = reference(cache, table, new_seq, q, S)
        for r in range(batch * S):
            got = {int(x) for x in out[0][0, r].tolist() if x >= 0}
            assert got == picks[r]
            assert int(out[2][r]) == ctx[r]


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
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        decode_score_mxfp4,
        fresh_index_score,
        index_score_buffer,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    batch, S = 6, 8
    cache, steps = _score_steps(batch, S, seed=21)
    max_block = steps[0][0].shape[1]
    wbuf = torch.empty(
        (index_work_map_capacity(32, max_block, S, HEADS), 2),
        dtype=torch.int32,
        device="cuda",
    )
    sbuf = index_score_buffer(32 * S, HEADS, max_block, "cuda")

    def run(table, seq, q, out):
        wmap = build_index_work_map(
            seq, max_block=max_block, max_query_len=S, num_idx_heads=HEADS, out=wbuf
        )
        score = decode_score_mxfp4(
            q, cache, table, seq, max_block, S, HEADS, SM_SCALE, wmap, out
        )
        sel = minimax_m3_index_topk_decode(
            q, cache, table, seq, max_block * 128, TOPK, 0, 1, HEADS, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            index_mxfp4=True, pages_per_block=1, index_score_work_map=wmap,
            index_score_max_block=max_block, index_score_out=out,
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


def test_a_captured_decode_follows_the_refilled_score_buffer():
    """The graph bakes the buffer's pointer; the refill between replays is outside it."""
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        build_index_work_map,
        fresh_index_score,
        index_score_buffer,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    batch, S = 5, 8
    cache, steps = _score_steps(batch, S, seed=33)
    table = steps[0][0]
    max_block = table.shape[1]
    wbuf = torch.empty(
        (index_work_map_capacity(32, max_block, S, HEADS), 2),
        dtype=torch.int32,
        device="cuda",
    )
    sbuf = index_score_buffer(32 * S, HEADS, max_block, "cuda")
    static_seq, static_q = steps[0][1].clone(), steps[0][2].clone()
    static_table = table.clone()
    rows = batch * S

    def prepare():
        out = fresh_index_score(sbuf, rows, HEADS, max_block, S)
        wmap = build_index_work_map(
            static_seq, max_block=max_block, max_query_len=S, num_idx_heads=HEADS,
            out=wbuf,
        )  # fmt: skip
        return out, wmap

    def run(out, wmap):
        return minimax_m3_index_topk_decode(
            static_q, cache, static_table, static_seq, max_block * 128, TOPK, 0, 1,
            HEADS, SM_SCALE, emit_sparse_block_table=True, max_query_len=S,
            block_page_stride=1, index_mxfp4=True, pages_per_block=1,
            index_score_work_map=wmap, index_score_max_block=max_block,
            index_score_out=out,
        )  # fmt: skip

    out, wmap = prepare()
    run(out, wmap)  # warm up: JIT compile outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = run(out, wmap)
    for t, s, q in (steps[0], steps[1]):  # the second is shorter, on permuted rows
        static_table.copy_(t)
        static_seq.copy_(s)
        static_q.copy_(q)
        prepare()  # the refill and the map, outside the graph
        graph.replay()
        torch.cuda.synchronize()
        want = minimax_m3_index_topk_decode(
            q, cache, t, s, max_block * 128, TOPK, 0, 1, HEADS, SM_SCALE,
            emit_sparse_block_table=True, max_query_len=S, block_page_stride=1,
            index_mxfp4=True, pages_per_block=1,
        )  # fmt: skip
        for g, w in zip(got, want):
            assert torch.equal(g, w)


def test_the_decode_metadata_refills_the_score_view_once():
    from atom.model_ops.minimax_m3.index_mxfp4 import (
        index_score_buffer,
        index_work_map_capacity,
    )
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    batch, S, bound = 3, 8, 64
    sbuf = index_score_buffer(8 * S, HEADS, bound, "cuda")
    sbuf.fill_(1.0)  # an earlier step's scores
    seq = torch.tensor([300, 1000, 5000], dtype=torch.int32, device="cuda")
    kw = {
        "seq_lens": seq,
        "block_table": torch.zeros((batch, bound), dtype=torch.int32, device="cuda"),
        "slot_mapping": torch.zeros(batch * S, dtype=torch.int64, device="cuda"),
        "max_seq_len": 5000,
        "num_idx_heads": HEADS,
        "max_query_len": S,
    }
    wmap = torch.empty(
        (index_work_map_capacity(8, bound, S, HEADS), 2),
        dtype=torch.int32,
        device="cuda",
    )
    md = make_sparse_decode_metadata(
        **kw, index_score_work_map_out=wmap, index_score_max_block=bound,
        index_score_buf=sbuf,
    )  # fmt: skip
    out = md.index_score_out
    assert tuple(out.shape) == (HEADS, batch * S, bound)
    assert out.data_ptr() == sbuf.data_ptr() and bool((out == float("-inf")).all())
    assert bool((sbuf[out.numel() :] == 1.0).all())  # nothing past this step's rows
    # another index cache: no map, so no score storage either
    assert make_sparse_decode_metadata(**kw).index_score_out is None
