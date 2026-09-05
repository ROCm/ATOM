# SPDX-License-Identifier: MIT
"""MiniMax-M3 lightning-indexer selection.

Split in two on purpose. The launch policies are plain Python, so CI runs them
on a box with no GPU and no aiter; the kernels need both, so they sit behind a
skip and only run where there is a device. Anything asserted about a kernel is
asserted against the op's definition rather than against another kernel -- a
comparison can only catch a defect the two implementations do not share, and
the second implementation was removed once it lost.
"""

from __future__ import annotations

import pytest
import torch

from atom.model_ops.minimax_m3.index_topk import (
    DECODE_SCORE_MAX_CHUNKS,
    DECODE_SCORE_TARGET_GRID,
    PREFILL_TOPK_MAX_BLOCK_SIZE_K,
    PREFILL_TOPK_MIN_BLOCK_SIZE_K,
    SPARSE_BLOCK_SIZE,
    _assert_packable,
    _decode_score_chunks,
    _prefill_topk_block_size_k,
)

TOPK = 16


class TestPrefillTileWidth:
    """clamp(next_pow2(max_block), MIN, MAX), and the two ends of the clamp."""

    @pytest.mark.parametrize(
        "max_block,want",
        [
            (1, PREFILL_TOPK_MIN_BLOCK_SIZE_K),
            (68, 128),
            (128, 128),
            (129, 256),
            (384, 512),
            (896, 1024),
            (2048, PREFILL_TOPK_MAX_BLOCK_SIZE_K),
            (100_000, PREFILL_TOPK_MAX_BLOCK_SIZE_K),
        ],
    )
    def test_width(self, max_block, want):
        assert _prefill_topk_block_size_k(max_block) == want

    def test_width_is_a_power_of_two(self):
        # The selector tiles with tl.arange and folds with tl.topk; both need it.
        for mb in range(1, 2000):
            w = _prefill_topk_block_size_k(mb)
            assert w & (w - 1) == 0

    def test_width_never_below_the_topk_it_must_hold(self):
        # tl.static_assert(BLOCK_SIZE_K >= BLOCK_SIZE_T) in the kernel.
        for mb in (1, 2, 15, 16, 17, 4096):
            assert _prefill_topk_block_size_k(mb) >= TOPK


class TestDecodeScoreChunks:
    """A grid dim, so: shape-constant, positive, and never past the blocks."""

    @pytest.mark.parametrize("batch", [1, 2, 7, 8, 50, 64, 256, 1024])
    @pytest.mark.parametrize("max_block", [1, 2, 63, 64, 256, 800, 8192])
    def test_bounds_and_coverage(self, batch, max_block):
        # Not a power of two: it is a grid dim, not a tile, so nothing indexes
        # it with tl.arange. What has to hold is that the chunks cover every
        # block and that none of them is empty by construction.
        n = _decode_score_chunks(batch, max_block)
        assert 1 <= n <= max_block
        assert n <= DECODE_SCORE_MAX_CHUNKS
        chunk_blocks = -(-max_block // n)
        assert chunk_blocks * n >= max_block
        assert chunk_blocks * (n - 1) < max_block

    def test_shrinks_with_batch(self):
        # The cap exists so a large batch does not multiply into a pointless
        # grid; monotonicity is what makes that statement true.
        prev = _decode_score_chunks(1, 4096)
        for batch in (2, 4, 16, 64, 256, 4096):
            cur = _decode_score_chunks(batch, 4096)
            assert cur <= prev
            prev = cur

    def test_grid_stays_near_the_target(self):
        # (request, chunk) is the grid, so batch * chunks is what it sizes.
        for batch in (1, 8, 50, 64):
            grid = batch * _decode_score_chunks(batch, 4096)
            assert grid <= DECODE_SCORE_TARGET_GRID * 2


class TestPackableBound:
    """The packed key spends its low 16 bits on a 1-based block id."""

    def test_accepts_what_fits(self):
        _assert_packable(0xFFFE)

    @pytest.mark.parametrize("max_block", [0xFFFF, 0x10000, 1 << 20])
    def test_rejects_what_does_not(self, max_block):
        with pytest.raises(AssertionError, match="packed top-k addresses"):
            _assert_packable(max_block)

    def test_the_bound_is_reachable_from_a_real_config(self):
        # 0xFFFE blocks is an 8.4M-token context; the assert is a guard rail,
        # not a limit anyone meets. Pin the arithmetic so a block-size change
        # that would bring it into range fails here first.
        assert 0xFFFE * SPARSE_BLOCK_SIZE > 8_000_000


# ---------------------------------------------------------------------------
# Kernel behaviour. Needs a GPU and triton.
# ---------------------------------------------------------------------------
gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the indexer kernels need a GPU"
)
HEAD_DIM, INIT, LOCAL = 128, 0, 1


def _inputs(qlens, prefixes, heads, device):
    """One (q_len, prefix_len) pair per request: ragged and chunked by default."""
    batch = len(qlens)
    seqs = [p + q for q, p in zip(qlens, prefixes)]
    nblk = max(-(-s // SPARSE_BLOCK_SIZE) for s in seqs)
    total_q = sum(qlens)
    g = torch.Generator(device=device).manual_seed(total_q)
    i32 = {"dtype": torch.int32, "device": device}

    def randn(*shape):
        return (torch.randn(*shape, generator=g, device=device) * 0.3).to(
            torch.bfloat16
        )

    return {
        "idx_q": randn(total_q, heads, HEAD_DIM),
        "index_kv_cache": randn(batch * nblk, SPARSE_BLOCK_SIZE, HEAD_DIM),
        "block_table": torch.arange(batch * nblk, **i32).view(batch, nblk),
        "cu_seqlens_q": torch.tensor(
            [0, *torch.tensor(qlens).cumsum(0).tolist()], **i32
        ),
        "seq_lens": torch.tensor(seqs, **i32),
        "prefix_lens": torch.tensor(prefixes, **i32),
        "max_query_len": max(qlens),
        "max_seq_len": max(seqs),
        "topk": TOPK,
        "init_blocks": INIT,
        "local_blocks": LOCAL,
        "num_kv_heads": heads,
        "sm_scale": HEAD_DIM**-0.5,
    }


@gpu
class TestForcedSelection:
    """A row with no more candidates than topk has only one right answer.

    Every block is in, so the emitted set must be exactly {0 .. valid-1} and
    sparse_ctx must be the row's causal length. That is checkable without a
    reference implementation, and it is the regime a short-prompt workload
    spends all of its time in.
    """

    @pytest.mark.parametrize(
        "qlens,prefixes",
        [
            ([1147] * 4, [0] * 4),  # whole prompt, uniform
            ([1100, 950, 1300, 890], [0] * 4),  # whole prompt, ragged
            ([600] * 4, [547] * 4),  # a middle chunk
            ([500, 600, 400, 700], [647, 580, 550, 600]),  # ragged and chunked
            ([128] * 4, [1019] * 4),  # the tail chunk
            ([1] * 4, [1146] * 4),  # a one-token chunk
        ],
    )
    def test_prefill(self, qlens, prefixes):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk

        kw = _inputs(qlens, prefixes, 1, "cuda")
        idx, _, sctx = minimax_m3_index_topk(**kw, emit_sparse_block_table=True)
        causal = torch.cat(
            [torch.arange(p + 1, p + n + 1) for n, p in zip(qlens, prefixes)]
        )
        self._check(idx, sctx, causal)

    @pytest.mark.parametrize("ctx", [1147, 1677, 2048])
    @pytest.mark.parametrize("q_per_req", [1, 4])
    def test_decode(self, ctx, q_per_req):
        from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

        batch, heads = 4, 1
        kw = _inputs([q_per_req] * batch, [ctx - q_per_req] * batch, heads, "cuda")
        idx, _, sctx = minimax_m3_index_topk_decode(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            ctx, TOPK, INIT, LOCAL, heads, kw["sm_scale"],
            emit_sparse_block_table=True, max_query_len=q_per_req,
        )  # fmt: skip
        causal = (
            torch.full((batch,), ctx - q_per_req).repeat_interleave(q_per_req)
            + torch.arange(q_per_req).repeat(batch) + 1
        )  # fmt: skip
        self._check(idx, sctx, causal)

    @staticmethod
    def _check(idx, sctx, causal):
        idx = idx.reshape(-1, idx.shape[-1]).cpu()
        sctx = sctx.reshape(-1).cpu()
        valid = torch.div(
            causal + SPARSE_BLOCK_SIZE - 1, SPARSE_BLOCK_SIZE, rounding_mode="floor"
        )
        forced = valid <= TOPK
        assert forced.any(), "the shape under test left the forced regime"
        for r in forced.nonzero().flatten().tolist():
            got = {int(v) for v in idx[r] if v >= 0}
            assert got == set(range(int(valid[r]))), f"row {r}"
            assert int(sctx[r]) == int(causal[r]), f"row {r} ctx"


@gpu
def test_emitted_order_is_full_blocks_by_score_then_the_partial_tail():
    """sparse_bt order is the attention's accumulation order, so it is contract.

    Checked against the score the kernel itself produced, not against a second
    selector: the tail block is the only one that can be partial, so it has to
    land last however it scored.
    """
    import triton

    from atom.model_ops.minimax_m3 import index_topk as m

    qlens, prefixes, heads = [600] * 4, [547] * 4, 1
    kw = _inputs(qlens, prefixes, heads, "cuda")
    total_q, max_block = sum(qlens), triton.cdiv(kw["max_seq_len"], SPARSE_BLOCK_SIZE)
    score = torch.empty((heads, total_q, max_block), dtype=torch.float32, device="cuda")
    q_tiles = triton.cdiv(kw["max_query_len"], m.SCORE_BLOCK_SIZE_Q)
    cb = m._score_chunk_blocks(max_block, q_tiles, len(qlens), heads, torch.device("cuda"))  # fmt: skip
    m._index_block_score_kernel[(q_tiles, len(qlens) * heads, triton.cdiv(max_block, cb))](
        kw["idx_q"], kw["index_kv_cache"], score, kw["block_table"], kw["cu_seqlens_q"],
        kw["seq_lens"], kw["prefix_lens"], heads, HEAD_DIM, kw["sm_scale"], cb,
        *kw["idx_q"].stride(), *kw["index_kv_cache"].stride(), *score.stride(),
        kw["block_table"].stride(0), BLOCK_SIZE_Q=m.SCORE_BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE, num_stages=m.SCORE_NUM_STAGES,
    )  # fmt: skip
    idx, sbt, _ = m.minimax_m3_index_topk(**kw, emit_sparse_block_table=True)

    s, emitted, sbt = score[0].cpu(), idx[0].cpu(), sbt.cpu()
    causal = torch.cat(
        [torch.arange(p + 1, p + n + 1) for n, p in zip(qlens, prefixes)]
    )
    req = torch.cat([torch.full((n,), b) for b, n in enumerate(qlens)])
    bt = kw["block_table"].cpu()
    for r in range(0, total_q, 97):  # every row is the same assertion; sample
        valid = -(-int(causal[r]) // SPARSE_BLOCK_SIZE)
        if valid > TOPK:
            continue
        tail = (int(causal[r]) - 1) // SPARSE_BLOCK_SIZE
        want = sorted(
            (b for b in range(valid) if b != tail), key=lambda b: -float(s[r, b])
        ) + [tail]
        pages = m.PAGES_PER_SPARSE_BLOCK
        expect = [
            int(bt[int(req[r]), b]) * pages * heads + pj * heads
            for b in want
            for pj in range(pages)
        ]
        expect += [0] * (TOPK * pages - len(expect))
        assert [int(v) for v in sbt[r]] == expect, f"row {r}"
        assert {int(v) for v in emitted[r] if v >= 0} == set(want)
