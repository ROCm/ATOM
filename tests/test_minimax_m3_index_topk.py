# SPDX-License-Identifier: MIT
"""MiniMax-M3 lightning-indexer selection.

Split in two: the launch policies are plain Python and run wherever triton is
installed, the kernels also need a device and sit behind `gpu` below. Anything
asserted about a kernel is asserted against the op's definition rather than
against another kernel -- a comparison can only catch a defect the two
implementations do not share, and the second implementation was removed once
it lost.

One test breaks that rule on purpose:
`test_the_two_selectors_agree_where_the_dispatch_switches`. Two selectors are
live again, chosen by row width, and the dispatch's whole claim is that the
cheaper one computes the same answer -- so agreement between them is the
property, not a stand-in for one.

The whole file needs triton, because the module under test defines
`@triton.jit` kernels and a decorator runs at import. CI has no triton, so
nothing here runs there; a skip rather than a collection error, which would
abort the run for every other test as well.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton", reason="index_topk defines @triton.jit kernels")

from atom.model_ops.minimax_m3 import index_topk as m
from atom.model_ops.minimax_m3.index_topk import (
    DECODE_SCORE_MIN_BLOCKS,
    DECODE_SCORE_TARGET_GRID,
    PREFILL_TOPK_MAX_BLOCK_SIZE_K,
    PREFILL_TOPK_MIN_BLOCK_SIZE_K,
    SPARSE_BLOCK_SIZE,
    _decode_score_chunks,
    _prefill_topk_block_size_k,
    _require_packable,
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
        # Two bounds, one per end of the batch range.
        assert n <= max(1, -(-max_block // DECODE_SCORE_MIN_BLOCKS))
        assert batch * n <= max(DECODE_SCORE_TARGET_GRID, batch)
        chunk_blocks = -(-max_block // n)
        assert chunk_blocks * n >= max_block
        assert chunk_blocks * (n - 1) < max_block

    def test_the_ceiling_binds_at_high_batch(self):
        """The floor alone would return the same count at every batch.

        Asserted against a literal rather than DECODE_SCORE_TARGET_GRID: a bound
        read from the constant moves with it, so raising the constant back out
        of range would satisfy the assertion instead of failing it.
        """
        assert _decode_score_chunks(8, 8192) > _decode_score_chunks(128, 8192)
        assert _decode_score_chunks(128, 8192) * 128 <= 16384

    @pytest.mark.parametrize("batch", [1, 64])
    def test_an_empty_bound_still_gives_a_grid(self, batch):
        """`cdiv(max_block, cdiv(max_block, chunks))` divides by its own inner
        result, which is zero when there is nothing to score. The exception
        lands outside any kernel launch, so on tp>1 one rank raises and the
        rest wait in the next collective -- the shape a hang takes."""
        assert _decode_score_chunks(batch, 0) == 1

    def test_shrinks_with_batch(self):
        # A larger batch must not buy more chunks per request: the grid is
        # (request, chunk), so that would multiply into a pointless grid.
        prev = _decode_score_chunks(1, 4096)
        for batch in (2, 4, 16, 64, 256, 4096):
            cur = _decode_score_chunks(batch, 4096)
            assert cur <= prev
            prev = cur

    @pytest.mark.parametrize("max_block", [3, 5, 64, 800, 2464, 8192, 65534])
    @pytest.mark.parametrize("batch", [1, 8, 64])
    def test_a_chunk_walks_at_least_min_blocks(self, batch, max_block):
        """The floor is the whole point of the split rule: a chunk down to one
        block pays the query-tile load, which sits outside the block loop, for
        a single block of work. Deleting the clamp must turn this red."""
        n = _decode_score_chunks(batch, max_block)
        assert -(-max_block // n) >= DECODE_SCORE_MIN_BLOCKS

    @pytest.mark.parametrize("max_block,want_chunks", [(1, 1), (2, 1), (4, 2)])
    def test_the_floor_is_a_target_not_a_guarantee(self, max_block, want_chunks):
        """Right above MIN_BLOCKS the floor cannot hand out a second full
        chunk: max_block=4 splits into 2 chunks of 2, under the floor. Nothing
        is wrong with that -- 2 blocks still amortize the query tile -- but the
        floor is a target, so `test_a_chunk_walks_at_least_min_blocks` skips
        this range rather than asserting something untrue about it."""
        assert _decode_score_chunks(1, max_block) == want_chunks


class TestPackableBound:
    """The packed key spends its low 16 bits on a 1-based block id."""

    def test_accepts_what_fits(self):
        _require_packable(0xFFFE)

    @pytest.mark.parametrize("max_block", [0xFFFF, 0x10000, 1 << 20])
    def test_rejects_what_does_not(self, max_block):
        # ValueError and not AssertionError: `max_block` comes from the launch
        # flags, so `python -O` must not be able to turn this into a wrap.
        with pytest.raises(ValueError, match="packed top-k addresses"):
            _require_packable(max_block)

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


@gpu
class TestMetadataRowBounds:
    """`MiniMaxM3SparseMetadata.n_valid_column_per_row`, and what it selects.

    The counts are `ceil(causal_len / SPARSE_BLOCK_SIZE)` repeated per index
    head; the selection they enable still owes the forced-regime answer.
    """

    @staticmethod
    def _want(causal, heads):
        valid = torch.div(
            causal + SPARSE_BLOCK_SIZE - 1, SPARSE_BLOCK_SIZE, rounding_mode="floor"
        )
        return valid.repeat(heads).to(torch.int32)

    @staticmethod
    def _dummy_slot_mapping():
        return torch.zeros(1, dtype=torch.int64, device="cuda")

    @pytest.mark.parametrize(
        "qlens,prefixes",
        [
            ([1147] * 4, [0] * 4),
            ([500, 600, 400, 700], [647, 580, 550, 600]),
        ],
    )
    @pytest.mark.parametrize("heads", [1, 2])
    def test_prefill_counts(self, qlens, prefixes, heads):
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_prefill_metadata

        kw = _inputs(qlens, prefixes, heads, "cuda")
        md = make_sparse_prefill_metadata(
            cu_seqlens_q=kw["cu_seqlens_q"],
            seq_lens=kw["seq_lens"],
            block_table=kw["block_table"],
            slot_mapping=self._dummy_slot_mapping(),
            max_query_len=kw["max_query_len"],
            max_seq_len=kw["max_seq_len"],
            num_prefills=len(qlens),
            num_prefill_tokens=sum(qlens),
            num_idx_heads=heads,
        )
        causal = torch.cat(
            [torch.arange(p + 1, p + n + 1) for n, p in zip(qlens, prefixes)]
        )
        assert torch.equal(md.n_valid_column_per_row.cpu(), self._want(causal, heads))

    @pytest.mark.parametrize("q_per_req", [1, 4])
    @pytest.mark.parametrize("heads", [1, 2])
    def test_decode_counts(self, q_per_req, heads):
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

        batch, ctx = 4, 1677
        kw = _inputs([q_per_req] * batch, [ctx - q_per_req] * batch, heads, "cuda")
        md = make_sparse_decode_metadata(
            seq_lens=kw["seq_lens"],
            block_table=kw["block_table"],
            slot_mapping=self._dummy_slot_mapping(),
            max_seq_len=ctx,
            max_query_len=q_per_req,
            num_idx_heads=heads,
        )
        causal = (
            torch.full((batch,), ctx - q_per_req).repeat_interleave(q_per_req)
            + torch.arange(q_per_req).repeat(batch) + 1
        )  # fmt: skip
        assert torch.equal(md.n_valid_column_per_row.cpu(), self._want(causal, heads))

    @pytest.mark.parametrize("q_per_req", [1, 4])
    def test_narrow_rows_still_answer_with_the_bounds_published(self, q_per_req):
        """Publishing the bounds must not change the answer where they are unused.

        14 columns is under `_AITER_MIN_WIDTH_WITH_EMIT`, so Triton serves it
        with a tensor in hand that it never reads -- the common decode shape.
        """
        from atom.model_ops.minimax_m3 import index_topk as m
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

        batch, heads, ctx = 4, 1, 1677
        kw = _inputs([q_per_req] * batch, [ctx - q_per_req] * batch, heads, "cuda")
        md = make_sparse_decode_metadata(
            seq_lens=kw["seq_lens"],
            block_table=kw["block_table"],
            slot_mapping=self._dummy_slot_mapping(),
            max_seq_len=ctx,
            max_query_len=q_per_req,
            num_idx_heads=heads,
        )
        idx, _, sctx = m.minimax_m3_index_topk_decode(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            ctx, TOPK, INIT, LOCAL, heads, kw["sm_scale"],
            emit_sparse_block_table=True, max_query_len=q_per_req,
            n_valid_column_per_row=md.n_valid_column_per_row,
        )  # fmt: skip
        causal = (
            torch.full((batch,), ctx - q_per_req).repeat_interleave(q_per_req)
            + torch.arange(q_per_req).repeat(batch) + 1
        )  # fmt: skip
        TestForcedSelection._check(idx, sctx, causal)

    def test_empty_batch_publishes_nothing(self):
        from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

        i32 = {"dtype": torch.int32, "device": "cuda"}
        md = make_sparse_decode_metadata(
            seq_lens=torch.empty(0, **i32),
            block_table=torch.empty((0, 4), **i32),
            slot_mapping=self._dummy_slot_mapping(),
            max_seq_len=0,
            num_idx_heads=2,
        )
        assert md.n_valid_column_per_row is None


@gpu
class TestPerForwardHoist:
    """`n_valid_column_per_row_for_forward`, the plugins' stand-in for the field.

    One build per forward, no crosstalk between a hybrid batch's two phases,
    and -- the load-bearing one -- a new owner builds fresh, since that is what
    caching with no content key rests on.
    """

    class _Owner:
        """Stands in for a framework's per-forward metadata / batch object."""

    @staticmethod
    def _call(owner, phase, batch, total_q, heads, decode_max_q):
        from atom.model_ops.minimax_m3.index_topk import (
            n_valid_column_per_row_for_forward,
        )

        i32 = {"dtype": torch.int32, "device": "cuda"}
        if decode_max_q:
            lens = torch.full((batch,), 1677, **i32)
            starts = prefix = lens
        else:
            q = total_q // batch
            starts = torch.arange(0, total_q + 1, q, **i32)
            prefix = torch.zeros(batch, **i32)
        return n_valid_column_per_row_for_forward(
            owner,
            phase,
            starts,
            prefix,
            batch=batch,
            total_q=total_q,
            num_idx_heads=heads,
            decode_max_q=decode_max_q,
        )

    def test_second_layer_reads_the_first_layer_s_tensor(self):
        owner = self._Owner()
        first = self._call(owner, "decode", 4, 4, 2, 1)
        assert self._call(owner, "decode", 4, 4, 2, 1) is first

    def test_the_two_phases_of_one_batch_do_not_share(self):
        owner = self._Owner()
        decode = self._call(owner, "decode", 4, 4, 2, 1)
        prefill = self._call(owner, "prefill", 4, 512, 2, 0)
        assert decode is not prefill
        assert self._call(owner, "decode", 4, 4, 2, 1) is decode

    def test_a_new_forward_builds_new(self):
        first = self._call(self._Owner(), "decode", 4, 4, 2, 1)
        assert self._call(self._Owner(), "decode", 4, 4, 2, 1) is not first

    def test_an_owner_that_refuses_attributes_still_answers(self):
        class Slotted:
            __slots__ = ()

        owner = Slotted()
        got = self._call(owner, "decode", 4, 4, 2, 1)
        # Correct, just rebuilt per layer: 1677 keys is 14 blocks of 128.
        assert torch.equal(
            got.cpu(),
            torch.full((8,), -(-1677 // SPARSE_BLOCK_SIZE), dtype=torch.int32),
        )
        assert self._call(owner, "decode", 4, 4, 2, 1) is not got


class TestSelectorDispatch:
    """`_aiter_selector_wins`: which selector a row width is cheaper on.

    The thresholds are a fit to measured device time (the table beside them);
    what is asserted is only that it is applied as stated -- monotone in width,
    stricter when the call also emits.
    """

    def test_narrow_rows_stay_on_triton(self):
        for width in (1, 64, 512, m._AITER_MIN_WIDTH - 1):
            assert not m._aiter_selector_wins(width, emit=False)
            assert not m._aiter_selector_wins(width, emit=True)

    def test_emission_raises_the_bar(self):
        between = m._AITER_MIN_WIDTH
        assert between < m._AITER_MIN_WIDTH_WITH_EMIT
        assert m._aiter_selector_wins(between, emit=False)
        assert not m._aiter_selector_wins(between, emit=True)

    def test_wide_rows_take_aiter(self):
        for width in (m._AITER_MIN_WIDTH_WITH_EMIT, 8192):
            assert m._aiter_selector_wins(width, emit=False)
            assert m._aiter_selector_wins(width, emit=True)


@gpu
def test_the_two_selectors_agree_where_the_dispatch_switches():
    """Above the width threshold both paths must return the same selection.

    The dispatch claims only that aiter computes the same answer more cheaply,
    so agreement is the property. 262144 tokens is 2048 columns, the first
    width `_aiter_selector_wins` takes with emission on.
    """
    from atom.model_ops.minimax_m3 import index_topk as m2
    from atom.model_ops.minimax_m3.sparse_attn import make_sparse_decode_metadata

    if m2.topk_per_row_small_k is None:
        pytest.skip("aiter's small-k selector is not installed")
    batch, heads, ctx = 2, 1, m2._AITER_MIN_WIDTH_WITH_EMIT * SPARSE_BLOCK_SIZE
    kw = _inputs([1] * batch, [ctx - 1] * batch, heads, "cuda")
    md = make_sparse_decode_metadata(
        seq_lens=kw["seq_lens"],
        block_table=kw["block_table"],
        slot_mapping=torch.zeros(1, dtype=torch.int64, device="cuda"),
        max_seq_len=ctx,
        max_query_len=1,
        num_idx_heads=heads,
    )

    def run(bounds):
        return m2.minimax_m3_index_topk_decode(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            ctx, TOPK, INIT, LOCAL, heads, kw["sm_scale"],
            emit_sparse_block_table=True, max_query_len=1,
            n_valid_column_per_row=bounds,
        )  # fmt: skip

    ait_idx, ait_bt, ait_ctx = run(md.n_valid_column_per_row)
    tri_idx, tri_bt, tri_ctx = run(None)
    rows = heads * batch
    assert torch.equal(
        torch.sort(ait_idx.reshape(rows, TOPK), dim=1).values,
        torch.sort(tri_idx.reshape(rows, TOPK), dim=1).values,
    )
    assert torch.equal(ait_ctx, tri_ctx)
    assert torch.equal(ait_bt, tri_bt)
