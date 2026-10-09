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
    PREFILL_TOPK_MAX_BLOCK_SIZE_K,
    PREFILL_TOPK_MIN_BLOCK_SIZE_K,
    SPARSE_BLOCK_SIZE,
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


class TestTheAiterDependencyIsDeferred:
    """A missing decode kernel must not break IMPORT, only use.

    The FlyDSL M3 index-score kernels are newer than the aiter ATOM's image
    builds (`AITER_COMMIT: HEAD` -> aiter main), so on an aiter that predates
    them these symbols are simply absent. That has to stay contained: this
    module is imported at module scope by `sparse_attn`, which is imported at
    module scope by `atom/plugin/vllm/attention/backend.py` for
    `SPARSE_BLOCK_SIZE` -- and that module carries the MLA prefill backends.
    A module-scope `from aiter.ops.flydsl import minimax_m3_*` would therefore
    take down far more than MiniMax-M3.

    Decode still has no fallback, so using it must fail loudly; these two tests
    pin both halves of that.
    """

    def test_the_symbols_are_not_imported_at_module_scope(self):
        import inspect

        body = inspect.getsource(m)
        offenders = [
            line
            for line in body.splitlines()
            if line.startswith(("from aiter", "import aiter"))
            and "minimax_m3_index_score" in line
        ]
        assert not offenders, f"module-scope aiter M3 import reintroduced: {offenders}"

    def test_using_it_without_the_kernel_names_the_dependency(self, monkeypatch):
        import sys
        import types

        real = sys.modules["aiter.ops.flydsl"]
        stub = types.ModuleType("aiter.ops.flydsl")
        # `__getattr__` is excluded deliberately, not incidentally: aiter's
        # flydsl package resolves its kernels through a module-level lazy
        # loader, so copying it would hand the stub the very symbols this test
        # is pretending do not exist.
        stub.__dict__.update(
            {
                k: v
                for k, v in real.__dict__.items()
                if "minimax" not in k.lower() and k not in ("__getattr__", "__dir__")
            }
        )
        # Both, and the attribute is the one that matters: the resolver says
        # `from aiter.ops import flydsl`, which reads the attribute off the
        # parent package rather than going through sys.modules.
        monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", stub)
        monkeypatch.setattr("aiter.ops.flydsl", stub)
        m._flydsl.cache_clear()
        m.index_score_config.cache_clear()
        try:
            with pytest.raises(RuntimeError, match="aiter#5626"):
                m.index_score_config()
        finally:
            # Both caches memoize across tests; a stubbed entry left behind
            # would fail every later GPU test in this file.
            m._flydsl.cache_clear()
            m.index_score_config.cache_clear()


# The shape the non-monotonicity shows up at: at max_block 256 a batch of 8
# needs 256 grid rows while a batch of 7 needs 455.
CAPACITY_SHAPE = {"max_block": 256, "max_query_len": 1, "num_idx_heads": 1}


class TestPersistentWorkMapCapacity:
    """A captured buffer must serve EVERY batch, not just the largest one.

    aiter's grid is non-monotonic in the batch, which is the whole reason it
    ships a separate capacity API. Sizing the persistent buffer with the exact
    size at `max_bs` therefore under-allocates for some smaller live batch, and
    the build rejects it at that step -- after capture, on a real request.
    """

    def test_the_exact_size_really_is_non_monotonic(self):
        # Self-validating: if aiter ever makes the grid monotonic, the tests
        # below stop proving anything and this one says so.
        sizes = [m.index_score_work_map_size(b, **CAPACITY_SHAPE) for b in range(1, 33)]
        assert any(
            sizes[i] > sizes[i + 1] for i in range(len(sizes) - 1)
        ), "the grid is monotonic now; this whole class can go"

    @pytest.mark.parametrize("max_batch", [8, 17, 32])
    def test_capacity_covers_every_batch_below_it(self, max_batch):
        cap = m.index_score_work_map_capacity(max_batch, **CAPACITY_SHAPE)
        worst = max(
            m.index_score_work_map_size(b, **CAPACITY_SHAPE)
            for b in range(1, max_batch + 1)
        )
        assert cap >= worst, f"capacity {cap} < worst exact size {worst}"

    def test_the_exact_size_would_not_have(self):
        # The bug this guards: at max_block 256 a batch of 8 needs 256 rows and
        # a batch of 7 needs 455, so `size(8)` is not a buffer a batch of 7 fits.
        assert m.index_score_work_map_size(7, **CAPACITY_SHAPE) > (
            m.index_score_work_map_size(8, **CAPACITY_SHAPE)
        )

    def test_empty_batch_asks_for_nothing(self):
        assert m.index_score_work_map_capacity(0, **CAPACITY_SHAPE) == 0


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
def test_a_capacity_buffer_serves_the_batch_the_exact_size_would_not():
    """End to end: build into one persistent buffer at both batches.

    Also pins the half this relies on but does not control -- aiter accepts an
    OVERSIZED `out` and hands back a `[rows, 2]` view of it, so the caller does
    not slice and `decode_index_score`'s exact-row check still sees the right
    shape.
    """
    shape = CAPACITY_SHAPE
    buf = torch.empty(
        (m.index_score_work_map_capacity(8, **shape), 2),
        dtype=torch.int32, device="cuda",
    )  # fmt: skip
    for batch in (8, 7):
        lens = torch.full(
            (batch,), shape["max_block"] * SPARSE_BLOCK_SIZE,
            dtype=torch.int32, device="cuda",
        )  # fmt: skip
        built = m.build_index_score_work_map(lens, out=buf, **shape)
        assert built.shape[0] == m.index_score_work_map_size(batch, **shape)
        assert built.data_ptr() == buf.data_ptr(), "aiter copied instead of slicing"


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


# ---------------------------------------------------------------------------
# The decode score kernel, and the hoist that makes it worth having.
# ---------------------------------------------------------------------------
def _score_oracle(idx_q, cache, block_table, seq_lens, max_query_len, heads, sm_scale):
    """Block scores in fp32 on the device, straight from the definition.

    `[heads, batch*max_query_len, nblk]`, NaN wherever the kernel is not
    required to write (`blk >= ceil(seq_len/128)`). Deliberately a transcription
    rather than a second kernel: there is one decode scorer now, so the only
    honest anchor is the rule it is supposed to implement.
    """
    batch = seq_lens.shape[0]
    dev, p = idx_q.device, SPARSE_BLOCK_SIZE
    nblk = -(-int(seq_lens.max()) // p)
    out = torch.full(
        (heads, batch * max_query_len, nblk), float("nan"), dtype=torch.float32,
        device=dev,
    )  # fmt: skip
    # The kernel folds log2(e) into the scale and works in base 2; the selection
    # is invariant to that, but a score comparison is not.
    scale = sm_scale * 1.4426950408889634
    tok = torch.arange(max_query_len, device=dev).repeat_interleave(heads)
    within = torch.arange(p, device=dev)
    for b in range(batch):
        length = int(seq_lens[b])
        q = idx_q[b * max_query_len : (b + 1) * max_query_len].reshape(-1, HEAD_DIM)
        cuts = length - max_query_len + tok + 1  # one causal cutoff per column
        for blk in range(-(-length // p)):
            # K is lifted to Q's dtype before fp32, not the other way round, so
            # an fp8 cache reproduces the kernel's rounding instead of skipping it.
            k = cache[int(block_table[b, blk])].to(idx_q.dtype).float()
            z = (k @ q.float().T) * scale
            z = z.masked_fill((blk * p + within)[:, None] >= cuts[None, :], -torch.inf)
            out[:, b * max_query_len : (b + 1) * max_query_len, blk] = (
                z.amax(0).reshape(max_query_len, heads).T
            )
    return out


@gpu
class TestDecodeScoreAgainstTheDefinition:
    """The decode scorer against `_score_oracle`, not against another kernel.

    This used to compare two kernels -- flydsl against the Triton one it beat --
    and assert on the SELECTION rather than the scores, because two accumulation
    orders may legally reorder blocks that tie. The Triton kernel is gone, so
    both halves of that change: the anchor is the definition, and scores are now
    comparable directly (with a tolerance, not bit-for-bit).
    """

    @staticmethod
    def _run(kw, work_map, max_block):
        return m.minimax_m3_index_topk_decode(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            kw["max_seq_len"], TOPK, INIT, LOCAL, kw["num_kv_heads"],
            kw["sm_scale"], emit_sparse_block_table=True,
            max_query_len=kw["max_query_len"],
            index_score_work_map=work_map, index_score_max_block=max_block,
        )  # fmt: skip

    @pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
    @pytest.mark.parametrize("max_query_len", [1, 8])
    @pytest.mark.parametrize("ragged", [False, True], ids=["uniform", "ragged"])
    def test_the_scores_are_the_ones_the_definition_asks_for(
        self, ragged, max_query_len, fp8
    ):
        batch, heads = 6, 2
        prefixes = [8192, 1024, 65536, 256, 16384, 512] if ragged else [8192] * batch
        kw = _inputs([max_query_len] * batch, prefixes, heads, "cuda")
        if fp8:
            kw["index_kv_cache"] = kw["index_kv_cache"].to(torch.float8_e4m3fn)
        max_block = -(-kw["max_seq_len"] // SPARSE_BLOCK_SIZE)
        work_map = m.build_index_score_work_map(
            kw["seq_lens"],
            max_block=max_block,
            max_query_len=max_query_len,
            num_idx_heads=heads,
        )
        score = m.decode_index_score(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            max_block, max_query_len, heads, kw["sm_scale"], work_map,
        )  # fmt: skip
        want = _score_oracle(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            max_query_len, heads, kw["sm_scale"],
        )  # fmt: skip
        assert score.shape == want.shape
        # Only where the kernel is required to write: the oracle marks the rest
        # NaN, and the kernel leaves those slots alone by design.
        live = ~torch.isnan(want)
        # Both sides carry -inf for a fully-masked block; subtracting them gives
        # NaN, so compare those positions by equality and the rest by tolerance.
        neg_inf = want == -torch.inf
        assert torch.equal(score[live & neg_inf], want[live & neg_inf])
        finite = live & ~neg_inf
        assert torch.allclose(score[finite], want[finite], rtol=2e-2, atol=2e-2)

    def test_a_map_built_for_a_wider_bound_still_agrees(self):
        """What a cudagraph replay does: score to `max_model_len`, not to the
        batch's longest request. The extra columns must not move the answer."""
        batch, heads = 4, 2
        kw = _inputs([1] * batch, [4096, 8192, 1024, 2048], heads, "cuda")
        narrow = -(-kw["max_seq_len"] // SPARSE_BLOCK_SIZE)
        wide = 4 * narrow
        # The table is widened with the bound, as production has it: the engine
        # sizes block tables from `max_model_len`, the same quantity
        # `_index_score_max_block` comes from, so a table narrower than the
        # bound never reaches the kernel -- and aiter rejects that pair rather
        # than reading past the row.
        wide_table = torch.zeros(batch, wide, dtype=torch.int32, device="cuda")
        wide_table[:, :narrow] = kw["block_table"]
        wide_kw = {**kw, "block_table": wide_table}
        work_map = m.build_index_score_work_map(
            kw["seq_lens"], max_block=wide, max_query_len=1, num_idx_heads=heads
        )
        wide_idx, _, _ = self._run(wide_kw, work_map, wide)
        narrow_idx, _, _ = self._run(kw, None, narrow)
        rows = heads * batch
        assert torch.equal(
            torch.sort(wide_idx.reshape(rows, TOPK), dim=1).values,
            torch.sort(narrow_idx.reshape(rows, TOPK), dim=1).values,
        )

    def test_a_map_built_for_another_bound_is_refused(self):
        """The one failure here that would otherwise be silent: the row count
        IS the grid, so a mismatched map is not a short map, it is the kernel
        reading rows that mean a different (request, chunk)."""
        kw = _inputs([1] * 4, [8192] * 4, 2, "cuda")
        max_block = -(-kw["max_seq_len"] // SPARSE_BLOCK_SIZE)
        work_map = m.build_index_score_work_map(
            kw["seq_lens"], max_block=2 * max_block, max_query_len=1, num_idx_heads=2
        )
        # ValueError and not AssertionError: the bound comes from launch
        # flags and a captured buffer, so `python -O` must not be able to turn
        # a stale map into unreported wrong scores. Same rule as
        # `TestPackableBound`.
        with pytest.raises(ValueError, match="different bounds"):
            self._run(kw, work_map, 0)


@gpu
class TestIndexScoreWorkMapHoist:
    """One build per decode step, not one per sparse layer.

    `make_work_map` is ~15 tiny device ops and costs ~120us whatever the shape
    -- launch floor, not work -- against a score kernel of 17-250us. Rebuilding
    it per layer is the failure that makes the whole substitution a net loss
    while every correctness test still passes.

    Every call path builds it in a metadata builder, one step upstream of this
    module. What is testable here is the half that lives here: given a map the
    op uses it as handed over and does not rebuild.

    The op does build one when none arrives -- there is no second scorer to fall
    back to any more, so declining would mean raising. That is the slow-but-
    correct path, and the count below is what keeps it from quietly becoming the
    fast path's behaviour too.
    """

    @pytest.mark.parametrize("hoisted", [True, False])
    def test_it_builds_exactly_when_the_caller_did_not(self, monkeypatch, hoisted):
        kw = _inputs([1] * 4, [8192] * 4, 2, "cuda")
        max_block = -(-kw["max_seq_len"] // SPARSE_BLOCK_SIZE)
        real = m.build_index_score_work_map
        work_map = real(
            kw["seq_lens"], max_block=max_block, max_query_len=1, num_idx_heads=2
        )
        builds = []

        def counted(*a, **k):
            builds.append(1)
            return real(*a, **k)

        monkeypatch.setattr(m, "build_index_score_work_map", counted)
        idx, _, _ = m.minimax_m3_index_topk_decode(
            kw["idx_q"], kw["index_kv_cache"], kw["block_table"], kw["seq_lens"],
            kw["max_seq_len"], TOPK, INIT, LOCAL, 2, kw["sm_scale"],
            emit_sparse_block_table=True, max_query_len=1,
            index_score_work_map=work_map if hoisted else None,
            index_score_max_block=max_block if hoisted else 0,
        )  # fmt: skip
        assert idx.shape == (2, 4, TOPK)
        assert len(builds) == (0 if hoisted else 1)


def _prefill_score_oracle(kw, heads, cache=None):
    """Prefill block scores in fp32 on the device, straight from the definition.

    `[heads, total_q, max_block]`, NaN wherever NEITHER scorer owes an answer:
    past a row's own causal window. The two tile the query axis differently --
    Triton by `SCORE_BLOCK_SIZE_Q`, flydsl by its own `tile_q` -- and each fills
    out to its TILE's window, so they legitimately disagree about columns past
    the row's. `n_valid_column_per_row` bounds the selector at exactly
    `ceil(causal_len / 128)` per row, so that window is the whole contract.

    A transcription of the rule rather than a third kernel, for the reason the
    module docstring gives.
    """
    idx_q = kw["idx_q"]
    cache = kw["index_kv_cache"] if cache is None else cache
    cu = kw["cu_seqlens_q"].tolist()
    dev, p = idx_q.device, SPARSE_BLOCK_SIZE
    out = torch.full(
        (heads, idx_q.shape[0], -(-kw["max_seq_len"] // p)),
        float("nan"), dtype=torch.float32, device=dev,
    )  # fmt: skip
    # The kernels fold log2(e) into the scale and work in base 2; the selection
    # is invariant to that, but a score comparison is not.
    scale = kw["sm_scale"] * 1.4426950408889634
    within = torch.arange(p, device=dev)
    for b in range(kw["seq_lens"].shape[0]):
        lo, hi = cu[b], cu[b + 1]
        prefix, length = int(kw["prefix_lens"][b]), int(kw["seq_lens"][b])
        # One causal cutoff per (token, head) column, in the order
        # `reshape(-1, HEAD_DIM)` lays the columns out.
        cuts = prefix + torch.arange(hi - lo, device=dev).repeat_interleave(heads) + 1
        q = idx_q[lo:hi].reshape(-1, HEAD_DIM)
        if cache.dtype != idx_q.dtype:
            # PREFILL rounds Q DOWN to the cache dtype -- the Triton kernel is
            # `tl.dot(q.to(k.dtype), k)` under `k.dtype.is_fp8()`, and flydsl's
            # `fp8_mfma` does the same. That is the opposite of decode, which
            # lifts K to Q, so this oracle must not borrow the decode rule: at
            # the 2e-2 tolerance the two roundings are close enough that an
            # fp8-Q regression shared by both scorers would pass unnoticed.
            q = q.to(cache.dtype)
        for blk in range(-(-length // p)):
            # K goes straight to fp32 -- exact from either dtype. The rounding
            # that matters here is Q's, applied above.
            k = cache[int(kw["block_table"][b, blk])].float()
            z = (k @ q.float().T) * scale
            z = z.masked_fill((blk * p + within)[:, None] >= cuts[None, :], -torch.inf)
            col = torch.where(
                blk < -(-cuts // p), z.amax(0), z.new_full(cuts.shape, float("nan"))
            )
            out[:, lo:hi, blk] = col.reshape(hi - lo, heads).T
    return out


def _prefill_score(kw, heads, backend, fp8=False):
    """`prefill_index_score` on `_inputs` kwargs, plus the oracle's live mask."""
    cache = kw["index_kv_cache"]
    if fp8:
        cache = cache.to(torch.float8_e4m3fn)
    score = m.prefill_index_score(
        kw["idx_q"], cache, kw["block_table"], kw["cu_seqlens_q"], kw["seq_lens"],
        kw["prefix_lens"], kw["max_query_len"], kw["max_seq_len"], kw["sm_scale"],
        backend=backend,
    )  # fmt: skip
    return score, _prefill_score_oracle(kw, heads, cache)


# (qlens, prefixes): fresh prompts, a ragged chunked-prefill batch, and the
# single-token tiles a chunk boundary leaves behind.
PREFILL_SHAPES = [
    ([600] * 4, [0] * 4),
    ([600, 137, 1024, 41], [547, 8192, 0, 65536]),
    ([1, 1, 1, 1], [8192, 1024, 256, 512]),
]
PREFILL_IDS = ["fresh", "ragged-chunked", "q1-tail"]


class TestPrefillScoreBackendChoice:
    """`ATOM_M3_PREFILL_INDEX_SCORE`, which is pure Python and needs no device."""

    @pytest.mark.parametrize("name", ["flydsl", "triton"])
    def test_an_explicit_backend_is_taken_literally(self, monkeypatch, name):
        monkeypatch.setattr(m, "score_prefill_flydsl", object())
        assert m._prefill_score_backend(name) == name

    def test_auto_follows_the_cache_dtype_not_mere_availability(self, monkeypatch):
        """flydsl is only a win on an fp8 cache.

        Its `k_lds` path -- a quarter of the K traffic -- is legal only on an
        fp8 k128 cache; on a bf16 one the kernel takes a register fallback that
        measures 3.6-4.8x SLOWER than Triton at 32k-116k context. So "aiter has
        it" is the wrong thing for auto to key on.
        """
        monkeypatch.setattr(m, "score_prefill_flydsl", object())
        assert m._prefill_score_backend("auto", fp8_cache=False) == "triton"
        assert m._prefill_score_backend("auto", fp8_cache=True) == "flydsl"

    def test_auto_falls_back_but_an_explicit_flydsl_never_does(self, monkeypatch):
        """The asymmetry is the point: a measured arm must not quietly swap."""
        monkeypatch.setattr(m, "score_prefill_flydsl", None)
        assert m._prefill_score_backend("auto", fp8_cache=True) == "triton"
        with pytest.raises(RuntimeError, match="no"):
            m._prefill_score_backend("flydsl")

    def test_an_unknown_name_is_rejected(self):
        with pytest.raises(ValueError, match="must be one of"):
            m._prefill_score_backend("fly")

    def test_none_reads_the_env(self, monkeypatch):
        monkeypatch.setattr(m, "score_prefill_flydsl", None)
        monkeypatch.setenv("ATOM_M3_PREFILL_INDEX_SCORE", "triton")
        assert m._prefill_score_backend() == "triton"


@gpu
class TestPrefillScoreAgainstTheDefinition:
    """Both prefill scorers against `_prefill_score_oracle`, not each other.

    `TestPrefillScorersAgree` below does compare them, which the module
    docstring's rule allows only because both are live and the dispatch's whole
    claim IS that they agree. These tests are what stops a defect they share
    from passing as agreement.
    """

    @pytest.mark.parametrize("backend", ["flydsl", "triton"])
    @pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
    @pytest.mark.parametrize("heads", [1, 2], ids=["cp-slice", "tp"])
    @pytest.mark.parametrize("qlens,prefixes", PREFILL_SHAPES, ids=PREFILL_IDS)
    def test_the_scores_are_the_ones_the_definition_asks_for(
        self, qlens, prefixes, heads, fp8, backend
    ):
        kw = _inputs(qlens, prefixes, heads, "cuda")
        score, want = _prefill_score(kw, heads, backend, fp8)
        assert score.shape == want.shape
        # Only where the scorer is required to write: the oracle marks the rest
        # NaN, and past its own tile's window a scorer leaves the slot
        # uninitialised by design.
        live = ~torch.isnan(want)
        assert live.any()
        # atol 1e-3, not 2e-2. Measured worst case over every shape, head count
        # and backend here is 1.7e-5 on an fp8 cache and ~0 on bf16, so 2e-2
        # was three orders of magnitude of slack -- enough that the oracle
        # could model the WRONG rounding (lifting K to Q, which is decode's
        # rule) and still pass at 1.69e-2, within 18% of the bound. 1e-3 keeps
        # ~60x headroom over the real error and still fails that mistake.
        assert torch.allclose(score[live], want[live], rtol=2e-2, atol=1e-3)

    @pytest.mark.parametrize("backend", ["flydsl", "triton"])
    def test_a_perturbed_cache_moves_the_scores(self, backend):
        """Negative control: the comparison above can actually fail."""
        kw = _inputs(*PREFILL_SHAPES[1], 2, "cuda")
        _, want = _prefill_score(kw, 2, backend)
        live = ~torch.isnan(want)
        kw["index_kv_cache"][kw["block_table"][1, 0]] += 1.0
        moved, _ = _prefill_score(kw, 2, backend)
        assert not torch.allclose(moved[live], want[live], rtol=2e-2, atol=2e-2)


@gpu
class TestPrefillScorersAgree:
    """The dispatch's claim: swapping the scorer does not move the answer.

    Scores with a tolerance -- two accumulation orders over the same 128 values
    are not bit-identical -- and then the SELECTION exactly, which is what the
    rest of the model actually consumes.
    """

    @pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
    @pytest.mark.parametrize("qlens,prefixes", PREFILL_SHAPES, ids=PREFILL_IDS)
    def test_the_two_scorers_agree_where_either_owes_an_answer(
        self, qlens, prefixes, fp8
    ):
        kw = _inputs(qlens, prefixes, 2, "cuda")
        fly, want = _prefill_score(kw, 2, "flydsl", fp8)
        tri, _ = _prefill_score(kw, 2, "triton", fp8)
        live = ~torch.isnan(want)
        # Same bound as the oracle comparison above, for the same reason: each
        # scorer sits within 1.7e-5 of the definition, so they sit within twice
        # that of each other and 2e-2 would hide a real divergence.
        assert torch.allclose(fly[live], tri[live], rtol=2e-2, atol=1e-3)

    @pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
    @pytest.mark.parametrize("qlens,prefixes", PREFILL_SHAPES, ids=PREFILL_IDS)
    def test_the_selection_is_the_same(self, monkeypatch, qlens, prefixes, fp8):
        kw = _inputs(qlens, prefixes, 2, "cuda")
        if fp8:
            # The case `auto` actually dispatches. `k_lds` -- and with it any
            # reason to run the flydsl prefill scorer at all -- is legal only
            # on an fp8 cache, so a bf16-only selection check covers everything
            # except the configuration this dispatch exists to reach. The score
            # comparison above runs both dtypes; this is the exact one, which
            # is the only test that can see a tie resolve to a different block.
            kw["index_kv_cache"] = kw["index_kv_cache"].to(torch.float8_e4m3fn)
        picked = []
        for backend in ("flydsl", "triton"):
            monkeypatch.setenv("ATOM_M3_PREFILL_INDEX_SCORE", backend)
            idx, _sbt, sctx = m.minimax_m3_index_topk(
                **kw, emit_sparse_block_table=True
            )
            # sbt is dropped: it is score ORDER, which ties may legally
            # permute; the selected set and the context length are the
            # properties that must not move.
            picked.append((idx, sctx))
        (ia, ca), (ib, cb) = picked
        assert torch.equal(ca, cb)
        for h in range(ia.shape[0]):
            for r in range(0, ia.shape[1], 37):
                sa = {int(v) for v in ia[h, r] if v >= 0}
                sb = {int(v) for v in ib[h, r] if v >= 0}
                assert sa == sb, f"head {h} row {r}"
