# SPDX-License-Identifier: MIT
"""Which requests the MXFP4 decode index scorer and selector run, and its
per-ubatch buffers (CPU).

The scorer takes its batch off the rows of ``idx_q``, so a padded decode step
(a graph bucket, or an eager step padded for DP) must hand it that many lengths
and block-table rows; and under decode TBO each ubatch needs a map and a score
storage of its own. The row rule (`index_batch`) is pure and runs everywhere. The
builder tests drive `AiterAttentionMetadataBuilder` with its collaborators stubbed
and need ``aiter`` importable (CI's no-aiter job skips them); the scorer itself is
in test_minimax_m3_mxfp4_index.py.
"""

import sys
import types

import pytest
import torch

BOUND, HEADS, MAX_BS = 64, 2, 16


@pytest.fixture
def mod():
    pytest.importorskip("aiter")
    from atom.model_ops.attentions import aiter_attention

    return aiter_attention


def builder(mod, mxfp4=True, tbo_decode=True, forward_vars=None):
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b._index_score_max_block = BOUND if mxfp4 else 0
    b._num_idx_heads = HEADS
    b.max_bs = MAX_BS
    b.device = "cpu"
    b.block_size = 128
    b.block_ratio = 1
    b.max_num_blocks_per_seq = 4
    b.block_table_cols = 4
    b._has_sparse_attention = True
    b.model_runner = types.SimpleNamespace(
        forward_vars={} if forward_vars is None else forward_vars,
        config=types.SimpleNamespace(
            max_model_len=BOUND * 128, enable_tbo=True, enable_tbo_decode=tbo_decode
        ),
    )
    return b


def _check_score_view(heads, S, transpose_min_f, neg_inf):
    from atom.model_ops.minimax_m3.index_mxfp4 import fresh_index_score

    batch, max_block, fill = 3, 6, 4
    rows = batch * S
    buf = torch.empty(heads * 8 * 8 * max_block, dtype=torch.float32).fill_(7.0)
    out = fresh_index_score(buf, rows, heads, max_block, S, fill_blocks=fill)
    assert out.shape == (heads, rows, max_block)
    assert out.data_ptr() == buf.data_ptr()  # the persistent storage, no copy
    if S * heads >= transpose_min_f:  # feature-contiguous, as the scorer allocates
        assert out.stride() == (1, heads, rows * heads)
    else:
        assert out.stride() == (rows * max_block, max_block, 1)
    assert (out[..., :fill] == neg_inf).all() and (out[..., fill:] == 7.0).all()
    return out


def _stand_in_scorer_module(monkeypatch):
    """A stand-in for aiter's scorer constants (threshold 16): local logic only."""
    parents = ("aiter", "aiter.ops", "aiter.ops.flydsl", "aiter.ops.flydsl.kernels")
    name = "aiter.ops.flydsl.kernels.minimax_m3_index_score"
    for mod_name in (*parents, name):
        monkeypatch.setitem(sys.modules, mod_name, types.ModuleType(mod_name))
    sys.modules[name].NEG_INF = float("-inf")
    sys.modules[name].TRANSPOSE_MIN_F = 16


def _stand_in_flydsl(monkeypatch, capacity, work_map=None):
    """aiter's work-map entry points, replaced for this test alone (the rest of
    aiter is left as it is: only ``aiter.ops.flydsl`` is looked up by name)."""
    name = "aiter.ops.flydsl"
    flydsl = types.ModuleType(name)

    def refuse(*a, **kw):
        raise AssertionError("aiter's work map must not be asked for")

    flydsl.minimax_m3_index_score_work_map_capacity = capacity
    flydsl.minimax_m3_index_score_work_map = work_map or refuse
    monkeypatch.setitem(sys.modules, name, flydsl)


@pytest.mark.parametrize("S", [1, 3, 4, 8])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_the_score_view_follows_the_layout_of_its_step_width_with_a_stand_in_threshold(
    monkeypatch, heads, S
):
    """Local logic only: the threshold is a stand-in (16), so this does NOT show that
    aiter's layout agrees; see the next test."""
    _stand_in_scorer_module(monkeypatch)
    _check_score_view(heads, S, 16, float("-inf"))


def test_an_eager_step_scores_at_its_own_block_count_a_graph_step_at_the_models(
    monkeypatch,
):
    from atom.model_ops.minimax_m3 import index_mxfp4

    _stand_in_scorer_module(monkeypatch)
    built = []
    _stand_in_flydsl(
        monkeypatch,
        lambda *a: 4,
        lambda seq_lens, max_block, *a, **kw: built.append(max_block)
        or torch.zeros(1, 2),
    )
    buffers = index_mxfp4.IndexScoreBuffers.alloc(4, 2, 1, 64, "cpu")
    seq = torch.tensor([300, 1000, 0])
    graph = buffers.begin_step(seq, 2, 1000)
    eager = buffers.begin_step(seq, 2, 1000, eager=True)
    huge = buffers.begin_step(seq, 2, 10**9, eager=True)  # never past the model's
    assert [graph.max_block, eager.max_block, huge.max_block] == [64, 8, 64]
    assert built == [64, 8, 64]
    for step in (graph, eager, huge):  # the view is as wide as the map's bound
        assert step.out.shape == (1, 6, step.max_block)
    # one persistent storage under every view
    assert graph.out.data_ptr() == eager.out.data_ptr() == buffers.scores.data_ptr()


def test_a_batch_of_zero_needs_nothing_from_aiter(monkeypatch):
    """No requests: no map rows asked for, no map built, and no step state (the
    scorer then builds its own) -- decided here, not by aiter's own empty case."""
    from atom.model_ops.minimax_m3 import index_mxfp4

    _stand_in_scorer_module(monkeypatch)
    _stand_in_flydsl(monkeypatch, lambda *a: pytest.fail("capacity asked for"))
    buffers = index_mxfp4.IndexScoreBuffers.alloc(0, 8, 2, 64, "cpu")
    assert buffers.work_map.shape == (0, 2) and buffers.work_map.dtype == torch.int32
    assert buffers.scores.numel() == 0 and buffers.scores.dtype == torch.float32
    empty = torch.empty(0, dtype=torch.int32)
    assert buffers.begin_step(empty, 8, 0) is None
    assert buffers.begin_step(empty, 8, 0, eager=True) is None


@pytest.mark.parametrize("S", [1, 3, 4, 8])
@pytest.mark.parametrize("heads", [1, 2, 4])
def test_the_score_view_has_the_strides_aiters_own_allocation_has(heads, S):
    """Same check against the installed aiter: its threshold, and the shape and
    strides of the score it would allocate. Skipped, never stubbed, when aiter is
    absent, too old for the scorer, or cannot probe a GPU."""
    try:
        from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
            NEG_INF,
            TRANSPOSE_MIN_F,
            alloc_score,
        )
    except (ImportError, RuntimeError) as e:
        pytest.skip(f"aiter's scorer is not importable here: {e!r}"[:200])
    out = _check_score_view(heads, S, TRANSPOSE_MIN_F, NEG_INF)
    own = alloc_score(3, S, heads, 6, "cpu")
    assert (out.shape, out.stride()) == (own.shape, own.stride())


# (scheduled_bs, running_bs, running_tokens, q) -> the scorer's batch
@pytest.mark.parametrize(
    "scheduled,running,tokens,q,want",
    [
        (3, 8, 8, 1, 8),  # graph bucket / eager pad, plain decode
        (3, 8, 64, 8, 8),  # ... verify step
        (3, 3, 3, 1, 3),  # nothing padded
        (0, 8, 8, 1, 8),  # an idle DP rank still runs the group's rows
        (3, 8, 24, 8, 3),  # DP step a prefilling peer made ragged: no pad rows
    ],
)
def test_the_scorer_runs_the_rows_idx_q_has(scheduled, running, tokens, q, want):
    from atom.model_ops.minimax_m3.index_mxfp4 import index_batch

    assert index_batch(scheduled, running, tokens, q) == want


@pytest.mark.parametrize(
    "scheduled,running,tokens,q",
    [
        (3, 8, 13, 8),  # not a whole number of requests
        (3, 8, 16, 8),  # fewer requests than were scheduled
        (3, 4, 64, 8),  # more than the step's padded batch
    ],
)
def test_rows_that_are_not_a_batch_are_refused_not_clamped(
    scheduled, running, tokens, q
):
    from atom.model_ops.minimax_m3.index_mxfp4 import index_batch

    with pytest.raises(ValueError, match="index decode"):
        index_batch(scheduled, running, tokens, q)


def test_each_ubatch_has_its_own_map_and_scores(mod):
    name = "sparse_attention_index_score"
    prefixes = ("", "ub0_", "ub1_")
    var = {p + name: object() for p in prefixes}
    b = builder(mod, forward_vars=var)
    got = [b._index_score_kwargs(p) for p in prefixes]
    for kw, p in zip(got, prefixes):
        assert kw == {"index_score_buffers": var[p + name]}
    # another index cache allocated none: nothing to pass
    assert builder(mod)._index_score_kwargs("ub0_") == {}


class _Buf:
    def __init__(self, *size, dtype=None, device=None, publication_group=None):
        self.cpu = self.gpu = torch.zeros(*size, dtype=dtype)

    def copy_to_gpu(self, *a):
        pass


@pytest.mark.parametrize("max_bs", [16, 15])
@pytest.mark.parametrize("tbo_decode", [False, True])
@pytest.mark.parametrize("mxfp4", [False, True])
def test_the_ubatch_buffers_are_for_a_decode_tbo_at_one_token_a_request(
    mod, monkeypatch, mxfp4, tbo_decode, max_bs
):
    from atom.model_ops.minimax_m3 import index_mxfp4

    capacity_args = []

    def capacity(*args):
        capacity_args.append(args)
        return 100

    monkeypatch.setattr(mod, "CpuGpuBuffer", _Buf)
    _stand_in_flydsl(monkeypatch, capacity)
    b = builder(mod, mxfp4=mxfp4, tbo_decode=tbo_decode)
    b.max_bs = max_bs
    max_qlen = 8  # a speculative config: the TBO sparse decode still runs one token
    var = b.model_runner.forward_vars
    if mxfp4:
        var["sparse_attention_index_score"] = index_mxfp4.IndexScoreBuffers.alloc(
            max_bs, max_qlen, HEADS, BOUND, "cpu"
        )
        capacity_args.clear()
    # each ubatch's requests, from the split every backend's ubatch metadata follows
    from atom.utils.tbo.ubatch_splitting import decode_ubatch_request_ranges

    rows = [hi - lo for lo, hi in decode_ubatch_request_ranges(max_bs, 2)]
    # the work plan sizes are not under test
    b._allocate_ubatch_buffers(max_qlen, *([1, torch.int32] * 6))
    # `enable_tbo` alone is the prefill split, which never scores a decode
    names = [f"ub{i}_sparse_attention_index_score" for i in (0, 1)]
    if not (mxfp4 and tbo_decode):
        assert not [n for n in names if n in var] and not capacity_args
        return
    main = var["sparse_attention_index_score"]
    for i in (0, 1):
        ub = var[names[i]]
        assert ub.work_map.shape == (100, 2) and ub.work_map.dtype == torch.int32
        assert ub.scores.dtype == torch.float32
        assert ub.scores.numel() == HEADS * rows[i] * BOUND  # q == 1, not max_qlen
        assert (ub.max_block, ub.num_idx_heads) == (BOUND, HEADS)
    assert capacity_args == [(rows[0], BOUND, 1, HEADS), (rows[1], BOUND, 1, HEADS)]
    # the score storage is the main buffer's, the two ubatches' views not overlapping
    base = main.scores.data_ptr()
    step = main.scores.element_size()
    assert var[names[0]].scores.data_ptr() == base
    assert var[names[1]].scores.data_ptr() == base + rows[0] * HEADS * BOUND * step
    assert var[names[1]].work_map.data_ptr() != var[names[0]].work_map.data_ptr()
    # the other per-ubatch buffers stay at max_bs
    assert var["ub0_context_lens"].gpu.numel() == max_bs


def test_the_ubatch_metadata_is_built_with_its_own_buffers(mod, monkeypatch):
    from atom.model_ops.minimax_m3 import sparse_attn

    var = {"max_qlen": 1}
    rows = 4
    for i in (0, 1):
        p = f"ub{i}_"
        for name in ("slot_mapping", "context_lens", "block_tables", "kv_indptr"):
            var[p + name] = types.SimpleNamespace(gpu=torch.zeros(rows + 1, 4))
        for name in ("cu_seqlens_q", "kv_indices"):
            var[p + name] = types.SimpleNamespace(gpu=torch.zeros(rows + 1))
        for name in (
            "work_meta_data",
            "work_indptr",
            "work_info_set",
            "reduce_indptr",
            "reduce_final_map",
            "reduce_partial_map",
        ):
            var[p + name] = None
        var[p + "sparse_attention_n_valid_column_per_row"] = torch.zeros(1)
        var[p + "sparse_attention_index_score"] = object()
    seen = []
    monkeypatch.setattr(
        sparse_attn, "make_sparse_decode_metadata", lambda **kw: seen.append(kw)
    )
    b = builder(mod, forward_vars=var)
    for i in (0, 1):
        b.build_ubatch_metadata(i, rows)
    for i, kw in enumerate(seen):
        assert kw["index_score_buffers"] is var[f"ub{i}_sparse_attention_index_score"]
    assert seen[0]["index_score_buffers"] is not seen[1]["index_score_buffers"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU and triton")
@pytest.mark.parametrize("qlen", [1, 4])
def test_fp4_kv_decode_rows_name_pool_pages_one_page_a_block(qlen):
    """Not run by the CPU gate. The selector's emit at ``pages_per_block=1`` /
    ``block_page_stride=1`` (what fp4 KV asks of it), over scattered physical pages,
    with a speculative verify (qlen 4: a row per token, its own causal length): row
    (token t, kv head h) reads ``block_table[b, blk] * Hkv + h`` for each selected
    block, the tail block last, for ``(n - 1) * 128 + tail tokens``. A BF16 index
    cache stands in for the index: the emit does not depend on it."""
    pytest.importorskip("triton")
    from atom.model_ops.minimax_m3.index_topk import minimax_m3_index_topk_decode

    hkv, batch, topk, pages, dim = 2, 3, 4, 40, 128
    g = torch.Generator(device="cuda").manual_seed(qlen)
    seq = torch.tensor([700, 1300, 2100], dtype=torch.int32, device="cuda")
    blocks = -(-int(seq.max()) // 128)
    table = torch.zeros(batch, blocks, dtype=torch.int32, device="cuda")
    for b in range(batch):  # physical pages in no particular order
        table[b] = torch.randperm(pages, device="cuda", generator=g)[:blocks]
    idx_q = torch.randn(batch * qlen, hkv, dim, device="cuda", generator=g)
    idx_q = idx_q.to(torch.bfloat16)
    cache = torch.randn(pages, 128, dim, device="cuda", generator=g)
    cache = cache.to(torch.bfloat16)
    topk_idx, sbt, sctx = minimax_m3_index_topk_decode(
        idx_q, cache, table, seq, int(seq.max()), topk, 1, 1, hkv, dim**-0.5,
        emit_sparse_block_table=True, max_query_len=qlen,
        block_page_stride=1, pages_per_block=1,
    )  # fmt: skip
    assert sbt.shape == (batch * qlen * hkv, topk)
    for b in range(batch):
        for t in range(qlen):
            cl = int(seq[b]) - qlen + t + 1  # this token's causal length
            tail = (cl - 1) // 128
            for h in range(hkv):
                row = (b * qlen + t) * hkv + h
                sel = [int(x) for x in topk_idx[h, b * qlen + t] if x >= 0]
                assert tail in sel  # local_blocks=1 pins the tail
                n = len(sel)
                want = [int(table[b, x]) * hkv + h for x in sel if x != tail]
                got = [int(x) for x in sbt[row, :n]]
                assert sorted(got[:-1]) == sorted(want)
                assert got[-1] == int(table[b, tail]) * hkv + h
                assert int(sctx[row]) == (n - 1) * 128 + cl - tail * 128
