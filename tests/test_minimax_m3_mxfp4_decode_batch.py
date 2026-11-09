# SPDX-License-Identifier: MIT
"""Which requests the sparse decode (scorer, selector, attention) runs -- one rule
for every index cache -- and the MXFP4 scorer's per-ubatch buffers (CPU).

The MXFP4 scorer takes its batch off the rows of ``idx_q``, so the lengths and
block-table rows it is handed must be that many, on a padded step (a graph bucket,
or an eager step padded for DP) and a ragged one alike; and under decode TBO each
ubatch needs a map and a score storage of its own. The row rule
(`decode_rows_batch`) is pure and runs everywhere. The
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


@pytest.mark.parametrize("S,heads", [(1, 1), (8, 4)])  # S * heads each side of 16
def test_the_score_view_follows_its_step_width_layout_at_a_stand_in_threshold(
    monkeypatch, S, heads
):
    """Local logic only (the threshold is a stand-in); the next-but-one test holds
    it to aiter's own."""
    _stand_in_scorer_module(monkeypatch)
    _check_score_view(heads, S, 16, float("-inf"))


def test_an_eager_step_scores_at_its_own_block_count_a_graph_step_at_the_models(
    monkeypatch,
):
    from atom.model_ops.minimax_m3 import index_mxfp4

    _stand_in_scorer_module(monkeypatch)
    built = []
    flydsl = sys.modules["aiter.ops.flydsl"]
    flydsl.minimax_m3_index_score_work_map_capacity = lambda *a: 4
    flydsl.minimax_m3_index_score_work_map = (
        lambda seq_lens, max_block, *a, **kw: built.append(max_block)
        or torch.zeros(1, 2)
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


# (scheduled_bs, running_bs, running_tokens, q) -> the requests whose rows run
@pytest.mark.parametrize(
    "scheduled,running,tokens,q,want",
    [
        (3, 8, 8, 1, 8),  # graph bucket / eager pad, plain decode
        (3, 8, 64, 8, 8),  # ... verify step
        (3, 3, 3, 1, 3),  # nothing padded
        (0, 8, 8, 1, 8),  # an idle DP rank still runs the group's rows
        (3, 8, 24, 8, 3),  # DP step a prefilling peer made ragged: no pad rows
        (2, 4, 2048, 1, 4),  # a decoding DP rank in a prefill bucket
        (2, 4, 2048, 3, 4),  # ... whose bucket is no multiple of the query width
        (3, 4, 64, 8, 4),  # a DSpark decode bucket wider than running_bs * q
    ],
)
def test_the_sparse_decode_runs_the_rows_the_metadata_covers(
    scheduled, running, tokens, q, want
):
    from atom.utils.forward_context import decode_rows_batch

    assert decode_rows_batch(scheduled, running, tokens, q) == want


@pytest.mark.parametrize(
    "scheduled,running,tokens,q",
    [
        (3, 8, 13, 8),  # not a whole number of requests
        (3, 8, 16, 8),  # fewer requests than were scheduled
    ],
)
def test_rows_that_are_not_a_batch_are_refused_not_clamped(
    scheduled, running, tokens, q
):
    from atom.utils.forward_context import decode_rows_batch

    with pytest.raises(ValueError, match="decode rows"):
        decode_rows_batch(scheduled, running, tokens, q)


class _Buf:
    def __init__(self, *size, dtype=None, device=None, publication_group=None):
        self.cpu = self.gpu = torch.zeros(*size, dtype=dtype)

    def copy_to_gpu(self, *a):
        pass


@pytest.mark.parametrize("max_bs", [16, 15])
@pytest.mark.parametrize(
    "mxfp4,tbo_decode", [(False, True), (True, False), (True, True)]
)
def test_the_ubatch_buffers_are_for_a_decode_tbo_at_one_token_a_request(
    mod, monkeypatch, mxfp4, tbo_decode, max_bs
):
    from atom.model_ops.minimax_m3 import index_mxfp4

    capacity_args = []

    def capacity(*args):
        capacity_args.append(args)
        return 100

    monkeypatch.setattr(mod, "CpuGpuBuffer", _Buf)
    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        capacity,
        raising=False,
    )
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
    # another index cache allocated none: nothing to pass
    assert builder(mod)._index_score_kwargs("ub0_") == {}


@pytest.mark.parametrize("kv", ["fp4", "fp8"])
@pytest.mark.parametrize("index", ["fp4", "fp8"])
@pytest.mark.parametrize("q_rows", [6, 12])  # exactly the metadata's rows / a DP pad
def test_the_sparse_decode_runs_only_the_rows_the_metadata_covers(
    monkeypatch, kv, index, q_rows
):
    """Whatever the KV and index caches, the scorer, the selector and the attention
    get the same ``seq_lens.shape[0] * q`` rows; pad rows past them are never read
    (the selector never wrote them) and the output is zero there. The skip-index
    layer after it keys its cached selection on the unsliced shape."""
    pytest.importorskip("aiter")
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn, sparse_attn_mxfp4

    batch, mq, heads, hq = 3, 2, 1, 4
    rows = batch * mq
    seen = {}

    def topk(idx_q, cache, block_table, seq_lens, *a, **kw):
        seen["scorer"] = (idx_q.shape[0], seq_lens.shape[0], block_table.shape[0])
        n = idx_q.shape[0] * heads
        return (
            torch.zeros(heads, idx_q.shape[0], 2, dtype=torch.int32),
            torch.zeros(n, 2, dtype=torch.int32),
            torch.zeros(n, dtype=torch.int32),
        )

    def attend_fp4(q, k_cache, v_cache, sparse_bt, sparse_ctx, *a, **kw):
        seen["attend"] = (q.shape[0], sparse_bt.shape[0] // heads, sparse_ctx.numel())
        return torch.ones(q.shape[0], hq, 8)

    def attend_fp8(
        q, k_cache, v_cache, topk_idx, bt, seq_lens, hkv, scale, output, **kw
    ):
        seen["attend"] = (
            q.shape[0],
            kw["sparse_bt"].shape[0] // heads,
            output.shape[0],
        )
        output.fill_(1)

    monkeypatch.setattr(index_topk, "minimax_m3_index_topk_decode", topk)
    monkeypatch.setattr(sparse_attn_mxfp4, "sparse_pa_mxfp4", attend_fp4)
    monkeypatch.setattr(sparse_attn, "minimax_m3_sparse_attn_decode_asm", attend_fp8)

    def impl(skip):
        i = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
        torch.nn.Module.__init__(i)
        i.kv_cache_dtype, i.index_cache_dtype = kv, index
        i.skip_index_topk, i.index_topk_cache_state = skip, {}
        i.indexer_cp_group = None
        i.topk, i.init_blocks, i.local_blocks, i.num_kv_heads, i.scale = 2, 1, 0, 1, 1.0
        i.index_cache, i.layer_num, i.sparse_layer_ordinal = None, 0, 0
        return i

    md = types.SimpleNamespace(
        decode=types.SimpleNamespace(
            seq_lens=torch.zeros(batch, dtype=torch.int32),
            block_table=torch.zeros(batch, 4, dtype=torch.int32),
            max_query_len=mq,
        ),
        max_seq_len=100,
        n_valid_column_per_row=None,
        index_score=None,
    )
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=md)
    )
    q = torch.zeros(q_rows, hq, 8)
    v = torch.zeros(q_rows, hq, 8)
    first = impl(False)
    first._index_q = torch.zeros(q_rows, heads, 8)
    first._index_q_cache_key_info = None
    out = first._sparse_decode(q, None, v, None, None, None, None, ctx)
    assert out.shape == q.shape
    assert seen["scorer"] == (rows, batch, batch)
    assert seen["attend"][0] == rows and seen["attend"][1] == rows
    assert (out[:rows] == 1).all() and (out[rows:] == 0).all()

    # a skip-index layer holds no index_q, only the unsliced shape's key
    seen.clear()
    skip = impl(True)
    skip._index_q = None
    skip._index_q_cache_key_info = (
        (q_rows, heads, 8),
        torch.float32,
        torch.device("cpu"),
    )
    out = skip._sparse_decode(q, None, v, None, None, None, None, ctx)
    assert "scorer" not in seen and seen["attend"][0] == rows
    assert (out[rows:] == 0).all()
