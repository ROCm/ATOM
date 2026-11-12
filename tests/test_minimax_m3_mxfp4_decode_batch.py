# SPDX-License-Identifier: MIT
"""Which requests the sparse decode (scorer, selector, attention) runs under the
MXFP4 index cache, and that scorer's per-ubatch buffers (CPU).

The MXFP4 scorer takes its batch off the rows of ``idx_q``, so the lengths and
block-table rows it is handed must be that many, on a padded step (a graph bucket,
or an eager step padded for DP) and a ragged one alike; and under decode TBO each
ubatch needs a work map of its own. The
builder tests drive `AiterAttentionMetadataBuilder` with its collaborators stubbed
and need ``aiter`` importable (CI's no-aiter job skips them); the scorer itself is
in test_minimax_m3_mxfp4_index.py.
"""

import dataclasses
import types

import pytest
import torch

from atom.model_ops.sparse_indexer_chunk import LogitsWorkspace

BOUND, HEADS, MAX_BS, PREFILL_ROWS = 64, 2, 16, 40


@pytest.fixture
def mod():
    pytest.importorskip("aiter")
    from atom.model_ops.attentions import aiter_attention

    return aiter_attention


def builder(mod, mxfp4=True, tbo_decode=True, forward_vars=None):
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b._index_scorers = {}
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
            max_model_len=BOUND * 128,
            max_num_batched_tokens=PREFILL_ROWS,
            index_cache_dtype="fp4" if mxfp4 else "fp8",
            enable_tbo=True,
            enable_tbo_decode=tbo_decode,
        ),
    )
    return b


class _Buf:
    def __init__(self, *size, dtype=None, device=None, publication_group=None):
        self.cpu = self.gpu = torch.zeros(*size, dtype=dtype)

    def copy_to_gpu(self, *a):
        pass


@pytest.mark.parametrize("max_bs", [16, 15])
@pytest.mark.parametrize(
    "mxfp4,tbo_decode", [(False, True), (True, False), (True, True)]
)
def test_the_scorer_buffers_are_one_workspace_and_a_map_per_decode_ubatch(
    monkeypatch, mxfp4, tbo_decode, max_bs
):
    pytest.importorskip("aiter")
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    capacity_args = []

    def capacity(*args):
        capacity_args.append(args)
        return 100

    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        capacity,
        raising=False,
    )
    max_qlen = 8  # a speculative config: the TBO sparse decode still runs one token
    config = types.SimpleNamespace(
        max_model_len=BOUND * 128,
        max_num_batched_tokens=PREFILL_ROWS,
        index_cache_dtype="fp4" if mxfp4 else "fp8",
        enable_tbo=True,
        enable_tbo_decode=tbo_decode,
    )
    scorers = allocate_index_scorers(
        config, "cpu", max_bs=max_bs, max_qlen=max_qlen, num_idx_heads=HEADS
    )
    if not mxfp4:
        assert scorers == {} and not capacity_args  # the switch is `"" in scorers`
        return
    main = scorers[""]
    # one workspace for every layer, as wide as the map and as tall as the larger of a
    # decode step's rows and a prefill chunk's
    assert isinstance(main.scores, LogitsWorkspace)
    assert main.scores.capacity == HEADS * max(max_bs * max_qlen, PREFILL_ROWS) * BOUND
    assert main.scores.nbytes == 4 * main.scores.capacity
    assert (main.max_block, main.work_map.shape) == (BOUND, (100, 2))
    # each ubatch's requests, from the split every backend's ubatch metadata follows
    import numpy as np

    from atom.utils.tbo.ubatch_splitting import cut_decode_ubatches

    cut = cut_decode_ubatches(max_bs, max_bs, np.ones(max_bs, dtype=np.int32))
    rows = [ub.running_bs for ub in cut]
    # `enable_tbo` alone is the prefill split, which never scores a decode
    if not tbo_decode:
        assert set(scorers) == {""}
        assert capacity_args == [(max_bs, BOUND, max_qlen, HEADS)]
        return
    assert capacity_args == [
        (max_bs, BOUND, max_qlen, HEADS),
        (rows[0], BOUND, 1, HEADS),
        (rows[1], BOUND, 1, HEADS),
    ]
    ub = [scorers[n] for n in ("ub0_", "ub1_")]
    for u in ub:
        assert (u.max_block, u.work_map.shape) == (BOUND, (100, 2))
        # the workspace is the step's: both ubatches run on one compute stream and a
        # layer scores and selects inside one call
        assert u.scores is main.scores
    # each ubatch has its own map: both ubatches' metadata are alive at once
    ptrs = {u.work_map.data_ptr() for u in ub} | {main.work_map.data_ptr()}
    assert len(ptrs) == 3


def test_the_ubatch_metadata_is_built_with_its_own_map(mod, monkeypatch):
    rows = 4
    var = {
        f"ub{i}_sparse_attention_n_valid_column_per_row": torch.zeros(1) for i in (0, 1)
    }
    seen = []
    monkeypatch.setattr(
        mod, "make_sparse_decode_metadata", lambda **kw: seen.append(kw)
    )
    monkeypatch.setattr(
        mod.AiterAttentionMetadataBuilder.__mro__[1],
        "finish_ubatch",
        lambda self, *a: None,
    )
    b = builder(mod, forward_vars=var)
    shared = object()
    b._index_scorers = {
        f"ub{i}_": types.SimpleNamespace(scores=shared, work_map=object())
        for i in (0, 1)
    }
    ub = types.SimpleNamespace(running_bs=rows)
    attn = types.SimpleNamespace(
        block_tables=torch.zeros(rows, 4),
        context_lens=torch.zeros(rows),
        slot_mapping=torch.zeros(rows),
        max_seqlen_k=8,
    )
    for i in (0, 1):
        b.finish_ubatch(i, ub, attn, 1)
    for i, kw in enumerate(seen):
        assert kw["index_scorer"] is b._index_scorers[f"ub{i}_"]
    assert seen[0]["index_scorer"].work_map is not seen[1]["index_scorer"].work_map
    # another index cache allocated none: nothing to pass
    seen.clear()
    other = builder(mod, mxfp4=False, forward_vars=var)
    other.finish_ubatch(0, ub, attn, 1)
    assert seen[0]["index_scorer"] is None


@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("kv", ["fp4", "fp8"])
@pytest.mark.parametrize("index", ["fp4", "fp8"])
@pytest.mark.parametrize("q_rows", [6, 12])  # exactly the metadata's rows / a DP pad
def test_the_sparse_attention_runs_only_the_rows_the_metadata_covers(
    monkeypatch, phase, kv, index, q_rows
):
    """One row rule for decode and prefill. With the MXFP4 index the scorer, the
    selector and the attention get the same rows the metadata covers (decode:
    ``seq_lens.shape[0] * q``; prefill: its tokens); pad rows past them are never read
    (the selector never wrote them) and the output is zero there. Another index cache
    keeps every row, as before. The skip-index layer after it keys its cached
    selection on the unsliced shape."""
    pytest.importorskip("aiter")
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn

    batch, mq, heads, hq = 3, 2, 1, 4
    rows = batch * mq if index == "fp4" else q_rows
    seen = {}

    def topk(idx_q, cache, block_table, *a, **kw):
        seq_lens = a[0] if phase == "decode" else a[1]
        seen["scorer"] = (idx_q.shape[0], seq_lens.shape[0], block_table.shape[0])
        seen["record"] = kw["index_scorer"]
        n = idx_q.shape[0] * heads
        return (
            torch.zeros(heads, idx_q.shape[0], 2, dtype=torch.int32),
            torch.zeros(n, 2, dtype=torch.int32),
            torch.zeros(n, dtype=torch.int32),
        )

    def attend_fp4(q, k_cache, v_cache, sparse_bt, sparse_ctx, *a, **kw):
        seen["attend"] = (q.shape[0], sparse_bt.shape[0] // heads, sparse_ctx.numel())
        kw["out"].fill_(1)  # written in place; the pad rows past it stay the caller's
        seen["out"] = (kw["out"].shape[0], kw["out"].dtype)

    def attend_fp8(q, k_cache, v_cache, topk_idx, bt, *a, **kw):
        output = a[3] if phase == "decode" else a[5]
        seen["attend"] = (
            q.shape[0],
            kw["sparse_bt"].shape[0] // heads,
            output.shape[0],
        )
        output.fill_(1)

    monkeypatch.setattr(
        index_topk, f"minimax_m3_index_topk{'_decode' * (phase == 'decode')}", topk
    )
    attend = f"minimax_m3_sparse_attn_{phase}_asm"
    monkeypatch.setattr(sparse_attn, attend, attend_fp8)

    def impl(skip):
        i = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
        torch.nn.Module.__init__(i)
        i.kv_cache_dtype, i.index_cache_dtype = kv, index
        i._sparse_pa_mxfp4, i._select_page_kwargs = attend_fp4, {}
        i.skip_index_topk, i.index_topk_cache_state = skip, {}
        i.indexer_cp_group = None
        i.topk, i.init_blocks, i.local_blocks, i.num_kv_heads, i.scale = 2, 1, 0, 1, 1.0
        i.index_cache, i.layer_num, i.sparse_layer_ordinal = None, 0, 0
        return i

    seq = torch.zeros(batch, dtype=torch.int32)
    table = torch.zeros(batch, 4, dtype=torch.int32)
    md = types.SimpleNamespace(
        max_seq_len=100,
        n_valid_column_per_row=None,
        index_scorer=object() if index == "fp4" else None,
        decode=types.SimpleNamespace(seq_lens=seq, block_table=table, max_query_len=mq),
        prefill=types.SimpleNamespace(
            cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32) * mq,
            seq_lens=seq,
            context_lens=seq,
            block_table=table,
            max_query_len=mq,
            max_seq_len=100,
            qo_indptr=torch.arange(batch * mq, dtype=torch.int32),  # a row per token
        ),
    )
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=md)
    )

    def run(layer, q, v):
        return getattr(layer, f"_sparse_{phase}")(
            q, None, v, None, None, None, None, ctx
        )

    q = torch.zeros(q_rows, hq, 8)
    v = torch.zeros(q_rows, hq, 8)
    first = impl(False)
    first._index_q = torch.zeros(q_rows, heads, 8)
    first._index_q_cache_key_info = None
    out = run(first, q, v)
    assert out.shape == q.shape
    assert seen["scorer"] == (rows, batch, batch)
    assert seen["record"] is md.index_scorer
    assert seen["attend"][0] == rows and seen["attend"][1] == rows
    assert (out[:rows] == 1).all() and (out[rows:] == 0).all()
    if kv == "fp4":
        assert seen["out"] == (rows, v.dtype)

    # a skip-index layer holds no index_q, only the unsliced shape's key
    seen.clear()
    skip = impl(True)
    skip._index_q = None
    skip._index_q_cache_key_info = (
        (q_rows, heads, 8),
        torch.float32,
        torch.device("cpu"),
    )
    out = run(skip, q, v)
    assert "scorer" not in seen and seen["attend"][0] == rows
    assert (out[rows:] == 0).all()


@pytest.mark.parametrize(
    "mxfp4, scheduled, rows, want",
    [
        (True, 3, 24, 3),  # a DP step padded past the requests it runs: rows decide
        (False, 3, 24, 8),  # the other index caches keep the padded batch
        (True, 9, 24, ValueError),  # fewer rows than scheduled requests: refused
    ],
)
def test_the_row_rule_and_its_refusal_are_the_mxfp4_indexs(
    mod, monkeypatch, mxfp4, scheduled, rows, want
):
    running_bs, q = 8, 8
    seen = []
    monkeypatch.setattr(
        mod, "make_sparse_decode_metadata", lambda **kw: seen.append(kw)
    )
    monkeypatch.setattr(mod, "kv_indices_generate_triton", lambda *a: None)
    monkeypatch.setattr(
        mod.AiterAttentionMetadataBuilder.__mro__[1],
        "finish_decode",
        lambda self, *a: None,
    )
    b = builder(mod, mxfp4=mxfp4)
    if mxfp4:
        b._index_scorers[""] = object()
    b.refresh_flydsl_plan = lambda *_a, **_k: None
    b._get_sparse_attention_block_tables = lambda bt, sl, bs: bt
    b._n_valid_column_per_row_buffer = lambda: None
    attn = types.SimpleNamespace(
        max_seqlen_k=8,
        block_tables=torch.zeros(running_bs, 4),
        context_lens=torch.zeros(running_bs),
        slot_mapping=torch.zeros(running_bs),
        kv_indices=None,
        kv_indptr=None,
        ubatches=None,
    )
    batch = types.SimpleNamespace(total_seqs_num_decode=scheduled)
    if want is ValueError:
        with pytest.raises(ValueError, match="do not cover"):
            b.finish_decode(batch, attn, running_bs, rows, q)
        return
    b.finish_decode(batch, attn, running_bs, rows, q)
    assert seen[0]["seq_lens"].shape[0] == want


@pytest.mark.parametrize("splits", [False, True])
def test_a_split_decode_builds_no_whole_batch_scorer_state(mod, monkeypatch, splits):
    """A decode that splits runs on its micro-batches' metadata (each with its own map,
    `finish_ubatch`), so the whole batch's is built without the scorer record -- no work
    map launched for nobody -- and an unsplit step keeps it. The rest of the metadata is
    the same either way."""
    seen = []
    monkeypatch.setattr(
        mod, "make_sparse_decode_metadata", lambda **kw: seen.append(kw)
    )
    monkeypatch.setattr(mod, "kv_indices_generate_triton", lambda *a: None)
    monkeypatch.setattr(
        mod.AiterAttentionMetadataBuilder.__mro__[1],
        "finish_decode",
        lambda self, *a: None,
    )
    b = builder(mod)
    record = b._index_scorers[""] = object()
    b.refresh_flydsl_plan = lambda *_a, **_k: None
    b._get_sparse_attention_block_tables = lambda bt, sl, bs: bt
    b._n_valid_column_per_row_buffer = lambda: None
    attn = types.SimpleNamespace(
        max_seqlen_k=8,
        block_tables=torch.zeros(8, 4),
        context_lens=torch.zeros(8),
        slot_mapping=torch.zeros(8),
        kv_indices=None,
        kv_indptr=None,
        ubatches=[object(), object()] if splits else None,  # `stage_ubatch_decode`'s
    )
    b.finish_decode(types.SimpleNamespace(total_seqs_num_decode=8), attn, 8, 8, 1)
    kw = seen[0]
    assert kw["index_scorer"] is (None if splits else record)
    assert kw["seq_lens"].shape[0] == 8 and kw["num_idx_heads"] == HEADS


def test_the_scorer_record_is_immutable_and_builds_without_aiter():
    """`IndexScoreBuffers` lives in a module that imports no aiter, so the non-GPU
    suite can build it; a step's record is the builder's with its map replaced."""
    from dataclasses import FrozenInstanceError, replace

    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    scores, wmap = LogitsWorkspace(4, "cpu"), torch.zeros(8, 2)
    builder_record = IndexScoreBuffers(scores, wmap, 37)
    step = replace(builder_record, work_map=wmap[:3])
    assert (
        step.scores is scores and step.max_block == 37 and step.work_map.shape == (3, 2)
    )
    assert builder_record.work_map is wmap
    with pytest.raises(FrozenInstanceError):
        builder_record.max_block = 1


def test_the_scorer_gets_the_step_records_of_the_metadata_not_the_impls_own():
    """The map, its bound and the workspace travel as one record on the metadata; the
    impl derives none of them (a length source of its own that drifted would fail
    every decode's exact check)."""
    pytest.importorskip("aiter")
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers
    from atom.model_ops.minimax_m3.sparse_attn import MiniMaxM3SparseMetadata

    impl = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
    impl.kv_cache_dtype, impl.index_cache_dtype = "bf16", "fp4"
    impl.max_model_len = 10**9  # not the bound
    record = IndexScoreBuffers(
        scores=LogitsWorkspace(4, "cpu"), work_map=torch.zeros(1, 2), max_block=37
    )
    md = MiniMaxM3SparseMetadata(
        seq_lens=None,
        max_seq_len=0,
        slot_mapping=None,
        num_prefills=0,
        index_scorer=record,
    )
    assert impl._index_scorer(md) is record
    # a layer of the MXFP4 index whose metadata has no record raises: the Triton scorer
    # would read its uint8 pages (a split decode's whole-batch metadata, an empty batch)
    with pytest.raises(ValueError, match="needs the step's index_scorer"):
        impl._index_scorer(dataclasses.replace(md, index_scorer=None))
    # another index cache selects without one
    impl.index_cache_dtype = "fp8"
    assert impl._index_scorer(md) is None


def test_the_decode_metadata_builds_the_map_into_the_records_buffer(monkeypatch):
    """`make_sparse_decode_metadata` builds the step's map into the record's capacity
    buffer at the record's bound and carries the record with that map; an empty batch
    builds none. The prefill metadata keeps the workspace only."""
    pytest.importorskip("aiter")
    from atom.model_ops.minimax_m3 import sparse_attn
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    built = []

    def work_map(seq_lens, max_block, q, heads, out=None):
        built.append((max_block, q, heads, out))
        return out[:3]

    monkeypatch.setattr(sparse_attn, "minimax_m3_index_score_work_map", work_map)
    monkeypatch.setattr(
        sparse_attn, "build_n_valid_column_per_row", lambda *a, **k: None
    )
    record = IndexScoreBuffers(
        scores=LogitsWorkspace(4, "cpu"), work_map=torch.zeros(8, 2), max_block=37
    )

    def decode(seq_lens):
        return sparse_attn.make_sparse_decode_metadata(
            seq_lens=seq_lens,
            block_table=None,
            slot_mapping=None,
            max_seq_len=5,
            num_idx_heads=2,
            max_query_len=4,
            index_scorer=record,
        )

    md = decode(torch.ones(2, dtype=torch.int32))
    assert built == [(37, 4, 2, record.work_map)]
    assert md.index_scorer.scores is record.scores and md.index_scorer.max_block == 37
    assert md.index_scorer.work_map.shape == (3, 2)  # the step's map, not the capacity
    assert decode(torch.ones(0, dtype=torch.int32)).index_scorer is None
    assert len(built) == 1
    arange = torch.arange  # the prefill metadata allocates its row index on "cuda"
    monkeypatch.setattr(torch, "arange", lambda *a, device=None, **k: arange(*a, **k))
    prefill = sparse_attn.make_sparse_prefill_metadata(
        cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([5], dtype=torch.int32),
        block_table=None,
        slot_mapping=None,
        max_query_len=2,
        max_seq_len=5,
        num_prefills=1,
        num_prefill_tokens=2,
        num_idx_heads=2,
        index_scorer=record,
    )
    assert prefill.index_scorer.scores is record.scores
    assert prefill.index_scorer.work_map is None


@pytest.mark.parametrize("planned", [True, False])
def test_the_caller_can_own_the_output(monkeypatch, planned):
    """`out=` is written in place and returned, on the planned and the plan-free path;
    its dtype is `out_dtype` (the query is e4m3), and nothing else is accepted.
    The kernels are doubles: the segments' views must land in the caller's buffer."""
    pytest.importorskip("aiter")
    from atom.model_ops.minimax_m3 import sparse_attn_mxfp4 as m

    def pa_decode(out, q, *a, **kw):
        out.copy_(torch.arange(out.shape[0], dtype=out.dtype).view(-1, 1, 1))

    monkeypatch.setattr(m, "pa_decode", pa_decode)
    monkeypatch.setattr(m, "get_recommended_splits", lambda *_a: 1)
    monkeypatch.setattr(m, "_FLYDSL_PLAN_MAX_BATCH", 4)  # 3 segments of 5 rows
    monkeypatch.setattr(m, "_flydsl_prepare_explicit_plan", lambda **kw: None)
    monkeypatch.setattr(m, "_flydsl_plan_scratch", lambda *a: (None, None, None))

    rows, hkv, heads = 5, 2, 4
    k = torch.zeros(3, hkv, 8704, dtype=torch.uint8)
    q = torch.zeros(rows, heads, 128, dtype=torch.float8_e4m3fn)
    bt = torch.zeros(rows * hkv, 2, dtype=torch.int32)
    ctx = torch.ones(rows * hkv, dtype=torch.int32)
    buf = torch.full((rows + 2, heads, 128), -1.0, dtype=torch.bfloat16)  # 2 pad rows
    args = (q, k, k, bt, ctx, hkv, 1.0, torch.bfloat16)

    got = m.sparse_pa_mxfp4(*args, planned=planned, out=buf[:rows])
    assert got.data_ptr() == buf.data_ptr() and got.shape == q.shape
    per_pair = got.view(rows * hkv, heads // hkv, 128)[:, 0, 0]
    # each segment's call numbers its own rows from 0, the plan-free one all of them
    want = [i % 4 for i in range(rows * hkv)] if planned else list(range(rows * hkv))
    assert per_pair.tolist() == [float(w) for w in want]
    assert (buf[rows:] == -1).all()  # the pad rows are not touched

    with pytest.raises(ValueError, match="out must be"):
        m.sparse_pa_mxfp4(*args, out=torch.empty(q.shape, dtype=torch.float32))


def _scorer_call(monkeypatch, phase):
    """`minimax_m3_index_topk[_decode]` with the aiter scorer and the selection
    doubled: returns (call(record), what the scorer was handed, what the selection read).
    """
    pytest.importorskip("aiter")
    from atom.model_ops.minimax_m3 import index_topk

    heads, rows, width = 4, 8, 6
    seen = {}

    def scorer(*args, out=None, **kw):
        seen["out"] = out
        return out

    monkeypatch.setattr(index_topk, "minimax_m3_index_score_fp4_flydsl", scorer)
    monkeypatch.setattr(index_topk, "minimax_m3_index_score_prefill_fp4_flydsl", scorer)
    monkeypatch.setattr(
        index_topk, "_launch_select", lambda score, *a, **kw: seen.update(sel=score)
    )
    monkeypatch.setattr(index_topk, "page_views", lambda c: (c, c))
    idx_q = torch.zeros(rows, heads, 128)
    table = torch.zeros(rows, width, dtype=torch.int32)
    seq = torch.ones(rows, dtype=torch.int32)
    cache = torch.zeros(1)

    def call(record):
        if phase == "decode":
            index_topk.minimax_m3_index_topk_decode(
                idx_q, cache, table, seq, width * 128, 2, 1, 1, heads, 1.0,
                max_query_len=1, index_scorer=record,
            )  # fmt: skip
        else:
            cu = torch.arange(rows + 1, dtype=torch.int32)
            index_topk.minimax_m3_index_topk(
                idx_q, cache, table, cu, seq, seq - 1, 1, width * 128, 2, 1, 1,
                heads, 1.0, index_scorer=record,
            )  # fmt: skip

    return call, seen, (heads, rows, width)


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_the_scorer_writes_a_contiguous_view_of_the_shared_workspace(
    monkeypatch, phase
):
    """The scorer, decode or prefill, is handed a contiguous [heads, rows, width] view
    of the workspace's head (the layout the aiter selector needs), the selection
    reads that same tensor, and a workspace too small for the step is refused rather
    than overrun."""
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    call, seen, (heads, rows, width) = _scorer_call(monkeypatch, phase)
    wmap = torch.zeros(2, 2)  # decode only
    storage = LogitsWorkspace(heads * rows * width + 5, "cpu")  # sized for more
    call(IndexScoreBuffers(storage, wmap, width))
    out = seen["out"]
    assert out.shape == (heads, rows, width) and out.is_contiguous()
    assert out.data_ptr() == storage.logits(1, 1).data_ptr() and seen["sel"] is out
    with pytest.raises(ValueError, match="exceeds its workspace"):
        call(
            IndexScoreBuffers(
                LogitsWorkspace(heads * rows * width - 1, "cpu"), wmap, width
            )
        )


def test_the_mxfp4_decode_scorer_refuses_a_record_without_its_map(monkeypatch):
    """No quiet per-call allocation: a record without the step's map (a prefill's)
    is refused by the decode scorer, and so is a context wider than the map's bound.
    A layer without any record is refused before it gets here (`_index_scorer`)."""
    from atom.model_ops.minimax_m3.index_buffers import IndexScoreBuffers

    call, _, (heads, rows, width) = _scorer_call(monkeypatch, "decode")
    ws = LogitsWorkspace(heads * rows * width, "cpu")
    with pytest.raises(ValueError, match="needs the step's index_scorer"):
        call(IndexScoreBuffers(ws))  # a prefill record has no map
    with pytest.raises(ValueError, match="wider than the 3 blocks"):
        call(IndexScoreBuffers(ws, torch.zeros(2, 2), width // 2))
