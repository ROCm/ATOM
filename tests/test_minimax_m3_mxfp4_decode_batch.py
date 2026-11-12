# SPDX-License-Identifier: MIT
"""Which requests the sparse decode runs under the MXFP4 index cache, and that
scorer's buffers (CPU, aiter importable; the scorer itself is in
test_minimax_m3_mxfp4_index.py)."""

import types

import pytest
import torch

HEADS = 2


@pytest.fixture
def mod():
    pytest.importorskip("aiter")
    from atom.model_ops.attentions import aiter_attention

    return aiter_attention


@pytest.mark.parametrize("max_bs", [16, 15])
@pytest.mark.parametrize("tbo_decode", [False, True])
def test_the_scorer_buffers_are_one_workspace_and_a_map_per_decode_ubatch(
    monkeypatch, tbo_decode, max_bs
):
    pytest.importorskip("aiter")
    from atom.model_ops.minimax_m3.index_buffers import allocate_index_scorers

    capacity_args = []
    monkeypatch.setattr(
        "aiter.ops.flydsl.minimax_m3_index_score_work_map_capacity",
        lambda *args: capacity_args.append(args) or 100,
        raising=False,
    )
    bound, prefill_rows, max_qlen = 64, 40, 8  # max_qlen: speculative; TBO runs q=1
    config = types.SimpleNamespace(
        max_model_len=bound * 128,
        max_num_batched_tokens=prefill_rows,
        index_cache_dtype="fp4",
        enable_tbo=True,
        enable_tbo_decode=tbo_decode,
    )
    scorers = allocate_index_scorers(
        config, "cpu", max_bs=max_bs, max_qlen=max_qlen, num_idx_heads=HEADS
    )
    main = scorers[""]
    # one workspace for every layer, as tall as the larger of decode rows / a chunk
    cap = HEADS * max(max_bs * max_qlen, prefill_rows) * bound
    assert main.scores.capacity == cap and main.max_block == bound
    assert capacity_args[0] == (max_bs, bound, max_qlen, HEADS)
    if not tbo_decode:  # `enable_tbo` alone is the prefill split: no decode ubatch
        assert set(scorers) == {""} and len(capacity_args) == 1
        return
    first = max_bs // 2  # an odd batch: the second ubatch takes the extra request
    assert [a[0] for a in capacity_args[1:]] == [first, max_bs - first]
    ub = [scorers[n] for n in ("ub0_", "ub1_")]
    # the workspace is the step's (one compute stream); each ubatch has its own map
    assert all(u.scores is main.scores for u in ub)
    assert len({u.work_map.data_ptr() for u in (*ub, main)}) == 3
    # another index cache allocates none: `"" in scorers` is the MXFP4 switch
    config.index_cache_dtype = "fp8"
    assert (
        allocate_index_scorers(config, "cpu", max_bs=4, max_qlen=1, num_idx_heads=1)
        == {}
    )


@pytest.mark.parametrize("phase", ["decode", "prefill"])
@pytest.mark.parametrize("kv,index", [("fp4", "fp4"), ("fp8", "fp4"), ("fp8", "fp8")])
@pytest.mark.parametrize("q_rows", [6, 12])  # exactly the metadata's rows / a DP pad
def test_the_sparse_attention_runs_only_the_rows_the_metadata_covers(
    monkeypatch, phase, kv, index, q_rows
):
    """One row rule for decode and prefill. With the MXFP4 index the scorer, selector
    and attention get the rows the metadata covers; the pad rows past them are never
    read and the output is zero there. Another index cache keeps every row."""
    pytest.importorskip("aiter")
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn

    batch, mq, heads = 3, 2, 1
    rows = batch * mq if index == "fp4" else q_rows
    seen = {}

    def topk(idx_q, cache, block_table, *a, **kw):
        seq_lens = a[0] if phase == "decode" else a[1]
        seen["scorer"] = (idx_q.shape[0], seq_lens.shape[0], block_table.shape[0])
        n = idx_q.shape[0] * heads
        return (
            torch.zeros(heads, idx_q.shape[0], 2, dtype=torch.int32),
            torch.zeros(n, 2, dtype=torch.int32),
            torch.zeros(n, dtype=torch.int32),
        )

    def attend_fp4(q, k_cache, v_cache, sparse_bt, sparse_ctx, *a, **kw):
        seen["attend"] = q.shape[0]
        kw["out"].fill_(1)  # in place; the pad rows past it stay the caller's

    def attend_fp8(q, k_cache, v_cache, topk_idx, bt, *a, **kw):
        seen["attend"] = q.shape[0]
        (a[3] if phase == "decode" else a[5]).fill_(1)

    monkeypatch.setattr(
        index_topk, f"minimax_m3_index_topk{'_decode' * (phase == 'decode')}", topk
    )
    monkeypatch.setattr(sparse_attn, f"minimax_m3_sparse_attn_{phase}_asm", attend_fp8)
    layer = object.__new__(attention_mha.SparseMHAPagedAttentionImpl)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype, layer.index_cache_dtype = kv, index
    layer._sparse_pa_mxfp4, layer._select_page_kwargs = attend_fp4, {}
    layer.skip_index_topk, layer.index_topk_cache_state = False, {}
    layer.indexer_cp_group = None
    layer.topk, layer.init_blocks, layer.local_blocks = 2, 1, 0
    layer.num_kv_heads, layer.scale = 1, 1.0
    layer.index_cache, layer.layer_num, layer.sparse_layer_ordinal = None, 0, 0
    layer._index_q, layer._index_q_cache_key_info = torch.zeros(q_rows, heads, 8), None

    seq = torch.zeros(batch, dtype=torch.int32)
    table = torch.zeros(batch, 4, dtype=torch.int32)
    cover = torch.arange(batch * mq, dtype=torch.int32)
    md = types.SimpleNamespace(
        max_seq_len=100,
        n_valid_column_per_row=None,
        index_scorer=object() if index == "fp4" else None,
        decode=types.SimpleNamespace(seq_lens=seq, block_table=table, max_query_len=mq),
        prefill=types.SimpleNamespace(
            cu_seqlens_q=torch.arange(batch + 1, dtype=torch.int32) * mq,
            seq_lens=seq, context_lens=seq, block_table=table, max_query_len=mq,
            max_seq_len=100, qo_indptr=cover,
        ),
    )  # fmt: skip
    ctx = types.SimpleNamespace(
        attn_metadata=types.SimpleNamespace(sparse_attention_metadata=md)
    )
    q = v = torch.zeros(q_rows, 4, 8)
    out = getattr(layer, f"_sparse_{phase}")(q, None, v, None, None, None, None, ctx)
    assert out.shape == q.shape
    assert seen["scorer"] == (rows, batch, batch) and seen["attend"] == rows
    assert (out[:rows] == 1).all() and (out[rows:] == 0).all()


@pytest.mark.parametrize(
    "mxfp4, scheduled, rows, q, ubatches, want",
    [
        (True, 3, 24, 8, None, 3),  # a DP step padded past its requests: rows decide
        (False, 3, 24, 8, None, 8),  # the other index caches keep the padded batch
        (True, 9, 24, 8, None, ValueError),  # fewer rows than scheduled: refused
        (True, 8, 8, 1, [1, 2], "split"),  # a split decode builds no whole-batch scorer
    ],
)
def test_the_decode_row_rule_and_the_split_step_are_the_mxfp4_indexs(
    mod, monkeypatch, mxfp4, scheduled, rows, q, ubatches, want
):
    running_bs = 8
    seen = []
    monkeypatch.setattr(
        mod, "make_sparse_decode_metadata", lambda **kw: seen.append(kw)
    )
    monkeypatch.setattr(mod, "kv_indices_generate_triton", lambda *a: None)
    base = mod.AiterAttentionMetadataBuilder.__mro__[1]
    monkeypatch.setattr(base, "finish_decode", lambda self, *a: None)
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    record = object()
    b._index_scorers = {"": record} if mxfp4 else {}
    b._num_idx_heads, b.block_size, b.block_ratio = HEADS, 128, 1
    b.refresh_flydsl_plan = lambda *_a, **_k: None
    b._get_sparse_attention_block_tables = lambda bt, sl, bs: bt
    b._n_valid_column_per_row_buffer = lambda: None
    b._has_sparse_attention, b.device, b.max_bs = True, "cpu", 16
    b.max_num_blocks_per_seq = b.block_table_cols = 4
    b.model_runner = types.SimpleNamespace(
        forward_vars={}, config=types.SimpleNamespace()
    )
    attn = types.SimpleNamespace(
        max_seqlen_k=8,
        block_tables=torch.zeros(running_bs, 4),
        context_lens=torch.zeros(running_bs),
        slot_mapping=torch.zeros(running_bs),
        kv_indices=None,
        kv_indptr=None,
        ubatches=ubatches,
    )
    batch = types.SimpleNamespace(total_seqs_num_decode=scheduled)
    if want is ValueError:
        with pytest.raises(ValueError, match="do not cover"):
            b.finish_decode(batch, attn, running_bs, rows, q)
        return
    b.finish_decode(batch, attn, running_bs, rows, q)
    if want == "split":  # each micro-batch has its own map; the whole batch's is unread
        assert seen[0]["index_scorer"] is None
    else:
        assert seen[0]["seq_lens"].shape[0] == want


@pytest.mark.parametrize("planned", [True, False])
def test_the_caller_can_own_the_output(monkeypatch, planned):
    """`out=` is written in place and returned, planned or plan-free, in the caller's
    buffer (the pad rows past it untouched); another dtype is refused."""
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
    assert (buf[rows:] == -1).all()
    with pytest.raises(ValueError, match="out must be"):
        m.sparse_pa_mxfp4(*args, out=torch.empty(q.shape, dtype=torch.float32))
