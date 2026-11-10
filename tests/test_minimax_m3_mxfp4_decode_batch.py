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
    b._index_work_map_blocks = BOUND if mxfp4 else 0
    b._index_work_maps = {}
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
    if mxfp4:
        b._index_work_maps[""] = b._alloc_index_work_map(max_bs, max_qlen)
        capacity_args.clear()
    # each ubatch's requests, from the split every backend's ubatch metadata follows
    import numpy as np

    from atom.utils.tbo.ubatch_splitting import cut_decode_ubatches

    cut = cut_decode_ubatches(max_bs, max_bs, np.ones(max_bs, dtype=np.int32))
    rows = [ub.running_bs for ub in cut]
    # the base's buffers are not under test
    monkeypatch.setattr(
        mod.AiterAttentionMetadataBuilder.__mro__[1],
        "_allocate_ubatch_buffers",
        lambda self, max_seqlen_q: None,
    )
    b._allocate_ubatch_buffers(max_qlen)
    # `enable_tbo` alone is the prefill split, which never scores a decode
    names = ["ub0_", "ub1_"]
    maps = b._index_work_maps
    if not (mxfp4 and tbo_decode):
        assert not [n for n in names if n in maps] and not capacity_args
        return
    for i in (0, 1):
        assert maps[names[i]].shape == (100, 2)
        assert maps[names[i]].dtype == torch.int32
    assert capacity_args == [(rows[0], BOUND, 1, HEADS), (rows[1], BOUND, 1, HEADS)]
    # each ubatch has its own map
    assert maps[names[1]].data_ptr() != maps[names[0]].data_ptr()


def test_the_ubatch_metadata_is_built_with_its_own_buffers(mod, monkeypatch):
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
    b._index_work_maps = {"ub0_": object(), "ub1_": object()}
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
        assert kw["index_work_map_out"] is b._index_work_maps[f"ub{i}_"]
        assert kw["index_work_map_max_block"] == BOUND
    assert seen[0]["index_work_map_out"] is not seen[1]["index_work_map_out"]
    # another index cache allocated none: nothing to pass
    assert builder(mod)._index_work_map_kwargs("ub0_") == {}


@pytest.mark.parametrize("kv", ["fp4", "fp8"])
@pytest.mark.parametrize("index", ["fp4", "fp8"])
@pytest.mark.parametrize("q_rows", [6, 12])  # exactly the metadata's rows / a DP pad
def test_the_sparse_decode_runs_only_the_rows_the_metadata_covers(
    monkeypatch, kv, index, q_rows
):
    """With the MXFP4 index the scorer, the selector and the attention get the same
    ``seq_lens.shape[0] * q`` rows; pad rows past them are never read (the selector
    never wrote them) and the output is zero there. Another index cache keeps every
    row, as before. The skip-index layer after it keys its cached selection on the
    unsliced shape."""
    pytest.importorskip("aiter")
    from atom.model_ops import attention_mha
    from atom.model_ops.minimax_m3 import index_topk, sparse_attn, sparse_attn_mxfp4

    batch, mq, heads, hq = 3, 2, 1, 4
    rows = batch * mq if index == "fp4" else q_rows
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
        index_work_map=None,
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
        b._index_work_maps[""] = object()
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
    )
    batch = types.SimpleNamespace(total_seqs_num_decode=scheduled)
    if want is ValueError:
        with pytest.raises(ValueError, match="do not cover"):
            b.finish_decode(batch, attn, running_bs, rows, q)
        return
    b.finish_decode(batch, attn, running_bs, rows, q)
    assert seen[0]["seq_lens"].shape[0] == want
