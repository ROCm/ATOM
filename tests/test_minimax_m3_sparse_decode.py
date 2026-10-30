# SPDX-License-Identifier: MIT
"""Which rows MiniMax-M3's sparse decode attends.

An eager decode of a batch padded up the capture ladder -- the q=1 step a P/D
producer that keeps a drafter runs -- cuts the decode tables to its requests
while q keeps the pad rows. The top-k writes a page table and a length for the
rows the tables describe and no others, so a pad row handed to the paged decode
reads a length and pages nobody wrote: on MI355X, a memory access fault.

The kernels are replaced here; what is asserted is which rows reach them. A
pad row's output is left as allocated -- it is dropped with the row -- so
nothing is asserted about it. The module imports triton and aiter, which the
CPU CI runner installs neither of, so this skips there.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="attention_mha imports triton kernels")
pytest.importorskip("aiter", reason="attention_mha imports the AITER runtime")

from atom.model_ops import attention_mha
from atom.model_ops.minimax_m3 import index_topk, sparse_attn

TOPK_PAGES = 8


def _layer(index_q):
    return SimpleNamespace(
        _index_q=index_q,
        _index_q_cache_key_info=None,
        _topk_cache_key=lambda *args: ("decode",),
        _load_cached_topk=lambda metadata, key: None,
        _store_cached_topk=lambda metadata, key, value: None,
        indexer_cp_group=None,
        index_cache=None,
        kv_mxfp4=False,
        kv_nvfp4=False,
        index_scale=None,
        _mxfp4_select_kwargs=lambda *args, **kwargs: {},
        topk=1,
        init_blocks=0,
        local_blocks=1,
        num_kv_heads=1,
        scale=1.0,
    )


def _decode(layer, q, num_reqs, max_query_len, monkeypatch):
    seen = {}

    def topk(idx_q, *args, **kwargs):
        seen["topk_rows"] = idx_q.shape[0]
        n = idx_q.shape[0]
        return (
            torch.zeros(1, n, 1, dtype=torch.int32),
            torch.zeros(n, TOPK_PAGES, dtype=torch.int32),
            torch.ones(n, dtype=torch.int32),
        )

    def attend(q, k_cache, v_cache, topk_idx, bt, seq_lens, h, scale, output, **kw):
        seen["attended_rows"] = q.shape[0]
        assert output.shape[0] == q.shape[0] == kw["sparse_bt"].shape[0]
        assert kw["sparse_ctx"].shape[0] == q.shape[0]
        output.fill_(1.0)

    monkeypatch.setattr(index_topk, "minimax_m3_index_topk_decode", topk)
    monkeypatch.setattr(sparse_attn, "minimax_m3_sparse_attn_decode_asm", attend)
    decode = SimpleNamespace(
        seq_lens=torch.full((num_reqs,), 100, dtype=torch.int32),
        block_table=torch.zeros(num_reqs, 4, dtype=torch.int32),
        max_query_len=max_query_len,
    )
    metadata = SimpleNamespace(
        decode=decode, max_seq_len=100, n_valid_column_per_row=None
    )
    layer._sparse_metadata = lambda fwd_ctx: metadata
    out = attention_mha.SparseMHAPagedAttentionImpl._sparse_decode(
        layer, q, None, None, None, None, None, None, None
    )
    return out, seen


def test_a_padded_eager_decode_attends_only_its_requests(monkeypatch):
    # 9 requests at q=1, run at the ladder's 16 rows.
    q = torch.randn(16, 4, 128)
    out, seen = _decode(_layer(torch.randn(16, 1, 128)), q, 9, 1, monkeypatch)
    assert seen == {"topk_rows": 9, "attended_rows": 9}
    assert out.shape == q.shape
    assert torch.equal(out[:9], torch.ones(9, 4, 128))


@pytest.mark.parametrize("num_reqs, max_query_len", [(16, 1), (2, 8)])
def test_a_decode_whose_tables_describe_every_row_attends_them_all(
    monkeypatch, num_reqs, max_query_len
):
    # A captured decode, or a speculative verify: 16 rows, all described.
    q = torch.randn(16, 4, 128)
    layer = _layer(torch.randn(16, 1, 128))
    out, seen = _decode(layer, q, num_reqs, max_query_len, monkeypatch)
    assert seen == {"topk_rows": 16, "attended_rows": 16}
    assert torch.equal(out, torch.ones(16, 4, 128))
