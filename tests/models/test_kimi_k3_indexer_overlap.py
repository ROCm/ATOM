# SPDX-License-Identifier: MIT
"""A Kimi-K3 DSA layer's DCP prefill: the indexer runs its top-k all-gather and
owner filter on the model's side stream while the q path runs, and the sparse
MLA waits for them before its query gather. Graph capture, TBO and an indexer
given no stream keep the serial order."""

from inspect import signature
from types import MethodType, SimpleNamespace

import pytest
import torch
from aiter_stub import atom_under_stubs

PREFIX = "layers.0.self_attn.indexer"


@pytest.fixture
def modules():
    with atom_under_stubs() as load:
        yield SimpleNamespace(
            attention_mla=load("atom.model_ops.attention_mla"),
            indexer_forward=load("atom.model_ops.indexer.indexer_forward"),
            AttentionMetaData=load("atom.utils.forward_context").AttentionMetaData,
        )


@pytest.fixture
def dcp_prefill(monkeypatch, modules):
    """`run(stream)` runs the indexer op on a DCP prefill step and returns the
    (step, stream) of each exchange call and the step's metadata."""
    calls = []
    indexer_forward = modules.indexer_forward
    AttentionMetaData = modules.AttentionMetaData
    current = torch.cuda.current_stream

    def score_topk(*args, out, **kwargs):
        calls.append(("score", current(), args[3].split.world > 1))
        return out

    monkeypatch.setattr(indexer_forward, "score_topk", score_topk)
    monkeypatch.setattr(
        indexer_forward,
        "_gather_split",
        lambda out, split: calls.append(("gather", current())),
        raising=False,
    )
    monkeypatch.setattr(
        indexer_forward,
        "triton_filter_and_convert_dcp_index_prefill",
        lambda *a, **k: calls.append(("filter", current())),
    )
    monkeypatch.setattr(
        indexer_forward, "indexer_k_quant_and_cache", lambda *a, **k: None
    )
    monkeypatch.setattr(indexer_forward, "_index_cache", lambda *a: None)
    monkeypatch.setattr(indexer_forward, "get_dcp_world_size", lambda: 2)
    monkeypatch.setattr(indexer_forward, "get_dcp_rank", lambda: 0)
    monkeypatch.setattr(indexer_forward, "pcp_is_enabled", lambda: False)

    def run(stream, *, passes=None):
        indexer = SimpleNamespace(
            _indexer_fp4=False, use_nope_cache_fusion=False, exchange_stream=stream
        )
        monkeypatch.setattr(
            indexer_forward,
            "get_current_atom_config",
            lambda: SimpleNamespace(
                kv_cache_block_size=1,
                dcp_config=SimpleNamespace(interleave_size=1),
                compilation_config=SimpleNamespace(
                    static_forward_context={PREFIX: indexer}
                ),
            ),
        )
        md = AttentionMetaData()
        md.run_sparse = True
        from atom.model_ops.indexer.rows import IndexRows, RowSplit

        md.index_rows = IndexRows(
            torch.zeros(7, device="cuda"), 16, (), RowSplit.of(7, 0, 2), passes=passes
        )
        monkeypatch.setattr(
            indexer_forward,
            "get_forward_context",
            lambda: SimpleNamespace(
                attn_metadata=md,
                context=SimpleNamespace(is_prefill=True, is_dummy_run=False),
            ),
        )
        x = torch.zeros(7, 4, device="cuda")
        args = dict.fromkeys(signature(indexer_forward.sparse_attn_indexer).parameters)
        args.update(hidden_states=x, kv_cache=x, q_input=x, k=x, weights=x)
        args.update(k_cache_prefix=f"{PREFIX}.k_cache", topk_tokens=16)
        indexer_forward.sparse_attn_indexer(**args)
        return calls, md

    return run


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_exchange_runs_on_the_side_stream_and_hands_its_event_to_mla(dcp_prefill):
    side = torch.cuda.Stream()
    calls, md = dcp_prefill(side)
    main = torch.cuda.current_stream()
    assert calls == [("score", main, False), ("gather", side), ("filter", side)]
    assert isinstance(md.index_exchange_done, torch.cuda.Event)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
@pytest.mark.parametrize("serial", ["no stream", "graph capture", "tbo"])
def test_capture_tbo_and_a_streamless_indexer_keep_the_serial_order(
    dcp_prefill, monkeypatch, serial, modules
):
    indexer_forward = modules.indexer_forward
    if serial == "graph capture":
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    if serial == "tbo":
        monkeypatch.setattr(indexer_forward, "tbo_active", lambda: True, raising=False)
    calls, md = dcp_prefill(None if serial == "no stream" else torch.cuda.Stream())
    main = torch.cuda.current_stream()
    assert calls == [("score", main, True), ("filter", main)]
    assert md.index_exchange_done is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA streams")
def test_packed_prefill_keeps_its_per_pass_gathers_in_the_scorer(dcp_prefill):
    passes = (
        SimpleNamespace(row_base=0, stride=2, world=2),
        SimpleNamespace(row_base=4, stride=2, world=2),
    )
    calls, md = dcp_prefill(torch.cuda.Stream(), passes=passes)
    main = torch.cuda.current_stream()
    assert calls == [("score", main, True), ("filter", main)]
    assert md.index_exchange_done is None


@pytest.mark.parametrize("use_qrep", [False, True])
def test_sparse_prefill_joins_the_exchange_before_its_query_gather(
    monkeypatch, use_qrep, modules
):
    calls = []
    attention_mla = modules.attention_mla
    AttentionMetaData = modules.AttentionMetaData
    monkeypatch.setattr(
        attention_mla,
        "dcp_all_gather_prefill_query_heads",
        lambda group, q, heads: calls.append("query gather") or q,
    )
    layer = SimpleNamespace(
        dcp_group=None,
        dcp_sparse_prefill_num_heads=8,
        dcp_prefill_merge_bf16_ok=True,
        _forward_prefill_mla=lambda *a, **k: calls.append("mla") or (None, None),
        _dcp_prefill_merge_out=lambda *a, **k: calls.append("merge"),
    )
    prefill = MethodType(attention_mla.MLAAttention._dcp_sparse_prefill, layer)
    md = AttentionMetaData()
    md.dcp_owned_counts = torch.zeros(4, dtype=torch.int32)
    md.index_exchange_done = SimpleNamespace(wait=lambda: calls.append("join"))
    gather = [] if use_qrep else ["query gather"]
    prefill(torch.zeros(4, 1), None, md, use_qrep=use_qrep)
    assert calls == ["join", *gather, "mla", "merge"]
    assert md.index_exchange_done is None
    calls.clear()
    prefill(torch.zeros(4, 1), None, md, use_qrep=use_qrep)
    assert calls == [*gather, "mla", "merge"]
