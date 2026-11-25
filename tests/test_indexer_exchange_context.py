# SPDX-License-Identifier: MIT
"""Only an eligible DCP prefill enters a stream context for its exchange."""

import ast
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "atom/model_ops/indexer/indexer_forward.py"


def _production(namespace, *names):
    tree = ast.parse(PATH.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(  # noqa: S102 - production bodies with CPU boundaries
        compile(ast.Module(body=nodes, type_ignores=[]), str(PATH), "exec"), namespace
    )
    return namespace


@pytest.mark.parametrize("capture,tbo", [(False, False), (True, False), (False, True)])
def test_exchange_guard_keeps_capture_and_tbo_serial(capture, tbo):
    ns = _production(
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(is_current_stream_capturing=lambda: capture)
            ),
            "tbo_active": lambda: tbo,
        },
        "_exchange_stream",
    )
    stream = object()
    assert ns["_exchange_stream"](stream) is (None if capture or tbo else stream)


@pytest.mark.parametrize("fp4", [False, True])
def test_exchange_scores_local_rows_then_gathers_and_filters_on_its_stream(fp4):
    calls, current, event = [], object(), object()

    class Exchange:
        def wait_stream(self, stream):
            assert stream is current
            calls.append("wait")

        def record_event(self):
            calls.append("event")
            return event

    exchange = Exchange()

    @contextmanager
    def context(stream):
        assert stream is exchange
        calls.append("enter")
        yield
        calls.append("exit")

    Split = namedtuple("Split", "world stripe")
    Rows = namedtuple("Rows", "split out_rows")
    rows = Rows(Split(2, False), 8)
    metadata = SimpleNamespace(
        index_rows=rows,
        sparse_kv_indptr=None,
        batch_id_per_q_token=None,
        block_tables=None,
        dcp_sparse_kv_indptr=None,
        dcp_owned_counts=None,
        sparse_kv_indices=None,
        index_exchange_done=None,
    )

    class Selection:
        def __getitem__(self, item):
            assert item == slice(None, 8)
            return self

        def record_stream(self, stream):
            assert stream is exchange
            calls.append("record")

    out = Selection()

    def score(query, weights, cache, local_rows, **kwargs):
        assert local_rows.split.world == 1 and rows.split.world == 2
        assert "gather" not in kwargs and kwargs["out"] is out
        calls.append("score")
        return out

    def gather(output, split):
        assert output is out and split is rows.split
        calls.append("gather")

    ns = _production(
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(current_stream=lambda: current, stream=context)
            ),
            "score_topk": score,
            "_index_cache": lambda *a: None,
            "_gather_split": gather,
            "triton_filter_and_convert_dcp_index_prefill": lambda *a, **k: calls.append(
                "filter"
            ),
            "get_dcp_rank": lambda: 0,
            "get_dcp_world_size": lambda: 2,
        },
        "_dcp_prefill_exchange",
    )
    ns["_dcp_prefill_exchange"](
        exchange,
        SimpleNamespace(_indexer_fp4=fp4),
        torch.zeros(7, 2),
        torch.ones(7, 1),
        None,
        None,
        metadata,
        out,
        4,
        1.0,
        False,
        1,
        1,
    )
    assert calls == [
        "score",
        "wait",
        "record",
        "enter",
        "gather",
        "filter",
        "exit",
        "event",
    ]
    assert metadata.index_exchange_done is event


@pytest.mark.parametrize(
    "prefill,dcp,stream,packed",
    [
        (True, 1, True, False),
        (False, 1, True, False),
        (True, 2, False, False),
        (True, 2, True, False),
        (True, 2, True, True),
    ],
)
@pytest.mark.parametrize("kv", [False, True])
def test_only_eligible_prefill_dispatches_the_exchange(
    prefill, dcp, stream, packed, kv
):
    tree = ast.parse(PATH.read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer"
    )
    start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "rows" for t in n.targets)
    )
    fn.body = fn.body[start:]
    fn.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg="topk_indices")],
        kwonlyargs=[],
        kw_defaults=[],
        defaults=[],
    )
    fn.returns, fn.decorator_list = None, []
    calls, scores = [], []
    side = object() if stream else None
    rows = SimpleNamespace(
        passes=(object(),) if packed else None, out_rows=1, concat=None
    )
    metadata = SimpleNamespace(
        index_rows=rows,
        max_seqlen_q=1,
        cu_seqlens_q=None,
        kv_indptr=None,
        sparse_kv_indptr=None,
        kv_indices=None,
        sparse_kv_indices=None,
        batch_id_per_q_token=None,
        block_tables=None,
        dcp_sparse_kv_indptr=None,
        dcp_owned_counts=None,
        sparse_cu_seqlens_q=None,
        cu_seqlens_k=None,
        dcp_sparse_prefill_kv=object() if kv and prefill and dcp > 1 else None,
    )
    eligible = prefill and dcp > 1 and stream and not packed and not kv

    def exchange(value):
        assert eligible and value is side
        calls.append("guard")
        return side

    ns = {
        "context": SimpleNamespace(is_prefill=prefill),
        "get_dcp_world_size": lambda: dcp,
        "get_dcp_rank": lambda: 0,
        "indexer_module": SimpleNamespace(exchange_stream=side),
        "_exchange_stream": exchange,
        "_dcp_prefill_exchange": lambda *a: calls.append("exchange"),
        "score_topk": lambda *a, **k: scores.append(k) or [0],
        "indexer_fp4": False,
        "q_quant": None,
        "q_fp4_scale": None,
        "weights_mqa": None,
        "kv_cache": None,
        "attn_metadata": metadata,
        "_index_cache": lambda *a: None,
        "topk_tokens": 1,
        "weights_scale": 1,
        "stable_topk": False,
        "topk_indices": [0],
        "runner_block_size": 1,
        "cp_kv_cache_interleave_size": 1,
        "triton_convert_req_index_to_global_index": lambda *a, **k: calls.append(
            "decode"
        ),
        "triton_filter_and_convert_dcp_index_prefill": lambda *a, **k: calls.append(
            "filter"
        ),
        "triton_dcp_sparse_prefill_kv_index": lambda *a, **k: calls.append("kv"),
        "triton_convert_req_index_to_global_index_dsa_prefill": lambda *a, **k: calls.append(
            "convert"
        ),
    }
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    exec(  # noqa: S102 - production dispatch tail
        compile(module, str(PATH), "exec"), ns
    )
    ns["sparse_attn_indexer"](ns["topk_indices"])
    assert calls == (
        ["guard", "exchange"]
        if eligible
        else [
            (
                "decode"
                if not prefill
                else ("kv" if kv else "filter") if dcp > 1 else "convert"
            )
        ]
    )
    assert len(scores) == (0 if eligible else 1)
    if scores:
        assert "gather" not in scores[0]
