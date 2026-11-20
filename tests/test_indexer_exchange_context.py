# SPDX-License-Identifier: MIT
"""A serial DCP prefill filter must not enter CUDA's default-stream context."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("has_exchange", [False, True])
def test_dcp_filter_enters_a_cuda_context_only_for_a_real_exchange(has_exchange):
    path = (
        Path(__file__).resolve().parents[1]
        / "atom/model_ops/indexer/indexer_forward.py"
    )
    tree = ast.parse(path.read_text())
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Call)
        and isinstance(node.test.left.func, ast.Name)
        and node.test.left.func.id == "get_dcp_world_size"
        and any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "triton_filter_and_convert_dcp_index_prefill"
            for call in ast.walk(node)
        )
    ]
    assert len(branches) == 1
    calls, current, event = [], object(), object()

    class Exchange:
        def wait_stream(self, stream):
            assert stream is current
            calls.append("wait")

        def record_event(self):
            calls.append("event")
            return event

    exchange = Exchange() if has_exchange else None

    @contextmanager
    def stream_context(stream):
        assert stream is not None, "streamless filters must not resolve a CUDA context"
        assert stream is exchange
        calls.append("enter")
        yield
        calls.append("exit")

    def record_stream(stream):
        assert stream is exchange
        calls.append("record")

    metadata = SimpleNamespace(
        sparse_kv_indptr=None,
        batch_id_per_q_token=None,
        block_tables=None,
        dcp_sparse_kv_indptr=None,
        dcp_owned_counts=None,
        sparse_kv_indices=None,
        index_exchange_done=None,
    )
    namespace = {
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(stream=stream_context, current_stream=lambda: current)
        ),
        "nullcontext": lambda: pytest.fail(
            "streamless filters must not create a context"
        ),
        "exchange": exchange,
        "attn_metadata": metadata,
        "topk_indices": SimpleNamespace(record_stream=record_stream),
        "rows": SimpleNamespace(split=None),
        "runner_block_size": 1,
        "topk_tokens": 4,
        "cp_kv_cache_interleave_size": 1,
        "get_dcp_rank": lambda: 0,
        "get_dcp_world_size": lambda: 2,
        "_gather_split": lambda *_: calls.append("gather"),
        "triton_filter_and_convert_dcp_index_prefill": lambda *a, **k: calls.append(
            "filter"
        ),
    }
    exec(  # noqa: S102 - execute the production DCP-filter branch with stream stand-ins
        compile(ast.Module(body=branches[0].body, type_ignores=[]), str(path), "exec"),
        namespace,
    )
    if has_exchange:
        assert calls == ["wait", "record", "enter", "gather", "filter", "exit", "event"]
        assert metadata.index_exchange_done is event
    else:
        assert calls == ["filter"]
        assert metadata.index_exchange_done is None


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
def test_only_dcp_prefill_reads_the_exchange_stream(prefill, dcp, stream, packed):
    path = (
        Path(__file__).resolve().parents[1]
        / "atom/model_ops/indexer/indexer_forward.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "sparse_attn_indexer"
    )
    start = next(
        i
        for i, n in enumerate(function.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "single_rank_prefill" for t in n.targets
        )
    )
    stop = start + next(
        i
        for i, n in enumerate(function.body[start:])
        if isinstance(n, ast.If) and ast.unparse(n.test) == "not context.is_prefill"
    )
    calls, scores = [], []
    side = object() if stream else None
    eligible = prefill and dcp > 1 and not packed

    class Indexer:
        @property
        def exchange_stream(self):
            assert eligible, "unrelated indexer path read the exchange stream"
            calls.append("stream")
            return side

    def exchange(*args):
        calls.append("exchange")
        return side

    ns = {
        "context": SimpleNamespace(is_prefill=prefill),
        "get_dcp_world_size": lambda: dcp,
        "indexer_module": Indexer(),
        "rows": SimpleNamespace(passes=(object(),) if packed else None, out_rows=1),
        "_exchange_stream": exchange,
        "score_topk": lambda *a, **k: scores.append(k),
        "indexer_fp4": False,
        "q_quant": None,
        "weights_mqa": None,
        "kv_cache": None,
        "attn_metadata": None,
        "_index_cache": lambda *a: None,
        "topk_tokens": 1,
        "weights_scale": None,
        "stable_topk": False,
        "topk_indices": [0],
    }
    exec(  # noqa: S102 - execute the production dispatch with stream-access guards
        compile(
            ast.Module(body=function.body[start:stop], type_ignores=[]),
            str(path),
            "exec",
        ),
        ns,
    )
    assert len(scores) == 1
    assert scores[0].get("gather", True) == (not eligible or not stream)
    assert calls == (
        ["stream", "exchange"]
        if eligible and stream
        else ["stream"] if eligible else []
    )
