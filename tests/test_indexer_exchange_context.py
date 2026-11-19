# SPDX-License-Identifier: MIT
"""A serial DCP prefill filter must not enter CUDA's default-stream context."""

import ast
from contextlib import contextmanager, nullcontext
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
        "nullcontext": nullcontext,
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
