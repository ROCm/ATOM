# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the vLLM padding adapter without importing vLLM or GPU kernels.

Compile the production override against a recording base, as in the other
CPU contract tests. This also runs in CI, which excludes tests/plugin.
"""

import ast
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

_SOURCE = Path(__file__).resolve().parents[1] / "atom/plugin/vllm/models/deepseek_v4.py"
_DECODE = object()


class _RecordingAttention:
    def _sparse_attention(self, qkn, positions, *args, index_topk=None):
        self.received = (qkn, positions, index_topk)
        # Return supplied indices to check forwarding and output padding.
        return qkn.q_sa if index_topk is None else index_topk


def _load_attention(fc):
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DeepseekV4AttentionVllm"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_sparse_attention"
    ]
    namespace = {
        "torch": torch,
        "dataclasses": dataclasses,
        "DeepseekV4AttentionBase": _RecordingAttention,
        "get_forward_context": lambda: fc,
        "AttnState": SimpleNamespace(DECODE=_DECODE),
    }
    exec(  # noqa: S102 - execute the production adapter without GPU imports
        compile(ast.Module(body=[cls], type_ignores=[]), str(_SOURCE), "exec"),
        namespace,
    )
    return namespace["DeepseekV4AttentionVllm"]()


@dataclasses.dataclass
class _QKN:
    q_sa: torch.Tensor
    kv: torch.Tensor | None = None


@pytest.mark.parametrize("with_topk", [False, True])
@pytest.mark.parametrize(
    "mode,num_real,expected_rows",
    [
        ("prefill", 5, 5),
        ("prefill", 8, 8),
        ("decode", 5, 8),
        ("dummy", 5, 8),
        ("no_metadata", 5, 8),
        ("no_batch_ids", 5, 8),
    ],
)
def test_sparse_attention_forwards_topk_with_matching_rows(
    mode, num_real, expected_rows, with_topk
):
    md = SimpleNamespace(
        state=_DECODE if mode == "decode" else object(),
        batch_id_per_q_token=(
            None if mode == "no_batch_ids" else torch.zeros(num_real)
        ),
    )
    fc = SimpleNamespace(
        context=SimpleNamespace(is_dummy_run=mode == "dummy"),
        attn_metadata=None if mode == "no_metadata" else md,
    )
    attention = _load_attention(fc)
    qkn = _QKN(torch.arange(16).reshape(8, 2))
    positions = torch.arange(8)
    topk = torch.arange(24, dtype=torch.int32).reshape(8, 3) if with_topk else None
    out = attention._sparse_attention(qkn, positions, index_topk=topk)

    received_qkn, received_positions, received_topk = attention.received
    torch.testing.assert_close(received_qkn.q_sa, qkn.q_sa[:expected_rows])
    torch.testing.assert_close(received_positions, positions[:expected_rows])
    assert received_qkn.kv is None
    if topk is None:
        assert received_topk is None
    else:
        torch.testing.assert_close(received_topk, topk[:expected_rows])
        assert received_topk.data_ptr() == topk.data_ptr()
        if expected_rows == 8:
            assert received_topk is topk

    expected = (qkn.q_sa if topk is None else topk).clone()
    expected[expected_rows:] = 0
    torch.testing.assert_close(out, expected)
