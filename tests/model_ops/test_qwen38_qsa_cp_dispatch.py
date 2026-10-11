# SPDX-License-Identifier: MIT
"""Measured QSA index context-partition decode gates."""

from types import SimpleNamespace

import pytest
import torch

import atom.distributed.qwen_qsa_cp as qsa_cp


@pytest.mark.parametrize(
    "context,tp,below_gate,on_gate",
    [
        (131072, 4, 15, 16),
        (131072, 2, 47, 48),
        (262144, 4, 1, 2),
        (262144, 2, 15, 16),
    ],
)
def test_decode_gate_uses_measured_context_bucket(
    monkeypatch, context, tp, below_gate, on_gate
):
    monkeypatch.setenv("ATOM_QWEN38_QSA_CP", "1")
    group = SimpleNamespace(world_size=tp)
    monkeypatch.setattr(qsa_cp, "get_qsa_cp_group", lambda: group)
    choose = lambda rows: qsa_cp.get_qsa_cp_decode_group(
        rows=rows, max_seq_len=context, tp_size=tp
    )
    assert choose(below_gate) is None
    assert choose(on_gate) is group


def test_decode_gate_does_not_exchange_short_context_or_when_disabled(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN38_QSA_CP", "1")
    monkeypatch.setattr(
        qsa_cp,
        "get_qsa_cp_group",
        lambda: pytest.fail("the fallback must not open a process group"),
    )
    assert (
        qsa_cp.get_qsa_cp_decode_group(rows=128, max_seq_len=65536, tp_size=4) is None
    )
    monkeypatch.setenv("ATOM_QWEN38_QSA_CP", "0")
    assert (
        qsa_cp.get_qsa_cp_decode_group(rows=128, max_seq_len=262144, tp_size=4) is None
    )


def test_plugin_mode_never_uses_native_qsa_exchange(monkeypatch):
    monkeypatch.setenv("ATOM_QWEN38_QSA_CP", "1")
    monkeypatch.setattr(qsa_cp, "is_plugin_mode", lambda: True)
    monkeypatch.setattr(
        qsa_cp,
        "get_qsa_cp_group",
        lambda: pytest.fail("plugin mode must not open a native TP exchange"),
    )
    assert (
        qsa_cp.get_qsa_cp_decode_group(rows=128, max_seq_len=262144, tp_size=4) is None
    )


@pytest.mark.parametrize(
    "pages,token_topk",
    [
        (5, 64),  # Uneven page count across TP4.
        (4, 128),  # Each page shard has fewer groups than the global top-k.
    ],
)
def test_index_cp_falls_back_when_shards_are_invalid(pages, token_topk):
    if not torch.cuda.is_available():
        pytest.skip("needs a ROCm GPU")
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_select_paged_tokens

    context = pages * 64
    q = torch.randn(1, 4, 128, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(pages, 16, 1, 128, device="cuda", dtype=torch.bfloat16)
    table = torch.randperm(pages, device="cuda").to(torch.int32)[None]
    requests = torch.zeros(1, device="cuda", dtype=torch.int32)
    positions = torch.tensor([context - 1], device="cuda", dtype=torch.int64)
    lengths = torch.tensor([context], device="cuda", dtype=torch.int32)

    def select(index_cp):
        return qsa_select_paged_tokens(
            q,
            cache,
            table,
            requests,
            positions,
            lengths,
            token_topk,
            max_seq_len=context,
            index_cp=index_cp,
        )

    torch.testing.assert_close(select((0, 4, None)), select(None), atol=0, rtol=0)
    select((0, 4, None))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = select((0, 4, None))
    q.copy_(torch.randn_like(q))
    cache.copy_(torch.randn_like(cache))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, select(None), atol=0, rtol=0)
