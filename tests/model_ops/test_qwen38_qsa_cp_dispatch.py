# SPDX-License-Identifier: MIT
"""Measured QSA index context-partition decode gates."""

from types import SimpleNamespace

import pytest

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
