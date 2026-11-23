# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step watchdog names a long forward by its kind (Mono kernels run only in
decode-only steps, so a long prefill or mixed forward must not read as a Mono
stall), and the runner keeps a GPU-guard-blocked step off Mono-captured graphs."""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from atom.model_engine.engine_core import EngineCore
from atom.model_engine.model_runner import ModelRunner


def _batch(prefill, decode, blocked=False, spec=0):
    return SimpleNamespace(
        total_seqs_num_prefill=prefill,
        total_seqs_num_decode=decode,
        total_seqs_num=prefill + decode,
        num_spec_step=spec,
        mono_blocked=blocked,
    )


def test_forward_kind():
    assert EngineCore._forward_kind(_batch(1, 0)) == "prefill"
    assert EngineCore._forward_kind(_batch(9, 0)) == "prefill"
    assert EngineCore._forward_kind(_batch(2, 3)) == "mixed"
    assert EngineCore._forward_kind(_batch(0, 1)) == "decode"
    assert EngineCore._forward_kind(_batch(0, 4, blocked=True)) == "decode (Mono off)"


def _blocked_step(batch, tokens):
    calls = []
    runner = SimpleNamespace(
        model=SimpleNamespace(
            mono_set_blocked=calls.append, mono_graph_tokens=set(tokens)
        )
    )
    return ModelRunner._mono_blocked_step(runner, batch), calls


def test_blocked_step_runs_eagerly_only_where_a_mono_graph_could_replay():
    assert _blocked_step(_batch(0, 4, blocked=True), {8, 16}) == (True, [True])
    # past the largest Mono graph, or a prefill: no Mono graph replays it
    assert _blocked_step(_batch(0, 20, blocked=True), {8, 16})[0] is False
    assert _blocked_step(_batch(1, 0, blocked=True), {8, 16})[0] is False
    # spec decode counts rows per sequence
    assert _blocked_step(_batch(0, 4, blocked=True, spec=3), {8, 16})[0] is True
    assert _blocked_step(_batch(0, 8, blocked=True, spec=3), {8, 16})[0] is False
    # not blocked: the flag is cleared and nothing is forced eager
    assert _blocked_step(_batch(0, 4), {8, 16}) == (False, [False])
