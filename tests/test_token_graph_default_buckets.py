# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's default prefill token buckets (`ATOM_TOKEN_GRAPH_SIZES` unset).

A DeepSeek-V4.1-Flash TP4 prefill is still host-bound past 3,072 tokens, so
a prefill of 3,073 to 7,168 tokens replays a recorded bucket, padded by
fewer than 512 rows, instead of launching every kernel eagerly; a longer
one still runs eagerly.

Run through the code that decides it: the buckets the runner records at
startup (`ModelRunner._record_tr_graphs`, its capture forwards stubbed) and
the bucket a prefill then pads to (`ForwardMode._pad_to_token_graph`).
"""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from aiter_stub import atom_without_aiter

# `--max-num-batched-tokens`'s default, and the V4.1-Flash recipes' value.
BATCHED_TOKENS = 16384


@pytest.fixture(scope="module")
def load():
    with atom_without_aiter() as load:
        yield load


@pytest.fixture
def recorded(load, monkeypatch):
    """The prefill buckets the runner records at the default, given the
    tokens one warmup forward carries."""
    monkeypatch.delenv("ATOM_TOKEN_GRAPH_SIZES", raising=False)
    model_runner = load("atom.model_engine.model_runner")
    monkeypatch.setattr(model_runner.torch.cuda, "synchronize", lambda: None)

    def record(max_tokens):
        passes = []
        runner = SimpleNamespace(
            attn_metadata_builder=SimpleNamespace(build_for_token_graph_capture=None),
            warmup_max_tokens=lambda: max_tokens,
            running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
            _build_capture=lambda build, tokens: (tokens, None),
            _run_capture_prefill=lambda tokens, context, mode: passes.append(
                (tokens, mode)
            ),
            rank=1,
        )
        model_runner.ModelRunner._record_tr_graphs(runner)
        return tuple(sorted(tokens for tokens, mode in passes if mode == "record"))

    return record


@pytest.fixture
def bucket(load):
    """The bucket a one-rank prefill of ``tokens`` replays, 0 for eager."""
    forward_context = load("atom.utils.forward_context")
    region_graph = load("atom.utils.region_graph")

    def pad(sizes, tokens):
        running = np.array([tokens], dtype=np.int32)
        sync = SimpleNamespace(prefill_across_dp=(True,))
        with mock.patch.object(
            region_graph.TokenGraph,
            "captured_sizes",
            staticmethod(lambda phase=None: sizes if phase == "prefill" else ()),
        ):
            return forward_context.ForwardMode._pad_to_token_graph(True, sync, running)

    return pad


def test_a_prefill_of_3073_to_7168_tokens_replays_a_bucket_under_512_rows_above_it(
    recorded, bucket
):
    sizes = recorded(BATCHED_TOKENS)
    buckets = {n: bucket(sizes, n) for n in range(3073, 7169)}
    eager = [n for n, b in buckets.items() if not b]
    assert not eager, f"prefills of {eager[0]} to {eager[-1]} tokens run eagerly"
    assert max(b - n for n, b in buckets.items()) < 512


def test_the_default_keeps_its_buckets_through_3072(recorded):
    assert recorded(BATCHED_TOKENS)[:5] == (256, 512, 1024, 2048, 3072)


def test_a_prefill_past_7168_tokens_runs_eagerly(recorded, bucket):
    assert bucket(recorded(BATCHED_TOKENS), 7169) == 0


def test_a_smaller_token_budget_records_the_default_buckets_up_to_it(recorded):
    """`--max-num-batched-tokens 4096`: no bucket a warmup forward cannot
    carry is recorded."""
    assert recorded(4096) == (256, 512, 1024, 2048, 3072, 3584, 4096)
