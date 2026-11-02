# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill token graphs from the attention builder's side: what
turns them on, the dummy prefill a bucket is recorded on, and which live
prefills still run eagerly.

The builders are made with ``__new__`` and given only the state each method
reads, so no device is needed. triton and aiter must still import, because the
builder's module pulls both."""

import logging
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

from atom.model_ops.attentions import aiter_attention as aa

_BUDGET = 8192


def _builder(enabled=True):
    builder = aa.AiterAttentionMetadataBuilder.__new__(aa.AiterAttentionMetadataBuilder)
    builder._m3_prefill_token_graph = enabled
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(max_num_batched_tokens=_BUDGET),
        forward_vars={"positions": SimpleNamespace(gpu=torch.arange(_BUDGET))},
    )
    return builder


_DP2 = "data_parallel_size must be 1, got 2"


def _gate(monkeypatch, env, refusal):
    monkeypatch.setattr(aa.envs, "ATOM_MINIMAX_M3_PREFILL_TOKEN_GRAPH", env)
    monkeypatch.setattr(
        aa, "minimax_m3_prefill_token_graph_refusal", lambda config: refusal
    )


@pytest.mark.parametrize(
    "sparse, env, refusal, enabled",
    [
        (True, True, None, True),
        (True, False, None, False),
        (True, True, _DP2, False),
        # Every other model this builder serves has no sparse attention.
        (False, True, None, False),
    ],
)
def test_token_graphs_need_m3_the_env_and_an_admitted_deployment(
    monkeypatch, sparse, env, refusal, enabled
):
    _gate(monkeypatch, env, refusal)
    assert aa._m3_prefill_token_graph_enabled(sparse, None) is enabled


def test_a_refused_deployment_says_why_it_stays_eager(monkeypatch, caplog):
    _gate(monkeypatch, True, _DP2)
    with caplog.at_level(logging.WARNING, logger="atom"):
        aa._m3_prefill_token_graph_enabled(True, None)
    assert f"refused: {_DP2}" in caplog.text


def test_only_an_enabled_builder_supports_token_graphs():
    assert _builder(enabled=True).supports_token_graph
    assert not _builder(enabled=False).supports_token_graph
    # A subclass that never ran this builder's __init__.
    bare = aa.AiterAttentionMetadataBuilder.__new__(aa.AiterAttentionMetadataBuilder)
    assert not bare.supports_token_graph


@pytest.mark.parametrize("tokens", [1, 256, _BUDGET])
def test_a_bucket_records_on_a_dummy_prefill_of_its_height(tokens):
    builder = _builder()
    attn_metadata, context = builder.build_for_token_graph_capture(tokens)

    assert context.is_prefill and context.is_dummy_run
    assert (context.scheduled_bs, context.running_bs) == (1, 1)
    assert (context.scheduled_tokens, context.running_tokens) == (tokens, tokens)
    assert (attn_metadata.max_seqlen_q, attn_metadata.max_seqlen_k) == (
        tokens,
        tokens,
    )
    # The recording reads the persistent positions a live step fills.
    buffer = builder.model_runner.forward_vars["positions"].gpu
    assert context.positions.data_ptr() == buffer.data_ptr()
    assert context.positions.shape == (tokens,) and not context.positions.any()


@pytest.mark.parametrize("tokens", [0, _BUDGET + 1])
def test_a_bucket_outside_the_token_budget_is_refused(tokens):
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _builder().build_for_token_graph_capture(tokens)


@pytest.mark.parametrize(
    "enabled, prefill_tokens, images, eager",
    [
        (True, 512, {7: {"pixel_values": None}}, True),
        (True, 512, {}, False),
        # Decode and a builder without token graphs keep the base answer.
        (True, 0, {7: {"pixel_values": None}}, False),
        (False, 512, {7: {"pixel_values": None}}, False),
    ],
)
def test_a_prefill_carrying_images_stays_eager(enabled, prefill_tokens, images, eager):
    batch = SimpleNamespace(
        total_tokens_num_prefill=prefill_tokens, multimodal_data=images
    )
    assert _builder(enabled).step_needs_eager(batch) is eager
