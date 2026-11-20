# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill token graphs, from the attention builder's side: the
deployments that record them, the dummy prefill a bucket is recorded on and
which live prefills still run eagerly.

The builders are made with ``__new__`` and given only the state each method
reads, so no device is needed; their tests still need triton and aiter, which
the builder's module imports."""

from types import SimpleNamespace

import pytest
import torch

_BUDGET = 8192


@pytest.fixture
def aa():
    pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
    pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")
    return pytest.importorskip("atom.model_ops.attentions.aiter_attention")


def _builder(aa, sparse=True, dp=1, sp=1, pcp=1):
    builder = aa.AiterAttentionMetadataBuilder.__new__(aa.AiterAttentionMetadataBuilder)
    builder._has_sparse_attention = sparse
    builder.model_runner = SimpleNamespace(
        config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=dp),
            sequence_parallel_size=sp,
            prefill_context_parallel_size=pcp,
            max_num_batched_tokens=_BUDGET,
        ),
        forward_vars={"positions": SimpleNamespace(gpu=torch.arange(_BUDGET))},
    )
    return builder


@pytest.mark.parametrize(
    "builder, enabled",
    [
        ({}, True),
        ({"dp": 2}, False),
        ({"sp": 2}, False),
        ({"pcp": 2}, False),
        # Every other model this builder serves has no sparse attention.
        ({"sparse": False}, False),
    ],
)
def test_token_graphs_are_on_for_m3_outside_dp_sp_and_pcp(aa, builder, enabled):
    assert _builder(aa, **builder).supports_token_graph is enabled


@pytest.mark.parametrize("tokens", [1, 256, _BUDGET])
def test_a_bucket_records_on_a_dummy_prefill_of_its_height(aa, tokens):
    builder = _builder(aa)
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
def test_a_bucket_outside_the_token_budget_is_refused(aa, tokens):
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        _builder(aa).build_for_token_graph_capture(tokens)


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
def test_a_prefill_carrying_images_stays_eager(
    aa, enabled, prefill_tokens, images, eager
):
    batch = SimpleNamespace(
        total_tokens_num_prefill=prefill_tokens, multimodal_data=images
    )
    assert _builder(aa, sparse=enabled).step_needs_eager(batch) is eager
