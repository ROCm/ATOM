# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""An extend segment can legitimately have nothing in front of its queries.

The MHA builder moves a ragged decode segment into the extend path so the
varlen kernels take it (see ``test_minimax_m3_decode_uniformity``). vLLM 0.29's
``profile_cudagraph_memory`` builds attention metadata from
``InputBatch.make_dummy``, whose requests carry ``seq_len == query_len`` and --
when ``num_tokens`` does not divide ``num_reqs`` -- disagreeing query lengths.
Both conditions hold at once, so the extend segment gets requests with zero
preceding KV and ``num_chunks`` is 0.

The producer half of that failed loudly (``max()`` over an empty tensor). This
half would not: ``chunked_output`` stays ``None`` and reaches
``merge_attn_states`` as ``prefix_output``. The property under test is that no
merge is attempted at all -- with no prefix context, the causal pass over the
new tokens is already the complete answer.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

import aiter
import vllm.v1.attention.ops.merge_attn_states as merge_mod

from atom.plugin.vllm.attention import layer_mha
from atom.plugin.vllm.attention.layer_mha import AttentionForVllmMHA

NUM_TOKENS, NUM_HEADS, HEAD_DIM = 6, 2, 8


def _layer():
    layer = object.__new__(AttentionForVllmMHA)
    layer.sliding_window = -1
    layer.scale = 0.125
    layer.sinks = None
    layer.alibi_slopes = None
    layer.kv_cache_dtype = "auto"
    layer.per_token_quant = False
    return layer


def _zero_chunk_metadata():
    """What the builder emits for a segment whose requests all start at 0."""
    empty = torch.zeros(0, dtype=torch.int32)
    return SimpleNamespace(
        extend_metadata=SimpleNamespace(
            chunk_context_metadata=SimpleNamespace(
                num_chunks=0,
                workspace=torch.zeros(2, 0, NUM_HEADS, HEAD_DIM),
                cu_seq_lens_chunk=empty,
                max_seq_lens=[],
                chunk_starts=empty,
                batch_id_per_k_token=empty,
                total_token_per_batch=[],
            )
        )
    )


def test_zero_chunk_extend_returns_the_causal_pass_untouched(monkeypatch):
    causal_out = torch.arange(
        NUM_TOKENS * NUM_HEADS * HEAD_DIM, dtype=torch.float32
    ).reshape(NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    lse = torch.zeros(NUM_HEADS, NUM_TOKENS)
    calls = []

    monkeypatch.setattr(
        aiter,
        "flash_attn_varlen_func",
        lambda **kw: (calls.append(kw) or (causal_out, lse)),
    )

    def _no_merge(**kw):
        raise AssertionError(
            "merge_attn_states must not run with no prefix context; it would "
            f"be handed prefix_output={kw.get('prefix_output')!r}"
        )

    # `extend_forward` imports merge_attn_states inside the function, so the
    # name is looked up on merge_mod per call. Patch layer_mha's namespace too,
    # so that hoisting that import to module scope -- an ordinary tidy-up --
    # cannot turn this assertion into a no-op that still passes.
    monkeypatch.setattr(merge_mod, "merge_attn_states", _no_merge)
    monkeypatch.setattr(layer_mha, "merge_attn_states", _no_merge, raising=False)

    q = torch.zeros(NUM_TOKENS, NUM_HEADS, HEAD_DIM)
    output = torch.full_like(causal_out, -1.0)

    _layer().extend_forward(
        attn_metadata=_zero_chunk_metadata(),
        query=q,
        key=q,
        value=q,
        key_cache=torch.zeros(1),
        value_cache=torch.zeros(1),
        output=output,
        cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
        max_seqlen_q=3,
        max_seqlen_k=3,
        min_seqlen_q=1,
        block_table=torch.zeros(2, 1, dtype=torch.int32),
        slot_mapping=torch.zeros(NUM_TOKENS, dtype=torch.int32),
        k_scale=None,
        v_scale=None,
    )

    assert len(calls) == 1, "the context-gathering pass must be skipped entirely"
    assert calls[0]["causal"] is True
    assert torch.equal(output, causal_out)
