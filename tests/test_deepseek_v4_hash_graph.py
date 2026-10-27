# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 hash routing must consume the current DP IDs across target replays."""

from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA graph capture"
)


@torch.inference_mode()
def test_hash_ids_refresh_after_prefill_and_across_decode_buckets(monkeypatch):
    from atom.models import deepseek_v4

    monkeypatch.setattr(deepseek_v4, "_pcp_active", lambda: False)
    calls = []
    caller_ids = []

    class HashReader(torch.nn.Module):
        def forward(self, input_ids, positions):
            calls.append(input_ids.data_ptr())
            ids = forward_context.get_forward_context().context.input_ids
            # A small token-to-expert mapping stands in for the real hash layers.
            return torch.stack((ids, ids.remainder(13)), dim=1)

    # Exercise the real decorated wrapper and its forward without loading the
    # full checkpoint. The downstream hash reader is the only fake operation.
    target = deepseek_v4.DeepseekV4ForCausalLM.__new__(
        deepseek_v4.DeepseekV4ForCausalLM
    )
    torch.nn.Module.__init__(target)
    target.model = HashReader()

    def context(mode=None, prefill=False, stream=None):
        ctx = SimpleNamespace(
            context=SimpleNamespace(is_prefill=prefill),
            tr_graph=mode,
            in_hipgraph=mode == "record",
            main_stream=stream or torch.cuda.current_stream(),
        )
        monkeypatch.setattr(forward_context, "_forward_context", ctx)

    def inputs(rows, offset):
        ids = torch.arange(rows, device="cuda", dtype=torch.int32) + offset
        positions = torch.arange(rows, device="cuda", dtype=torch.int64)
        # Keep older caller allocations live: changing the published context
        # must not depend on allocator reuse refreshing a captured address.
        caller_ids.append(ids)
        return ids, positions

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for rows in (32, 8):
            ids, positions = inputs(rows, 0)
            context(stream=stream)
            target(ids, positions)
            context("record", stream=stream)
            target(ids, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = target.__dict__["_token_graph"]
    try:
        for rows, offset in ((32, 101), (8, 209), (32, 307)):
            # A real prefill replaces the forward context and caller's input-ID
            # allocation between decode replays.
            context(prefill=True)
            target(*inputs(19, offset + 1000))
            eager_calls = len(calls)
            ids, positions = inputs(rows, offset)
            context("replay")
            actual = target(ids, positions).clone()
            expected = torch.stack((ids, ids.remainder(13)), dim=1)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert len(calls) == eager_calls, "the decode must replay its graph"
    finally:
        graph.release_graphs()
