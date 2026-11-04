# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The MHA token-graph break written into the next segment's buffer
(`unified_attention_into_output`): a recorded forward replays bitwise to
what the eager forward computes through `unified_attention_with_output_base`,
with no replay-time copy of the attention result, and the prefill
unified-attention path writes the caller's buffer instead of a fresh one."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops import base_attention, paged_attention
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.utils import forward_context, region_graph
from atom.utils.region_graph import _EagerStep, _record_segments

HEADS, HEAD_DIM, TOKENS = 4, 8, 64
WIDTH = HEADS * HEAD_DIM
LAYER = "test_into_output.attn"


class _FakeImpl:
    """Stands in for `PagedAttentionImpl`: host-sized work (only the
    scheduled rows) whose values depend on the live step, written into
    ``output`` when ``writes`` and handed in, else into a fresh tensor whose
    pad rows hold garbage, as the real prefill paths do."""

    accepts_output = True
    num_heads, head_dim = HEADS, HEAD_DIM

    def __init__(self, writes: bool):
        self.writes = writes
        self.scale = 1.0

    def forward(self, query, key, value, position, q_scale, qkv, output=None):
        n = forward_context.get_forward_context().context.scheduled_tokens
        if output is not None and self.writes:
            o = output
        else:
            o = torch.full_like(query, float("nan"))
        torch.addcmul(key[:n], query[:n], value[:n] * self.scale, out=o[:n])
        return o


class _Layer(torch.nn.Module):
    def __init__(self, writes: bool):
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, 3 * WIDTH, device="cuda")
        self.impl = _FakeImpl(writes)
        # What `paged_attention.Attention.forward` reads off its module.
        self.attn = SimpleNamespace(use_mla=False, impl=self.impl, layer_name=LAYER)

    def forward(self, x):
        q, k, v = self.proj(x).split(WIDTH, dim=-1)
        o = paged_attention.Attention.forward(self.attn, q, k, v, positions=None)
        return o * 2.0 + 1.0


@pytest.fixture
def step(monkeypatch):
    """The forward context and the config the ops look the layer up in."""
    if not torch.cuda.is_available():
        pytest.skip("region graphs record CUDA graphs")
    ctx = SimpleNamespace(
        in_hipgraph=False,
        main_stream=torch.cuda.current_stream(),
        context=SimpleNamespace(
            is_prefill=True, is_dummy_run=False, scheduled_tokens=TOKENS
        ),
    )
    monkeypatch.setattr(forward_context, "_forward_context", ctx)
    static = {}
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=static),
        torch_dtype=torch.float32,
    )
    for module in (base_attention, paged_attention):
        monkeypatch.setattr(module, "get_current_atom_config", lambda: config)
    return ctx, static


def _record(layer, x):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        layer(x)  # warmup of the same shapes
        segments, out = _record_segments(layer, x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return segments, out


@pytest.mark.parametrize("writes", [True, False], ids=["into_output", "fallback"])
@torch.inference_mode()
def test_recorded_attention_replays_bitwise_without_a_replay_copy(step, writes):
    ctx, static = step
    torch.manual_seed(0)
    layer = _Layer(writes)
    static[LAYER] = layer
    x = torch.randn(TOKENS, WIDTH, device="cuda")
    segments, out = _record(layer, x)

    regions = [s for s in segments._steps if isinstance(s, _EagerStep)]
    assert len(regions) == 1
    # The region wrote an input of the next segment: nothing to copy at replay.
    assert regions[0]._target is None

    for scale, rows in ((1.0, TOKENS), (0.5, 37), (2.0, 1), (-3.0, TOKENS - 1)):
        x.copy_(torch.randn_like(x))
        layer.impl.scale = scale
        ctx.context.scheduled_tokens = rows
        replayed = segments.replay().clone()
        torch.cuda.synchronize()
        # The eager forward, not recording, takes the returning op.
        assert not region_graph.recording()
        eager = layer(x)
        assert torch.equal(replayed, eager)
        assert torch.equal(out, replayed)
        # Pad rows: the zeroed attention, through `o * 2 + 1`.
        assert torch.equal(replayed[rows:], torch.ones_like(replayed[rows:]))


@torch.inference_mode()
def test_returning_op_recording_copies_at_replay_and_matches(step, monkeypatch):
    """The recording before this change: the region's fresh result reaches
    the next segment through a replay-time copy, with the same values."""
    ctx, static = step
    torch.manual_seed(0)
    layer = _Layer(writes=True)
    static[LAYER] = layer
    x = torch.randn(TOKENS, WIDTH, device="cuda")
    new, _ = _record(layer, x)
    monkeypatch.setattr(region_graph, "recording", lambda: False)
    old, _ = _record(layer, x)
    (old_region,) = [s for s in old._steps if isinstance(s, _EagerStep)]
    assert old_region._target is not None

    for scale, rows in ((0.25, 50), (1.5, TOKENS)):
        x.copy_(torch.randn_like(x))
        layer.impl.scale = scale
        ctx.context.scheduled_tokens = rows
        a = new.replay().clone()
        b = old.replay().clone()
        torch.cuda.synchronize()
        assert torch.equal(a, b)


def _triton_impl(**overrides):
    attrs = {
        "kv_cache_dtype": "bf16",
        "sliding_window": -1,
        "use_flash_layout": True,
        "k4v8": False,
        "kv_scale": None,
        "scale": 1.0,
        "sinks": None,
    }
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


@pytest.mark.parametrize("rows", [TOKENS, 21])
def test_prefill_unified_attention_writes_the_callers_buffer(monkeypatch, rows):
    """`prefill_attention_triton` with ``out``: the kernel writes ``out``
    itself, the scheduled rows hold what the fresh-output call computes, and
    the pad rows are left to the caller (which zeroes them)."""
    from atom.model_ops import attention_mha

    def fake_unified_attention(q, k, v, out, **kwargs):
        out.copy_(q * 3.0 + 1.0)

    monkeypatch.setattr(attention_mha, "unified_attention", fake_unified_attention)
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    fwd_ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(
            has_cached=False,
            cu_seqlens_q=torch.zeros(2, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            max_seqlen_q=rows,
            max_seqlen_k=rows,
            block_tables=None,
        ),
        context=SimpleNamespace(scheduled_bs=1, scheduled_tokens=rows),
    )
    q = torch.randn(TOKENS, HEADS, HEAD_DIM, dtype=torch.bfloat16)
    k = torch.randn(TOKENS, 1, HEAD_DIM, dtype=torch.bfloat16)
    impl = _triton_impl()
    fresh = PagedAttentionImpl.prefill_attention_triton(
        impl, q, k, k, None, None, None, None, fwd_ctx
    )
    sentinel = torch.full((TOKENS, WIDTH), 7.0, dtype=torch.bfloat16)
    written = PagedAttentionImpl.prefill_attention_triton(
        impl, q, k, k, None, None, None, None, fwd_ctx, out=sentinel
    )
    assert written.data_ptr() == sentinel.data_ptr()
    assert torch.equal(written[:rows], fresh[:rows])
    assert torch.equal(fresh[rows:], torch.zeros_like(fresh[rows:]))
    # Pad rows untouched: `unified_attention_into_output` zeroes them.
    assert torch.equal(
        sentinel.view(written.shape)[rows:], torch.full_like(written[rows:], 7.0)
    )

    # A buffer of another dtype is not written; a fresh one is returned.
    wrong = torch.zeros(TOKENS, WIDTH, dtype=torch.float32)
    other = PagedAttentionImpl.prefill_attention_triton(
        impl, q, k, k, None, None, None, None, fwd_ctx, out=wrong
    )
    assert other.data_ptr() != wrong.data_ptr()
    assert torch.equal(other, fresh)
    assert not wrong.any()
