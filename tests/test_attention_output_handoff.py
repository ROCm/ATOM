# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A token-graph replay of the generic attention op hands MHA the rows the
next segment reads, and the BF16 varlen prefill writes them in place: no
padded copy, no copy into the target. Checked on the CPU with the attention
kernels replaced; triton and aiter must still import."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton", reason="base_attention defines @triton.jit kernels")
pytest.importorskip("aiter", reason="base_attention imports the AITER runtime")

from atom.model_ops import attention_mha, base_attention
from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.utils import forward_context, region_graph

BUCKET, ROWS, HEADS, HEAD_DIM = 16, 11, 2, 8
WIDTH = HEADS * HEAD_DIM


class _Impl:
    def __init__(self):
        self.calls = []

    def forward(self, query, key, value, position, q_scale, qkv, output=None):
        self.calls.append(SimpleNamespace(query=query, position=position, out=output))
        return torch.mul(query, 2.0, out=output)


@pytest.fixture
def prefill(monkeypatch):
    impl = _Impl()
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": SimpleNamespace(impl=impl)}
        )
    )
    monkeypatch.setattr(base_attention, "get_current_atom_config", lambda: config)
    monkeypatch.setattr(
        forward_context,
        "_forward_context",
        SimpleNamespace(
            context=SimpleNamespace(is_prefill=True, scheduled_tokens=ROWS)
        ),
    )
    return impl


def _attend(target=None):
    """The op's body as `_EagerStep` replays it, delivering to ``target``."""
    qkv = torch.randn(BUCKET, 2 * WIDTH)
    q, kv = qkv.split([WIDTH, WIDTH], dim=-1)
    region_graph._local.delivery = target
    try:
        got = base_attention.unified_attention_with_output_base.__wrapped__(
            q, None, kv, kv, torch.arange(BUCKET), "layer", False, qkv
        )
    finally:
        region_graph._local.delivery = None
    return q, got


def test_a_replayed_mha_prefill_attends_into_the_next_segments_rows(prefill):
    target = torch.full((BUCKET, WIDTH), float("nan"))
    q, got = _attend(target)

    assert got is target
    (call,) = prefill.calls
    assert call.query.shape[0] == ROWS and call.position.shape == (ROWS,)
    assert call.out.data_ptr() == target.data_ptr()
    assert call.out.shape == (ROWS, WIDTH)
    torch.testing.assert_close(got[:ROWS], q[:ROWS] * 2.0, rtol=0, atol=0)
    assert torch.equal(got[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))


def test_an_eager_mha_prefill_still_pads_its_own_result(prefill):
    q, got = _attend()

    (call,) = prefill.calls
    assert call.out is None
    assert got.shape == (BUCKET, WIDTH)
    torch.testing.assert_close(got[:ROWS], q[:ROWS] * 2.0, rtol=0, atol=0)
    assert torch.equal(got[ROWS:], torch.zeros(BUCKET - ROWS, WIDTH))


def _paged_impl(monkeypatch, is_prefill=True, is_dummy_run=False):
    impl = PagedAttentionImpl.__new__(PagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.num_heads, impl.num_kv_heads, impl.head_dim = HEADS, 1, HEAD_DIM
    impl.scale, impl.sinks, impl.sliding_window = HEAD_DIM**-0.5, None, -1
    impl.use_flash_layout = False
    impl.rope_cache = lambda q, k, v, qkv, position, ctx: (
        q,
        k,
        v,
        None,
        None,
        None,
        None,
    )
    impl._can_use_fp8_prefill_attention = lambda *args: False
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_prefill=is_prefill, is_dummy_run=is_dummy_run),
        attn_metadata=SimpleNamespace(
            has_cached=False,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            max_seqlen_q=ROWS,
            max_seqlen_k=ROWS,
            min_seqlen_q=0,
            dropout_p=0.0,
        ),
    )
    monkeypatch.setattr(attention_mha, "get_forward_context", lambda: ctx)
    monkeypatch.setattr(attention_mha.envs, "ATOM_USE_UNIFIED_ATTN", False)
    monkeypatch.setattr(attention_mha.envs, "ATOM_FORCE_ATTN_TRITON", False)
    seen = {}

    def varlen(q, k, v, out=None, **kwargs):
        seen["out"] = out
        return torch.mul(q, 2.0, out=out)

    monkeypatch.setattr(attention_mha.aiter, "flash_attn_varlen_func", varlen)
    return impl, seen


def _rows():
    return (
        torch.randn(ROWS, WIDTH),
        torch.randn(ROWS, HEAD_DIM),
        torch.randn(ROWS, HEAD_DIM),
    )


def test_the_bf16_varlen_prefill_writes_the_given_output(monkeypatch):
    impl, seen = _paged_impl(monkeypatch)
    q, k, v = _rows()
    output = torch.full((ROWS, WIDTH), float("nan"))

    got = impl.forward(q, k, v, position=torch.arange(ROWS), output=output)

    assert seen["out"].data_ptr() == output.data_ptr()
    assert seen["out"].shape == (ROWS, HEADS, HEAD_DIM)
    assert got.data_ptr() == output.data_ptr() and got.shape == (ROWS, WIDTH)
    torch.testing.assert_close(output, q * 2.0, rtol=0, atol=0)


def test_a_prefill_without_output_allocates_as_before(monkeypatch):
    impl, seen = _paged_impl(monkeypatch)
    q, k, v = _rows()

    got = impl.forward(q, k, v, position=torch.arange(ROWS))

    assert seen["out"] is None
    torch.testing.assert_close(got, q * 2.0, rtol=0, atol=0)


def test_another_backend_returns_its_own_result(monkeypatch):
    impl, seen = _paged_impl(monkeypatch, is_prefill=False)
    calls = []

    def decode(q, k, v, k_cache, v_cache, k_scale, v_scale, fwd_ctx):
        calls.append(q)
        return q * 3.0

    impl.dispatch_backend = lambda ctx, q, k, v: decode
    q, k, v = _rows()
    output = torch.zeros(ROWS, WIDTH)

    got = impl.forward(q, k, v, position=torch.arange(ROWS), output=output)

    assert calls and "out" not in seen
    assert got.data_ptr() != output.data_ptr()
    torch.testing.assert_close(got, q * 3.0, rtol=0, atol=0)


def test_a_dummy_run_hands_back_the_given_output(monkeypatch):
    impl, _ = _paged_impl(monkeypatch, is_dummy_run=True)
    q, k, v = _rows()
    output = torch.zeros(ROWS, WIDTH)
    got = impl.forward(q, k, v, position=torch.arange(ROWS), output=output)
    assert got is output
