# SPDX-License-Identifier: MIT
"""The fp8 cache write of the plain `rope_cache` branch under unified attention.

With an fp8 cache Qwen3.5 leaves the fused Gluon norm + MRoPE + cache kernel
(bf16 only): the model runs the norm and the MRoPE and calls `attn(q, k, v)`,
so `rope_cache` takes its last branch. That branch wrote every fp8 cache with
per-token scales, which `unified_attention` refuses (one descale for the whole
tensor), so ATOM_USE_UNIFIED_ATTN=1 with --kv_cache_dtype fp8 raised on the
first prefill. Under unified it now writes per tensor with the layer's
kv_scale and casts Q to the cache dtype, as the Triton RoPE branch does.

Dispatch only, no device: both aiter writers are stubbed. Like the other
attention tests, this file skips where triton or aiter does not import.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton", reason="attention_mha defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mha imports the AITER runtime")

from atom.model_ops import attention_mha as mha

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, TOKENS = 4, 1, 256, 3
FP8 = torch.float8_e4m3fn


@pytest.fixture
def writers(monkeypatch):
    """Both aiter cache writers, recorded."""
    writers = SimpleNamespace(plain=Mock(), per_token=Mock())
    monkeypatch.setattr(mha.aiter, "reshape_and_cache", writers.plain)
    monkeypatch.setattr(
        mha.aiter, "reshape_and_cache_with_pertoken_quant", writers.per_token
    )
    return writers


def _impl(kv_cache_dtype):
    """A Qwen3.5 full-attention impl as an fp8 cache builds it: no rotary_emb,
    q_norm or k_norm, the model having run them. Built without the GPU its
    __init__ reads."""
    impl = mha.PagedAttentionImpl.__new__(mha.PagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.head_dim = HEAD_DIM
    impl.kv_cache_dtype = kv_cache_dtype
    impl.layer_num = 0
    impl.sliding_window = -1
    impl.rotary_emb = impl.q_norm = impl.k_norm = None
    impl.sinks = None
    impl.use_flash_layout = False
    impl.per_token_quant = True
    impl.kv_scale = torch.tensor(1.0)
    return impl


def _rope_cache(impl, q, cache_dtype):
    """`rope_cache` on one decode batch, over a 5-D SHUFFLE cache holding
    per-token scale planes, as the native fp8 pool binds it."""
    k = torch.randn(TOKENS, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    v = torch.randn(TOKENS, NUM_KV_HEADS * HEAD_DIM, dtype=torch.bfloat16)
    layer = SimpleNamespace(
        k_cache=torch.empty(1, 1, 1, 1, 1, dtype=cache_dtype),
        v_cache=torch.empty(1, 1, 1, 1, 1, dtype=cache_dtype),
        k_scale=torch.empty(1, NUM_KV_HEADS, 16),
        v_scale=torch.empty(1, NUM_KV_HEADS, 16),
    )
    fwd_ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=torch.arange(TOKENS)),
        kv_cache_data={"layer_0": layer},
        context=SimpleNamespace(is_prefill=False),
    )
    q = q.view(-1, NUM_HEADS, HEAD_DIM)
    k = k.view(-1, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(-1, NUM_KV_HEADS, HEAD_DIM)
    out = impl.rope_cache(q, k, v, None, None, fwd_ctx)
    return layer, fwd_ctx.attn_metadata.slot_mapping, out


def _query():
    return torch.randn(TOKENS, NUM_HEADS * HEAD_DIM, dtype=torch.bfloat16)


def test_unified_fp8_cache_is_written_per_tensor(writers, monkeypatch):
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    impl = _impl("fp8")
    layer, slots, (_q, _k, _v, _kc, _vc, k_scale, v_scale) = _rope_cache(
        impl, _query(), FP8
    )

    writers.per_token.assert_not_called()
    writers.plain.assert_called_once()
    args, kwargs = writers.plain.call_args
    assert args[2] is layer.k_cache and args[3] is layer.v_cache
    assert args[4] is slots
    assert kwargs == {
        "kv_cache_dtype": "fp8",
        "k_scale": impl.kv_scale,
        "v_scale": impl.kv_scale,
        "asm_layout": True,
    }
    # The scales handed on are the per-tensor one, which unified accepts.
    assert k_scale is impl.kv_scale and v_scale is impl.kv_scale
    impl._reject_per_token_scales_on_unified(k_scale, v_scale, "decode")
    assert impl.per_token_quant is False


def test_unified_fp8_query_is_clamped_into_the_cache_dtype(writers, monkeypatch):
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    q = _query()
    q[0, 0], q[0, 1] = 1000.0, -1000.0
    _layer, _slots, (q_out, *_rest) = _rope_cache(_impl("fp8"), q, FP8)

    assert q_out.dtype == FP8
    assert q_out.shape == (TOKENS, NUM_HEADS, HEAD_DIM)
    fp8_max = torch.finfo(FP8).max
    assert q_out[0, 0, 0].float() == fp8_max
    assert q_out[0, 0, 1].float() == -fp8_max
    assert torch.isfinite(q_out.float()).all()


def test_fp8_cache_off_unified_keeps_the_per_token_scales(writers, monkeypatch):
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    q = _query()
    layer, slots, (q_out, _k, _v, _kc, _vc, k_scale, v_scale) = _rope_cache(
        _impl("fp8"), q, FP8
    )

    writers.plain.assert_not_called()
    writers.per_token.assert_called_once()
    args, kwargs = writers.per_token.call_args
    expected = (layer.k_cache, layer.v_cache, layer.k_scale, layer.v_scale, slots)
    assert all(a is b for a, b in zip(args[2:], expected, strict=True))
    assert kwargs == {"asm_layout": True}
    assert k_scale is layer.k_scale and v_scale is layer.v_scale
    assert q_out.dtype == torch.bfloat16


def test_unified_bf16_cache_is_written_without_scales(writers, monkeypatch):
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "1")
    _layer, _slots, (q_out, *_rest) = _rope_cache(
        _impl("bf16"), _query(), torch.bfloat16
    )

    writers.per_token.assert_not_called()
    writers.plain.assert_called_once()
    assert writers.plain.call_args.kwargs == {
        "kv_cache_dtype": "auto",
        "k_scale": None,
        "v_scale": None,
        "asm_layout": True,
    }
    assert q_out.dtype == torch.bfloat16
