# SPDX-License-Identifier: MIT
"""The cache write of a long batch on the Gemma q/k norm + RoPE path.

With ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1, Qwen3-Next and Qwen3.5 send
a batch above ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS to
`_gemma_norm_rope_then_cache`. It wrote every cache with the per-token FP8
writer. A bf16 cache has no scales, so each rank raised a TypeError on
`k_scale=None` and the engine hung.

Dispatch only, no device: the norm, the RoPE and both aiter writers are
stubbed. Like the other attention tests, this file skips where triton or aiter
does not import.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton", reason="attention_mha defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mha imports the AITER runtime")

from atom.model_ops import attention_mha as mha
from atom.model_ops import triton_mrope
from atom.model_ops.layernorm import GemmaRMSNorm

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 1, 256
MAX_TOKENS = 2


@pytest.fixture
def writers(monkeypatch):
    """Both aiter cache writers, recorded. The norm and the RoPE give back
    their inputs."""
    writers = SimpleNamespace(plain=Mock(), per_token=Mock())
    monkeypatch.setattr(mha.aiter, "reshape_and_cache", writers.plain)
    monkeypatch.setattr(
        mha.aiter, "reshape_and_cache_with_pertoken_quant", writers.per_token
    )
    monkeypatch.setattr(
        triton_mrope, "try_mrope_qk_fused", lambda _rotary, _pos, q, k, *_: (q, k)
    )
    monkeypatch.setenv("ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS", str(MAX_TOKENS))
    return writers


def _impl(kv_cache_dtype):
    """A Qwen3.5 full-attention impl, built without the GPU its __init__ reads."""
    impl = mha.PagedAttentionImpl.__new__(mha.PagedAttentionImpl)
    torch.nn.Module.__init__(impl)
    impl.num_heads = NUM_HEADS
    impl.num_kv_heads = NUM_KV_HEADS
    impl.head_dim = HEAD_DIM
    impl.kv_cache_dtype = kv_cache_dtype
    impl.layer_num = 0
    impl.sliding_window = -1
    impl.q_norm = GemmaRMSNorm(HEAD_DIM)
    impl.k_norm = GemmaRMSNorm(HEAD_DIM)
    impl.rotary_emb = SimpleNamespace()
    impl._dual_qk_norm = lambda q, k: (q, k)
    return impl


def _rope_cache(impl, k_scale=None, v_scale=None):
    """`rope_cache` on one batch above the limit. The V cache is 5-D, as the
    native SHUFFLE pool binds it."""
    tokens = MAX_TOKENS + 1
    q_size, kv_size = NUM_HEADS * HEAD_DIM, NUM_KV_HEADS * HEAD_DIM
    qkv = torch.randn(tokens, q_size + 2 * kv_size, dtype=torch.bfloat16)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    layer = SimpleNamespace(
        k_cache=torch.empty(1, 1, 1, 1, 1),
        v_cache=torch.empty(1, 1, 1, 1, 1),
        k_scale=k_scale,
        v_scale=v_scale,
    )
    attn_metadata = SimpleNamespace(slot_mapping=torch.arange(tokens))
    fwd_ctx = SimpleNamespace(
        attn_metadata=attn_metadata, kv_cache_data={"layer_0": layer}
    )
    positions = torch.zeros(3, tokens, dtype=torch.int64)
    impl.rope_cache(q, k, v, qkv, positions, fwd_ctx)
    return layer, attn_metadata.slot_mapping


def test_bf16_cache_is_written_without_scales(writers):
    layer, slots = _rope_cache(_impl("bf16"))

    writers.per_token.assert_not_called()
    writers.plain.assert_called_once()
    args, kwargs = writers.plain.call_args
    assert args[0].shape == (MAX_TOKENS + 1, NUM_KV_HEADS, HEAD_DIM)
    assert args[2] is layer.k_cache and args[3] is layer.v_cache
    assert args[4] is slots
    assert kwargs == {
        "kv_cache_dtype": "auto",
        "k_scale": None,
        "v_scale": None,
        "asm_layout": True,
    }


def test_fp8_cache_keeps_the_per_token_scales(writers):
    k_scale, v_scale = torch.empty(1, 1, 16), torch.empty(1, 1, 16)
    layer, slots = _rope_cache(_impl("fp8"), k_scale, v_scale)

    writers.plain.assert_not_called()
    writers.per_token.assert_called_once()
    args, kwargs = writers.per_token.call_args
    assert args[0].shape == (MAX_TOKENS + 1, NUM_KV_HEADS, HEAD_DIM)
    expected = (layer.k_cache, layer.v_cache, k_scale, v_scale, slots)
    assert all(a is b for a, b in zip(args[2:], expected, strict=True))
    assert kwargs == {"asm_layout": True}
