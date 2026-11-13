# SPDX-License-Identifier: MIT
"""The fused norm + RoPE + cache-write kernels treat tokens past the slot
table as padding.

Token t < len(slot_mapping) goes to slot slot_mapping[t]. A token past the
table has no slot. The kernels read one anyway, from the buffer behind the
view, where other sequences' slots can be, and wrote the token's K/V there.
Each kernel runs twice: with a short table over a buffer of real slots, and
with a full-length table that marks the same tail -1. Both runs must give the
same cache and the same outputs.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)

TOKENS, SLOTS = 12, 5
PAGES, PAGE = 4, 16
X = 8  # bf16 values in 16 bytes
HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM = 2, 1, 128, 64


def _slot_tables():
    """A short table over a buffer of distinct real slots, and a full-length
    table with the same tail marked -1."""
    slots = torch.randperm(PAGES * PAGE, device="cuda")[:TOKENS]
    tail = torch.arange(TOKENS, device="cuda") >= SLOTS
    return slots[:SLOTS], slots.masked_fill(tail, -1)


def _shuffle_caches():
    """Zeroed SHUFFLE K and V caches, the layout of
    `reshape_and_cache(asm_layout=True)`."""
    k_cache = torch.zeros(
        PAGES, KV_HEADS, HEAD_DIM // X, PAGE, X, device="cuda", dtype=torch.bfloat16
    )
    v_cache = torch.zeros(
        PAGES, KV_HEADS, PAGE // X, HEAD_DIM, X, device="cuda", dtype=torch.bfloat16
    )
    return k_cache, v_cache


def _assert_same(short, padded):
    for a, b in zip(short, padded, strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_triton_fused_norm_rope_cache_skips_padding_tokens():
    from atom.model_ops.layernorm import GemmaRMSNorm
    from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
        triton_fused_norm_rope_cache,
    )

    torch.manual_seed(0)
    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    qkv = torch.randn(TOKENS, q_size + 2 * kv_size, device="cuda", dtype=torch.bfloat16)
    q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    positions = torch.randint(0, 64, (TOKENS,), device="cuda")
    angles = torch.randn(64, ROTARY_DIM // 2, device="cuda", dtype=torch.float32)
    rotary = SimpleNamespace(
        rotary_dim=ROTARY_DIM, cos_cache=angles.cos(), sin_cache=angles.sin()
    )
    q_norm = GemmaRMSNorm(HEAD_DIM).cuda()
    k_norm = GemmaRMSNorm(HEAD_DIM).cuda()

    results = []
    for slot_mapping in _slot_tables():
        k_cache, v_cache = _shuffle_caches()
        q_out, k_out = triton_fused_norm_rope_cache(
            q,
            k,
            v,
            positions,
            q_norm=q_norm,
            k_norm=k_norm,
            rotary_emb=rotary,
            num_heads=HEADS,
            num_kv_heads=KV_HEADS,
            head_dim=HEAD_DIM,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=None,
            v_scale=None,
            slot_mapping=slot_mapping,
            kv_cache_dtype="bf16",
        )
        results.append((q_out, k_out, k_cache, v_cache))
    short, padded = results
    assert padded[2].any() and padded[3].any(), "the real tokens wrote nothing"
    _assert_same(short, padded)


def test_minimax_m3_kv_insert_skips_padding_tokens():
    from atom.model_ops.minimax_m3.sparse_attn import (
        minimax_m3_fused_qknorm_rope_kv_insert_shuffle,
    )

    torch.manual_seed(1)
    index_heads, index_dim = 1, 128
    q_size, kv_size = HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM
    iq_size = index_heads * index_dim
    qkv = torch.randn(
        TOKENS,
        q_size + 2 * kv_size + iq_size + index_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_w, k_w, iq_w, ik_w = (
        torch.randn(n, device="cuda", dtype=torch.bfloat16)
        for n in (HEAD_DIM, HEAD_DIM, index_dim, index_dim)
    )
    cos_sin = torch.randn(64, ROTARY_DIM, device="cuda", dtype=torch.float32)
    positions = torch.randint(0, 64, (TOKENS,), device="cuda")

    results = []
    for slot_mapping in _slot_tables():
        k_cache, v_cache = _shuffle_caches()
        index_cache = torch.zeros(
            PAGES * PAGE, index_dim, device="cuda", dtype=torch.bfloat16
        )
        q_out = torch.empty(TOKENS, q_size, device="cuda", dtype=torch.bfloat16)
        index_q_out = torch.empty(TOKENS, iq_size, device="cuda", dtype=torch.bfloat16)
        minimax_m3_fused_qknorm_rope_kv_insert_shuffle(
            qkv,
            q_w,
            k_w,
            cos_sin,
            positions,
            HEADS,
            KV_HEADS,
            ROTARY_DIM,
            1e-6,
            iq_w,
            ik_w,
            index_heads,
            slot_mapping,
            k_cache,
            v_cache,
            index_cache,
            q_out,
            index_q_out,
            index_dim,
        )
        results.append((q_out, index_q_out, k_cache, v_cache, index_cache))
    short, padded = results
    assert padded[2].any() and padded[4].any(), "the real tokens wrote nothing"
    _assert_same(short, padded)
