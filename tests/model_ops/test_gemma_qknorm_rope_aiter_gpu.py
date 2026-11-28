# SPDX-License-Identifier: MIT
"""The aiter route of the Gemma q/k norm + RoPE of a long prefill on gfx1250.

Above ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS and with the pre-shuffled (ASM)
KV cache layout (`_use_asm_cache_layout`), `PagedAttentionImpl.rope_cache` hands
the Gemma q/k norm and NeoX RoPE to aiter `fused_qknorm_idxrqknorm` (k in place
in qkv) before the cache write, where the op takes the layer (the MiniMax-M3
dense layers; not head_dim 256), asked once per layer. It must leave the
attention the q, k and K/V cache of the Triton kernel.
Not bitwise: the sum of squares is reduced in another order and aiter rounds k
to bf16 before the per-token FP8 quantization, so a cache is compared
dequantized, within one step at the row maximum (32 of the 448 an FP8 row
maximum is scaled to, below 1/8; 1/128 for bf16).
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")

import aiter
from aiter.jit.utils.build_targets import is_gfx1250_family
from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops import attention_mha
from atom.utils import envs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_gfx1250_family(get_gfx()),
    reason="the aiter route runs on the gfx1250 family only",
)

NQ, NKV, HD, ROT, MAX_POS = 16, 1, 128, 64, 1 << 16
FUSED_MAX = envs.ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS


def make_impl(kv_cache_dtype, hd=HD):
    from aiter.rotary_embedding import get_rope

    from atom.model_ops.layernorm import GemmaRMSNorm

    norms = []
    for _ in range(2):
        norm = GemmaRMSNorm(hd, eps=1e-6)
        norm.weight.data = (torch.randn(hd) * 0.1).to(torch.bfloat16).cuda()
        norms.append(norm)
    rope = get_rope(hd, ROT, MAX_POS, 10000, is_neox_style=True, dtype=torch.bfloat16)
    return attention_mha.PagedAttentionImpl(
        NQ,
        hd,
        hd**-0.5,
        NKV,
        None,
        kv_cache_dtype=kv_cache_dtype,
        rotary_emb=rope.cuda(),
        q_norm=norms[0],
        k_norm=norms[1],
    )


def make_cache(kv_cache_dtype, num_blocks, block_size):
    """SHUFFLE: K [blocks, heads, D / x, block, x], V [blocks, heads, block / x,
    D, x], x elements per 16 bytes; per-token scales for FP8."""
    dtype = torch.float8_e4m3fn if kv_cache_dtype == "fp8" else torch.bfloat16
    x = 16 // dtype.itemsize
    k = torch.zeros(num_blocks, NKV, HD // x, block_size, x, dtype=dtype, device="cuda")
    v = torch.zeros(num_blocks, NKV, block_size // x, HD, x, dtype=dtype, device="cuda")
    if kv_cache_dtype != "fp8":
        return k, v, None, None
    k_scale, v_scale = torch.zeros(2, num_blocks, NKV, block_size, device="cuda")
    return k, v, k_scale, v_scale


def read_cache(cache):
    """[slots, heads, D] K and V, dequantized."""
    k, v, k_scale, v_scale = cache
    nb, bs = k.shape[0], k.shape[3]
    k = k.permute(0, 3, 1, 2, 4).reshape(nb * bs, NKV, HD).float()
    v = v.permute(0, 2, 4, 1, 3).reshape(nb * bs, NKV, HD).float()
    if k_scale is not None:
        k = k * k_scale.permute(0, 2, 1).reshape(nb * bs, NKV, 1)
        v = v * v_scale.permute(0, 2, 1).reshape(nb * bs, NKV, 1)
    return k, v


def assert_close_to_row_max(got, ref, frac):
    err = (got.float() - ref.float()).abs()
    bound = frac * ref.float().abs().amax(-1, keepdim=True)
    assert (err <= bound).all(), f"max excess {(err - bound).max().item():.3e}"


def run_rope_cache(impl, qkv, position, slots, cache, is_prefill):
    fwd_ctx = SimpleNamespace(
        attn_metadata=SimpleNamespace(slot_mapping=slots),
        kv_cache_data={
            f"layer_{impl.layer_num}": SimpleNamespace(
                k_cache=cache[0], v_cache=cache[1], k_scale=cache[2], v_scale=cache[3]
            )
        },
        context=SimpleNamespace(is_prefill=is_prefill),
    )
    q_raw, k_raw, v_raw = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
    return impl.rope_cache(q_raw, k_raw, v_raw, qkv, position, fwd_ctx)[:3]


@pytest.mark.parametrize("block_size", [16, 128])
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "bf16"])
@pytest.mark.parametrize(
    ("num_tokens", "aiter_calls"), [(FUSED_MAX, 0), (FUSED_MAX + 1, 1)]
)
def test_rope_cache_matches_the_triton_kernel(
    monkeypatch, kv_cache_dtype, block_size, num_tokens, aiter_calls
):
    from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
        triton_fused_norm_rope_cache,
    )

    torch.manual_seed(0)
    impl = make_impl(kv_cache_dtype)
    qkv = torch.randn(num_tokens, (NQ + 2 * NKV) * HD, device="cuda").bfloat16()
    position = torch.randint(0, MAX_POS, (num_tokens,), device="cuda")
    num_blocks = num_tokens // block_size + 8
    slots = torch.randperm(num_blocks * block_size, device="cuda")[:num_tokens]

    ref_cache = make_cache(kv_cache_dtype, num_blocks, block_size)
    q_raw, k_raw, v_raw = qkv.split([NQ * HD, NKV * HD, NKV * HD], dim=-1)
    ref_q, ref_k = triton_fused_norm_rope_cache(
        q_raw,
        k_raw,
        v_raw,
        position,
        q_norm=impl.q_norm,
        k_norm=impl.k_norm,
        rotary_emb=impl.rotary_emb,
        num_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HD,
        k_cache=ref_cache[0],
        v_cache=ref_cache[1],
        k_scale=ref_cache[2],
        v_scale=ref_cache[3],
        slot_mapping=slots,
        kv_cache_dtype=kv_cache_dtype,
    )

    calls = []
    op = aiter.fused_qknorm_idxrqknorm
    monkeypatch.setattr(
        aiter,
        "fused_qknorm_idxrqknorm",
        lambda *a, **kw: calls.append(1) or op(*a, **kw),
    )
    cache = make_cache(kv_cache_dtype, num_blocks, block_size)
    qkv_in = qkv.clone()
    q, k, v = run_rope_cache(impl, qkv, position, slots, cache, True)

    assert len(calls) == aiter_calls
    torch.testing.assert_close(q.reshape(num_tokens, -1), ref_q.reshape(num_tokens, -1))
    frac = 1 / 8 if kv_cache_dtype == "fp8" else 1 / 128
    for got, ref in zip(read_cache(cache), read_cache(ref_cache)):
        assert_close_to_row_max(got, ref, frac)
    # prefill attention reads k
    torch.testing.assert_close(k.reshape(num_tokens, -1), ref_k.reshape(num_tokens, -1))
    torch.testing.assert_close(v.reshape(num_tokens, -1), qkv_in[:, -NKV * HD :])


def test_aiter_route_declines_what_the_op_does_not_take(monkeypatch):
    asked = []
    supports = aiter.fused_qknorm_idxrqknorm_supports
    monkeypatch.setattr(
        aiter,
        "fused_qknorm_idxrqknorm_supports",
        lambda *a, **kw: asked.append(1) or supports(*a, **kw),
    )
    position = torch.zeros(8, dtype=torch.long, device="cuda")
    for hd, taken in ((HD, True), (256, False)):  # 256: Qwen3-Next, Qwen3.5
        impl = make_impl("bf16", hd)
        qkv = torch.zeros(8, (NQ + 2 * NKV) * hd, dtype=torch.bfloat16, device="cuda")
        asked.clear()
        for _ in range(2):  # the layer asks aiter once and keeps the answer
            assert (impl._aiter_gemma_norm_rope(qkv, position) is not None) == taken
        assert len(asked) == 1
    # GPT-J: not the rotation the op computes
    from aiter.rotary_embedding import get_rope

    impl = make_impl("bf16")
    impl.rotary_emb = get_rope(
        HD, ROT, MAX_POS, 10000, is_neox_style=False, dtype=torch.bfloat16
    ).cuda()
    qkv = torch.zeros(8, (NQ + 2 * NKV) * HD, dtype=torch.bfloat16, device="cuda")
    assert impl._aiter_gemma_norm_rope(qkv, position) is None
