# SPDX-License-Identifier: MIT
"""The aiter route of the Gemma q/k norm + RoPE + cache write on gfx1250.

For a Gemma q/k norm layer with a plain NeoX RoPE and head_dim 128 (the
MiniMax-M3 dense layers), `PagedAttentionImpl.rope_cache` hands the norm and
RoPE to aiter `fused_qknorm_idxrqknorm`: a decode batch as one insert into the
cache, a prefill batch above ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS as the
norm and RoPE (k in place in qkv) before the cache write. Either must leave the
attention the q and K/V cache of the Triton kernel, and a prefill its k too.
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
from aiter.jit.utils.chip_info import get_gfx

from atom.model_ops import attention_mha
from atom.utils import envs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_gfx() != "gfx1250"
    or not attention_mha.AITER_QKNORM_TAKES_DENSE_LAYERS,
    reason="gfx1250 only, with an aiter that inserts layers without index heads",
)

NQ, NKV, HD, ROT, MAX_POS = 16, 1, 128, 64, 1 << 16
FUSED_MAX = envs.ATOM_QK_NORM_ROPE_CACHE_FUSION_MAX_TOKENS


def make_impl(kv_cache_dtype):
    from aiter.rotary_embedding import get_rope

    from atom.model_ops.layernorm import GemmaRMSNorm

    norms = []
    for _ in range(2):
        norm = GemmaRMSNorm(HD, eps=1e-6)
        norm.weight.data = (torch.randn(HD) * 0.1).to(torch.bfloat16).cuda()
        norms.append(norm)
    rope = get_rope(HD, ROT, MAX_POS, 10000, is_neox_style=True, dtype=torch.bfloat16)
    return attention_mha.PagedAttentionImpl(
        NQ,
        HD,
        HD**-0.5,
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
    ("is_prefill", "num_tokens", "aiter_calls"),
    [
        (False, 1, ["insert"]),
        (False, 64, ["insert"]),
        (False, 3000, ["insert"]),
        (True, 64, []),
        (True, FUSED_MAX, []),
        (True, FUSED_MAX + 1, ["norm_rope"]),
    ],
)
def test_rope_cache_matches_the_triton_kernel(
    monkeypatch, kv_cache_dtype, block_size, is_prefill, num_tokens, aiter_calls
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
    if not is_prefill:  # CUDA graph padding rows
        slots[num_tokens - num_tokens // 8 :] = -1

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
        lambda *a, **kw: calls.append("insert" if "kv_cache_k" in kw else "norm_rope")
        or op(*a, **kw),
    )
    cache = make_cache(kv_cache_dtype, num_blocks, block_size)
    qkv_in = qkv.clone()
    q, k, v = run_rope_cache(impl, qkv, position, slots, cache, is_prefill)

    assert calls == aiter_calls
    torch.testing.assert_close(q.reshape(num_tokens, -1), ref_q.reshape(num_tokens, -1))
    frac = 1 / 8 if kv_cache_dtype == "fp8" else 1 / 128
    for got, ref in zip(read_cache(cache), read_cache(ref_cache)):
        assert_close_to_row_max(got, ref, frac)
    if is_prefill:
        # prefill attention reads k
        torch.testing.assert_close(
            k.reshape(num_tokens, -1), ref_k.reshape(num_tokens, -1)
        )
    else:
        # the insert leaves qkv alone: the decode backends read the cache
        assert torch.equal(qkv, qkv_in)
    torch.testing.assert_close(v.reshape(num_tokens, -1), qkv_in[:, -NKV * HD :])


def test_aiter_route_needs_a_plain_rope_a_packed_qkv_and_a_shuffle_cache(
    monkeypatch,
):
    impl = make_impl("fp8")
    qkv = torch.zeros(8, (NQ + 2 * NKV) * HD, dtype=torch.bfloat16, device="cuda")
    position = torch.zeros(8, dtype=torch.long, device="cuda")
    k_cache, v_cache, k_scale, _ = make_cache("fp8", 4, 128)
    ok = impl._takes_aiter_gemma_norm_rope
    assert ok(qkv, position, k_cache, v_cache, k_scale)
    # MRoPE: one row of positions per section
    assert not ok(qkv, position.expand(3, 8), k_cache, v_cache, k_scale)
    assert not ok(qkv, position.int(), k_cache, v_cache, k_scale)
    # a qkv with more columns than [q | k | v], or not contiguous
    wide = torch.zeros(8, (NQ + 2 * NKV + 1) * HD, dtype=torch.bfloat16, device="cuda")
    assert not ok(wide, position, k_cache, v_cache, k_scale)
    assert not ok(wide[:, : qkv.shape[1]], position, k_cache, v_cache, k_scale)
    # an FP8 cache without its per-token scales, a V cache not in SHUFFLE order
    assert not ok(qkv, position, k_cache, v_cache, None)
    assert not ok(qkv, position, k_cache, v_cache.flatten(2, 3), k_scale)
    # caches the op does not write: MXFP4 pages, a bf16 cache for an fp16 qkv
    impl.kv_cache_dtype = "fp4"
    assert not ok(qkv, position, k_cache, v_cache, k_scale)
    impl.kv_cache_dtype = "bf16"
    bf16_k, bf16_v, _, _ = make_cache("bf16", 4, 128)
    assert ok(qkv, position, bf16_k, bf16_v, None)
    assert not ok(qkv.half(), position, bf16_k, bf16_v, None)
    # an aiter without ROCm/aiter#8011
    monkeypatch.setattr(attention_mha, "AITER_QKNORM_TAKES_DENSE_LAYERS", False)
    assert not ok(qkv, position, bf16_k, bf16_v, None)
