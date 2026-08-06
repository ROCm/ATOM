# SPDX-License-Identifier: MIT
"""Fused MLA context-row KV write vs the per-op chain it replaces.

The fused kernel inlines RMSNorm, the k-only RoPE and ``concat_and_cache_mla``,
so what is asserted is the only thing either path produces: the bytes landing in
the paged cache. Equality is exact -- the two differ only in the fp32
sum-of-squares reduction tree and in ``rsqrt``'s approximation, both ~1e-7
relative on a value that is then rounded to bf16's 8 mantissa bits. If this test
ever fails on the last bit of a few rows, that is the claim in the module
docstring breaking, not a wiring bug; check the failure count before relaxing it.

Shapes are Kimi-K3-DSpark's: a 512-wide latent plus a 64-wide positional lane,
cached bf16.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused context-KV write is a GPU kernel"
)

KV_LORA_RANK = 512
PE_DIM = 64
ENTRY = KV_LORA_RANK + PE_DIM
Q_LORA_RANK = 1536
EPS = 1e-6


def _rope(max_position=4096):
    from aiter.rotary_embedding import get_rope

    return get_rope(
        PE_DIM,
        rotary_dim=PE_DIM,
        max_position=max_position,
        base=10000.0,
        rope_scaling=None,
        is_neox_style=False,
    )


def _reference_write(kv_lora, norm, rope, positions, slot_mapping, cache):
    from aiter import concat_and_cache_mla

    kv_c, k_pe = kv_lora.split([KV_LORA_RANK, PE_DIM], dim=-1)
    kv_c = norm(kv_c)
    k_pe = k_pe.view(-1, 1, PE_DIM)
    _, k_pe = rope(positions, torch.empty_like(k_pe), k_pe)
    concat_and_cache_mla(
        kv_c,
        k_pe.squeeze(1),
        cache,
        slot_mapping.flatten(),
        kv_cache_dtype="auto",
        scale=torch.tensor(1.0, dtype=torch.float32, device=cache.device),
    )


def _run_both(num_tokens, num_blocks, block_size, slot_mapping, kv_lora=None):
    from atom.model_ops.layernorm import RMSNorm
    from atom.model_ops.triton_fused_mla_ctx_kv import fused_mla_ctx_norm_rope_cache

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    if kv_lora is None:
        kv_lora = torch.randn(
            num_tokens, ENTRY, generator=gen, device="cuda", dtype=torch.float32
        ).to(torch.bfloat16)
    norm = RMSNorm(KV_LORA_RANK, eps=EPS).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(mean=1.0, std=0.1, generator=gen)
    rope = _rope()
    positions = torch.randint(
        0, 4096, (num_tokens,), generator=gen, device="cuda", dtype=torch.int64
    )

    # Pre-filled with a sentinel so an untouched slot is distinguishable from a
    # written one -- that is what the slot < 0 case below checks.
    fused_cache = torch.full(
        (num_blocks, block_size, ENTRY), -7.0, device="cuda", dtype=torch.bfloat16
    )
    ref_cache = fused_cache.clone()

    fused_mla_ctx_norm_rope_cache(
        kv_lora,
        norm.weight,
        positions,
        rope.cos_cache,
        rope.sin_cache,
        slot_mapping,
        fused_cache,
        torch.tensor(1.0, dtype=torch.float32, device="cuda"),
        EPS,
        KV_LORA_RANK,
        PE_DIM,
        rope.is_neox_style,
        False,
    )
    _reference_write(kv_lora, norm, rope, positions, slot_mapping, ref_cache)
    return fused_cache, ref_cache


@pytest.mark.parametrize(
    "num_tokens,num_blocks,block_size",
    [
        (512, 64, 128),  # a decode step's bs*(1+T) rows
        (4096, 64, 128),  # a prefill step
        (7, 8, 1),  # per-token cache (block_size 1)
    ],
)
def test_matches_per_op_chain(num_tokens, num_blocks, block_size):
    slots = torch.randperm(num_blocks * block_size, device="cuda")[:num_tokens]
    fused_cache, ref_cache = _run_both(
        num_tokens, num_blocks, block_size, slots.to(torch.int64)
    )
    torch.testing.assert_close(fused_cache, ref_cache, rtol=0, atol=0)


def test_strided_kv_lora_slice():
    """The caller passes the [..., q_lora_rank:] half of a fused projection."""
    num_tokens, num_blocks, block_size = 128, 16, 128
    gen = torch.Generator(device="cuda").manual_seed(0)
    qkv_lora = torch.randn(
        num_tokens,
        Q_LORA_RANK + ENTRY,
        generator=gen,
        device="cuda",
        dtype=torch.float32,
    ).to(torch.bfloat16)
    kv_lora = qkv_lora[..., Q_LORA_RANK:]
    assert not kv_lora.is_contiguous() and kv_lora.stride(1) == 1

    slots = torch.randperm(num_blocks * block_size, device="cuda")[:num_tokens]
    fused_cache, ref_cache = _run_both(
        num_tokens, num_blocks, block_size, slots.to(torch.int64), kv_lora=kv_lora
    )
    torch.testing.assert_close(fused_cache, ref_cache, rtol=0, atol=0)


def test_negative_slots_are_skipped():
    """Padded rows carry slot -1 and must leave the cache alone, as aiter does."""
    num_tokens, num_blocks, block_size = 32, 4, 128
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    slots[::3] = -1
    fused_cache, ref_cache = _run_both(num_tokens, num_blocks, block_size, slots)
    torch.testing.assert_close(fused_cache, ref_cache, rtol=0, atol=0)
    assert (fused_cache[0, 0] == -7.0).all()


def test_kv_only_projection_matches_the_merged_gemm():
    """forward_shard's narrow GEMM against slicing the merged result.

    Not exact: a different N can pick a different tuned solution, so the fp32
    accumulation order differs. bf16's resolution is the bar that matters.
    """
    from atom.model_ops.linear import MergedReplicatedLinear

    proj = MergedReplicatedLinear(
        7168, [Q_LORA_RANK, ENTRY], bias=False, prefix="fused_qkv_a_proj"
    )
    proj = proj.cuda().to(torch.bfloat16)
    proj.weight.data.normal_(std=0.02)
    x = torch.randn(256, 7168, device="cuda", dtype=torch.bfloat16)

    merged = proj(x)
    merged = merged[0] if isinstance(merged, tuple) else merged
    torch.testing.assert_close(
        proj.forward_shard(x, 1),
        merged[..., Q_LORA_RANK:],
        rtol=1e-2,
        atol=1e-2,
    )
