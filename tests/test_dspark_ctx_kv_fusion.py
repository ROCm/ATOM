# SPDX-License-Identifier: MIT
"""Fused MLA context-row KV write vs the per-op chain it replaces.

The fused kernel inlines RMSNorm, the k-only RoPE and ``concat_and_cache_mla``,
so what is asserted is the only thing either path produces: the bytes landing in
the paged cache. The two differ only in the fp32 sum-of-squares reduction tree
and in ``rsqrt``'s approximation, both ~1e-7 relative on a value then rounded to
bf16's 8 mantissa bits, so almost every element is bitwise equal and the rest
are one ulp apart (2^-7 relative at the bottom of a binade): measured on gfx950,
1 element of 4,718,592 over 512 tokens, none at all over 4096. That is the bar below -- a handful of last-bit
disagreements passes, a wiring bug (wrong slot, wrong lane, wrong position)
moves whole rows and does not.

Shapes are Kimi-K3-DSpark's: a 512-wide latent plus a 64-wide positional lane.
The draft shares the engine's fp8 cache, so the fp8 store -- scale then cast,
which the bf16 store must not do -- is covered alongside the bf16 one.
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

def _fp8_dtype():
    """Whichever fp8 this build caches in -- gfx950 is OCP e4m3, gfx942 is fnuz."""
    from aiter import dtypes

    return dtypes.fp8


FP8_DTYPE = _fp8_dtype()


def _assert_cache_matches(fused, ref):
    """Bitwise equal but for a sprinkling of last-bit rounding (see module doc)."""
    if fused.dtype is torch.uint8:
        fused, ref = fused.view(FP8_DTYPE), ref.view(FP8_DTYPE)
    mismatched = int((fused != ref).sum())
    # One in a hundred thousand, i.e. room for the rounding and none for a bug.
    assert mismatched <= max(1, fused.numel() // 100_000), (
        f"{mismatched} of {fused.numel()} cache elements differ, which is more "
        "than last-bit rounding can account for"
    )
    # One bf16 ulp is 2^-8 relative at the top of a binade and 2^-7 at the
    # bottom, so 2^-7 is the bound that means "last bit". An fp8 e4m3 mantissa
    # is 3 bits, so its last bit is 2^-3. atol covers the subnormal end, where a
    # relative bound has nothing to hold on to.
    if fused.dtype is FP8_DTYPE:
        fused, ref = fused.float(), ref.float()
        rtol = 2**-3
    else:
        rtol = 2**-7
    torch.testing.assert_close(fused, ref, rtol=rtol, atol=1e-6)


def _rope(max_position=4096):
    from aiter.rotary_embedding import get_rope

    rope = get_rope(
        PE_DIM,
        rotary_dim=PE_DIM,
        max_position=max_position,
        base=10000.0,
        rope_scaling=None,
        is_neox_style=False,
    )
    # get_rope builds the tables on CPU in fp32 and moves them to the query's
    # device and dtype on its first forward. Do it up front so the fused path,
    # which reads the buffers directly, sees the same ones the reference will
    # -- and the same ones production has, where the model's activation dtype
    # has long since coerced them.
    rope.cos_cache = rope.cos_cache.cuda().to(torch.bfloat16)
    rope.sin_cache = rope.sin_cache.cuda().to(torch.bfloat16)
    return rope


def _reference_write(
    kv_lora, norm_weight, rope, positions, slot_mapping, cache, k_scale
):
    from aiter import concat_and_cache_mla, rmsnorm2d_fwd

    kv_c, k_pe = kv_lora.split([KV_LORA_RANK, PE_DIM], dim=-1)
    # rmsnorm2d_fwd, not an RMSNorm module: the module's unquantized path is a
    # call to exactly this (layernorm.rmsnorm2d_fwd_), and constructing one
    # would drag in a TP group this test has no use for.
    kv_c = rmsnorm2d_fwd(kv_c.contiguous(), norm_weight, EPS)
    k_pe = k_pe.view(-1, 1, PE_DIM)
    _, k_pe = rope(positions, torch.empty_like(k_pe), k_pe)
    concat_and_cache_mla(
        kv_c,
        k_pe.squeeze(1),
        cache,
        slot_mapping.flatten(),
        kv_cache_dtype="fp8" if cache.dtype is torch.uint8 else "auto",
        scale=k_scale,
    )


def _run_both(
    num_tokens, num_blocks, block_size, slot_mapping, kv_lora=None, cache_dtype=None
):
    from atom.model_ops.triton_fused_mla_ctx_kv import fused_mla_ctx_norm_rope_cache

    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    if kv_lora is None:
        kv_lora = torch.randn(
            num_tokens, ENTRY, generator=gen, device="cuda", dtype=torch.float32
        ).to(torch.bfloat16)
    norm_weight = torch.empty(KV_LORA_RANK, device="cuda", dtype=torch.bfloat16)
    norm_weight.normal_(mean=1.0, std=0.1, generator=gen)
    rope = _rope()
    positions = torch.randint(
        0, 4096, (num_tokens,), generator=gen, device="cuda", dtype=torch.int64
    )

    cache_dtype = cache_dtype or torch.bfloat16
    is_fp8 = cache_dtype is FP8_DTYPE
    # Pre-filled with a sentinel so an untouched slot is distinguishable from a
    # written one -- that is what the slot < 0 case below checks.
    fused_cache = torch.full(
        (num_blocks, block_size, ENTRY), -7.0, device="cuda", dtype=cache_dtype
    )
    ref_cache = fused_cache.clone()
    if is_fp8:
        # As vLLM holds it: an fp8 pool is allocated as raw bytes, and both
        # stores have to reinterpret rather than convert.
        fused_cache = fused_cache.view(torch.uint8)
        ref_cache = ref_cache.view(torch.uint8)
    # Not 1.0: a unit scale would let a store that forgets to divide by it pass.
    k_scale = torch.tensor(0.5 if is_fp8 else 1.0, dtype=torch.float32, device="cuda")

    fused_mla_ctx_norm_rope_cache(
        kv_lora,
        norm_weight,
        positions,
        rope.cos_cache,
        rope.sin_cache,
        slot_mapping,
        fused_cache,
        k_scale,
        EPS,
        KV_LORA_RANK,
        PE_DIM,
        rope.is_neox_style,
        is_fp8,
    )
    _reference_write(
        kv_lora, norm_weight, rope, positions, slot_mapping, ref_cache, k_scale
    )
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
    slots = torch.randperm(
        num_blocks * block_size,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(num_tokens),
    )[:num_tokens]
    fused_cache, ref_cache = _run_both(
        num_tokens, num_blocks, block_size, slots.to(torch.int64)
    )
    _assert_cache_matches(fused_cache, ref_cache)


@pytest.mark.parametrize("num_tokens,num_blocks,block_size", [(512, 64, 128), (7, 8, 1)])
def test_matches_per_op_chain_on_an_fp8_cache(num_tokens, num_blocks, block_size):
    """The draft caches fp8 with the engine, so the store scales before casting.

    aiter's kFp8 branch divides by ``k_scale`` and the kAuto branch does not, and
    the fused kernel has to pick the same branch off the same flag -- getting it
    backwards is silent, since both produce plausible-looking numbers.
    """
    slots = torch.randperm(
        num_blocks * block_size,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(num_tokens),
    )[:num_tokens]
    fused_cache, ref_cache = _run_both(
        num_tokens,
        num_blocks,
        block_size,
        slots.to(torch.int64),
        cache_dtype=FP8_DTYPE,
    )
    _assert_cache_matches(fused_cache, ref_cache)


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

    slots = torch.randperm(
        num_blocks * block_size,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(0),
    )[:num_tokens]
    fused_cache, ref_cache = _run_both(
        num_tokens, num_blocks, block_size, slots.to(torch.int64), kv_lora=kv_lora
    )
    _assert_cache_matches(fused_cache, ref_cache)


def test_negative_slots_are_skipped():
    """Padded rows carry slot -1 and must leave the cache alone, as aiter does."""
    num_tokens, num_blocks, block_size = 32, 4, 128
    slots = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    slots[::3] = -1
    fused_cache, ref_cache = _run_both(num_tokens, num_blocks, block_size, slots)
    _assert_cache_matches(fused_cache, ref_cache)
    assert (fused_cache[0, 0] == -7.0).all()


def test_kv_only_projection_matches_the_merged_gemm():
    """The claim behind `MergedReplicatedLinear.forward_shard`.

    The method is glue; what has to hold is that narrowing the merged weight's
    rows gives the merged output's columns. Driven through `tgemm.mm` rather
    than the module so the test needs no TP group, since that is the one call
    the unquantized path makes.

    Not exact: a different N can pick a different tuned solution, so the fp32
    accumulation order differs. bf16's resolution is the bar that matters.
    """
    from aiter import dtypes
    from aiter.tuned_gemm import tgemm

    weight = torch.empty(
        Q_LORA_RANK + ENTRY, 7168, device="cuda", dtype=torch.bfloat16
    ).normal_(std=0.02)
    x = torch.randn(256, 7168, device="cuda", dtype=torch.bfloat16)

    merged = tgemm.mm(x, weight, None, otype=dtypes.bf16)
    shard = tgemm.mm(x, weight.narrow(0, Q_LORA_RANK, ENTRY), None, otype=dtypes.bf16)
    # One bf16 ulp, which is what the measured disagreement is: 27 elements of
    # 147,456, none further out than 0.015625.
    torch.testing.assert_close(shard, merged[..., Q_LORA_RANK:], rtol=2**-7, atol=0.02)
