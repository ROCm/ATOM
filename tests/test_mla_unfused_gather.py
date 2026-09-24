# SPDX-License-Identifier: MIT
"""Numerics of the unfused `gather_kv_b_proj` fallback.

The fallback exists because the fused Triton kernel cannot be compiled on some
targets for chunked-prefill shapes. That makes it the only implementation there,
with nothing to cross-check against at runtime -- and every way it can be wrong
is silent: a mis-addressed row, a k_pe that landed in the wrong lane, or a cache
scale applied to a bf16 cache all produce plausible logits, not an error.

So these compare it against a deliberately naive per-token reference on CPU.
No device, no aiter: `_mm` falls back to `torch.matmul` when the Triton GEMM
cannot be imported.
"""

from __future__ import annotations

import pytest
import torch

from atom.model_ops.mla_unfused_gather import unfused_gather_kv_b_proj

NUM_HEADS = 3
NOPE = 4
V_DIM = 5
PE_DIM = 2
KV_C_DIM = 6


def _reference(k_buffer, k_scale, kv_indptr, kv_indices, cu_seqlens_k, weight, scale):
    """One token at a time, written for legibility rather than speed."""
    block_size = k_buffer.shape[1]
    total_kv = int(cu_seqlens_k[-1])
    w = weight.to(torch.float32)
    if scale is not None:
        w = w * scale.reshape(-1, 1).to(torch.float32)
    k = torch.zeros(total_kv, NUM_HEADS, NOPE + PE_DIM, dtype=torch.float32)
    v = torch.zeros(total_kv, NUM_HEADS, V_DIM, dtype=torch.float32)
    for token in range(total_kv):
        seq = int((cu_seqlens_k[1:] <= token).sum())
        within = token - int(cu_seqlens_k[seq])
        block = int(kv_indices[int(kv_indptr[seq]) + within // block_size])
        latent = k_buffer[block, within % block_size].to(torch.float32)
        kv_c, k_pe = latent[:KV_C_DIM], latent[KV_C_DIM:]
        if k_scale is not None and k_buffer.dtype is not torch.bfloat16:
            kv_c = kv_c * float(k_scale)
            k_pe = k_pe * float(k_scale)
        projected = (kv_c @ w.t()).view(NUM_HEADS, NOPE + V_DIM)
        k[token] = torch.cat((projected[:, :NOPE], k_pe.expand(NUM_HEADS, -1)), dim=-1)
        v[token] = projected[:, NOPE:]
    return k, v


def _case(block_size, seq_lens, cache_dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    blocks_per_seq = [(n + block_size - 1) // block_size for n in seq_lens]
    num_blocks = sum(blocks_per_seq) + 2  # unused blocks: addressing must not drift
    k_buffer = torch.randn(num_blocks, block_size, KV_C_DIM + PE_DIM).to(cache_dtype)
    # Shuffle so block ids are not the identity map -- an off-by-one in the
    # indirection would otherwise still land on plausible data.
    order = torch.randperm(num_blocks)[: sum(blocks_per_seq)]
    kv_indices = order.to(torch.int32)
    kv_indptr = torch.tensor([0, *torch.cumsum(torch.tensor(blocks_per_seq), 0)]).to(
        torch.int32
    )
    cu_seqlens_k = torch.tensor([0, *torch.cumsum(torch.tensor(seq_lens), 0)]).to(
        torch.int32
    )
    weight = torch.randn(NUM_HEADS * (NOPE + V_DIM), KV_C_DIM)
    return k_buffer, kv_indptr, kv_indices, cu_seqlens_k, weight


def _run(case, k_scale=None, scale=None, **kw):
    k_buffer, kv_indptr, kv_indices, cu_seqlens_k, weight = case
    total_kv = int(cu_seqlens_k[-1])
    k_out = torch.zeros(total_kv, NUM_HEADS, NOPE + PE_DIM)
    v_out = torch.zeros(total_kv, NUM_HEADS, V_DIM)
    unfused_gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        cu_seqlens_k,
        weight,
        scale,
        k_out,
        v_out,
        **kw,
    )
    return k_out, v_out


@pytest.mark.parametrize(
    "block_size, seq_lens",
    [
        (1, [5]),  # page_size 1 skips the block arithmetic entirely
        (4, [4]),  # exact multiple of the block
        (4, [7]),  # ragged tail
        (4, [7, 1, 13]),  # several sequences, one of them a single token
        (16, [33, 5]),
    ],
)
def test_matches_reference(block_size, seq_lens):
    case = _case(block_size, seq_lens)
    got_k, got_v = _run(case)
    want_k, want_v = _reference(case[0], None, case[1], case[2], case[3], case[4], None)
    torch.testing.assert_close(got_k, want_k)
    torch.testing.assert_close(got_v, want_v)


def test_chunking_does_not_change_the_result():
    """rows_per_chunk is a memory knob; crossing a chunk must be invisible."""
    case = _case(4, [7, 13])
    whole = _run(case, rows_per_chunk=4096)
    for rows in (1, 3, 8, 19):
        torch.testing.assert_close(_run(case, rows_per_chunk=rows)[0], whole[0])
        torch.testing.assert_close(_run(case, rows_per_chunk=rows)[1], whole[1])


def test_per_row_weight_scale_is_applied():
    case = _case(4, [9])
    scale = torch.rand(NUM_HEADS * (NOPE + V_DIM)) + 0.5
    got_k, got_v = _run(case, scale=scale)
    want_k, want_v = _reference(
        case[0], None, case[1], case[2], case[3], case[4], scale
    )
    torch.testing.assert_close(got_k, want_k)
    torch.testing.assert_close(got_v, want_v)


def test_block_weight_scale_matches_expanded_scale():
    """The 128x128 block scale, checked against the row-scale path it expands to."""
    case = _case(4, [9])
    n, k = case[4].shape
    blocked = torch.rand(n // 3, k // 2) + 0.5
    got_k, got_v = _run(case, scale=blocked)
    expanded = blocked.repeat_interleave(3, 0).repeat_interleave(2, 1)
    want_k, want_v = _run((case[0], case[1], case[2], case[3], case[4] * expanded))
    torch.testing.assert_close(got_k, want_k)
    torch.testing.assert_close(got_v, want_v)


def test_bf16_cache_is_not_scaled_by_k_scale():
    """The subtlest way to be wrong.

    The fused kernel uses 1.0 for a bf16 cache and only loads k_scale for a
    quantized one. Scaling anyway multiplies all of K and V by a constant --
    which no assertion downstream would catch.
    """
    case = _case(4, [9], cache_dtype=torch.bfloat16)
    scaled = _run(case, k_scale=torch.tensor(7.0))
    unscaled = _run(case, k_scale=None)
    torch.testing.assert_close(scaled[0], unscaled[0])
    torch.testing.assert_close(scaled[1], unscaled[1])


def test_quantized_cache_is_scaled_by_k_scale():
    case = _case(4, [9], cache_dtype=torch.float16)
    k_scale = torch.tensor(3.0)
    got_k, got_v = _run(case, k_scale=k_scale)
    want_k, want_v = _reference(
        case[0], k_scale, case[1], case[2], case[3], case[4], None
    )
    torch.testing.assert_close(got_k, want_k)
    torch.testing.assert_close(got_v, want_v)


@pytest.mark.parametrize(
    "kwargs", [{"shuffled_kv_cache": True}, {"weight_preshuffle": True}]
)
def test_private_layouts_are_refused_not_guessed(kwargs):
    """Wrong K/V is worse than no K/V: these layouts belong to the fused kernel."""
    with pytest.raises(NotImplementedError):
        _run(_case(4, [4]), **kwargs)
