from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def reshape_and_cache_shuffle_kernel(
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, block_size // x, head_size, x]
    slot_mapping_ptr,  # [num_tokens]
    k_scale_ptr,
    v_scale_ptr,
    x,
    k_stride0,
    v_stride0,
    block_size,
    head_size,
    num_kv_heads,
    BLOCK_SIZE: tl.constexpr,
    QUANT: tl.constexpr,
):
    tid = tl.program_id(0)
    head_id = tl.program_id(1)
    offset = tl.arange(0, BLOCK_SIZE)
    src_offset_k = tid * k_stride0 + head_id * head_size
    src_offset_v = tid * v_stride0 + head_id * head_size
    slot_id = tl.load(slot_mapping_ptr + tid)
    if slot_id < 0:
        return
    block_id = slot_id // block_size
    block_offset = slot_id % block_size
    dst_offset = (
        block_id * num_kv_heads * head_size * block_size
        + head_id * head_size * block_size
    )
    dst_k_shuffle_offset = (
        dst_offset + offset // x * block_size * x + block_offset * x + offset % x
    )
    dst_v_shuffle_offset = (
        dst_offset + block_offset // x * head_size * x + offset * x + block_offset % x
    )
    k_val = tl.load(key_ptr + src_offset_k + offset)
    v_val = tl.load(value_ptr + src_offset_v + offset)
    if QUANT:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)
        k_dtype = key_cache_ptr.type.element_ty
        v_dtype = value_cache_ptr.type.element_ty
        k_val = (k_val.to(tl.float32) / k_scale).to(k_dtype)
        v_val = (v_val.to(tl.float32) / v_scale).to(v_dtype)
    tl.store(key_cache_ptr + dst_k_shuffle_offset, k_val)
    tl.store(value_cache_ptr + dst_v_shuffle_offset, v_val)


def reshape_and_cache_shuffle_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
):
    num_tokens = slot_mapping.shape[0]
    _, num_kv_heads, head_size = key.shape
    num_blocks, block_size, _, _ = key_cache.shape
    x = 16 // key_cache.element_size()
    k_cache_template = torch.empty(
        [num_blocks, num_kv_heads, head_size // x, block_size, x],
        dtype=key_cache.dtype,
        device="meta",
    )
    v_cache_template = torch.empty(
        [num_blocks, num_kv_heads, block_size // x, head_size, x],
        dtype=value_cache.dtype,
        device="meta",
    )
    new_key_cache = key_cache.view_as(k_cache_template)
    new_value_cache = value_cache.view_as(v_cache_template)
    quant = kv_cache_dtype.startswith("fp8")
    grid = (
        num_tokens,
        num_kv_heads,
    )
    reshape_and_cache_shuffle_kernel[grid](
        key,
        value,
        new_key_cache,
        new_value_cache,
        slot_mapping,
        k_scales,
        v_scales,
        x,
        key.stride(0),
        value.stride(0),
        block_size,
        head_size,
        num_kv_heads,
        BLOCK_SIZE=head_size,
        QUANT=quant,
    )


def set_kv_buffer_with_layout_shuffle(
    cache_loc,
    k,
    v,
    k_buffer,
    v_buffer,
    k_scale,
    v_scale,
    block_size,
):
    num_slots, num_kv_heads, head_dim = k_buffer.shape
    num_blocks = num_slots // block_size
    num_slots_with_block = num_blocks * block_size
    k_buffer = k_buffer[:num_slots_with_block].view(
        num_blocks, block_size, num_kv_heads, head_dim
    )
    v_buffer = v_buffer[:num_slots_with_block].view(
        num_blocks, block_size, num_kv_heads, head_dim
    )
    kv_cache_dtype = "auto"
    if k_buffer.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        kv_cache_dtype = "fp8"
    if kv_cache_dtype.startswith("fp8"):
        import aiter

        x = 16 // k_buffer.element_size()
        key_cache = k_buffer.view(
            num_blocks, num_kv_heads, head_dim // x, block_size, x
        )
        value_cache = v_buffer.view(
            num_blocks, num_kv_heads, block_size // x, head_dim, x
        ).view(num_blocks, num_kv_heads, head_dim, block_size)
        aiter.reshape_and_cache_with_pertoken_quant(
            k,
            v,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            cache_loc,
            asm_layout=True,
        )
        return
    reshape_and_cache_shuffle_triton(
        k,
        v,
        k_buffer,
        v_buffer,
        cache_loc,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )
