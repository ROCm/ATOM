# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def block_table_convert_kernel(
    blk_table_ptr,
    output_ptr,
    context_lens_ptr,
    ratio: tl.constexpr,
    n_input_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_input_elements

    # Load original values
    val = tl.load(blk_table_ptr + offsets, mask=mask, other=-1)

    # Compute row (batch) and column indices from flattened offsets
    row = offsets // n_cols
    col = offsets % n_cols

    # Load per-batch context length
    ctx = tl.load(context_lens_ptr + row, mask=mask, other=0)

    # Vectorized expansion over ratio
    i = tl.arange(0, ratio)
    val_exp = val[:, None]
    is_neg_one = val_exp == -1
    new_val = val_exp * ratio + i[None, :]

    # For each expanded position, check against context length limit
    expanded_col = col[:, None] * ratio + i[None, :]
    valid = expanded_col < ctx[:, None]

    # If original was -1 or exceeds context, write -1
    write_val = tl.where(is_neg_one | (~valid), -1, new_val)

    # Compute output indices in flattened space
    out_idx = offsets[:, None] * ratio + i[None, :]
    tl.store(output_ptr + out_idx, write_val, mask=mask[:, None])


def block_table_convert_triton(block_table, block_table_convert, context_lens, ratio):
    if not block_table.is_contiguous():
        block_table = block_table.contiguous()
    assert block_table.shape[1] * ratio == block_table_convert.shape[1]
    assert context_lens.shape[0] == block_table.shape[0]

    n_input_elements = block_table.numel()
    n_cols = block_table.shape[1]
    grid = lambda meta: (triton.cdiv(n_input_elements, meta["BLOCK_SIZE"]),)

    block_table_convert_kernel[grid](
        block_table,
        block_table_convert,
        context_lens,
        ratio,
        n_input_elements,
        n_cols,
        BLOCK_SIZE=256,
    )

    return block_table_convert


@triton.jit
def kv_indices_convert_kernel(
    kv_indices_ptr,
    output_ptr,
    kv_indptr_convert_ptr,
    ratio: tl.constexpr,
    ori_block_size,
    bs,
    n_input_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_input_elements

    # Load original values
    val = tl.load(kv_indices_ptr + offsets, mask=mask, other=-1)

    col = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    ctx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    out_bs_start = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    acc_block = 0
    for j in range(0, bs):
        start_j = tl.load(kv_indptr_convert_ptr + j)
        end_j = tl.load(kv_indptr_convert_ptr + (j + 1))
        ctx_len = end_j - start_j
        num_block = tl.cdiv(ctx_len, ori_block_size)
        in_this = (offsets >= acc_block) & (offsets < (acc_block + num_block)) & mask
        col = tl.where(in_this, (offsets - acc_block) * ratio, col)
        ctx = tl.where(in_this, ctx_len, ctx)
        out_bs_start = tl.where(in_this, start_j, out_bs_start)
        acc_block += num_block

    for r_i in range(0, ratio):
        cond = (col + r_i) < ctx
        cand = val * ratio + r_i
        tl.store(output_ptr + out_bs_start + col + r_i, cand, mask=mask & cond)


def kv_indices_convert_triton(
    kv_indices, kv_indices_convert, kv_indptr_convert, ratio, block_size
):
    ori_block_size = block_size * ratio
    bs = kv_indptr_convert.shape[0] - 1
    n_input_elements = kv_indices.numel()
    grid = lambda meta: (triton.cdiv(n_input_elements, meta["BLOCK_SIZE"]),)

    kv_indices_convert_kernel[grid](
        kv_indices,
        kv_indices_convert,
        kv_indptr_convert,
        ratio,
        ori_block_size,
        bs,
        n_input_elements,
        BLOCK_SIZE=256,
    )

    return kv_indices_convert


if __name__ == "__main__":
    # Example usage and test

    block_table = torch.tensor(
        [[0, 1, 2, -1], [3, 4, -1, -1], [5, 6, 7, 8]], dtype=torch.int32
    ).cuda()
    context_lens = torch.tensor([10, 5, 15], dtype=torch.int32).cuda()
    ratio = 4
    block_table_converted = torch.empty(
        (block_table.shape[0], block_table.shape[1] * ratio),
        dtype=torch.int32,
    ).cuda()

    block_table_convert_triton(block_table, block_table_converted, context_lens, ratio)

    print("Original Block Table:")
    print(block_table.cpu().numpy())
    print("Converted Block Table:")
    print(block_table_converted.cpu().numpy())

    kv_indices = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32).cuda()
    kv_indptr_convert = torch.tensor([0, 3, 9, 11, 18], dtype=torch.int32).cuda()
    ori_block_size = 4
    kv_indices_converted = torch.zeros(
        (kv_indices.shape[0] * ratio,), dtype=torch.int32
    ).cuda()
    kv_indices_convert_triton(
        kv_indices, kv_indices_converted, kv_indptr_convert, ratio, ori_block_size
    )
    print("Original KV Indices:")
    print(kv_indices.cpu().numpy())
    print("Converted KV Indices:")
    print(kv_indices_converted.cpu().numpy())
