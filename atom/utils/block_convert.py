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
    context_lens_ptr,
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

    # Determine batch (row) and local column (col) using context_lens and ori_block_size.
    # Each batch contributes num_blocks = ceil(ctx_len / ori_block_size) entries to kv_indices.
    row = tl.full([BLOCK_SIZE], -1, dtype=tl.int32)
    col = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.int32)  # running prefix of num_blocks

    # Iterate over batches to place each offset into its batch segment
    # n_rows is small enough to loop; Triton will unroll if constexpr.
    for j in range(0, bs):
        ctx_j = tl.load(context_lens_ptr + j)
        num_blocks_j = (ctx_j + ori_block_size - 1) // ori_block_size

        in_this = (offsets >= acc) & (offsets < acc + num_blocks_j) & mask & (row == -1)
        row = tl.where(in_this, j, row)
        col = tl.where(in_this, offsets - acc, col)

        acc += num_blocks_j

    # For any not found (e.g., offsets beyond total), set safe defaults
    row = tl.maximum(row, 0)
    ctx = tl.load(context_lens_ptr + row, mask=mask, other=0)
    num_blocks = (ctx + ori_block_size - 1) // ori_block_size  # per-thread vector

    # Vectorized expansion over ratio
    r = tl.arange(0, ratio)
    val_exp = val[:, None]
    is_neg_one = val_exp == -1
    new_val = val_exp * ratio + r[None, :]

    # Validate against available blocks after expansion
    # After expansion, each original block splits into `ratio` sub-blocks.
    # Valid if original block index < num_blocks for the batch.
    valid = (col[:, None] < num_blocks[:, None]) & mask[:, None]

    # If original was -1 or exceeds context-derived blocks, write -1
    write_val = tl.where(is_neg_one | (~valid), -1, new_val)

    # Compute output indices in flattened space
    out_idx = offsets[:, None] * ratio + r[None, :]
    tl.store(output_ptr + out_idx, write_val, mask=mask[:, None])


def kv_indices_convert_triton(
    kv_indices, kv_indices_convert, context_lens, ratio, ori_block_size
):
    n_input_elements = kv_indices.numel()
    bs = context_lens.shape[0]
    grid = lambda meta: (triton.cdiv(n_input_elements, meta["BLOCK_SIZE"]),)

    kv_indices_convert_kernel[grid](
        kv_indices,
        kv_indices_convert,
        context_lens,
        ratio,
        ori_block_size,
        bs,
        n_input_elements,
        BLOCK_SIZE=256,
    )

    return kv_indices_convert
