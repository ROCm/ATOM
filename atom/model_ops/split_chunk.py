"""
Optimized operations for Qwen3.5 model.

Fused Triton kernel with coalesced memory access for ba and z.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def fused_split_chunk_kernel(
    # Inputs
    mixed_qkvz_ptr,  # [num_tokens, qkv_size + z_size]
    ba_ptr,  # [num_tokens, num_v_heads_tp * 2]
    # Outputs
    mixed_qkv_ptr,  # [num_tokens, qkv_size]
    z_ptr,  # [num_tokens, num_v_heads_tp, head_v_dim]
    b_ptr,  # [num_tokens, num_v_heads_tp]
    a_ptr,  # [num_tokens, num_v_heads_tp]
    core_attn_out_ptr,  # [num_tokens, num_v_heads_tp, head_v_dim]
    # Dimensions
    num_tokens,
    qkv_size: tl.constexpr,
    z_size: tl.constexpr,
    head_v_dim: tl.constexpr,
    num_v_heads_tp: tl.constexpr,
    # Block sizes
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel with coalesced memory access.

    Grid: (num_tokens, num_v_heads_tp + num_qkv_chunks + 1)
    - work_id < num_v_heads_tp: Load head_v_dim elements for z[head_id] (coalesced)
    - work_id == num_v_heads_tp: Load entire ba (2*num_v_heads_tp elements, coalesced)
    - work_id > num_v_heads_tp: Load chunks of mixed_qkv (coalesced)

    Each thread block has BLOCK_SIZE threads for coalesced access.
    """
    token_id = tl.program_id(0)
    work_id = tl.program_id(1)

    if work_id < num_v_heads_tp:
        # Process one head's z and core_attn_out
        # All threads in block cooperatively load head_v_dim elements (coalesced)
        head_id = work_id

        # Load z: coalesced read of head_v_dim consecutive elements
        # Source: mixed_qkvz[token_id, qkv_size + head_id*head_v_dim : qkv_size + (head_id+1)*head_v_dim]
        dim_idx = tl.arange(0, BLOCK_SIZE)
        mask = dim_idx < head_v_dim

        qkvz_base = token_id * (qkv_size + z_size) + qkv_size + head_id * head_v_dim
        z_vals = tl.load(mixed_qkvz_ptr + qkvz_base + dim_idx, mask=mask, other=0.0)

        # Store z: coalesced write
        z_out_base = token_id * num_v_heads_tp * head_v_dim + head_id * head_v_dim
        tl.store(z_ptr + z_out_base + dim_idx, z_vals, mask=mask)

        # Store zeros to core_attn_out: coalesced write
        zeros = tl.zeros([BLOCK_SIZE], dtype=z_vals.dtype)
        tl.store(core_attn_out_ptr + z_out_base + dim_idx, zeros, mask=mask)

    elif work_id == num_v_heads_tp:
        # Special block: process entire ba in one coalesced load
        # Load 2*num_v_heads_tp elements with threads cooperatively
        ba_idx = tl.arange(0, BLOCK_SIZE)
        ba_size = num_v_heads_tp * 2
        mask = ba_idx < ba_size

        ba_base = token_id * ba_size
        ba_vals = tl.load(ba_ptr + ba_base + ba_idx, mask=mask, other=0.0)

        # Split and store: first half to b, second half to a
        # This creates two separate stores but threads access consecutive addresses
        b_mask = ba_idx < num_v_heads_tp
        a_mask = (ba_idx >= num_v_heads_tp) & (ba_idx < ba_size)

        b_base = token_id * num_v_heads_tp
        a_base = token_id * num_v_heads_tp

        # Store b (first num_v_heads_tp elements)
        tl.store(b_ptr + b_base + ba_idx, ba_vals, mask=b_mask)

        # Store a (last num_v_heads_tp elements, shift indices)
        a_idx = ba_idx - num_v_heads_tp
        tl.store(a_ptr + a_base + a_idx, ba_vals, mask=a_mask)

    else:
        # Mixed_qkv blocks: process chunks with coalesced access
        # Threads cooperatively load BLOCK_SIZE consecutive elements
        chunk_id = work_id - num_v_heads_tp - 1
        chunk_start = chunk_id * BLOCK_SIZE

        if chunk_start < qkv_size:  # Guard at block level
            cols = tl.arange(0, BLOCK_SIZE)
            mask = (chunk_start + cols) < qkv_size

            qkvz_base = token_id * (qkv_size + z_size) + chunk_start
            qkv_out_base = token_id * qkv_size + chunk_start

            vals = tl.load(mixed_qkvz_ptr + qkvz_base + cols, mask=mask, other=0.0)
            tl.store(mixed_qkv_ptr + qkv_out_base + cols, vals, mask=mask)


def fused_split_chunk_zeros(
    mixed_qkvz: torch.Tensor,  # [num_tokens, qkv_size + z_size]
    ba: torch.Tensor,  # [num_tokens, num_v_heads_tp * 2]
    qkv_size: int,
    z_size: int,
    head_v_dim: int,
    num_v_heads_tp: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused operation with coalesced memory access.

    Grid design for high occupancy and coalescing:
    - num_v_heads_tp blocks: Each loads one head's z data (head_v_dim elements, coalesced)
    - 1 block: Loads entire ba (2*num_v_heads_tp elements, coalesced)
    - num_qkv_blocks: Each loads chunk of mixed_qkv (BLOCK_SIZE elements, coalesced)

    Args:
        mixed_qkvz: Input tensor from in_proj_qkvz [M, qkv_size + z_size]
        ba: Input tensor from in_proj_ba [M, num_v_heads_tp * 2]
        qkv_size: Size of qkv portion
        z_size: Size of z portion (= num_v_heads_tp * head_v_dim)
        head_v_dim: Dimension per value head
        num_v_heads_tp: Number of value heads after TP split

    Returns:
        Tuple of (mixed_qkv, z, b, a, core_attn_out)
    """
    num_tokens = mixed_qkvz.size(0)
    dtype = mixed_qkvz.dtype
    device = mixed_qkvz.device

    # Allocate outputs
    mixed_qkv = torch.empty(num_tokens, qkv_size, dtype=dtype, device=device)
    z = torch.empty(num_tokens, num_v_heads_tp, head_v_dim, dtype=dtype, device=device)
    b = torch.empty(num_tokens, num_v_heads_tp, dtype=dtype, device=device)
    a = torch.empty(num_tokens, num_v_heads_tp, dtype=dtype, device=device)
    core_attn_out = torch.empty(
        num_tokens, num_v_heads_tp, head_v_dim, dtype=dtype, device=device
    )

    # Block size for vectorized access
    # Must be >= head_v_dim (128) and >= 2*num_v_heads_tp (32)
    BLOCK_SIZE = 128

    # Calculate number of blocks needed for mixed_qkv
    num_qkv_blocks = (qkv_size + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Grid:
    # - num_v_heads_tp blocks for z processing (16 for Qwen3.5 with tp=2)
    # - 1 block for ba processing
    # - num_qkv_blocks for mixed_qkv processing (32 for qkv_size=4096)
    # Total for M=1: (1, 16 + 1 + 32) = 49 blocks
    grid = (num_tokens, num_v_heads_tp + 1 + num_qkv_blocks)

    # Launch kernel
    fused_split_chunk_kernel[grid](
        mixed_qkvz,
        ba,
        mixed_qkv,
        z,
        b,
        a,
        core_attn_out,
        num_tokens,
        qkv_size,
        z_size,
        head_v_dim,
        num_v_heads_tp,
        BLOCK_SIZE,
    )

    return mixed_qkv, z, b, a, core_attn_out


@triton.jit
def fused_split_chunk_kernel_qwen3_5_qkvzba(
    qkvzba_ptr,
    mixed_qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    core_attn_out_ptr,
    num_tokens,
    qkv_size: tl.constexpr,
    z_size: tl.constexpr,
    head_v_dim: tl.constexpr,
    num_v_heads_tp: tl.constexpr,
    total_dim: tl.constexpr,
    block_size: tl.constexpr,
):
    token_id = tl.program_id(0)
    work_id = tl.program_id(1)

    if work_id < num_v_heads_tp:
        head_id = work_id
        cols = tl.arange(0, block_size)
        mask = cols < head_v_dim
        src = token_id * total_dim + qkv_size + head_id * head_v_dim + cols
        vals = tl.load(qkvzba_ptr + src, mask=mask, other=0.0)
        dst = token_id * num_v_heads_tp * head_v_dim + head_id * head_v_dim + cols
        tl.store(z_ptr + dst, vals, mask=mask)
        zeros = tl.zeros([block_size], dtype=core_attn_out_ptr.type.element_ty)
        tl.store(core_attn_out_ptr + dst, zeros, mask=mask)
    elif work_id == num_v_heads_tp:
        cols = tl.arange(0, block_size)
        ba_size = num_v_heads_tp * 2
        mask = cols < ba_size
        src = token_id * total_dim + qkv_size + z_size + cols
        vals = tl.load(qkvzba_ptr + src, mask=mask, other=0.0)
        b_mask = cols < num_v_heads_tp
        a_mask = (cols >= num_v_heads_tp) & (cols < ba_size)
        dst = token_id * num_v_heads_tp
        tl.store(b_ptr + dst + cols, vals, mask=b_mask)
        tl.store(a_ptr + dst + (cols - num_v_heads_tp), vals, mask=a_mask)
    else:
        chunk_id = work_id - num_v_heads_tp - 1
        chunk_start = chunk_id * block_size
        if chunk_start < qkv_size:
            cols = tl.arange(0, block_size)
            mask = (chunk_start + cols) < qkv_size
            src = token_id * total_dim + chunk_start + cols
            dst = token_id * qkv_size + chunk_start + cols
            vals = tl.load(qkvzba_ptr + src, mask=mask, other=0.0)
            tl.store(mixed_qkv_ptr + dst, vals, mask=mask)


def fused_split_chunk_zeros_qwen3_5_qkvzba(
    qkvzba: torch.Tensor,
    num_k_heads_tp: int,
    num_v_heads_tp: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, dtype, device = qkvzba.shape[0], qkvzba.dtype, qkvzba.device
    qkv_size = 2 * num_k_heads_tp * head_k_dim + num_v_heads_tp * head_v_dim
    z_size = num_v_heads_tp * head_v_dim
    total_dim = qkv_size + z_size + num_v_heads_tp * 2

    mixed_qkv = torch.empty(num_tokens, qkv_size, dtype=dtype, device=device)
    z = torch.empty(num_tokens, num_v_heads_tp, head_v_dim, dtype=dtype, device=device)
    b = torch.empty(num_tokens, num_v_heads_tp, dtype=dtype, device=device)
    a = torch.empty(num_tokens, num_v_heads_tp, dtype=dtype, device=device)
    core_attn_out = torch.empty(
        num_tokens, num_v_heads_tp, head_v_dim, dtype=dtype, device=device
    )

    block_size = 128
    num_qkv_blocks = (qkv_size + block_size - 1) // block_size
    grid = (num_tokens, num_v_heads_tp + 1 + num_qkv_blocks)

    fused_split_chunk_kernel_qwen3_5_qkvzba[grid](
        qkvzba,
        mixed_qkv,
        z,
        b,
        a,
        core_attn_out,
        num_tokens,
        qkv_size,
        z_size,
        head_v_dim,
        num_v_heads_tp,
        total_dim,
        block_size,
    )

    return mixed_qkv, z, b, a, core_attn_out


@triton.jit
def fused_split_chunk_kernel_qwen_next_qkvz_ba(
    qkvz_ptr,  # [num_tokens, (num_k_heads//tp) * (2*head_k_dim + 2*head_v_dim*kv_ratio)]
    ba_ptr,  # [num_tokens, (num_v_heads//tp) * 2]
    qkv_ptr,  # [num_tokens, 2*num_k_heads*head_k_dim + num_v_heads*head_v_dim] (output)
    z_ptr,  # [num_tokens, num_v_heads, head_v_dim] (output)
    b_ptr,  # [num_tokens, num_v_heads] (output)
    a_ptr,  # [num_tokens, num_v_heads] (output)
    core_attn_out_ptr,  # [num_tokens, num_v_heads, head_v_dim] (output, zeros)
    num_tokens,
    num_k_heads: tl.constexpr,  # After TP split
    num_v_heads: tl.constexpr,  # After TP split
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
):
    """
    Process separate qkvz and ba projections into mixed_qkv, z, b, a, core_attn_out.

    Optimized with 1D grid for better memory coalescing and cache locality.
    Grid: (num_tokens * (num_k_heads + 1) + num_tokens * num_v_heads,)
    """
    pid = tl.program_id(0)

    # Split work into three phases using single PID for better coalescing
    qkvz_process_limit = num_tokens * num_k_heads
    ba_process_limit = qkvz_process_limit + num_tokens

    KV_HEAD_RATIO: tl.constexpr = num_v_heads // num_k_heads
    QKVZ_DIM_SIZE: tl.constexpr = 2 * head_k_dim + 2 * head_v_dim * KV_HEAD_RATIO

    if pid < qkvz_process_limit:
        # Phase 1: Process qkvz - sequential PIDs process sequential tokens for coalescing
        token_id = pid // num_k_heads
        head_id = pid % num_k_heads

        qkvz_base = token_id * num_k_heads * QKVZ_DIM_SIZE + head_id * QKVZ_DIM_SIZE
        qkv_base = token_id * (2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim)
        z_base = token_id * num_v_heads * head_v_dim

        k_dim_offset = tl.arange(0, head_k_dim)
        v_dim_offset = tl.arange(0, head_v_dim)

        # Coalesced loads and stores
        q_val = tl.load(qkvz_ptr + qkvz_base + k_dim_offset)
        q_store = qkv_base + head_id * head_k_dim
        tl.store(qkv_ptr + q_store + k_dim_offset, q_val)

        k_val = tl.load(qkvz_ptr + qkvz_base + head_k_dim + k_dim_offset)
        k_store = qkv_base + num_k_heads * head_k_dim + head_id * head_k_dim
        tl.store(qkv_ptr + k_store + k_dim_offset, k_val)

        # v and z with static unrolling for better performance
        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            v_val = tl.load(
                qkvz_ptr
                + qkvz_base
                + 2 * head_k_dim
                + sub_head * head_v_dim
                + v_dim_offset
            )
            v_head_id = head_id * KV_HEAD_RATIO + sub_head
            v_store = qkv_base + 2 * num_k_heads * head_k_dim + v_head_id * head_v_dim
            tl.store(qkv_ptr + v_store + v_dim_offset, v_val)

        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            z_val = tl.load(
                qkvz_ptr
                + qkvz_base
                + 2 * head_k_dim
                + KV_HEAD_RATIO * head_v_dim
                + sub_head * head_v_dim
                + v_dim_offset
            )
            z_head_id = head_id * KV_HEAD_RATIO + sub_head
            z_store = z_base + z_head_id * head_v_dim
            tl.store(z_ptr + z_store + v_dim_offset, z_val)

    elif pid < ba_process_limit:
        # Phase 2: Process ba - one token per thread for coalescing
        token_id = pid - qkvz_process_limit

        ba_offset = token_id * num_v_heads * 2
        b_offset = tl.arange(0, num_v_heads) * 2
        a_offset = b_offset + 1

        b_val = tl.load(ba_ptr + ba_offset + b_offset)
        a_val = tl.load(ba_ptr + ba_offset + a_offset)

        store_offset = token_id * num_v_heads + tl.arange(0, num_v_heads)
        tl.store(b_ptr + store_offset, b_val)
        tl.store(a_ptr + store_offset, a_val)

    else:
        # Phase 3: Initialize core_attn_out to zeros
        zero_fill_pid = pid - ba_process_limit
        token_id = zero_fill_pid // num_v_heads
        v_head_id = zero_fill_pid % num_v_heads

        core_out_base = token_id * num_v_heads * head_v_dim + v_head_id * head_v_dim
        v_dim_offset = tl.arange(0, head_v_dim)
        val = tl.zeros([head_v_dim], dtype=core_attn_out_ptr.type.element_ty)
        tl.store(core_attn_out_ptr + core_out_base + v_dim_offset, val)


def fused_split_chunk_qwen_next_qkvz_ba(
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, dtype, device = qkvz.shape[0], qkvz.dtype, qkvz.device
    mixed_qkv = torch.empty(
        [
            num_tokens,
            2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim,
        ],
        dtype=dtype,
        device=device,
    )
    z = torch.empty([num_tokens, num_v_heads, head_v_dim], dtype=dtype, device=device)
    b = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
    a = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
    core_attn_out = torch.empty(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=dtype,
        device=device,
    )

    grid = (num_tokens * (num_k_heads + 1) + num_tokens * num_v_heads,)
    fused_split_chunk_kernel_qwen_next_qkvz_ba[grid](
        qkvz,
        ba,
        mixed_qkv,
        z,
        b,
        a,
        core_attn_out,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    return mixed_qkv, z, b, a, core_attn_out


@triton.jit
def fused_split_chunk_kernel_qwen_next_qkvzba(
    qkvzba_ptr,
    qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    core_attn_out_ptr,
    num_tokens,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
):
    # token_id = tl.program_id(0)
    # head_id = tl.program_id(1)
    pid = tl.program_id(0)
    qkvzba_process_limit = num_tokens * (num_k_heads + 1)
    if pid >= qkvzba_process_limit:
        zero_fill_pid = pid - qkvzba_process_limit
        token_id = zero_fill_pid // num_v_heads
        head_id = zero_fill_pid % num_v_heads
        core_attn_out_base_ptr = (
            core_attn_out_ptr
            + token_id * num_v_heads * head_v_dim
            + head_id * head_v_dim
        )
        v_dim_offset = tl.arange(0, head_v_dim)
        val = tl.zeros([head_v_dim], dtype=core_attn_out_ptr.type.element_ty)
        tl.store(core_attn_out_base_ptr + v_dim_offset, val)
        return
    num_heads_for_qkvzba: tl.constexpr = num_k_heads + 1
    token_id = pid // num_heads_for_qkvzba
    head_id = pid % num_heads_for_qkvzba
    QKVZ_TOTAL_SIZE = 2 * num_k_heads * head_k_dim + 2 * num_v_heads * head_v_dim
    QKVZ_DIM_SIZE = 2 * head_k_dim + 2 * head_v_dim * num_v_heads // num_k_heads
    BA_TOTAL_SIZE = 2 * num_v_heads
    KV_HEAD_RATIO: tl.constexpr = num_v_heads // num_k_heads
    ROW_SIZE = QKVZ_TOTAL_SIZE + BA_TOTAL_SIZE
    if head_id == num_k_heads:  # ba
        load_ptr = qkvzba_ptr + token_id * ROW_SIZE + QKVZ_TOTAL_SIZE
        b_offset = (
            tl.arange(0, num_v_heads) // KV_HEAD_RATIO * KV_HEAD_RATIO * 2
            + tl.arange(0, num_v_heads) % KV_HEAD_RATIO
        )
        a_offset = b_offset + KV_HEAD_RATIO
        b_val = tl.load(load_ptr + b_offset)
        a_val = tl.load(load_ptr + a_offset)
        store_offset = tl.arange(0, num_v_heads)
        tl.store(b_ptr + token_id * num_v_heads + store_offset, b_val)
        tl.store(a_ptr + token_id * num_v_heads + store_offset, a_val)
    else:
        base_ptr = qkvzba_ptr + token_id * ROW_SIZE + head_id * QKVZ_DIM_SIZE
        qkv_base_ptr = qkv_ptr + token_id * (
            2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        )
        k_dim_offset = tl.arange(0, head_k_dim)
        v_dim_offset = tl.arange(0, head_v_dim)
        q_val = tl.load(base_ptr + k_dim_offset)
        q_store_ptr = qkv_base_ptr + head_id * head_k_dim
        tl.store(q_store_ptr + k_dim_offset, q_val)

        k_val = tl.load(base_ptr + head_k_dim + k_dim_offset)
        k_store_ptr = qkv_base_ptr + num_k_heads * head_k_dim + head_id * head_k_dim
        tl.store(k_store_ptr + k_dim_offset, k_val)

        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            v_val = tl.load(
                base_ptr + 2 * head_k_dim + sub_head * head_v_dim + v_dim_offset
            )
            v_store_ptr = (
                qkv_base_ptr
                + 2 * num_k_heads * head_k_dim
                + (head_id * KV_HEAD_RATIO + sub_head) * head_v_dim
            )
            tl.store(v_store_ptr + v_dim_offset, v_val)

        for sub_head in tl.static_range(0, KV_HEAD_RATIO):
            z_val = tl.load(
                base_ptr
                + 2 * head_k_dim
                + KV_HEAD_RATIO * head_v_dim
                + sub_head * head_v_dim
                + v_dim_offset
            )
            z_store_ptr = (
                z_ptr
                + token_id * num_v_heads * head_v_dim
                + (head_id * KV_HEAD_RATIO + sub_head) * head_v_dim
            )
            tl.store(z_store_ptr + v_dim_offset, z_val)


def fused_split_chunk_qwen_next_qkvzba(
    qkvzba: torch.Tensor,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, dtype, device = qkvzba.shape[0], qkvzba.dtype, qkvzba.device
    mixed_qkv = torch.empty(
        [
            num_tokens,
            2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim,
        ],
        dtype=dtype,
        device=device,
    )
    z = torch.empty([num_tokens, num_v_heads, head_v_dim], dtype=dtype, device=device)
    b = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
    a = torch.empty([num_tokens, num_v_heads], dtype=dtype, device=device)
    core_attn_out = torch.empty(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=dtype,
        device=device,
    )
    grid = (num_tokens * (num_k_heads + 1) + num_tokens * num_v_heads,)

    fused_split_chunk_kernel_qwen_next_qkvzba[grid](
        qkvzba,
        mixed_qkv,
        z,
        b,
        a,
        core_attn_out,
        num_tokens,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    return mixed_qkv, z, b, a, core_attn_out
