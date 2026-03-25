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
