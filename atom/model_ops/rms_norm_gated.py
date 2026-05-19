import triton
import triton.language as tl

# decode
@triton.jit
def _rmsnorm_gated_contiguous_128_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_ptr,
    num_heads: tl.constexpr,
    eps: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offsets = tl.arange(0, 128)
    row_offset = (token_id * num_heads + head_id) * 128

    x = tl.load(x_ptr + row_offset + offsets, cache_modifier=".ca").to(tl.float32)
    z = tl.load(z_ptr + row_offset + offsets, cache_modifier=".ca").to(tl.float32)
    weight = tl.load(weight_ptr + offsets, cache_modifier=".ca").to(tl.float32)

    variance = tl.sum(x * x, axis=0) * 0.0078125
    inv_rms = tl.rsqrt(variance + eps)
    gate = z * tl.sigmoid(z)
    out = x * inv_rms * weight * gate

    tl.store(out_ptr + row_offset + offsets, out)

# prefill
@triton.jit
def _rmsnorm_gated_contiguous_128_tiled_rows_kernel(
    x_ptr,
    z_ptr,
    weight_ptr,
    out_ptr,
    num_rows: tl.constexpr,
    eps: tl.constexpr,
    block_rows: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    dim_offsets = tl.arange(0, 128)
    mask_rows = row_offsets < num_rows
    offsets = row_offsets[:, None] * 128 + dim_offsets[None, :]

    x = tl.load(x_ptr + offsets, mask=mask_rows[:, None], other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offsets, mask=mask_rows[:, None], other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + dim_offsets, cache_modifier=".ca").to(tl.float32)

    variance = tl.sum(x * x, axis=1) * 0.0078125
    inv_rms = tl.rsqrt(variance + eps)
    gate = z * tl.sigmoid(z)
    out = x * inv_rms[:, None] * weight[None, :] * gate

    tl.store(out_ptr + offsets, out, mask=mask_rows[:, None])
