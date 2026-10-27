# Vendored from SGLang (Apache-2.0), python/sglang/kernels/ops/attention/minimax_sparse/,
# image lmsysorg/sglang@sha256:2d74f6ea..., for the MiniMax-M3 NVFP4 KV path.
import triton
import triton.language as tl


@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH, D] - partials in, merged out at chunk 0
    lse_ptr,  # [NUM_TOPK_CHUNKS, BS, NQH]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr = 128,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)
    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    # Plain pointers: triton 3.8 removed block pointers.
    o_ptrs = (
        o_ptr
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + off_c[:, None] * stride_o_c
        + off_d[None, :] * stride_o_d
    )
    lse_ptrs = lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c
    o = tl.load(o_ptrs, mask=off_d[None, :] < head_dim, other=0.0)
    lse = tl.load(lse_ptrs)  # empty chunks contribute -inf -> weight 0
    lse_max = tl.max(lse, axis=0)
    weights = tl.exp(lse - lse_max)
    weights = weights / tl.sum(weights, axis=0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    o_out_ptrs = o_ptr + pid_b * stride_o_b + pid_h * stride_o_h + off_d * stride_o_d
    tl.store(o_out_ptrs, o_merged.to(o_ptr.dtype.element_ty), mask=off_d < head_dim)
