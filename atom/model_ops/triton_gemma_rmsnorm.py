"""Fused add + GemmaRMSNorm Triton kernel.

Replaces the torch.compile'd GemmaRMSNorm.forward_static with a single Triton
kernel that fuses residual add and RMS normalization with the Gemma-style
weight offset: out = rmsnorm(x + residual) * (1 + w).

Based on aiter's _fused_add_rmsnorm_kernel with the (g + 1.0) Gemma offset.

Two custom ops are registered so that torch.compile (Dynamo) can trace through
them without falling back to the PyTorch implementation that contains
x.float() / x.to(orig_dtype) dtype-cast copy kernels:
  - ``fused_gemma_rmsnorm``  (no residual)
  - ``fused_gemma_add_rmsnorm``  (with residual add)
"""

import torch
import triton
import triton.language as tl

# ── Triton kernel ────────────────────────────────────────────────────────────


@triton.jit
def _gemma_rmsnorm_kernel(
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    epsilon,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """Fused add + GemmaRMSNorm (weight offset x * (1 + w)).

    Each program handles multiple rows in a persistent-thread loop.
    Assumes n_cols <= BLOCK_SIZE (single-tile per row).
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pre-load weight (shared across all rows)
    g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    g = g + 1.0  # Gemma offset

    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        input_ptrs = tl.multiple_of(input_ptrs, (16,))
        x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")

        if HAS_RESIDUAL:
            res_in_ptrs = res_in_ptr + row_idx * input_row_stride + col_offsets
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
            x = x + res
            # Store residual_out (needed by next layer)
            res_out_ptrs = res_out_ptr + row_idx * input_row_stride + col_offsets
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            tl.store(res_out_ptrs, x.to(res_out_ptr.dtype.element_ty), mask=mask)

        x = x.to(tl.float32)

        # RMSNorm
        row_norm = tl.sum(x * x, axis=-1)
        norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)

        out = x * norm_factor * g

        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        output_ptrs = tl.multiple_of(output_ptrs, (16,))
        tl.store(output_ptrs, out.to(output_ptr.dtype.element_ty), mask=mask)


# ── Kernel launcher (shared by both custom ops) ─────────────────────────────


def gemma_rmsnorm_triton(x, weight, eps, residual):
    """Launch the Triton kernel. Returns out or (out, residual_out)."""
    ori_shape = x.shape
    x = x.view(-1, ori_shape[-1])
    n_rows, n_cols = x.shape

    out = torch.empty_like(x)

    has_residual = residual is not None
    if has_residual:
        residual = residual.view(-1, ori_shape[-1])
        residual_out = torch.empty_like(residual)
    else:
        residual_out = x  # dummy, won't be written

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    NUM_PRGMS = min(n_rows, 304)  # MI355X has 304 CUs

    _gemma_rmsnorm_kernel[(NUM_PRGMS,)](
        x,
        out,
        residual if has_residual else x,  # dummy for res_in when no residual
        residual_out,
        weight,
        x.stride(0),
        out.stride(0),
        n_rows,
        n_cols,
        eps,
        HAS_RESIDUAL=has_residual,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )

    out = out.view(ori_shape)
    if has_residual:
        return out, residual_out.view(ori_shape)
    return out
