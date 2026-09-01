"""GLM/DeepSeek MTP embedding + dual RMSNorm + FP8 quant fusion."""

import aiter
import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

_FP8_DTYPE = aiter.dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)


@triton.jit
def _fused_mtp_embedding_dual_rmsnorm_fp8_quant_kernel(
    input_ids_ptr,
    embedding_weight_ptr,
    hidden_ptr,
    enorm_weight_ptr,
    hnorm_weight_ptr,
    out_ptr,
    scale_ptr,
    vocab_size,
    hidden_size,
    embedding_stride_row,
    hidden_stride_row,
    out_stride_row,
    eps,
    FP8_MAX: tl.constexpr,
    FP8_INV: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_H)
    col_offsets = cols.to(tl.int64)
    mask = cols < hidden_size

    token_id = tl.load(input_ids_ptr + row).to(tl.int64)
    token_valid = (token_id >= 0) & (token_id < vocab_size)
    safe_token_id = tl.where(token_valid, token_id, 0)
    embed = tl.load(
        embedding_weight_ptr + safe_token_id * embedding_stride_row + col_offsets,
        mask=token_valid & mask,
        other=0.0,
    ).to(tl.float32)
    hidden = tl.load(
        hidden_ptr + row * hidden_stride_row + col_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    enorm_weight = tl.load(enorm_weight_ptr + col_offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    hnorm_weight = tl.load(hnorm_weight_ptr + col_offsets, mask=mask, other=0.0).to(
        tl.float32
    )

    embed_rstd = tl.rsqrt(tl.sum(embed * embed, axis=0) / hidden_size + eps)
    hidden_rstd = tl.rsqrt(tl.sum(hidden * hidden, axis=0) / hidden_size + eps)

    # Match the unfused path's BF16 materialization before dynamic quantization.
    norm_embed = (embed * embed_rstd * enorm_weight).to(
        embedding_weight_ptr.dtype.element_ty
    )
    norm_hidden = (hidden * hidden_rstd * hnorm_weight).to(hidden_ptr.dtype.element_ty)
    norm_embed_f32 = norm_embed.to(tl.float32)
    norm_hidden_f32 = norm_hidden.to(tl.float32)

    absmax = tl.maximum(
        tl.max(tl.abs(norm_embed_f32), axis=0),
        tl.max(tl.abs(norm_hidden_f32), axis=0),
    )
    scale = absmax * FP8_INV
    inv_scale = tl.where(absmax > 0.0, 1.0 / scale, 0.0)

    quant_embed = tl.clamp(norm_embed_f32 * inv_scale, -FP8_MAX, FP8_MAX)
    quant_hidden = tl.clamp(norm_hidden_f32 * inv_scale, -FP8_MAX, FP8_MAX)
    tl.store(
        out_ptr + row * out_stride_row + col_offsets,
        quant_embed.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        out_ptr + row * out_stride_row + hidden_size + col_offsets,
        quant_hidden.to(out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(scale_ptr + row, scale)


def _fused_mtp_prologue_fake(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del input_ids, enorm_weight, hnorm_weight, eps
    rows, hidden_size = previous_hidden_states.shape
    quantized = torch.empty(
        (rows, 2 * hidden_size),
        dtype=_FP8_DTYPE,
        device=previous_hidden_states.device,
    )
    scale = torch.empty(
        (rows, 1), dtype=torch.float32, device=previous_hidden_states.device
    )
    return quantized, scale


@torch_compile_guard(gen_fake=_fused_mtp_prologue_fake)
def fused_mtp_embedding_dual_rmsnorm_fp8_quant(
    input_ids: torch.Tensor,
    embedding_weight: torch.Tensor,
    previous_hidden_states: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Produce the pre-quantized ``eh_proj`` input in one Triton launch."""
    assert input_ids.ndim == 1
    assert previous_hidden_states.ndim == 2
    rows, hidden_size = previous_hidden_states.shape
    assert input_ids.shape[0] == rows
    assert embedding_weight.shape[1] == hidden_size
    assert enorm_weight.shape == (hidden_size,)
    assert hnorm_weight.shape == (hidden_size,)
    assert embedding_weight.dtype == previous_hidden_states.dtype
    assert previous_hidden_states.is_contiguous()

    quantized, scale = _fused_mtp_prologue_fake(
        input_ids,
        embedding_weight,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        eps,
    )
    if rows == 0:
        return quantized, scale

    block_h = triton.next_power_of_2(hidden_size)
    _fused_mtp_embedding_dual_rmsnorm_fp8_quant_kernel[(rows,)](
        input_ids,
        embedding_weight,
        previous_hidden_states,
        enorm_weight,
        hnorm_weight,
        quantized,
        scale,
        embedding_weight.shape[0],
        hidden_size,
        embedding_weight.stride(0),
        previous_hidden_states.stride(0),
        quantized.stride(0),
        eps,
        FP8_MAX=_FP8_MAX,
        FP8_INV=1.0 / _FP8_MAX,
        BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4,
        num_stages=2,
    )
    return quantized, scale
