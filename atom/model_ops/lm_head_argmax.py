import torch
import triton
import triton.language as tl

from aiter.jit.utils.torch_guard import torch_compile_guard

_MAX_BLOCK_M = 131072
# One program reduces one row, so small row counts underutilize the GPU.
_MIN_ROWS_FOR_FUSED_ARGMAX = 16


@triton.jit
def _lm_head_argmax_pack_kernel(
    logits_ptr,
    packed_ptr,
    vocab_start_idx,
    M: tl.constexpr,
    stride_logits_n: tl.constexpr,
    stride_logits_m: tl.constexpr,
    stride_packed_n: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    mask = offs < M
    vals = tl.load(
        logits_ptr + row * stride_logits_n + offs * stride_logits_m,
        mask=mask,
        other=-float("inf"),
    ).to(tl.float32)

    max_val = tl.max(vals, axis=0)
    idxs = offs.to(tl.int64)
    masked_idxs = tl.where((vals == max_val) & mask, idxs, idxs + BLOCK_M)
    local_idx = tl.min(masked_idxs, axis=0)
    global_idx = local_idx + vocab_start_idx

    tl.store(packed_ptr + row * stride_packed_n, max_val)
    tl.store(packed_ptr + row * stride_packed_n + 1, global_idx.to(tl.float32))


def _lm_head_argmax_pack_fake(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 2), dtype=torch.float32, device=logits.device)


def _torch_lm_head_argmax_pack(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    local_max_val, local_idx = logits.max(dim=-1)
    global_idx = local_idx + vocab_start_idx
    return torch.stack([local_max_val.float(), global_idx.float()], dim=-1)


@torch_compile_guard(gen_fake=_lm_head_argmax_pack_fake)
def lm_head_argmax_pack(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    """Reduce local LM-head logits and pack (max_val, global_idx) as fp32."""
    if logits.dim() != 2:
        raise ValueError("lm_head_argmax_pack expects a 2-D logits tensor")

    N, M = logits.shape
    if N == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=logits.device)
    if N < _MIN_ROWS_FOR_FUSED_ARGMAX or M > _MAX_BLOCK_M:
        return _torch_lm_head_argmax_pack(logits, vocab_start_idx)

    packed = torch.empty((N, 2), dtype=torch.float32, device=logits.device)
    block_m = triton.next_power_of_2(M)
    num_warps = 8 if block_m >= 2048 else 4

    _lm_head_argmax_pack_kernel[(N,)](
        logits,
        packed,
        vocab_start_idx,
        M=M,
        stride_logits_n=logits.stride(0),
        stride_logits_m=logits.stride(1),
        stride_packed_n=packed.stride(0),
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=2,
    )
    return packed
