import torch
import triton
import triton.language as tl
from aiter.jit.utils.torch_guard import torch_compile_guard

# Cap the vector width; larger vocabularies stream through multiple tiles.
_MAX_BLOCK_M = 131072


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
    block_offs = tl.arange(0, BLOCK_M)
    best_val = tl.full((), -float("inf"), dtype=tl.float32)
    invalid_idx = tl.full((), M, dtype=tl.int64)
    best_idx = invalid_idx

    for block_start in tl.range(0, M, BLOCK_M):
        offs = block_start + block_offs
        mask = offs < M
        vals = tl.load(
            logits_ptr + row * stride_logits_n + offs * stride_logits_m,
            mask=mask,
            other=-float("inf"),
        ).to(tl.float32)

        block_max = tl.max(vals, axis=0)
        idxs = offs.to(tl.int64)
        block_idx = tl.min(
            tl.where((vals == block_max) & mask, idxs, invalid_idx), axis=0
        )
        take_block = (block_max > best_val) | (
            (block_max == best_val) & (block_idx < best_idx)
        )
        best_val = tl.where(take_block, block_max, best_val)
        best_idx = tl.where(take_block, block_idx, best_idx)

    global_idx = best_idx + vocab_start_idx
    tl.store(packed_ptr + row * stride_packed_n, best_val)
    tl.store(packed_ptr + row * stride_packed_n + 1, global_idx.to(tl.float32))


def _lm_head_argmax_pack_fake(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    return torch.empty((logits.shape[0], 2), dtype=torch.float32, device=logits.device)


@torch_compile_guard(gen_fake=_lm_head_argmax_pack_fake)
def lm_head_argmax_pack(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    """Reduce and fp32-pack local LM-head argmax in one Triton launch."""
    if logits.dim() != 2:
        raise ValueError("lm_head_argmax_pack expects a 2-D logits tensor")

    N, M = logits.shape
    if N == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=logits.device)
    if M == 0:
        raise ValueError("lm_head_argmax_pack requires a non-empty vocabulary")

    packed = torch.empty((N, 2), dtype=torch.float32, device=logits.device)
    block_m = min(_MAX_BLOCK_M, triton.next_power_of_2(M))
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
