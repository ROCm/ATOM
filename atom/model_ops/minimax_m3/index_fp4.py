"""NVFP4 index-K cache for MiniMax-M3 (``--index-cache-dtype fp4``).

One cached index key (head dim D = 128) is two planes of the same page-128 block:

- payload ``[blocks, 128, D // 2]`` uint8: two e2m1 codes per byte, the even
  element in the low nibble;
- scale ``[blocks, 128, D // 16]`` uint8: one E4M3 block scale per 16 elements,
  unit outer scale.

This is the format SGLang's MiniMax-M3 NVFP4 index pool stores (72 bytes per
token against 128 for FP8), quantized the same way: per 16 elements
``scale = e4m3(amax / 6)`` and codes rounded half-to-even, saturating at 6.

The fused qk-norm / rope / cache kernel (aiter ``fused_qknorm_idxrqknorm``) only
writes BF16 or FP8 index keys. It writes this step's keys BF16 into a token-major
staging buffer (``index_slot_mapping`` = token number), and
:func:`index_fp4_store` quantizes them into the paged FP4 planes at the step's
real slots. The scorers read the planes through the gfx950 hardware e2m1
decode (``sglang_nvfp4.nvfp4.nvfp4_load_even_odd``).
"""

import torch
import triton
import triton.language as tl

FP4_GROUP = 16


def fp4_index_row_bytes(head_dim: int) -> tuple[int, int]:
    """(payload, scale) bytes of one cached index key."""
    return head_dim // 2, head_dim // FP4_GROUP


@triton.jit
def _f32_to_e2m1(x):
    """Round fp32 to e2m1 codes (round half to even, saturating at 6)."""
    a = tl.abs(x)
    mag = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    return mag | tl.where(x < 0, 8, 0)


@triton.jit
def _nvfp4_store_kernel(
    stage_ptr,  # staged BF16 rows of this step's tokens (any page-major layout)
    slot_ptr,  # [tokens] int64 cache slot of token t (-1 = padding)
    payload_ptr,  # [blocks, (heads,) PAGE, D // 2] uint8
    scale_ptr,  # [blocks, (heads,) PAGE, D // 16] uint8
    num_tokens,
    stride_st_blk,  # stage stride per 128-token stage page
    stride_st_tok,  # stage stride per token inside a stage page
    stride_st_head,
    stride_p_blk,
    stride_p_head,
    stride_p_pos,
    stride_s_blk,
    stride_s_head,
    stride_s_pos,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    HALF: tl.constexpr = D // 2
    NBLK: tl.constexpr = D // 16
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < num_tokens
    slot = tl.load(slot_ptr + t, mask=t_mask, other=-1).to(tl.int64)
    valid = t_mask & (slot >= 0)
    t64 = t.to(tl.int64)
    st_row = (t64 // PAGE) * stride_st_blk + (t64 % PAGE) * stride_st_tok
    page = slot // PAGE
    pos = slot % PAGE
    d = tl.arange(0, D)
    pc = tl.arange(0, HALF)
    sc = tl.arange(0, NBLK)
    for h in tl.static_range(NUM_HEADS):
        x = tl.load(
            stage_ptr + (st_row + h * stride_st_head)[:, None] + d[None, :],
            mask=valid[:, None],
            other=0.0,
        ).to(tl.float32)
        blocks = tl.reshape(x, (BLOCK_T, NBLK, 16))
        amax = tl.max(tl.abs(blocks), axis=2)
        scale_e4m3 = (amax / 6.0).to(tl.float8e4nv)
        scale = scale_e4m3.to(tl.float32)
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        codes = _f32_to_e2m1(blocks * inv[:, :, None])
        lo, hi = tl.split(tl.reshape(codes, (BLOCK_T, HALF, 2)))
        payload = (lo | (hi << 4)).to(tl.uint8)
        p_row = page * stride_p_blk + h * stride_p_head + pos * stride_p_pos
        tl.store(
            payload_ptr + p_row[:, None] + pc[None, :], payload, mask=valid[:, None]
        )
        s_row = page * stride_s_blk + h * stride_s_head + pos * stride_s_pos
        tl.store(
            scale_ptr + s_row[:, None] + sc[None, :],
            scale_e4m3.to(tl.uint8, bitcast=True),
            mask=valid[:, None],
        )


def _nvfp4_store(
    stage, stage_strides, num_heads, head_dim, slot_mapping, payload, scale
):
    """``stage_strides`` = (per stage page, per token, per head) in elements;
    ``payload`` / ``scale`` are ``[blocks, PAGE, x]`` or ``[blocks, heads, PAGE, x]``.
    """
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    if payload.dim() == 3:
        p_strides = (payload.stride(0), 0, payload.stride(1))
        s_strides = (scale.stride(0), 0, scale.stride(1))
        page = payload.shape[1]
    else:
        p_strides = (payload.stride(0), payload.stride(1), payload.stride(2))
        s_strides = (scale.stride(0), scale.stride(1), scale.stride(2))
        page = payload.shape[2]
    block_t = 16
    _nvfp4_store_kernel[(triton.cdiv(num_tokens, block_t),)](
        stage,
        slot_mapping,
        payload,
        scale,
        num_tokens,
        *stage_strides,
        *p_strides,
        *s_strides,
        D=head_dim,
        PAGE=page,
        NUM_HEADS=num_heads,
        BLOCK_T=block_t,
    )


def index_fp4_store(
    stage: torch.Tensor,  # [>= tokens, D] bf16, token-major
    slot_mapping: torch.Tensor,  # [tokens]
    payload: torch.Tensor,  # [blocks, PAGE, D // 2] uint8
    scale: torch.Tensor,  # [blocks, PAGE, D // 16] uint8
) -> None:
    """Quantize this step's staged index keys into the paged FP4 planes."""
    page = payload.shape[1]
    _nvfp4_store(
        stage,
        (page * stage.stride(0), stage.stride(0), 0),
        1,
        stage.shape[-1],
        slot_mapping,
        payload,
        scale,
    )


def index_fp4_dequant_reference(
    payload: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Torch reference: FP4 planes ``[..., D/2]`` / ``[..., D/16]`` -> fp32 ``[..., D]``."""
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=payload.device,
    )
    p = payload.to(torch.int64)
    vals = torch.stack([lut[p & 15], lut[p >> 4]], dim=-1).flatten(-2)
    s = scale.view(torch.float8_e4m3fn).to(torch.float32)
    return (vals.unflatten(-1, (-1, FP4_GROUP)) * s[..., None]).flatten(-2)
