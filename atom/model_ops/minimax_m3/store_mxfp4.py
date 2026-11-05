# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Norm-free K/V write into the MXFP4 KV pool (``--kv-cache-dtype fp4``).

The counterpart of ``gather_mxfp4`` for a layer that has no q/k norm (EAGLE3's
draft attention): ``aiter.fused_qknorm_idxrqknorm`` always RMS-norms, so it cannot
write such a layer. The caller rotates q/k (RoPE, bf16) and this kernel quantizes
the already-rotated K and the raw V rows into the pages, one (token, kv head) a
program, at the slot ``slot_mapping`` names.

Page contract (``mxfp4_pages``): a page of one head is
``[128 tokens x 64 B e2m1][128 tokens x 4 B e8m0]`` -- the low nibble the even
element, one e8m0 per 32 consecutive head-dim elements. The quantization is the
one the aiter fused writer and ``per_1x32_f4_quant`` make (``_mxfp4_quant_op``,
``SCALING_MODE=1``: ``2^ceil(log2(amax / 6))``), so a written scale byte is in
1..254: an all-zero row gets byte 1 (the op floors amax at ``6 * 2^-126``), never
0, which ``pa_decode`` does not support in a valid row; only a non-finite amax
gives 255. The elements are bf16 as given: the same rounding the aiter writer
applies before it quantizes.

``slot_mapping[t] < 0`` skips the token and leaves its page bytes untouched:
the pad row of a captured draft step, a graph-capture blank, a DCP rank's foreign
token. The grid is the row count of ``k`` (fixed at capture) and the slots are
read from device memory, so a captured step replays with new slots; nothing here
syncs the host or allocates.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

from atom.model_ops.minimax_m3.mxfp4_pages import (
    DATA_BYTES,
    HEAD_DIM,
    PAGE_BYTES,
    PAGE_TOKENS,
    SCALE_BYTES,
)

# jit code reads globals only as constexpr
_HEAD_DIM = tl.constexpr(HEAD_DIM)
_DATA_BYTES = tl.constexpr(DATA_BYTES)
_SCALE_BYTES = tl.constexpr(SCALE_BYTES)
_PAGE_TOKENS = tl.constexpr(PAGE_TOKENS)


@triton.jit
def _store_row(src_ptr, pages_ptr, page, tok, head, stride_page, stride_head):
    x = tl.load(src_ptr + tl.arange(0, _HEAD_DIM)).to(tl.float32)
    packed, e8m0 = _mxfp4_quant_op(
        tl.reshape(x, (1, _HEAD_DIM)), _HEAD_DIM, 1, 32,  # one e8m0 per 32 elements
        SCALING_MODE=1, USE_ASM=True,
    )  # fmt: skip
    base = pages_ptr + page * stride_page + head * stride_head
    tl.store(
        base + tok * _DATA_BYTES + tl.arange(0, _DATA_BYTES),
        tl.reshape(packed, (_DATA_BYTES,)),
    )
    tl.store(
        base
        + _PAGE_TOKENS * _DATA_BYTES
        + tok * _SCALE_BYTES
        + tl.arange(0, _SCALE_BYTES),
        tl.reshape(e8m0, (_SCALE_BYTES,)),
    )


@triton.jit
def _store_kv_mxfp4_kernel(
    k_ptr,
    v_ptr,
    k_pages_ptr,
    v_pages_ptr,
    slot_ptr,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    kp_stride_page,
    kp_stride_head,
    vp_stride_page,
    vp_stride_head,
):
    # int64: a token's offset into a packed qkv row exceeds int32 on long chunks
    token = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    slot = tl.load(slot_ptr + token).to(tl.int64)
    if slot < 0:
        return
    page = slot // _PAGE_TOKENS
    tok = slot % _PAGE_TOKENS
    _store_row(
        k_ptr + token * k_stride_t + head * k_stride_h,
        k_pages_ptr, page, tok, head, kp_stride_page, kp_stride_head,
    )  # fmt: skip
    _store_row(
        v_ptr + token * v_stride_t + head * v_stride_h,
        v_pages_ptr, page, tok, head, vp_stride_page, vp_stride_head,
    )  # fmt: skip


@torch.no_grad()
def store_kv_mxfp4(
    k: torch.Tensor,  # bf16 [tokens, Hkv, 128], unit last stride (a view of qkv is fine)
    v: torch.Tensor,  # bf16 [tokens, Hkv, 128]
    k_pages: torch.Tensor,  # uint8 [pages, Hkv, 8704], Mxfp4KvLayout.kv_views
    v_pages: torch.Tensor,
    slot_mapping: torch.Tensor,  # int32 / int64 [>= tokens]; < 0 skips the token
) -> None:
    """Quantize ``k`` / ``v`` rows into the pages at their slots (see the module).

    ``k`` / ``v`` keep their own row and head strides; the pages are addressed by
    their real page and head strides, so a pool field that is not tightly packed
    is fine. Only the first ``k.shape[0]`` slots are read.
    """
    for name, t in (("k", k), ("v", v)):
        if (
            t.dtype != torch.bfloat16
            or t.dim() != 3
            or t.shape[-1] != HEAD_DIM
            or t.stride(-1) != 1
        ):
            raise ValueError(
                f"{name} must be bf16 [tokens, Hkv, {HEAD_DIM}] with a unit last "
                f"stride, got {t.dtype} {tuple(t.shape)} strides {tuple(t.stride())}"
            )
    if k.shape != v.shape:
        raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} must match")
    tokens, heads, _ = k.shape
    for name, pages in (("k", k_pages), ("v", v_pages)):
        if (
            pages.dtype != torch.uint8
            or pages.dim() != 3
            or pages.shape[1] != heads
            or pages.shape[2] != PAGE_BYTES
            or pages.stride(2) != 1
        ):
            raise ValueError(
                f"{name}_pages must be uint8 [pages, {heads}, {PAGE_BYTES}] with a "
                f"contiguous byte axis, got {pages.dtype} {tuple(pages.shape)} "
                f"strides {tuple(pages.stride())}"
            )
    if k_pages.shape[0] != v_pages.shape[0]:
        raise ValueError("k and v pages must have the same page count")
    if (
        slot_mapping.dtype not in (torch.int32, torch.int64)
        or slot_mapping.dim() != 1
        or not slot_mapping.is_contiguous()
        or slot_mapping.shape[0] < tokens
    ):
        raise ValueError(
            f"slot_mapping must be contiguous int32/int64 [>= {tokens}], got "
            f"{slot_mapping.dtype} {tuple(slot_mapping.shape)}"
        )
    if tokens == 0:
        return
    _store_kv_mxfp4_kernel[(tokens, heads)](
        k,
        v,
        k_pages,
        v_pages,
        slot_mapping,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        k_pages.stride(0),
        k_pages.stride(1),
        v_pages.stride(0),
        v_pages.stride(1),
        num_warps=1,
    )


def require_prefill_slots_written(
    slot_mapping: torch.Tensor, cu_seqlens_q: torch.Tensor, tokens: int, num_pages: int
) -> None:
    """Raise unless every query token of a request has a slot inside the pool.

    A prefill that gathers the full KV back out of the pages reads whatever sits
    where a skipped token (slot < 0) should have been written: zero bytes, or
    another request's page, as if it were this request's key. So the gather must
    never run past one. The tokens past the last ``cu_seqlens_q`` entry belong to
    no request (a TBO ubatch's pad) and may be -1.

    One host sync (the answer is a Python bool); prefill is eager. Raised, not
    asserted, so ``python -O`` keeps it.
    """
    if slot_mapping.shape[0] < tokens:
        raise RuntimeError(
            f"slot_mapping covers {slot_mapping.shape[0]} tokens, the prefill has "
            f"{tokens}"
        )
    slots = slot_mapping[:tokens]
    covered = torch.arange(tokens, device=slots.device) < cu_seqlens_q[-1]
    bad = covered & ((slots < 0) | (slots >= num_pages * PAGE_TOKENS))
    if bool(bad.any()):
        first = int(bad.nonzero()[0])
        raise RuntimeError(
            f"MXFP4 prefill reads every token back from the KV pages, but token "
            f"{first} of a request has slot {int(slots[first])} (pool holds "
            f"{num_pages * PAGE_TOKENS} slots): its K/V was not written, so the "
            "gather would read an unwritten page"
        )
