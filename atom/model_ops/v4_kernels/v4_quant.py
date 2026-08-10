# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 MLA on-the-fly quantization helpers for the hipkittens decode kernel.

The hipkittens kernel ``aiter.mla.mla_v40_decode_fwd`` consumes Q/KV in a
two-buffer layout:

    nope_scale_buff: [..., 512]  FP8 (1 byte/elem)
        bytes [0   , 448): NOPE FP8 (per-token feature, position-agnostic)
        bytes [448 , 462): 14 duplicated E8M0 scales (7 per-64-elt scales x2)
        bytes [462 , 512): 50 bytes unused trailing pad
    rope_buff:       [..., 64]   BF16  (per-token RoPE-rotated, kept BF16)

ATOM currently stores ``unified_kv`` as a single contiguous bf16 tensor of
shape ``[..., 512]`` (NoPE 448 + RoPE 64 concatenated). For the PoC we
convert bf16 -> (fp8+scale+pad, bf16-rope) on every decode call. A future
phase 2 may change ``unified_kv`` to store the packed layout directly so the
runtime quantization is skipped.

All constants and pack arithmetic mirror
``/mnt/raid0/ruitang3/git_repo/aiter/op_tests/test_mla_v4_persistent.py``
(``V4_*`` constants and ``pack_v4_nope_scale``/``quantize_v4_nope_bpad8``).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from aiter import dtypes

V4_DIM_NOPE = 448
V4_DIM_ROPE = 64
V4_DIM_QK = V4_DIM_NOPE + V4_DIM_ROPE  # 512
V4_TILE = 64
V4_NUM_TILES = V4_DIM_NOPE // V4_TILE  # 7
V4_DIM_SCALE_DUP = V4_DIM_NOPE // (V4_TILE // 2)  # 14
V4_DIM_QK_PACKED = 512
V4_PACK_OFF_NOPE = 0
V4_PACK_OFF_SCALE = V4_DIM_NOPE  # 448
V4_PACK_OFF_PAD = V4_DIM_NOPE + V4_DIM_SCALE_DUP  # 462


def _fp32_pow2_to_e8m0(pow2_fp32: torch.Tensor) -> torch.Tensor:
    """Pack a power-of-2 fp32 scale into a 1-byte E8M0 exponent
    (byte B encodes 2^(B-127); B=0 -> 0.0, B=255 -> INF)."""
    safe = torch.where(pow2_fp32 > 0, pow2_fp32, torch.ones_like(pow2_fp32))
    biased = torch.log2(safe).round().to(torch.int32) + 127
    biased = torch.clamp(biased, 0, 254)
    biased = torch.where(pow2_fp32 > 0, biased, torch.zeros_like(biased))
    return biased.to(torch.uint8)


def _cast_scale_inv_to_ue8m0_pow2(scales_inv: torch.Tensor) -> torch.Tensor:
    """amax/FP8_AMAX -> ceil-log2 -> power-of-2 fp32."""
    return torch.pow(2.0, torch.clamp_min(scales_inv, 1e-4).log2().ceil()).to(
        torch.float32
    )


def _duplicate_each_lastdim(x: torch.Tensor) -> torch.Tensor:
    """[..., N] -> [..., 2*N] with each element written twice."""
    return x.unsqueeze(-1).expand(*x.shape, 2).reshape(*x.shape[:-1], x.shape[-1] * 2)


def quantize_v4_nope_bpad8(
    nope_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tile (64-elt) E8M0 quantization of the NOPE segment.

    Returns ``(nope_fp8 [..., 448], scale_e8m0 [..., 7] uint8)``.
    """
    fp8_amax = float(torch.finfo(dtypes.fp8).max)
    nope_fp32 = nope_src.float()
    leading = nope_fp32.shape[:-1]
    tiled = nope_fp32.reshape(*leading, V4_NUM_TILES, V4_TILE)
    active_scale_pow2 = _cast_scale_inv_to_ue8m0_pow2(
        tiled.abs().amax(dim=-1) / fp8_amax
    )
    nope_fp8 = (
        (tiled / active_scale_pow2.unsqueeze(-1))
        .to(dtypes.fp8)
        .reshape(*leading, V4_DIM_NOPE)
    )
    scale_e8m0 = _fp32_pow2_to_e8m0(active_scale_pow2)  # [..., 7] uint8
    return nope_fp8, scale_e8m0


def pack_v4_nope_scale(
    nope_fp8: torch.Tensor, scale_e8m0: torch.Tensor
) -> torch.Tensor:
    """Pack NOPE + duplicated E8M0 scale into a single 512-byte/token FP8 tensor.

    nope_fp8:   [..., 448] FP8
    scale_e8m0: [..., 7]   uint8 (E8M0 byte per quant tile)
    returns:    [..., 512] FP8 (NoPE | dup-scale x14 | pad x50)
    """
    leading = nope_fp8.shape[:-1]
    assert nope_fp8.shape[-1] == V4_DIM_NOPE
    assert scale_e8m0.shape[-1] == V4_NUM_TILES
    assert scale_e8m0.shape[:-1] == leading

    packed = torch.zeros(
        (*leading, V4_DIM_QK_PACKED), dtype=torch.uint8, device=nope_fp8.device
    )
    packed[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE] = nope_fp8.view(
        torch.uint8
    )
    packed[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP] = (
        _duplicate_each_lastdim(scale_e8m0)
    )
    return packed.view(dtypes.fp8)


def _e8m0_to_fp32_pow2(scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_fp32_pow2_to_e8m0``: E8M0 byte B -> fp32 2^(B-127).

    B == 0 decodes to 0.0 (the zero-scale sentinel produced by the forward
    path for all-zero tiles)."""
    biased = scale_e8m0.to(torch.int32)
    pow2 = torch.pow(2.0, (biased - 127).float())
    return torch.where(biased > 0, pow2, torch.zeros_like(pow2))


def dequantize_v4_2buff_to_bf16(
    packed_fp8: torch.Tensor,
    rope_bf16: torch.Tensor,
) -> torch.Tensor:
    """Inverse of ``quantize_bf16_to_v4_2buff``.

    Takes the two-buffer layout ``(packed_fp8 [..., 512], rope_bf16 [..., 64])``
    and reconstructs the bf16 ``[..., 512]`` row (NoPE 448 + RoPE 64).

    The NoPE half is dequantized per 64-elt tile: ``fp8_val * 2^(B-127)`` where
    ``B`` is the tile's E8M0 scale byte (read from the dup-scale region; we use
    the first of each duplicated pair). Round-trips
    ``quantize_bf16_to_v4_2buff`` to within fp8 per-tile quantization error.
    """
    assert packed_fp8.shape[-1] == V4_DIM_QK_PACKED, (
        f"dequantize_v4_2buff_to_bf16: packed last dim must be "
        f"{V4_DIM_QK_PACKED}, got {tuple(packed_fp8.shape)}"
    )
    assert rope_bf16.shape[-1] == V4_DIM_ROPE, (
        f"dequantize_v4_2buff_to_bf16: rope last dim must be {V4_DIM_ROPE}, "
        f"got {tuple(rope_bf16.shape)}"
    )
    leading = packed_fp8.shape[:-1]
    packed_u8 = packed_fp8.view(torch.uint8)

    nope_fp8 = packed_u8[..., V4_PACK_OFF_NOPE : V4_PACK_OFF_NOPE + V4_DIM_NOPE].view(
        dtypes.fp8
    )
    nope_fp32 = nope_fp8.float().reshape(*leading, V4_NUM_TILES, V4_TILE)

    # Dup-scale region holds each of the 7 tile scales twice; take the even
    # entries to recover the 7 per-tile E8M0 bytes.
    scale_dup = packed_u8[..., V4_PACK_OFF_SCALE : V4_PACK_OFF_SCALE + V4_DIM_SCALE_DUP]
    scale_e8m0 = scale_dup[..., 0::2]  # [..., 7]
    scale_pow2 = _e8m0_to_fp32_pow2(scale_e8m0)  # [..., 7] fp32

    nope_bf16 = (
        (nope_fp32 * scale_pow2.unsqueeze(-1))
        .reshape(*leading, V4_DIM_NOPE)
        .to(torch.bfloat16)
    )
    rope = rope_bf16.to(torch.bfloat16)
    return torch.cat([nope_bf16, rope], dim=-1)


def quantize_bf16_to_v4_2buff(
    bf16_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """End-to-end helper: bf16 [..., 512] -> (packed_fp8 [..., 512], rope_bf16 [..., 64]).

    Splits the input on the NoPE/RoPE boundary, quantizes the NoPE half via
    ``quantize_v4_nope_bpad8`` + ``pack_v4_nope_scale``, and keeps the RoPE
    half in bf16 (contiguous).
    """
    assert bf16_src.shape[-1] == V4_DIM_QK, (
        f"quantize_bf16_to_v4_2buff: last dim must be {V4_DIM_QK}, "
        f"got {tuple(bf16_src.shape)}"
    )
    nope_src = bf16_src[..., :V4_DIM_NOPE]
    rope_src = bf16_src[..., V4_DIM_NOPE:].to(torch.bfloat16).contiguous()
    nope_fp8, scale_e8m0 = quantize_v4_nope_bpad8(nope_src)
    packed_fp8 = pack_v4_nope_scale(nope_fp8, scale_e8m0)
    return packed_fp8, rope_src


# ---------------------------------------------------------------------------
# Triton port of quantize_bf16_to_v4_2buff.
#
# The torch helper above chains ~6 tensor ops (reshape, amax, log2/ceil,
# to(fp8), uint8 bit-views, scatter-pack) — fine as an eager reference, but it
# graph-breaks / is CUDAGraph-hostile inside a `@support_torch_compile` region.
# This single-kernel version is the compile-safe path used to build the fp8
# 2buff Q/K without the aiter HIP op (op1). Output is bit-compatible with
# ``quantize_bf16_to_v4_2buff`` (same round-to-nearest fp8 cast + same
# ceil-log2 e8m0 scale), so ``dequantize_v4_2buff_to_bf16`` round-trips either.
# ---------------------------------------------------------------------------


@triton.jit
def _bf16_to_v4_2buff_kernel(
    src_ptr,  # [N, 512] bf16 (NoPE 448 | RoPE 64)
    packed_fp8_ptr,  # [N, 512] fp8 out — NoPE fp8 written here (rest pre-zeroed)
    packed_u8_ptr,  # same buffer, uint8 view — e8m0 dup-scale bytes written here
    rope_ptr,  # [N, 64] bf16 out
    src_row_stride,
    N,
    FP8_AMAX: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    TILE: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PACK_OFF_SCALE: tl.constexpr,
    PACKED_ROW: tl.constexpr,  # 512 (packed row stride)
    BLOCK_N: tl.constexpr,
):
    """One program per ``BLOCK_N``-row tile. For each of the 7 NoPE tiles:
    per-64-elt amax -> ceil-log2 e8m0 power-of-2 scale -> fp8 quant, storing the
    fp8 bytes into the NoPE region and the E8M0 byte (duplicated) into the
    dup-scale region. The RoPE tail is copied through in bf16. The 50-byte pad
    is left as the caller's pre-zeroed value."""
    pid = tl.program_id(0)
    row = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = row < N

    d_tile = tl.arange(0, TILE)
    # Loop is unrolled at compile time (NUM_TILES=7 constexpr).
    for t in tl.static_range(NUM_TILES):
        cols = t * TILE + d_tile
        x = tl.load(
            src_ptr + row[:, None] * src_row_stride + cols[None, :],
            mask=row_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        amax = tl.max(tl.abs(x), axis=1)  # [BLOCK_N]
        scale_inv = amax / FP8_AMAX
        # ceil-log2 power-of-2 scale (matches _cast_scale_inv_to_ue8m0_pow2:
        # clamp_min(1e-4) guards log2 of an all-zero tile).
        e = tl.ceil(tl.log2(tl.maximum(scale_inv, 1e-4)))  # integer-valued fp32
        scale_pow2 = tl.exp2(e)

        xq = (x / scale_pow2[:, None]).to(packed_fp8_ptr.dtype.element_ty)
        tl.store(
            packed_fp8_ptr + row[:, None] * PACKED_ROW + cols[None, :],
            xq,
            mask=row_mask[:, None],
        )

        # E8M0 byte = clamp(round(log2(scale_pow2)) + 127, 0, 254). scale_pow2 is
        # an exact power of two so round(log2)==e. Written twice (dup layout).
        byte_f = tl.minimum(tl.maximum(e + 127.0, 0.0), 254.0)
        byte = byte_f.to(tl.uint8)
        off = PACK_OFF_SCALE + 2 * t
        tl.store(packed_u8_ptr + row * PACKED_ROW + off, byte, mask=row_mask)
        tl.store(packed_u8_ptr + row * PACKED_ROW + off + 1, byte, mask=row_mask)

    # RoPE tail passthrough (bf16, unquantized).
    r_cols = tl.arange(0, ROPE)
    r = tl.load(
        src_ptr + row[:, None] * src_row_stride + (NOPE + r_cols)[None, :],
        mask=row_mask[:, None],
    )
    tl.store(
        rope_ptr + row[:, None] * ROPE + r_cols[None, :],
        r,
        mask=row_mask[:, None],
    )


def quantize_bf16_to_v4_2buff_triton(
    bf16_src: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton port of :func:`quantize_bf16_to_v4_2buff` (compile-safe, single
    launch). ``bf16_src`` may have any leading shape; the last dim must be 512.

    Returns ``(packed_fp8 [..., 512], rope_bf16 [..., 64])`` — bit-compatible
    with the torch helper (same fp8 rounding + e8m0 ceil-log2 scale).
    """
    assert bf16_src.shape[-1] == V4_DIM_QK, (
        f"quantize_bf16_to_v4_2buff_triton: last dim must be {V4_DIM_QK}, "
        f"got {tuple(bf16_src.shape)}"
    )
    assert bf16_src.dtype == torch.bfloat16, (
        f"quantize_bf16_to_v4_2buff_triton: input must be bf16, "
        f"got {bf16_src.dtype}"
    )
    leading = bf16_src.shape[:-1]
    src = bf16_src.reshape(-1, V4_DIM_QK)
    src = src.contiguous() if src.stride(-1) != 1 else src
    N = src.shape[0]

    # Zeroed so the 50-byte trailing pad (and any untouched byte) is 0, matching
    # pack_v4_nope_scale's torch.zeros allocation.
    packed = torch.zeros((N, V4_DIM_QK_PACKED), dtype=dtypes.fp8, device=src.device)
    rope = torch.empty((N, V4_DIM_ROPE), dtype=torch.bfloat16, device=src.device)

    if N > 0:
        block_n = 8
        while block_n > 1 and block_n > N:
            block_n //= 2
        grid = (triton.cdiv(N, block_n),)
        _bf16_to_v4_2buff_kernel[grid](
            src,
            packed,
            packed.view(torch.uint8),
            rope,
            src.stride(0),
            N,
            FP8_AMAX=float(torch.finfo(dtypes.fp8).max),
            NOPE=V4_DIM_NOPE,
            ROPE=V4_DIM_ROPE,
            TILE=V4_TILE,
            NUM_TILES=V4_NUM_TILES,
            PACK_OFF_SCALE=V4_PACK_OFF_SCALE,
            PACKED_ROW=V4_DIM_QK_PACKED,
            BLOCK_N=block_n,
        )

    return (
        packed.view(*leading, V4_DIM_QK_PACKED),
        rope.view(*leading, V4_DIM_ROPE),
    )


# ---------------------------------------------------------------------------
# Torch reference for aiter.mla.mla_decode_fwd_v4_nm
#
# Mirrors op_tests/test_mla_v4_nm.py::_torch_attn_decode_fp8_dequant_ref +
# _torch_attn_decode_bf16_golden: dequantize the exact 2-buffer FP8 tensors the
# asm kernel consumes, then run plain-torch scaled-dot-product attention with a
# per-batch loop, GQA broadcast, and an attention sink. Reusing this module's
# `dequantize_v4_2buff_to_bf16` (the inverse of the packing above) keeps the
# reference self-consistent with ATOM's own quant path.
#
# This isolates "kernel math bug" from "FP8 quant noise": feed the SAME packed
# bytes to the kernel and to this reference, and any diff is kernel math.
# ---------------------------------------------------------------------------


def _attn_decode_bf16_golden(
    q_bf16: torch.Tensor,  # [total_q, num_heads, D]
    kv_bf16: torch.Tensor,  # [num_page, page_size, num_kv_heads, D]
    qo_indptr: torch.Tensor,  # [num_seqs+1]   q rows per seq (cumulative)
    kv_indptr: torch.Tensor,  # [num_seqs+1]   pages per seq (cumulative)
    kv_page_indices: torch.Tensor,  # [total_pages_used]
    kv_last_page_lens: torch.Tensor,  # [num_seqs]
    sm_scale: float,
    v_head_dim: int,
    attn_sink: Optional[torch.Tensor] = None,  # [num_heads] or None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-torch BF16 attention reference (FP32 accum).

    Per-batch loop, scaled-dot-product attention over the full D=NoPE+RoPE
    query/key dim, with GQA broadcast (each KV head serves gqa_ratio Q heads)
    and an optional per-head attention sink (virtual K-column of logit
    ``sink[h]`` shared by every query token). V is the first ``v_head_dim``
    columns of the (dequantized) KV row.

    Returns:
        out  [total_q, num_heads, v_head_dim]  FP32
        lse  [total_q, num_heads]              FP32
    """
    total_q, num_heads, d = q_bf16.shape
    num_kv_heads = kv_bf16.size(2)
    gqa_ratio = num_heads // num_kv_heads
    page_size = kv_bf16.size(1)
    device = q_bf16.device

    out = torch.zeros(
        (total_q, num_heads, v_head_dim), dtype=torch.float32, device=device
    )
    lse_full = torch.full(
        (total_q, num_heads), float("inf"), dtype=torch.float32, device=device
    )

    batch = qo_indptr.numel() - 1
    qo_cpu = qo_indptr.cpu().tolist()
    kv_cpu = kv_indptr.cpu().tolist()
    last_cpu = kv_last_page_lens.cpu().tolist()

    for b in range(batch):
        qs, qe = qo_cpu[b], qo_cpu[b + 1]
        ps, pe = kv_cpu[b], kv_cpu[b + 1]
        num_pages_b = pe - ps
        if num_pages_b == 0:
            continue

        page_ids = kv_page_indices[ps:pe]
        kv_pages = kv_bf16[page_ids]  # [num_pages_b, page_size, num_kv_heads, D]
        # Flatten (page, slot) -> token, then trim the partial last page.
        kv_flat = kv_pages.reshape(-1, num_kv_heads, d)
        total_tokens = (num_pages_b - 1) * page_size + last_cpu[b]
        kv_b = kv_flat[:total_tokens].float()  # [seq_k, num_kv_heads, D]

        # GQA broadcast: replicate each KV head across its gqa_ratio Q heads.
        kv_heads = kv_b.repeat_interleave(gqa_ratio, dim=1)  # [seq_k, num_heads, D]

        q_b = q_bf16[qs:qe].float()  # [s_q, num_heads, D]
        scores = torch.einsum("shd,khd->shk", q_b, kv_heads) * sm_scale  # [s_q,H,seq_k]

        lse = scores.logsumexp(dim=-1)  # [s_q, H]
        if attn_sink is not None:
            sink_b = attn_sink.view(1, num_heads).float()  # [1, H]
            m = torch.maximum(lse, sink_b)
            denom = torch.exp(lse - m) + torch.exp(sink_b - m)
            lse_final = m + torch.log(denom)
        else:
            lse_final = lse
        probs = torch.exp(scores - lse_final.unsqueeze(-1))  # [s_q, H, seq_k]

        v_heads = kv_heads[..., :v_head_dim]  # MLA: V == first v_head_dim of K
        out_b = torch.einsum("shk,khv->shv", probs, v_heads)  # [s_q, H, v_head_dim]
        out[qs:qe] = out_b
        lse_full[qs:qe] = lse_final

    return out, lse_full


def mla_decode_fwd_v4_nm_ref(
    q,  # [total_q, num_heads, 512]  FP8 packed (NoPE 448 | dup-scale 14 | pad 50)
    qrope,  # [total_q, num_heads, 64]   BF16
    kv_buffer,  # [num_page, page_size, num_kv_heads, 512]  FP8 packed
    kvrope,  # [num_page, page_size, num_kv_heads, 64]    BF16
    output,  # [total_q, num_heads, v_head_dim]  BF16 (written in-place)
    qo_indptr,  # [num_seqs+1]
    kv_indptr,  # [num_seqs+1]
    kv_page_indices,  # [num_page_used]
    kv_last_page_lens,  # [num_seqs]
    max_seqlen_q,
    *,
    sink=None,  # [num_heads] FP32, or None for "no sink" math
    split_indptr=None,  # accepted for signature parity; math is split-invariant
    sm_scale=None,  # None -> 1/sqrt(D) matching the kernel's hardcoded scale
    out_16_nosplit=0,  # accepted for parity; reference always writes both buffers
    num_kv_splits=1,  # accepted for parity; KV splitting does not change the math
    logits=None,
    attn_lse=None,
):
    """Torch reference for ``aiter.mla.mla_decode_fwd_v4_nm``.

    Dequantizes the same 2-buffer FP8 tensors the asm kernel reads (via
    :func:`dequantize_v4_2buff_to_bf16`) and computes full MLA decode attention
    in plain torch. Matches the kernel's public contract:

      - Returns ``(logits, attn_lse)`` in kernel-native layout
        ``logits [total_q, num_kv_splits, num_heads, v_head_dim]`` FP32 and
        ``attn_lse [total_q, num_kv_splits, num_heads, 1]`` FP32. The final
        attention result lands in the ``[:, 0]`` split slot (single-pass
        contract; any remaining split slots are left zero).
      - Also writes the BF16 result into ``output`` in-place, so callers that
        read the merged/``out_16_nosplit`` buffer see the same answer.

    ``num_kv_splits`` / ``split_indptr`` / ``out_16_nosplit`` are accepted only
    for signature parity: the KV split is a kernel-side perf optimization whose
    logsumexp-merged result is mathematically identical to the un-split full
    attention this reference computes. ``sm_scale`` defaults to ``1/sqrt(D)``
    (D = NoPE + RoPE = 512) to mirror the kernel, which ignores the passed
    value and hardcodes that scale.
    """
    num_seqs = qo_indptr.numel() - 1
    num_heads = q.size(1)
    num_kv_heads = kv_buffer.size(2)
    gqa_ratio = num_heads // num_kv_heads
    v_head_dim = output.size(2)
    total_q = num_seqs * max_seqlen_q
    device = q.device

    # ---- sink validation (mirror the kernel wrapper's contract) ----
    if sink is not None:
        if sink.dtype != dtypes.fp32:
            raise ValueError(
                f"mla_decode_fwd_v4_nm_ref: `sink` must be FP32, got {sink.dtype}."
            )
        if not sink.is_contiguous():
            raise ValueError(
                "mla_decode_fwd_v4_nm_ref: `sink` must be contiguous "
                f"(got strides={sink.stride()})."
            )
        if sink.numel() != num_heads:
            raise ValueError(
                f"mla_decode_fwd_v4_nm_ref: `sink` numel {sink.numel()} != "
                f"num_heads {num_heads} (= num_kv_heads({num_kv_heads}) * "
                f"gqa_ratio({gqa_ratio}))."
            )
        if sink.device != device:
            raise ValueError(
                "mla_decode_fwd_v4_nm_ref: `sink` must be on the same device "
                f"as `q` (got sink={sink.device}, q={device})."
            )
        # An all -inf sink is the documented "no sink" no-op; treat it as None
        # so exp(-inf - m) = 0 contributes nothing (and avoids NaN when m=-inf).
        attn_sink = None if bool(torch.all(torch.isneginf(sink))) else sink
    else:
        attn_sink = None

    # ---- dequantize the exact packed bytes the kernel consumes ----
    q_bf16 = dequantize_v4_2buff_to_bf16(q, qrope)  # [total_q, num_heads, 512]
    kv_bf16 = dequantize_v4_2buff_to_bf16(
        kv_buffer, kvrope
    )  # [num_page, page_size, num_kv_heads, 512]

    d = q_bf16.size(-1)
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    out_fp32, lse_fp32 = _attn_decode_bf16_golden(
        q_bf16,
        kv_bf16,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        float(sm_scale),
        v_head_dim,
        attn_sink=attn_sink,
    )

    # ---- write outputs in the kernel's public layout ----
    output.copy_(out_fp32.to(output.dtype))

    if logits is None:
        logits = torch.zeros(
            (total_q, num_kv_splits, num_heads, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
    else:
        logits.zero_()
    if attn_lse is None:
        attn_lse = torch.zeros(
            (total_q, num_kv_splits, num_heads, 1),
            dtype=dtypes.fp32,
            device=device,
        )
    else:
        attn_lse.zero_()

    logits[:, 0] = out_fp32
    attn_lse[:, 0, :, 0] = lse_fp32

    return logits, attn_lse
