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

from typing import Tuple

import torch
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
