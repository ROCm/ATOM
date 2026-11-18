# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 MLA query quantization in one Triton launch.

Does what ``mla_nvfp4_quantize_q`` does, byte for byte, but reads the absorbed
NoPE ``[B, 128, 512]`` and the roped RoPE ``[B, 128, 64]`` where they are, so
the decode path no longer concatenates them into ``[B, 128, 576]`` first.
"""

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.nvfp4_ring_triton import (
    PAGE,
    TILE,
    _append_quantize_token,
    _fp4,
    _pack,
    _rope_gptj,
)

Q_GAIN = tl.constexpr(6.72)
ROWS = 16
NOPE = 512
ROPE = 64


@triton.jit
def _quantize_q_rows(
    pid,  # this program's block of ROWS (batch, head) rows
    nope_ptr,  # [B, H, 512] BF16, unit inner stride
    rope_ptr,  # [B, H, 64] BF16, unit inner stride
    qn_ptr,  # [B * H, 256] uint8 out
    qa_ptr,  # [B * H, 128] uint8 out
    rows,
    heads,
    stride_nb,
    stride_nh,
    stride_rb,
    stride_rh,
    pos_ptr,  # [B] positions, read only under ROPE_FUSED
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    ROWS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
    CH: tl.constexpr = 128,
):
    """ROWS rows of `quantize_q`, NoPE CH channels at a time (its 16-channel
    blocks are independent)."""
    r = pid * ROWS + tl.arange(0, ROWS)
    live = r < rows
    b = (r // heads).to(tl.int64)
    h = (r % heads).to(tl.int64)
    row = r.to(tl.int64)[:, None]
    m = live[:, None]
    aux = qa_ptr + row * 128
    nope_rows = nope_ptr + (b * stride_nb + h * stride_nh)[:, None]

    # QN: per row, 16-channel blocks, codes over the rounded scale.
    for c0 in tl.range(0, NOPE, CH):
        cc = c0 + tl.arange(0, CH)
        xn = tl.load(nope_rows + cc[None, :], mask=m, other=0.0).to(tl.float32)
        nb = tl.reshape(xn, (ROWS, CH // 16, 16))
        _, codes, scale = _fp4(
            nb, tl.max(tl.abs(nb), axis=2)[:, :, None], False, Q_GAIN
        )
        nbytes = c0 // 2 + tl.arange(0, CH // 2)
        tl.store(
            qn_ptr + row * (NOPE // 2) + nbytes[None, :],
            _pack(tl.reshape(codes, (ROWS, CH)), ROWS, CH),
            mask=m,
        )
        sc = c0 // 16 + tl.arange(0, CH // 16)
        tl.store(
            aux + 64 + sc[None, :],
            tl.reshape(scale, (ROWS, CH // 16)).to(tl.uint8),
            mask=m,
        )

    # QR and its residual QE.
    rc = tl.arange(0, ROPE)
    xr = tl.load(
        rope_ptr + (b * stride_rb + h * stride_rh)[:, None] + rc[None, :],
        mask=m,
        other=0.0,
    )
    if ROPE_FUSED:
        # `q_rope` is pre-RoPE here; rotate it as the RoPE kernel would.
        pos = tl.load(pos_ptr + b, mask=live, other=0).to(tl.int64)
        xr = _rope_gptj(xr, pos, cos_ptr, sin_ptr, stride_cos_t, ROWS, ROPE)
    rb = tl.reshape(xr.to(tl.float32), (ROWS, ROPE // 16, 16))
    base, b_codes, b_scale = _fp4(
        rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, Q_GAIN
    )
    res = rb - base
    _, e_codes, e_scale = _fp4(
        res, tl.max(tl.abs(res), axis=2)[:, :, None], False, Q_GAIN
    )
    # q_aux: QR codes | QE codes | 32 QN scales | 4 QR scales | 4 QE scales | 0x7F.
    rbytes = tl.arange(0, ROPE // 2)
    tl.store(
        aux + rbytes[None, :],
        _pack(tl.reshape(b_codes, (ROWS, ROPE)), ROWS, ROPE),
        mask=m,
    )
    tl.store(
        aux + ROPE // 2 + rbytes[None, :],
        _pack(tl.reshape(e_codes, (ROWS, ROPE)), ROWS, ROPE),
        mask=m,
    )
    blk = tl.arange(0, ROPE // 16)
    tl.store(
        aux + 96 + blk[None, :],
        tl.reshape(b_scale, (ROWS, ROPE // 16)).to(tl.uint8),
        mask=m,
    )
    tl.store(
        aux + 100 + blk[None, :],
        tl.reshape(e_scale, (ROWS, ROPE // 16)).to(tl.uint8),
        mask=m,
    )
    pad = tl.arange(0, 32)
    tl.store(
        aux + 104 + pad[None, :],
        tl.full((ROWS, 32), 127, tl.uint8),
        mask=m & (pad[None, :] < 24),
    )


@triton.jit
def _quantize_q_kernel(
    nope_ptr,
    rope_ptr,
    qn_ptr,
    qa_ptr,
    rows,
    heads,
    stride_nb,
    stride_nh,
    stride_rb,
    stride_rh,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    ROWS: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
):
    _quantize_q_rows(
        tl.program_id(0),
        nope_ptr,
        rope_ptr,
        qn_ptr,
        qa_ptr,
        rows,
        heads,
        stride_nb,
        stride_nh,
        stride_rb,
        stride_rh,
        pos_ptr,
        cos_ptr,
        sin_ptr,
        stride_cos_t,
        ROWS,
        NOPE,
        ROPE,
        ROPE_FUSED,
    )


def quantize_q(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    positions: torch.Tensor | None = None,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
):
    """``mla_nvfp4_quantize_q(torch.cat((q_nope, q_rope), -1))`` without the cat.

    ``q_nope [B, H, 512]`` and roped ``q_rope [B, H, 64]`` are BF16 with a unit
    inner stride. With ``positions [B]`` and the BF16 cos/sin caches (front
    half, ``[max_pos, ..., 32]``), ``q_rope`` is pre-RoPE and the kernel
    applies interleaved RoPE itself. Returns ``q_nope [B, H, 256]`` and
    ``q_aux [B, H, 128]`` uint8. Graph-safe.
    """
    batch, heads, nope = q_nope.shape
    if nope != NOPE or q_rope.shape != (batch, heads, ROPE):
        raise ValueError(
            f"NVFP4 Q wants [B, H, {NOPE}] + [B, H, {ROPE}], got "
            f"{tuple(q_nope.shape)} + {tuple(q_rope.shape)}."
        )
    if q_nope.dtype != torch.bfloat16 or q_rope.dtype != torch.bfloat16:
        raise ValueError("NVFP4 Q quantization takes BF16 inputs.")
    if q_nope.stride(-1) != 1 or q_rope.stride(-1) != 1:
        raise ValueError("NVFP4 Q inputs need a unit inner stride.")
    qn = torch.empty((batch, heads, NOPE // 2), dtype=torch.uint8, device=q_nope.device)
    qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_nope.device)
    fused = positions is not None
    if fused:
        _check_rope_cache(cos, sin)
    rows = batch * heads
    if rows:
        _quantize_q_kernel[(triton.cdiv(rows, ROWS),)](
            q_nope,
            q_rope,
            qn,
            qa,
            rows,
            heads,
            q_nope.stride(0),
            q_nope.stride(1),
            q_rope.stride(0),
            q_rope.stride(1),
            positions if fused else q_nope,
            cos if fused else q_nope,
            sin if fused else q_nope,
            cos.stride(0) if fused else 0,
            ROWS=ROWS,
            NOPE=NOPE,
            ROPE=ROPE,
            ROPE_FUSED=fused,
            num_warps=4,
        )
    return qn, qa


@triton.jit
def _decode_prep_kernel(
    # Q: [B, H, 512] NoPE and [B, H, 64] RoPE in, q_nope / q_aux out
    q_nope_ptr,
    q_rope_ptr,
    qn_ptr,
    qa_ptr,
    q_rows,
    heads,
    stride_qnb,
    stride_qnh,
    stride_qrb,
    stride_qrh,
    q_programs,
    # K: one new token per batch row into the tail and the FP4 pages
    ring_ptr,
    k_nope_ptr,
    k_rope_ptr,
    req_ptr,
    slot_ptr,
    nope_ptr,
    aux_ptr,
    stride_req,
    stride_slot,
    stride_knope,
    stride_krope,
    # shared
    pos_ptr,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    ROWS: tl.constexpr,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < q_programs:
        _quantize_q_rows(
            pid,
            q_nope_ptr,
            q_rope_ptr,
            qn_ptr,
            qa_ptr,
            q_rows,
            heads,
            stride_qnb,
            stride_qnh,
            stride_qrb,
            stride_qrh,
            pos_ptr,
            cos_ptr,
            sin_ptr,
            stride_cos_t,
            ROWS,
            NOPE,
            ROPE,
            ROPE_FUSED,
        )
    else:
        _append_quantize_token(
            pid - q_programs,
            ring_ptr,
            k_nope_ptr,
            k_rope_ptr,
            req_ptr,
            pos_ptr,
            slot_ptr,
            nope_ptr,
            aux_ptr,
            stride_req,
            stride_slot,
            stride_knope,
            stride_krope,
            cos_ptr,
            sin_ptr,
            stride_cos_t,
            RING,
            PAGE,
            TILE,
            NOPE,
            ROPE,
            ROPE_FUSED,
        )


def decode_prep(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    ring: torch.Tensor,
    req_ids: torch.Tensor,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    slots: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_aux: torch.Tensor,
    positions: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
):
    """`quantize_q` and `ring_append_quantize` in one launch.

    ``q_nope [B, H, 512]``, ``q_rope [B, H, 64]``, ``k_nope [B, 512]`` and
    ``k_rope [B, 64]`` are BF16 with a unit inner stride; with the BF16
    cos/sin caches both RoPE parts are pre-RoPE and rotated in the kernel,
    without them they are already roped. ``ring`` is one layer of the tail,
    ``kv_nope`` / ``kv_aux`` one layer of the cache. Returns ``q_nope``
    ``[B, H, 256]`` and ``q_aux [B, H, 128]``. Same bytes as the two
    separate launches. Graph-safe.
    """
    batch, heads, nope = q_nope.shape
    if nope != NOPE or q_rope.shape != (batch, heads, ROPE):
        raise ValueError(f"NVFP4 Q wants [B, H, {NOPE}] + [B, H, {ROPE}].")
    if k_nope.shape != (batch, NOPE) or k_rope.shape != (batch, ROPE):
        raise ValueError(f"NVFP4 K wants [B, {NOPE}] + [B, {ROPE}].")
    tensors = (q_nope, q_rope, k_nope, k_rope, ring)
    if any(t.dtype != torch.bfloat16 for t in tensors):
        raise ValueError("NVFP4 decode prep takes BF16 Q, K and tail.")
    if any(t.stride(-1) != 1 for t in tensors):
        raise ValueError("NVFP4 decode prep needs unit inner strides.")
    if not (kv_nope.is_contiguous() and kv_aux.is_contiguous()):
        raise ValueError("NVFP4 cache planes must be contiguous.")
    fused = cos is not None
    if fused:
        _check_rope_cache(cos, sin)
    qn = torch.empty((batch, heads, NOPE // 2), dtype=torch.uint8, device=q_nope.device)
    qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_nope.device)
    q_rows = batch * heads
    q_programs = triton.cdiv(q_rows, ROWS)
    if batch:
        _decode_prep_kernel[(q_programs + batch,)](
            q_nope,
            q_rope,
            qn,
            qa,
            q_rows,
            heads,
            q_nope.stride(0),
            q_nope.stride(1),
            q_rope.stride(0),
            q_rope.stride(1),
            q_programs,
            ring,
            k_nope,
            k_rope,
            req_ids,
            slots,
            kv_nope,
            kv_aux,
            ring.stride(0),
            ring.stride(1),
            k_nope.stride(0),
            k_rope.stride(0),
            positions,
            cos if fused else q_nope,
            sin if fused else q_nope,
            cos.stride(0) if fused else 0,
            ROWS=ROWS,
            RING=ring.shape[1],
            PAGE=PAGE,
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
            ROPE_FUSED=fused,
            num_warps=4,
        )
    return qn, qa


def _check_rope_cache(cos: torch.Tensor, sin: torch.Tensor) -> None:
    if (
        cos is None
        or sin is None
        or cos.dtype != torch.bfloat16
        or sin.dtype != torch.bfloat16
    ):
        raise ValueError("Fused NVFP4 RoPE needs the BF16 cos/sin caches.")
    if (
        cos.shape[-1] != ROPE // 2
        or cos.stride(-1) != 1
        or cos.stride() != sin.stride()
    ):
        raise ValueError(
            f"Fused NVFP4 RoPE wants [max_pos, ..., {ROPE // 2}] caches with a unit "
            f"inner stride, got {tuple(cos.shape)} {cos.stride()}."
        )
