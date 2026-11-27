# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""NVFP4 MLA decode-step Q quantization and cache write.

`kup_quantize_q_nope` absorbs W_K into the NoPE query and quantizes it,
`decode_prep` quantizes the RoPE part (or the whole Q) and appends the step's
KV token. Both use AITER's gfx1250 kernels when present
(``fused_bf16_bmm_nvfp4_quant_mla`` and ``mla_nvfp4_decode_prep``), else the
Triton kernels here; the bytes are the same as ``mla_nvfp4_quantize_q`` and
``mla_nvfp4_update_cache`` (N16 AUX) either way.
"""

import functools

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
    rcp32,
)

Q_GAIN = tl.constexpr(6.72)
Q_RCP = tl.constexpr(rcp32(6.72))
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
    NOPE_DONE: tl.constexpr = False,
    CH: tl.constexpr = 128,
):
    """ROWS rows of `quantize_q`, NoPE CH channels at a time (its 16-channel
    blocks are independent). Under NOPE_DONE the NoPE codes and scales are
    already in place (`kup_quantize_q_nope`) and only the RoPE part is written."""
    r = pid * ROWS + tl.arange(0, ROWS)
    live = r < rows
    b = (r // heads).to(tl.int64)
    h = (r % heads).to(tl.int64)
    row = r.to(tl.int64)[:, None]
    m = live[:, None]
    aux = qa_ptr + row * 128
    nope_rows = nope_ptr + (b * stride_nb + h * stride_nh)[:, None]

    # QN: per row, 16-channel blocks, codes over the rounded scale.
    for c0 in tl.range(0, 0 if NOPE_DONE else NOPE, CH):
        cc = c0 + tl.arange(0, CH)
        xn = tl.load(nope_rows + cc[None, :], mask=m, other=0.0).to(tl.float32)
        nb = tl.reshape(xn, (ROWS, CH // 16, 16))
        _, codes, scale = _fp4(
            nb, tl.max(tl.abs(nb), axis=2)[:, :, None], False, Q_GAIN, Q_RCP
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
        rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, Q_GAIN, Q_RCP
    )
    res = rb - base
    _, e_codes, e_scale = _fp4(
        res, tl.max(tl.abs(res), axis=2)[:, :, None], False, Q_GAIN, Q_RCP
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
def _zero_q_rows(
    pid,
    qn_ptr,
    qa_ptr,
    ROWS: tl.constexpr,
    NOPE: tl.constexpr,
    NOPE_DONE: tl.constexpr,
    CB: tl.constexpr = 64,
):
    """ROWS rows of a zero Q: codes and scales 0, the q_aux padding 0x7F.
    CB bytes at a time, so the address tiles stay small. Under NOPE_DONE the
    NoPE codes and scales (q_aux 64..95) are left as written."""
    row = (pid * ROWS + tl.arange(0, ROWS)).to(tl.int64)[:, None]
    cb = tl.arange(0, CB)[None, :]
    zeros = tl.zeros((ROWS, CB), tl.uint8)
    if not NOPE_DONE:
        for c0 in tl.static_range(0, NOPE // 2, CB):
            tl.store(qn_ptr + row * (NOPE // 2) + c0 + cb, zeros)
    for c0 in tl.static_range(0, 128, CB):
        keep = NOPE_DONE & (c0 + cb >= 64) & (c0 + cb < 96)
        tl.store(
            qa_ptr + row * 128 + c0 + cb,
            tl.where(c0 + cb < 104, 0, 127).to(tl.uint8) + zeros,
            mask=~keep,
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
    NOPE_DONE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if pid < q_programs:
        # A padding batch row (slot -1) has no context and the shader never
        # reads its Q: write a zero Q instead of quantizing. ROWS divides
        # heads, so a program's rows share one batch row.
        if tl.load(slot_ptr + (pid * ROWS) // heads) < 0:
            _zero_q_rows(pid, qn_ptr, qa_ptr, ROWS, NOPE, NOPE_DONE)
        else:
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
                NOPE_DONE,
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


@functools.cache
def _aiter_decode_prep():
    """AITER's gfx1250 one-launch decode prep, if this AITER has it."""
    try:
        from aiter.ops.mla_nvfp4 import mla_nvfp4_decode_prep
    except ImportError:
        return None
    return mla_nvfp4_decode_prep


def warm(device) -> None:
    """Build or load AITER's decode prep and run it once, eagerly: its code
    object must be loaded before a CUDA graph captures the first decode step.
    One padding row (slot -1): only its Q bytes are written."""
    op = _aiter_decode_prep()
    if op is None:
        return
    b16 = {"dtype": torch.bfloat16, "device": device}
    u8 = {"dtype": torch.uint8, "device": device}
    i64 = {"dtype": torch.int64, "device": device}
    op(
        torch.zeros((1, 128, ROPE), **b16),
        torch.zeros((1, 128, 128), **u8),
        torch.zeros((1, TILE, NOPE + ROPE), **b16),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros((1, NOPE), **b16),
        torch.zeros((1, ROPE), **b16),
        torch.zeros(1, **i64),
        torch.full((1,), -1, **i64),
        torch.zeros((1, 16384), **u8),
        torch.zeros((1, 8192), **u8),
    )


def decode_prep(
    q_nope: torch.Tensor | None,
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
    rows: int | None = None,
    q_planes: tuple[torch.Tensor, torch.Tensor] | None = None,
    use_aiter: bool = True,
):
    """`quantize_q` and `ring_append_quantize` in one launch.

    ``q_nope [B, H, 512]``, ``q_rope [B, H, 64]``, ``k_nope [B, 512]`` and
    ``k_rope [B, 64]`` are BF16 with a unit inner stride; with the BF16
    cos/sin caches both RoPE parts are pre-RoPE and rotated in the kernel,
    without them they are already roped. ``ring`` is one layer of the tail,
    ``kv_nope`` / ``kv_aux`` one layer of the cache. Returns ``q_nope``
    ``[B, H, 256]`` and ``q_aux [B, H, 128]``. Same bytes as the two
    separate launches, except that a padding row (slot -1) gets a zero Q.
    ``rows`` is the Q rows per program of the Triton kernel (default by batch
    width). With ``q_planes = (q_nope, q_aux)`` already holding the NoPE codes
    and scales (`kup_quantize_q_nope`), ``q_nope`` is unused and only the RoPE
    part of ``q_aux`` is written, in place; AITER's ``mla_nvfp4_decode_prep``
    does that when present and ``use_aiter``. Graph-safe.
    """
    nope_done = q_planes is not None
    batch, heads, rope_dim = q_rope.shape
    if rope_dim != ROPE or (not nope_done and q_nope.shape != (batch, heads, NOPE)):
        raise ValueError(f"NVFP4 Q wants [B, H, {NOPE}] + [B, H, {ROPE}].")
    if k_nope.shape != (batch, NOPE) or k_rope.shape != (batch, ROPE):
        raise ValueError(f"NVFP4 K wants [B, {NOPE}] + [B, {ROPE}].")
    if nope_done:
        qn, qa = q_planes
        if qn.shape != (batch, heads, NOPE // 2) or qa.shape != (batch, heads, 128):
            raise ValueError(
                "NVFP4 decode prep q_planes must be [B, H, 256] + [B, H, 128]."
            )
        if not (qn.is_contiguous() and qa.is_contiguous()):
            raise ValueError("NVFP4 decode prep q_planes must be contiguous.")
        q_nope = q_rope  # placeholder pointer; NOPE_DONE never reads it
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
    if not nope_done:
        qn = torch.empty(
            (batch, heads, NOPE // 2), dtype=torch.uint8, device=q_rope.device
        )
        qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_rope.device)
    if rows is None:
        # Q rows per program: 32 schedules better from ~384 batch rows up.
        rows = 32 if batch >= 384 else ROWS
    if heads % rows:
        raise ValueError(
            f"NVFP4 decode prep needs {rows} rows to divide {heads} heads."
        )
    if not slots.is_contiguous():
        raise ValueError("NVFP4 decode prep needs contiguous slots.")
    aiter_prep = _aiter_decode_prep() if use_aiter else None
    if (
        batch
        and nope_done
        and aiter_prep is not None
        and heads == 128
        and ring.shape[1] == TILE
        and q_rope.stride(0) == heads * q_rope.stride(1)
    ):
        aiter_prep(
            q_rope,
            qa,
            ring,
            req_ids,
            k_nope,
            k_rope,
            positions,
            slots,
            kv_nope,
            kv_aux,
            cos_cache=cos,
            sin_cache=sin,
        )
        return qn, qa
    q_rows = batch * heads
    q_programs = triton.cdiv(q_rows, rows)
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
            ROWS=rows,
            RING=ring.shape[1],
            PAGE=PAGE,
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
            ROPE_FUSED=fused,
            NOPE_DONE=nope_done,
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


@triton.jit
def _kup_quantize_kernel(
    q_ptr,  # [B, H, P] BF16 q_nope (P = qk_nope_head_dim), unit inner stride
    w_ptr,  # [H, L, P] BF16 W_K, unit inner stride
    qn_ptr,  # [B, H, L // 2] uint8 out
    qa_ptr,  # [B, H, 128] uint8 out, NoPE scales at 64..
    B,
    stride_qb,
    stride_qh,
    stride_wh,
    stride_wl,
    stride_qnb,
    stride_qnh,
    stride_qab,
    stride_qah,
    P: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """One head's ``q_nope @ W_K^T`` for BLOCK_M rows and BLOCK_N latent
    channels, rounded to BF16 as the BF16 BMM stores it, then quantized like
    `quantize_q`'s NoPE part: per row, 16-channel blocks."""
    h = tl.program_id(0).to(tl.int64)
    rows = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    n0 = tl.program_id(2) * BLOCK_N
    live = rows < B
    p = tl.arange(0, P)
    a = tl.load(
        q_ptr + rows.to(tl.int64)[:, None] * stride_qb + h * stride_qh + p[None, :],
        mask=live[:, None],
        other=0.0,
    )
    cols = n0 + tl.arange(0, BLOCK_N)
    w = tl.load(w_ptr + h * stride_wh + cols[None, :] * stride_wl + p[:, None])
    x = tl.dot(a, w).to(tl.bfloat16).to(tl.float32)
    blocks = tl.reshape(x, (BLOCK_M, BLOCK_N // 16, 16))
    _, codes, scale = _fp4(
        blocks, tl.max(tl.abs(blocks), axis=2)[:, :, None], False, Q_GAIN, Q_RCP
    )
    r = rows.to(tl.int64)[:, None]
    nb = n0 // 2 + tl.arange(0, BLOCK_N // 2)
    tl.store(
        qn_ptr + r * stride_qnb + h * stride_qnh + nb[None, :],
        _pack(tl.reshape(codes, (BLOCK_M, BLOCK_N)), BLOCK_M, BLOCK_N),
        mask=live[:, None],
    )
    sc = n0 // 16 + tl.arange(0, BLOCK_N // 16)
    tl.store(
        qa_ptr + r * stride_qab + h * stride_qah + 64 + sc[None, :],
        tl.reshape(scale, (BLOCK_M, BLOCK_N // 16)).to(tl.uint8),
        mask=live[:, None],
    )


@functools.cache
def _aiter_kup():
    """AITER's gfx1250 fused K-up GEMM and NVFP4 quantization, if present."""
    try:
        from aiter.ops.triton.fusions.fused_bf16_bmm_nvfp4_quant_mla import (
            fused_bf16_bmm_nvfp4_quant_mla,
        )
    except ImportError:
        return None
    return fused_bf16_bmm_nvfp4_quant_mla


def _aiter_kup_applies(q_nope, w_k, qn, qa) -> bool:
    """The shapes and strides ``fused_bf16_bmm_nvfp4_quant_mla`` takes."""
    return (
        q_nope.shape[2] == 128
        and w_k.is_contiguous()
        and qn.is_contiguous()
        and qa.is_contiguous()
        and q_nope.data_ptr() % 16 == 0
        and q_nope.stride(0) % 8 == 0
        and q_nope.stride(1) % 8 == 0
    )


def kup_quantize_q_nope(
    q_nope: torch.Tensor,
    w_k: torch.Tensor,
    qn: torch.Tensor | None = None,
    qa: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 4,
    use_aiter: bool = True,
):
    """The absorbed ``q_nope [B, H, P] @ W_K[H, L, P]^T`` straight into the
    NVFP4 Q planes: ``qn [B, H, L // 2]`` codes and the NoPE scales of
    ``qa [B, H, 128]`` (bytes 64..), the same bytes as the BF16 BMM followed by
    `quantize_q`. The RoPE part of ``qa`` is left to `decode_prep`. Graph-safe.

    AITER's ``fused_bf16_bmm_nvfp4_quant_mla`` when present, it applies and
    ``use_aiter``; else the Triton kernel here (``block_m``, ``block_n``,
    ``num_warps``).
    """
    batch, heads, p = q_nope.shape
    hw, lat, pw = w_k.shape
    if (hw, pw) != (heads, p) or lat != NOPE:
        raise ValueError(
            f"NVFP4 K-up wants W_K [{heads}, {NOPE}, {p}], got {tuple(w_k.shape)}."
        )
    if q_nope.dtype != torch.bfloat16 or w_k.dtype != torch.bfloat16:
        raise ValueError("NVFP4 K-up takes BF16 q_nope and W_K.")
    if q_nope.stride(-1) != 1 or w_k.stride(-1) != 1:
        raise ValueError("NVFP4 K-up needs unit inner strides.")
    if qn is None:
        qn = torch.empty(
            (batch, heads, NOPE // 2), dtype=torch.uint8, device=q_nope.device
        )
    if qa is None:
        qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_nope.device)
    aiter_kup = _aiter_kup() if use_aiter else None
    if aiter_kup is not None and _aiter_kup_applies(q_nope, w_k, qn, qa):
        aiter_kup(q_nope, w_k, qn, qa)
    elif batch:
        _kup_quantize_kernel[(heads, triton.cdiv(batch, block_m), NOPE // block_n)](
            q_nope,
            w_k,
            qn,
            qa,
            batch,
            q_nope.stride(0),
            q_nope.stride(1),
            w_k.stride(0),
            w_k.stride(1),
            qn.stride(0),
            qn.stride(1),
            qa.stride(0),
            qa.stride(1),
            P=p,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
        )
    return qn, qa
