# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 MLA query quantization in one Triton launch.

Does what ``mla_nvfp4_quantize_q`` does, byte for byte, but reads the absorbed
NoPE ``[B, 128, 512]`` and the roped RoPE ``[B, 128, 64]`` where they are, so
the decode path no longer concatenates them into ``[B, 128, 576]`` first.
"""

import os

import torch
import triton
import triton.language as tl

from atom.model_ops.attentions.nvfp4_ring_triton import (
    PAGE,
    TILE,
    _append_quantize_token,
    _fp4,
    _k_nope_finish,
    _k_rope,
    _k_token_setup,
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
        _, codes, scale = _fp4(nb, tl.max(tl.abs(nb), axis=2)[:, :, None], False, Q_GAIN, Q_RCP)
        nbytes = c0 // 2 + tl.arange(0, CH // 2)
        tl.store(qn_ptr + row * (NOPE // 2) + nbytes[None, :], _pack(tl.reshape(codes, (ROWS, CH)), ROWS, CH), mask=m)
        sc = c0 // 16 + tl.arange(0, CH // 16)
        tl.store(aux + 64 + sc[None, :], tl.reshape(scale, (ROWS, CH // 16)).to(tl.uint8), mask=m)

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
    base, b_codes, b_scale = _fp4(rb, tl.max(tl.abs(rb), axis=2)[:, :, None], False, Q_GAIN, Q_RCP)
    res = rb - base
    _, e_codes, e_scale = _fp4(res, tl.max(tl.abs(res), axis=2)[:, :, None], False, Q_GAIN, Q_RCP)
    # q_aux: QR codes | QE codes | 32 QN scales | 4 QR scales | 4 QE scales | 0x7F.
    rbytes = tl.arange(0, ROPE // 2)
    tl.store(aux + rbytes[None, :], _pack(tl.reshape(b_codes, (ROWS, ROPE)), ROWS, ROPE), mask=m)
    tl.store(
        aux + ROPE // 2 + rbytes[None, :],
        _pack(tl.reshape(e_codes, (ROWS, ROPE)), ROWS, ROPE),
        mask=m,
    )
    blk = tl.arange(0, ROPE // 16)
    tl.store(aux + 96 + blk[None, :], tl.reshape(b_scale, (ROWS, ROPE // 16)).to(tl.uint8), mask=m)
    tl.store(aux + 100 + blk[None, :], tl.reshape(e_scale, (ROWS, ROPE // 16)).to(tl.uint8), mask=m)
    pad = tl.arange(0, 32)
    tl.store(
        aux + 104 + pad[None, :],
        tl.full((ROWS, 32), 127, tl.uint8),
        mask=m & (pad[None, :] < 24),
    )

@triton.jit
def _zero_q_rows(
    pid, qn_ptr, qa_ptr, ROWS: tl.constexpr, NOPE: tl.constexpr, NOPE_DONE: tl.constexpr,
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
        tl.program_id(0), nope_ptr, rope_ptr, qn_ptr, qa_ptr, rows, heads,
        stride_nb, stride_nh, stride_rb, stride_rh, pos_ptr, cos_ptr, sin_ptr,
        stride_cos_t, ROWS, NOPE, ROPE, ROPE_FUSED,
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
    N16: tl.constexpr,
    NOPE_DONE: tl.constexpr = False,
    KSPLIT: tl.constexpr = 1,
    amax_ptr=None,
    tag_ptr=None,
    stride_amax=0,
    stride_tag=0,
    INCR: tl.constexpr = False,
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
                pid, q_nope_ptr, q_rope_ptr, qn_ptr, qa_ptr, q_rows, heads,
                stride_qnb, stride_qnh, stride_qrb, stride_qrh, pos_ptr, cos_ptr,
                sin_ptr, stride_cos_t, ROWS, NOPE, ROPE, ROPE_FUSED, NOPE_DONE,
            )
    else:
        # KSPLIT programs per token, each a slice of the NoPE channels.
        k = pid - q_programs
        _append_quantize_token(
            k // KSPLIT, ring_ptr, k_nope_ptr, k_rope_ptr, req_ptr, pos_ptr,
            slot_ptr, nope_ptr, aux_ptr, stride_req, stride_slot, stride_knope,
            stride_krope, cos_ptr, sin_ptr, stride_cos_t,
            RING, PAGE, TILE, NOPE, ROPE, ROPE_FUSED, N16, KSPLIT, k % KSPLIT,
            amax_ptr=amax_ptr, tag_ptr=tag_ptr, stride_amax=stride_amax,
            stride_tag=stride_tag, INCR=INCR,
        )


@triton.jit
def _decode_prep_qk_kernel(
    q_nope_ptr,
    q_rope_ptr,
    qn_ptr,
    qa_ptr,
    q_rows,
    stride_qnb,
    stride_qnh,
    stride_qrb,
    stride_qrh,
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
    pos_ptr,
    cos_ptr,
    sin_ptr,
    stride_cos_t,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    RING: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    NOPE: tl.constexpr,
    ROPE: tl.constexpr,
    ROPE_FUSED: tl.constexpr,
    N16: tl.constexpr,
    NOPE_DONE: tl.constexpr,
    KSPLIT: tl.constexpr,
):
    """`_decode_prep_kernel` with each token's Q in its K programs: part
    ``p`` of KSPLIT issues its NoPE slice's tail loads, quantizes Q heads
    ``[p, p + 1) * HEADS / KSPLIT`` while they are in flight, then finishes
    the slice (and the last part the RoPE). Same bytes out."""
    pid = tl.program_id(0)
    n = pid // KSPLIT
    part = pid % KSPLIT
    _, pos, slot, rows, valid, ring_rows, new_row, tile, nope_page, aux_page = _k_token_setup(
        n, ring_ptr, req_ptr, pos_ptr, slot_ptr, nope_ptr, aux_ptr, stride_req, stride_slot,
        RING, PAGE, TILE,
    )
    live = slot >= 0
    CHK: tl.constexpr = NOPE // KSPLIT
    c0 = part * CHK
    cc = c0 + tl.arange(0, CHK)
    new = tl.load(k_nope_ptr + n * stride_knope + cc, mask=live & (cc < NOPE), other=0.0)
    old = tl.load(ring_rows + cc[None, :], mask=((rows < valid - 1) & live)[:, None], other=0.0)
    HPP: tl.constexpr = HEADS // KSPLIT
    for g in tl.static_range(0, HPP, ROWS):
        qpid = (n * HEADS + part * HPP + g) // ROWS
        # A padding batch row has no context and the shader never reads its Q.
        if live:
            _quantize_q_rows(
                qpid, q_nope_ptr, q_rope_ptr, qn_ptr, qa_ptr, q_rows, HEADS,
                stride_qnb, stride_qnh, stride_qrb, stride_qrh, pos_ptr, cos_ptr,
                sin_ptr, stride_cos_t, ROWS, NOPE, ROPE, ROPE_FUSED, NOPE_DONE,
            )
        else:
            _zero_q_rows(qpid, qn_ptr, qa_ptr, ROWS, NOPE, NOPE_DONE)
    # A padding row (slot -1) has nothing to keep: no tail write, no tile.
    if live:
        tl.store(new_row + cc, new)
        _k_nope_finish(c0, new, old, rows, valid, tile, nope_page, aux_page, TILE, CHK, N16)
        if part == KSPLIT - 1:
            _k_rope(
                n, pos, rows, valid, new_row, tile, aux_page, k_rope_ptr, stride_krope,
                cos_ptr, sin_ptr, stride_cos_t, TILE, NOPE, ROPE, ROPE_FUSED, N16,
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
    n16: bool = False,
    rows: int | None = None,
    q_planes: tuple[torch.Tensor, torch.Tensor] | None = None,
    k_split: int = 1,
    interleave_q: bool = False,
    tile_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    hip_q: bool | None = None,
):
    """`quantize_q` and `ring_append_quantize` in one launch.

    ``q_nope [B, H, 512]``, ``q_rope [B, H, 64]``, ``k_nope [B, 512]`` and
    ``k_rope [B, 64]`` are BF16 with a unit inner stride; with the BF16
    cos/sin caches both RoPE parts are pre-RoPE and rotated in the kernel,
    without them they are already roped. ``ring`` is one layer of the tail,
    ``kv_nope`` / ``kv_aux`` one layer of the cache. Returns ``q_nope``
    ``[B, H, 256]`` and ``q_aux [B, H, 128]``. Same bytes as the two
    separate launches, except that a padding row (slot -1) gets a zero Q.
    ``rows`` is the Q rows per program (default by batch width). With
    ``q_planes = (q_nope, q_aux)`` already holding the NoPE codes and scales
    (`kup_quantize_q_nope`), ``q_nope`` is unused and only the RoPE part of
    ``q_aux`` is written, in place. ``k_split`` programs share each K token,
    a slice of its NoPE channels each. With ``interleave_q`` each token's
    Q heads are split over those programs too, quantized while their tail
    loads are in flight (`_decode_prep_qk_kernel`). ``tile_state = (amax
    [rows, 32] f32, tag [rows] i32)``, one row per tail row, keeps each open
    tile's block amax so a token rewrites only the blocks it raises
    (`_k_nope_incr`); same bytes. With ``q_planes`` and ``hip_q`` (default
    ``ATOM_MLA_NVFP4_HIP_Q``), the gfx1250 HIP kernel does it (same bytes):
    Q and K in one launch (`nvfp4_hip.decode_prep`) for the N16 layout, one K
    program per token and no tile state, else the RoPE part of Q
    (`nvfp4_hip.q_rope_quantize`) with this launch doing K only. Graph-safe.
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
            raise ValueError("NVFP4 decode prep q_planes must be [B, H, 256] + [B, H, 128].")
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
        qn = torch.empty((batch, heads, NOPE // 2), dtype=torch.uint8, device=q_rope.device)
        qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_rope.device)
    if rows is None:
        # Q rows per program: 32 schedules better from ~384 batch rows up.
        rows = 32 if batch >= 384 else ROWS
    if heads % rows:
        raise ValueError(f"NVFP4 decode prep needs {rows} rows to divide {heads} heads.")
    if not slots.is_contiguous():
        raise ValueError("NVFP4 decode prep needs contiguous slots.")
    if NOPE % (128 * k_split):
        raise ValueError(f"NVFP4 decode prep k_split={k_split} must split {NOPE} into 128s.")
    q_rows = batch * heads
    q_programs = triton.cdiv(q_rows, rows)
    if hip_q is None:
        hip_q = os.environ.get("ATOM_MLA_NVFP4_HIP_Q", "0") == "1"
    if batch and hip_q and nope_done and not interleave_q:
        from atom.model_ops.attentions import nvfp4_hip

        if n16 and tile_state is None and k_split == 1 and ring.shape[1] == TILE:
            nvfp4_hip.decode_prep(q_rope, qa, ring, req_ids, k_nope, k_rope, slots, kv_nope,
                                  kv_aux, positions, cos, sin)
            return qn, qa
        nvfp4_hip.q_rope_quantize(q_rope, qa, positions, slots, cos, sin)
        q_programs = 0
    incr = tile_state is not None
    if incr:
        t_amax, t_tag = tile_state
        if interleave_q:
            raise ValueError("NVFP4 decode prep tile_state does not take interleave_q.")
        if t_amax.dtype != torch.float32 or t_amax.shape[-1] != NOPE // 16 or t_amax.stride(-1) != 1:
            raise ValueError("NVFP4 tile_state amax must be f32 [rows, 32] with a unit inner stride.")
        if t_tag.dtype != torch.int32:
            raise ValueError("NVFP4 tile_state tag must be int32 [rows].")
    if batch and interleave_q:
        if heads % (k_split * rows):
            raise ValueError(f"NVFP4 decode prep interleave_q needs {k_split} x {rows} to divide {heads} heads.")
        _decode_prep_qk_kernel[(batch * k_split,)](
            q_nope, q_rope, qn, qa, q_rows,
            q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1),
            ring, k_nope, k_rope, req_ids, slots, kv_nope, kv_aux,
            ring.stride(0), ring.stride(1), k_nope.stride(0), k_rope.stride(0),
            positions,
            cos if fused else q_nope,
            sin if fused else q_nope,
            cos.stride(0) if fused else 0,
            ROWS=rows,
            HEADS=heads,
            RING=ring.shape[1],
            PAGE=PAGE,
            TILE=TILE,
            NOPE=NOPE,
            ROPE=ROPE,
            ROPE_FUSED=fused,
            N16=n16,
            NOPE_DONE=nope_done,
            KSPLIT=k_split,
            num_warps=4,
        )
        return qn, qa
    if batch:
        _decode_prep_kernel[(q_programs + batch * k_split,)](
            q_nope, q_rope, qn, qa, q_rows, heads,
            q_nope.stride(0), q_nope.stride(1), q_rope.stride(0), q_rope.stride(1),
            q_programs,
            ring, k_nope, k_rope, req_ids, slots, kv_nope, kv_aux,
            ring.stride(0), ring.stride(1), k_nope.stride(0), k_rope.stride(0),
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
            N16=n16,
            NOPE_DONE=nope_done,
            KSPLIT=k_split,
            amax_ptr=t_amax if incr else ring,
            tag_ptr=t_tag if incr else ring,
            stride_amax=t_amax.stride(0) if incr else 0,
            stride_tag=t_tag.stride(0) if incr else 0,
            INCR=incr,
            num_warps=4,
        )
    return qn, qa


def _check_rope_cache(cos: torch.Tensor, sin: torch.Tensor) -> None:
    if cos is None or sin is None or cos.dtype != torch.bfloat16 or sin.dtype != torch.bfloat16:
        raise ValueError("Fused NVFP4 RoPE needs the BF16 cos/sin caches.")
    if cos.shape[-1] != ROPE // 2 or cos.stride(-1) != 1 or cos.stride() != sin.stride():
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
    _, codes, scale = _fp4(blocks, tl.max(tl.abs(blocks), axis=2)[:, :, None], False, Q_GAIN, Q_RCP)
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


def _aiter_kup_nvfp4():
    """AITER's gfx1250 BF16 batched GEMM with the NVFP4 epilogue, if present."""
    import inspect

    try:
        from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16
    except ImportError:
        return None
    params = inspect.signature(batched_gemm_bf16).parameters
    return batched_gemm_bf16 if "nvfp4_out" in params else None


def kup_quantize_q_nope(
    q_nope: torch.Tensor,
    w_k: torch.Tensor,
    qn: torch.Tensor | None = None,
    qa: torch.Tensor | None = None,
    block_m: int = 64,
    block_n: int = 128,
    num_warps: int = 4,
    backend: str = "auto",
    config: dict | None = None,
):
    """The absorbed ``q_nope [B, H, P] @ W_K[H, L, P]^T`` straight into the
    NVFP4 Q planes: ``qn [B, H, L // 2]`` codes and the NoPE scales of
    ``qa [B, H, 128]`` (bytes 64..), the same bytes as the BF16 BMM followed by
    `quantize_q`. The RoPE part of ``qa`` is left to `decode_prep`. Graph-safe.

    ``backend``: "gluon" is AITER's batched GEMM with its NVFP4 epilogue
    (``nvfp4_out``, ``config`` its tuning dict), "full" the whole-row Gluon
    kernel (`nvfp4_kup_gluon.kup_nvfp4`), "triton" the kernel here
    (``block_m``, ``block_n``, ``num_warps``), "auto" "full" under
    ``ATOM_MLA_NVFP4_KUP=full`` where it applies, else gluon when available.
    """
    batch, heads, p = q_nope.shape
    hw, lat, pw = w_k.shape
    if (hw, pw) != (heads, p) or lat != NOPE:
        raise ValueError(f"NVFP4 K-up wants W_K [{heads}, {NOPE}, {p}], got {tuple(w_k.shape)}.")
    if q_nope.dtype != torch.bfloat16 or w_k.dtype != torch.bfloat16:
        raise ValueError("NVFP4 K-up takes BF16 q_nope and W_K.")
    if q_nope.stride(-1) != 1 or w_k.stride(-1) != 1:
        raise ValueError("NVFP4 K-up needs unit inner strides.")
    if qn is None:
        qn = torch.empty((batch, heads, NOPE // 2), dtype=torch.uint8, device=q_nope.device)
    if qa is None:
        qa = torch.empty((batch, heads, 128), dtype=torch.uint8, device=q_nope.device)
    if backend == "full" or (backend == "auto" and os.environ.get("ATOM_MLA_NVFP4_KUP") == "full"):
        from atom.model_ops.attentions.nvfp4_kup_gluon import kup_nvfp4_applies, kup_nvfp4_persist, kup_nvfp4_v2

        if kup_nvfp4_applies(q_nope, w_k, qn, qa):
            # The persistent kernel at every batch: kup_nvfp4_v2 writes its
            # codes into its A load buffer and has been seen to leave A bytes
            # in them.
            if os.environ.get("ATOM_MLA_NVFP4_KUP_PERSIST", "1") == "1":
                kup_nvfp4_persist(q_nope, w_k, qn, qa)
            else:
                kup_nvfp4_v2(q_nope, w_k, qn, qa, tdm_out=True, fast_block=True, chunk=256)
            if os.environ.get("ATOM_MLA_NVFP4_KUP_CHECK") == "1" and not torch.cuda.is_current_stream_capturing():
                from atom.model_ops.attentions.nvfp4_kup_check import check

                check(q_nope, w_k, qn, qa)
            return qn, qa
        if backend == "full":
            raise ValueError("NVFP4 whole-row K-up needs P = 128, L = 512 and aligned planes.")
    gluon = _aiter_kup_nvfp4() if backend in ("auto", "gluon") else None
    if backend == "gluon" and gluon is None:
        raise RuntimeError("This aiter's batched_gemm_bf16 has no nvfp4_out.")
    if batch and gluon is not None:
        planes = (qn.transpose(0, 1), qa[..., 64:96].transpose(0, 1))
        gluon(q_nope.transpose(0, 1), w_k, config=None if config is None else dict(config),
              nvfp4_out=planes)
    elif batch:
        _kup_quantize_kernel[(heads, triton.cdiv(batch, block_m), NOPE // block_n)](
            q_nope, w_k, qn, qa, batch,
            q_nope.stride(0), q_nope.stride(1), w_k.stride(0), w_k.stride(1),
            qn.stride(0), qn.stride(1), qa.stride(0), qa.stride(1),
            P=p, BLOCK_M=block_m, BLOCK_N=block_n, num_warps=num_warps,
        )
    return qn, qa
