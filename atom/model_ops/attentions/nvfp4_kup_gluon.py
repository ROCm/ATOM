# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NVFP4 K-up on gfx1250 with whole rows per workgroup (Gluon).

``q_nope [B, H, P] @ W_K[H, L, P]^T`` quantized like `quantize_q`'s NoPE part,
the same bytes as AITER's batched_gemm_bf16 NVFP4 epilogue. One workgroup per
(head, BLOCK_M rows) does all L = 2 * BLOCK_N columns: A is read once, the two
B halves in turn (the second loads while the first half quantizes), and each
row's codes (L / 2 bytes) and NoPE scales (L / 16) leave in one TDM store per
plane. The output rows of one head are a head apart in ``[B, H, ...]``, so
per-lane stores would be narrow scattered writes; whole row segments through
TDM are not.
"""

import os

import torch
import triton
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

_Q_GAIN = gl.constexpr(6.72)
# RN(1 / 6) and RN(1 / fp32(6.72)): the constant divisions are a multiply and
# an FMA correction (Markstein), exact for every BF16 amax and E4M3 scale.
_RCP_6 = gl.constexpr(0.1666666716337204)
_RCP_GAIN = gl.constexpr(0.1488095223903656)


@gluon.jit
def _e4m3(v):
    # Round v in [2**-9, 448] to E4M3, nearest even: value and code.
    bits = v.to(gl.int32, bitcast=True)
    e = gl.maximum(((bits >> 23) & 0xFF) - 127, -6)
    step = ((e - 3 + 127) << 23).to(gl.float32, bitcast=True)
    x = v * ((3 - e + 127) << 23).to(gl.float32, bitcast=True)
    f = gl.floor(x)
    d = x - f
    odd = (f.to(gl.int32) & 1) == 1
    m = f + gl.where((d > 0.5) | ((d == 0.5) & odd), 1.0, 0.0)
    code = (e + 7) * 8 + (m.to(gl.int32) - 8)
    return m * step, code


@gluon.jit
def _rcp_approx(x):
    return gl.inline_asm_elementwise("v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1)


@gluon.jit
def _block_scale(amax, FAST: gl.constexpr):
    """A 16-column block's E4M3 scale code (amax / 6 at the gain), its divisor
    scale / gain and RN(1 / divisor). FAST: the E4M3 rounding by the hardware
    conversion (nearest even) and the reciprocal as v_rcp_f32 plus one Newton
    step, both checked equal to the exact forms over every BF16 amax. In the
    K-up kernel FAST writes AITER's bytes for every block of the BF16 product
    that holds no Inf; a block with an Inf gets other bytes."""
    rcp6 = gl.full_like(amax, _RCP_6)
    q6 = amax * rcp6
    ideal = gl.fma(gl.fma(gl.full_like(amax, -6.0), q6, amax), rcp6, q6)
    ideal = gl.where(amax == 0, 1.0, ideal)
    gain = gl.full_like(amax, _Q_GAIN)
    t = gl.minimum(gl.maximum(ideal * gain, 1.0 / 512), 448.0)
    if FAST:
        s8 = t.to(gl.float8e4nv)
        scale = s8.to(gl.float32)
        scale_code = s8.to(gl.uint8, bitcast=True)
    else:
        scale, scale_code = _e4m3(t)
    rcp_gain = gl.full_like(amax, _RCP_GAIN)
    qg = scale * rcp_gain
    div = gl.fma(gl.fma(-gain, qg, scale), rcp_gain, qg)
    if FAST:
        # The negation on r, not on div, so a NaN divisor keeps its sign, as
        # through div_rn.
        r = _rcp_approx(div)
        recip = gl.fma(gl.fma(div, -r, gl.full_like(div, 1.0)), r, r)
    else:
        recip = gl.div_rn(gl.full_like(div, 1.0), div)
    return scale_code, div, recip


@gluon.jit
def _quantize_half(acc, stage, c_blocks, s_smem, blk0, BLOCK_M: gl.constexpr,
                   BLOCK_N: gl.constexpr, num_warps: gl.constexpr, CHUNK: gl.constexpr = 64,
                   FAST: gl.constexpr = False, PROBE: gl.constexpr = 0):
    """``acc [BLOCK_M, BLOCK_N]`` rounded to BF16 (as a BF16 output stores it)
    and quantized into the code and scale tiles from 16-column block ``blk0``
    on: per row, 16-column blocks, E4M3 scale of amax / 6 at the gain, codes
    over the rounded scale. PROBE (timing only, wrong bytes): 1 the block
    scales as constants, 2 no quantization after the stage store."""
    G: gl.constexpr = CHUNK // 16
    gl.barrier()  # the previous half's readers are done with the stage
    stage.store(acc.to(gl.bfloat16))
    gl.barrier()
    # Each thread owns one 16-column block of a row: the block amax and its
    # E2M1 conversion stay inside the thread.
    L3: gl.constexpr = gl.BlockedLayout([1, 1, 16], [32 // G, G, 1], [num_warps, 1, 1], [2, 1, 0])
    blocks = stage.reshape([BLOCK_M, BLOCK_N // 16, 16])
    if PROBE < 2:
        for c0 in gl.static_range(0, BLOCK_N, CHUNK):
            x3 = blocks.slice(c0 // 16, G, dim=1).load(L3).to(gl.float32)
            amax = gl.max(gl.abs(x3), axis=2)
            if PROBE == 1:
                scale_code = (amax * 0.0).to(gl.uint8)
                div = amax * 0.0 + 1.0
                recip = div
            else:
                scale_code, div, recip = _block_scale(amax, FAST)
            # x / div correctly rounded: q = x RN(1 / div) is within an ulp and one
            # FMA correction rounds it exactly. Signed x, with the remainder negated
            # (exact) so +-0 keep their sign.
            div3 = gl.expand_dims(div, 2) + gl.zeros_like(x3)
            recip3 = gl.expand_dims(recip, 2) + gl.zeros_like(x3)
            q0 = x3 * recip3
            q = gl.fma(-gl.fma(div3, q0, -x3), recip3, q0)
            # The hardware FP32 -> E2M1 conversion (nearest even, saturating) of q
            # at scale 2**0 packs two codes per byte, low nibble first.
            one = gl.expand_dims(gl.full_like(amax, 127, dtype=gl.uint8), 2)
            packed = gl.amd.gfx1250.scaled_downcast(q, one, "e2m1", axis=2)
            c_blocks.slice(blk0 + c0 // 16, G, dim=1).store(packed)
            s_smem.slice(blk0 + c0 // 16, G, dim=1).store(scale_code.to(gl.uint8))


@gluon.jit
def _quantize_regs(acc, c_blocks, s_smem, blk0, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
    """`_quantize_half` (fast block scales) on the accumulator where it is: in
    the transposed WMMA layout a lane holds 8 consecutive columns of a row and
    the lane 16 away the other 8 of its 16-column block, so the block amax is a
    lane-local max and one exchange, and each lane's 8 codes are 4 bytes."""
    x3 = gl.reshape(acc.to(gl.bfloat16).to(gl.float32), [BLOCK_M, BLOCK_N // 16, 16])
    amax = gl.max(gl.abs(x3), axis=2)
    scale_code, div, recip = _block_scale(amax, True)
    div3 = gl.expand_dims(div, 2) + gl.zeros_like(x3)
    recip3 = gl.expand_dims(recip, 2) + gl.zeros_like(x3)
    q0 = x3 * recip3
    q = gl.fma(-gl.fma(div3, q0, -x3), recip3, q0)
    one = gl.expand_dims(gl.full_like(amax, 127, dtype=gl.uint8), 2)
    packed = gl.amd.gfx1250.scaled_downcast(q, one, "e2m1", axis=2)
    c_blocks.slice(blk0, BLOCK_N // 16, dim=1).store(packed)
    s_smem.slice(blk0, BLOCK_N // 16, dim=1).store(scale_code.to(gl.uint8))


@gluon.jit
def _kup_nvfp4_kernel(
    a_ptr, b_ptr, c_ptr, s_ptr, M, N, K,
    stride_ab, stride_am, stride_bb, stride_bn, stride_cb, stride_cm, stride_sb, stride_sm,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr, OPERAND_LAYOUT_B: gl.constexpr, num_warps: gl.constexpr,
    TWO_B: gl.constexpr = False,
    B_REGULAR: gl.constexpr = False,
    SEQ_B: gl.constexpr = False,
    PAD_LDS: gl.constexpr = False,
    B_GLOBAL: gl.constexpr = False,
):
    # One K tile (K == BLOCK_K), A rows and B rows unit-stride along K.
    batch_id = gl.program_id(axis=0)
    pid_m = gl.program_id(axis=1)
    NC: gl.constexpr = BLOCK_N  # code bytes per row: 2 * BLOCK_N columns, two per byte
    NS: gl.constexpr = BLOCK_N // 8  # scales per row
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + batch_id * stride_ab + pid_m * BLOCK_M * stride_am,
        shape=(M - pid_m * BLOCK_M, K), strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + batch_id * stride_bb, shape=(N, K), strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_buf)
    if not B_GLOBAL:
        b_buf = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b_buf)
    if TWO_B:
        b_buf2 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
        gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b_buf2)
    if PAD_LDS:
        # Probe: one workgroup per WGP, as with two B buffers, without using them.
        pad = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
        gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], pad)

    # 16 bytes of padding per row: the rows would otherwise hit the same LDS
    # banks when a warp reads 8 of them at once.
    STAGE: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    CODES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NC, 16]], [BLOCK_M, NC], [1, 0])
    SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NS, 16]], [BLOCK_M, NS], [1, 0])
    stage = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_N], STAGE)
    c_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, NC], CODES)
    s_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, NS], SCALES)
    c_blocks = c_smem.reshape([BLOCK_M, NS, 8])

    # A TDM load is split over the warps and the wait covers only this warp's
    # part: the barrier makes the whole A and B tiles visible to every warp
    # (the relaxed loads below insert none).
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buf, OPERAND_LAYOUT_A)
    if B_GLOBAL:
        # B straight from global memory (W_K rows are K-contiguous and L2
        # resident) into the WMMA operand layout: no B tile in LDS.
        offs_k = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, OPERAND_LAYOUT_B))
        offs_n = gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, OPERAND_LAYOUT_B))
        b_ptrs = b_ptr + batch_id * stride_bb + offs_n[None, :] * stride_bn + offs_k[:, None]
        cur_b = gl.load(b_ptrs)
    elif B_REGULAR:
        # A plain shared load: the compiler orders it before the TDM load
        # that overwrites the buffer with the second half.
        cur_b = b_buf.permute([1, 0]).load(OPERAND_LAYOUT_B)
    else:
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buf.permute([1, 0]), OPERAND_LAYOUT_B)
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    acc = gl.amd.gfx1250.wmma(cur_a, cur_b, zero)
    if B_GLOBAL:
        # The second half's B loads while the first half quantizes.
        cur_b = gl.load(b_ptrs + BLOCK_N * stride_bn)
        _quantize_half(acc, stage, c_blocks, s_smem, 0, BLOCK_M, BLOCK_N, num_warps)
    elif TWO_B:
        _quantize_half(acc, stage, c_blocks, s_smem, 0, BLOCK_M, BLOCK_N, num_warps)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buf2.permute([1, 0]), OPERAND_LAYOUT_B)
    elif SEQ_B:
        # Probe: the second half's load only after the first half quantizes.
        _quantize_half(acc, stage, c_blocks, s_smem, 0, BLOCK_M, BLOCK_N, num_warps)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b_buf)
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buf.permute([1, 0]), OPERAND_LAYOUT_B)
    else:
        # Every wave has its B operand: the second half may overwrite the buffer
        # while the first half quantizes.
        gl.barrier()
        gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b_buf)
        _quantize_half(acc, stage, c_blocks, s_smem, 0, BLOCK_M, BLOCK_N, num_warps)
        gl.amd.gfx1250.tdm.async_wait(0)
        gl.barrier()
        if B_REGULAR:
            cur_b = b_buf.permute([1, 0]).load(OPERAND_LAYOUT_B)
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_buf.permute([1, 0]), OPERAND_LAYOUT_B)
    acc = gl.amd.gfx1250.wmma(cur_a, cur_b, zero)
    _quantize_half(acc, stage, c_blocks, s_smem, BLOCK_N // 16, BLOCK_M, BLOCK_N, num_warps)

    # All waves' LDS writes land before TDM reads the tiles; rows past M fall
    # outside the descriptors and are not written.
    gl.barrier()
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + batch_id * stride_cb, shape=(M, NC), strides=(stride_cm, 1),
        block_shape=(BLOCK_M, NC), layout=CODES)
    s_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=s_ptr + batch_id * stride_sb, shape=(M, NS), strides=(stride_sm, 1),
        block_shape=(BLOCK_M, NS), layout=SCALES)
    gl.amd.gfx1250.tdm.async_store(c_desc, [pid_m * BLOCK_M, 0], c_smem)
    gl.amd.gfx1250.tdm.async_store(s_desc, [pid_m * BLOCK_M, 0], s_smem)
    gl.amd.gfx1250.tdm.async_wait(0)


@gluon.jit
def _quantize_store_half(acc, stage, c_ptr, s_ptr, batch_id, row0, M, n0, stride_cb, stride_cm,
                         stride_sb, stride_sm, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                         num_warps: gl.constexpr):
    """``acc [BLOCK_M, BLOCK_N]`` (columns ``n0 ..``) rounded to BF16 through
    ``stage`` and quantized, then stored straight from registers: RP rows at a
    time, a thread holding 32 adjacent columns (two blocks) and a row's lanes
    in address order, so a store writes whole BLOCK_N / 2-byte code rows."""
    NB: gl.constexpr = BLOCK_N // 16
    TPR: gl.constexpr = NB // 2
    RPW: gl.constexpr = 32 // TPR
    RP: gl.constexpr = RPW * num_warps
    gl.barrier()  # every warp's reads of the buffer under ``stage`` are done
    stage.store(acc.to(gl.bfloat16))
    gl.barrier()
    L3: gl.constexpr = gl.BlockedLayout([1, 2, 16], [RPW, TPR, 1], [num_warps, 1, 1], [2, 1, 0])
    LS: gl.constexpr = gl.SliceLayout(2, L3)
    blocks = stage.reshape([BLOCK_M, NB, 16])
    for r0 in gl.static_range(0, BLOCK_M, RP):
        x3 = blocks.slice(r0, RP, dim=0).load(L3).to(gl.float32)
        amax = gl.max(gl.abs(x3), axis=2)
        rcp6 = gl.full_like(amax, _RCP_6)
        q6 = amax * rcp6
        ideal = gl.fma(gl.fma(gl.full_like(amax, -6.0), q6, amax), rcp6, q6)
        ideal = gl.where(amax == 0, 1.0, ideal)
        gain = gl.full_like(amax, _Q_GAIN)
        scale, scale_code = _e4m3(gl.minimum(gl.maximum(ideal * gain, 1.0 / 512), 448.0))
        rcp_gain = gl.full_like(amax, _RCP_GAIN)
        qg = scale * rcp_gain
        div = gl.fma(gl.fma(-gain, qg, scale), rcp_gain, qg)
        recip = gl.div_rn(gl.full_like(div, 1.0), div)
        div3 = gl.expand_dims(div, 2) + gl.zeros_like(x3)
        recip3 = gl.expand_dims(recip, 2) + gl.zeros_like(x3)
        q0 = x3 * recip3
        q = gl.fma(-gl.fma(div3, q0, -x3), recip3, q0)
        one = gl.expand_dims(gl.full_like(amax, 127, dtype=gl.uint8), 2)
        packed = gl.amd.gfx1250.scaled_downcast(q, one, "e2m1", axis=2)
        rows_p = row0 + r0 + gl.arange(0, RP, layout=gl.SliceLayout(1, gl.SliceLayout(2, packed.type.layout)))
        blk_p = gl.arange(0, NB, layout=gl.SliceLayout(0, gl.SliceLayout(2, packed.type.layout)))
        byte_p = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, packed.type.layout)))
        r3 = gl.expand_dims(gl.expand_dims(rows_p, 1), 2)
        col3 = n0 // 2 + gl.expand_dims(gl.expand_dims(blk_p, 0), 2) * 8 + gl.expand_dims(gl.expand_dims(byte_p, 0), 1)
        gl.store(c_ptr + batch_id * stride_cb + r3.to(gl.int64) * stride_cm + col3, packed, mask=r3 < M)
        r2 = gl.expand_dims(row0 + r0 + gl.arange(0, RP, layout=gl.SliceLayout(1, LS)), 1)
        blk_s = gl.expand_dims(gl.arange(0, NB, layout=gl.SliceLayout(0, LS)), 0)
        gl.store(s_ptr + batch_id * stride_sb + r2.to(gl.int64) * stride_sm + n0 // 16 + blk_s,
                 scale_code.to(gl.uint8), mask=r2 < M)


@gluon.jit
def _kup_nvfp4_v2_kernel(
    a_ptr, b_ptr, c_ptr, s_ptr, M, N, K,
    stride_ab, stride_am, stride_bb, stride_bn, stride_cb, stride_cm, stride_sb, stride_sm,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr, OPERAND_LAYOUT_B: gl.constexpr, num_warps: gl.constexpr,
    OVERLAP_B1: gl.constexpr = True,
    RELAXED: gl.constexpr = False,
    OWN_STAGE: gl.constexpr = False,
    TDM_OUT: gl.constexpr = False,
    FAST_BLOCK: gl.constexpr = False,
    PROBE: gl.constexpr = 0,
    EARLY_B1: gl.constexpr = False,
    CHUNK: gl.constexpr = 64,
):
    # One K tile; A and both B halves load once, up front, and nothing is
    # loaded asynchronously after that. Each B half's buffer, once its WMMA
    # operand is in registers, becomes the stage for that half's accumulator
    # (no separate stage), so A + 2 B halves is all the LDS.
    batch_id = gl.program_id(axis=0)
    pid_m = gl.program_id(axis=1)
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + batch_id * stride_ab + pid_m * BLOCK_M * stride_am,
        shape=(M - pid_m * BLOCK_M, K), strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + batch_id * stride_bb, shape=(N, K), strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b0 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    b1 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    gl.amd.gfx1250.tdm.async_load(a_desc, [0, 0], a_buf)
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b0)
    gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b1)
    # The loads are split over the warps and a wait covers only this warp's
    # part: a barrier after it makes the whole tiles visible. The second B
    # half stays in flight through the first half (its own, unread buffer).
    gl.amd.gfx1250.tdm.async_wait(1 if OVERLAP_B1 else 0)
    gl.barrier()
    # A view of a B buffer has to keep its padding pattern (8 elements every
    # BLOCK_K), here two runs per BLOCK_N-element stage row.
    STAGE: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_K, 8]], [BLOCK_M, BLOCK_N], [1, 0])
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    if OWN_STAGE:
        OWN: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0])
        own = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_N], OWN)
    if RELAXED:
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_buf, OPERAND_LAYOUT_A)
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b0.permute([1, 0]), OPERAND_LAYOUT_B)
    else:
        cur_a = a_buf.load(OPERAND_LAYOUT_A)
        cur_b = b0.permute([1, 0]).load(OPERAND_LAYOUT_B)
    acc = gl.amd.gfx1250.wmma(cur_a, cur_b, zero)
    if TDM_OUT:
        # The one-B kernel's epilogue: column chunks into code / scale tiles,
        # one TDM store per plane at the end. A is in registers from here on,
        # so its buffer (the same bytes per row: 256 + 16) holds the codes.
        NC: gl.constexpr = BLOCK_N
        NS: gl.constexpr = BLOCK_N // 8
        CODES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NC, 16]], [BLOCK_M, NC], [1, 0])
        SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NS, 16]], [BLOCK_M, NS], [1, 0])
        c_smem = a_buf.reinterpret(gl.uint8, [BLOCK_M, NC], CODES)
        s_smem = gl.allocate_shared_memory(gl.uint8, [BLOCK_M, NS], SCALES)
        c_blocks = c_smem.reshape([BLOCK_M, NS, 8])
        if EARLY_B1:
            # The second half's WMMA goes out before the first half quantizes,
            # so the matrix unit works under that half's VALU code.
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()
            acc1 = gl.amd.gfx1250.wmma(cur_a, b1.permute([1, 0]).load(OPERAND_LAYOUT_B), zero)
        _quantize_half(acc, b0.reinterpret(gl.bfloat16, [BLOCK_M, BLOCK_N], STAGE), c_blocks, s_smem, 0,
                       BLOCK_M, BLOCK_N, num_warps, CHUNK=CHUNK, FAST=FAST_BLOCK, PROBE=PROBE)
    else:
        _quantize_store_half(acc, own if OWN_STAGE else b0.reinterpret(gl.bfloat16, [BLOCK_M, BLOCK_N], STAGE), c_ptr, s_ptr,
                             batch_id, pid_m * BLOCK_M, M, 0, stride_cb, stride_cm, stride_sb, stride_sm,
                             BLOCK_M, BLOCK_N, num_warps)
    if EARLY_B1 and TDM_OUT:
        acc = acc1
    else:
        if OVERLAP_B1:
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()
        if RELAXED:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(b1.permute([1, 0]), OPERAND_LAYOUT_B)
        else:
            cur_b = b1.permute([1, 0]).load(OPERAND_LAYOUT_B)
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, zero)
    if TDM_OUT:
        _quantize_half(acc, b1.reinterpret(gl.bfloat16, [BLOCK_M, BLOCK_N], STAGE), c_blocks, s_smem,
                       BLOCK_N // 16, BLOCK_M, BLOCK_N, num_warps, CHUNK=CHUNK, FAST=FAST_BLOCK, PROBE=PROBE)
        gl.barrier()
        c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=c_ptr + batch_id * stride_cb, shape=(M, NC), strides=(stride_cm, 1),
            block_shape=(BLOCK_M, NC), layout=CODES)
        s_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=s_ptr + batch_id * stride_sb, shape=(M, NS), strides=(stride_sm, 1),
            block_shape=(BLOCK_M, NS), layout=SCALES)
        gl.amd.gfx1250.tdm.async_store(c_desc, [pid_m * BLOCK_M, 0], c_smem)
        gl.amd.gfx1250.tdm.async_store(s_desc, [pid_m * BLOCK_M, 0], s_smem)
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        _quantize_store_half(acc, own if OWN_STAGE else b1.reinterpret(gl.bfloat16, [BLOCK_M, BLOCK_N], STAGE), c_ptr, s_ptr,
                             batch_id, pid_m * BLOCK_M, M, BLOCK_N, stride_cb, stride_cm, stride_sb,
                             stride_sm, BLOCK_M, BLOCK_N, num_warps)


def kup_nvfp4_v2(q_nope, w_k, qn, qa, block_m: int = 64, num_warps: int = 4, overlap_b1: bool = True,
                 relaxed: bool = False, own_stage: bool = False, tdm_out: bool = False,
                 fast_block: bool = False, probe: int = 0, early_b1: bool = False, chunk: int = 64):
    """`kup_nvfp4` without asynchronous loads after the start and without a
    separate stage or output tiles in LDS (stores straight from registers)."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    batch, heads, p = q_nope.shape
    lat = w_k.shape[1]
    block_n = lat // 2
    wmma, op_a, op_b = create_wmma_layouts(num_warps)
    sh_a, sh_b = create_shared_layouts(block_m, block_n, p, "TN", 16)
    if batch:
        _kup_nvfp4_v2_kernel[(heads, triton.cdiv(batch, block_m))](
            q_nope, w_k, qn, qa[..., 64:], batch, lat, p,
            q_nope.stride(1), q_nope.stride(0), w_k.stride(0), w_k.stride(1),
            qn.stride(1), qn.stride(0), qa.stride(1), qa.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=p,
            SHARED_LAYOUT_A=sh_a, SHARED_LAYOUT_B=sh_b, WMMA_LAYOUT=wmma,
            OPERAND_LAYOUT_A=op_a, OPERAND_LAYOUT_B=op_b, num_warps=num_warps, OVERLAP_B1=overlap_b1,
            RELAXED=relaxed, OWN_STAGE=own_stage, TDM_OUT=tdm_out, FAST_BLOCK=fast_block, PROBE=probe,
            EARLY_B1=early_b1, CHUNK=chunk,
        )
    return qn, qa


@gluon.jit
def _kup_nvfp4_persist_kernel(
    a_ptr, b_ptr, c_ptr, s_ptr, M, N, K, TILES,
    stride_ab, stride_am, stride_bb, stride_bn, stride_cb, stride_cm, stride_sb, stride_sm,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr, SHARED_LAYOUT_B: gl.constexpr, WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr, OPERAND_LAYOUT_B: gl.constexpr, num_warps: gl.constexpr,
):
    # One workgroup per (head, split): the head's W_K stays in LDS while the
    # workgroup walks TILES row tiles. The next tile's A loads while this one
    # runs both halves, quantized in registers; each tile's codes and scales
    # leave by TDM stores from double buffers, one tile behind.
    head = gl.program_id(axis=0)
    m0 = gl.program_id(axis=1) * TILES * BLOCK_M
    n_tiles = gl.minimum(TILES, gl.cdiv(M - m0, BLOCK_M))
    NC: gl.constexpr = BLOCK_N
    NS: gl.constexpr = BLOCK_N // 8
    CODES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NC, 16]], [BLOCK_M, NC], [1, 0])
    SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[NS, 16]], [BLOCK_M, NS], [1, 0])
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + head * stride_ab, shape=(M, K), strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K), layout=SHARED_LAYOUT_A)
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + head * stride_bb, shape=(N, K), strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K), layout=SHARED_LAYOUT_B)
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + head * stride_cb, shape=(M, NC), strides=(stride_cm, 1),
        block_shape=(BLOCK_M, NC), layout=CODES)
    s_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=s_ptr + head * stride_sb, shape=(M, NS), strides=(stride_sm, 1),
        block_shape=(BLOCK_M, NS), layout=SCALES)
    a_buf = gl.allocate_shared_memory(a_ptr.type.element_ty, [2, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A)
    b0 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    b1 = gl.allocate_shared_memory(b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B)
    c_buf = gl.allocate_shared_memory(gl.uint8, [2, BLOCK_M, NC], CODES)
    s_buf = gl.allocate_shared_memory(gl.uint8, [2, BLOCK_M, NS], SCALES)
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b0)
    gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b1)
    gl.amd.gfx1250.tdm.async_load(a_desc, [m0, 0], a_buf.index(0))
    # A TDM load is split over the warps and a wait covers only this warp's
    # part: a barrier after each wait makes the whole tiles visible.
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    for t in range(n_tiles):
        if t + 1 < n_tiles:
            # Every warp read this buffer (tile t - 1) before the last barrier.
            gl.amd.gfx1250.tdm.async_load(a_desc, [m0 + (t + 1) * BLOCK_M, 0], a_buf.index((t + 1) % 2))
        cur_a = a_buf.index(t % 2).load(OPERAND_LAYOUT_A)
        acc0 = gl.amd.gfx1250.wmma(cur_a, b0.permute([1, 0]).load(OPERAND_LAYOUT_B), zero)
        acc1 = gl.amd.gfx1250.wmma(cur_a, b1.permute([1, 0]).load(OPERAND_LAYOUT_B), zero)
        c_blocks = c_buf.index(t % 2).reshape([BLOCK_M, NS, 8])
        _quantize_regs(acc0, c_blocks, s_buf.index(t % 2), 0, BLOCK_M, BLOCK_N)
        _quantize_regs(acc1, c_blocks, s_buf.index(t % 2), BLOCK_N // 16, BLOCK_M, BLOCK_N)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(c_desc, [m0 + t * BLOCK_M, 0], c_buf.index(t % 2))
        gl.amd.gfx1250.tdm.async_store(s_desc, [m0 + t * BLOCK_M, 0], s_buf.index(t % 2))
        # TDM ops complete in order: with at most this tile's two stores left,
        # the next A tile is in and tile t - 1's stores (issued before it) are
        # out, so their buffers can be refilled.
        gl.amd.gfx1250.tdm.async_wait(2)
        gl.barrier()
    gl.amd.gfx1250.tdm.async_wait(0)


_CUS: dict = {}


def kup_nvfp4_persist(q_nope, w_k, qn, qa, splits: int | None = None, block_m: int = 64, num_warps: int = 4):
    """`kup_nvfp4_v2` (TDM out, fast block scales) as one persistent workgroup
    per (head, split), W_K resident in LDS, quantizing in registers;
    ``splits`` defaults to the device's CUs per head (one workgroup per CU).
    Same bytes."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    batch, heads, p = q_nope.shape
    if not batch:
        return qn, qa
    if splits is None:
        dev = q_nope.device
        if dev not in _CUS:
            _CUS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
        splits = max(1, _CUS[dev] // heads)
    lat = w_k.shape[1]
    block_n = lat // 2
    wmma, op_a, op_b = create_wmma_layouts(num_warps)
    sh_a, sh_b = create_shared_layouts(block_m, block_n, p, "TN", 16)
    tiles = triton.cdiv(triton.cdiv(batch, block_m), splits)
    _kup_nvfp4_persist_kernel[(heads, triton.cdiv(triton.cdiv(batch, block_m), tiles))](
        q_nope, w_k, qn, qa[..., 64:], batch, lat, p, tiles,
        q_nope.stride(1), q_nope.stride(0), w_k.stride(0), w_k.stride(1),
        qn.stride(1), qn.stride(0), qa.stride(1), qa.stride(0),
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=p,
        SHARED_LAYOUT_A=sh_a, SHARED_LAYOUT_B=sh_b, WMMA_LAYOUT=wmma,
        OPERAND_LAYOUT_A=op_a, OPERAND_LAYOUT_B=op_b, num_warps=num_warps,
    )
    return qn, qa


def kup_nvfp4_applies(q_nope, w_k, qn, qa) -> bool:
    """The shapes and strides `kup_nvfp4` takes: P = 128, L = 512, unit inner
    strides, contiguous W_K and output planes, TDM-aligned rows."""
    return (
        q_nope.shape[2] == 128 and w_k.shape[1:] == (512, 128) and q_nope.stride(2) == 1
        and w_k.is_contiguous() and qn.is_contiguous() and qa.is_contiguous()
        and q_nope.data_ptr() % 16 == 0 and q_nope.stride(0) % 8 == 0 and q_nope.stride(1) % 8 == 0
    )


def kup_nvfp4(q_nope, w_k, qn, qa, block_m: int = 64, num_warps: int = 4, two_b: bool | None = None):
    """``q_nope [B, H, 128] @ w_k[H, 512, 128]^T`` into ``qn [B, H, 256]`` (codes)
    and ``qa [B, H, 128]`` bytes 64..95 (NoPE scales); the rest of ``qa`` is
    left as found. ``block_m = 64``, 4 warps: the fastest from 512 to 1536
    rows. Graph-safe."""
    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16 import (
        create_shared_layouts,
        create_wmma_layouts,
    )

    batch, heads, p = q_nope.shape
    lat = w_k.shape[1]
    block_n = lat // 2
    wmma, op_a, op_b = create_wmma_layouts(num_warps)
    sh_a, sh_b = create_shared_layouts(block_m, block_n, p, "TN", 16)
    if batch:
        _kup_nvfp4_kernel[(heads, triton.cdiv(batch, block_m))](
            q_nope, w_k, qn, qa[..., 64:], batch, lat, p,
            q_nope.stride(1), q_nope.stride(0), w_k.stride(0), w_k.stride(1),
            qn.stride(1), qn.stride(0), qa.stride(1), qa.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=p,
            SHARED_LAYOUT_A=sh_a, SHARED_LAYOUT_B=sh_b, WMMA_LAYOUT=wmma,
            OPERAND_LAYOUT_A=op_a, OPERAND_LAYOUT_B=op_b, num_warps=num_warps,
            TWO_B=(os.environ.get("ATOM_NVFP4_KUP_TWO_B", "0") == "1") if two_b is None else two_b,
            B_REGULAR=os.environ.get("ATOM_NVFP4_KUP_B_REGULAR", "0") == "1",
            SEQ_B=os.environ.get("ATOM_NVFP4_KUP_SEQ_B", "0") == "1",
            PAD_LDS=os.environ.get("ATOM_NVFP4_KUP_PAD_LDS", "0") == "1",
            B_GLOBAL=os.environ.get("ATOM_NVFP4_KUP_B_GLOBAL", "0") == "1",
        )
    return qn, qa
