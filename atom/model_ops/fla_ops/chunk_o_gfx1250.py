# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chunked Gated DeltaNet output with a fused gated RMSNorm on gfx1250.

Each program computes one chunk of one value head:
    o = scale * (exp(g) * (q @ h) + tril(exp(g_i - g_j) * (q @ k.T)) @ v)
The gated epilogue computes RMSNorm(bf16(o)) * weight * SiLU(z).

OUT_FP8 (opt-in, for an FP8 per_1x128 e8m0 out_proj): instead of the bf16 output, write the bf16-rounded result as
fp8 e4m3 [rows, H * V] plus one e8m0 scale per (row, value head) -- V = 128 is one quant group -- in the transposed
layout (byte of row m, head g at g * scale_rows + m), bit-identical to the bf16 output followed by aiter's
per_group_quant_hip(group_size=128, transpose_scale=True, scale_type=fp8_e8m0); see model_ops/mxfp8_act_quant.py.
"""

import functools

import torch

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from atom.utils import envs

_BLOCK_K = 32
_NUM_WARPS = 4
_WAVES_PER_EU = 4


@gluon.jit
def _chunk_o_gfx1250_kernel(
    q,
    k,
    v,
    h,
    g,
    o,
    z,
    norm_weight,
    o_fp8,
    o_scale,
    scale_rows,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    scale,
    eps,
    T,
    stride_zm,
    H: gl.constexpr,
    Hg: gl.constexpr,
    K: gl.constexpr,
    V: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    USE_G: gl.constexpr,
    GATED_NORM: gl.constexpr,
    IS_VARLEN: gl.constexpr,
    HAS_NT: gl.constexpr,
    FAST_NORM: gl.constexpr,
    WMMA: gl.constexpr,
    OP_A: gl.constexpr,
    OP_B: gl.constexpr,
    SMEM_ROW_K: gl.constexpr,
    SMEM_ROW_V: gl.constexpr,
    SMEM_H: gl.constexpr,
    SMEM_A: gl.constexpr,
    SMEM_G: gl.constexpr,
    NORM_LAYOUT: gl.constexpr,
    FAST_NORM_LAYOUT: gl.constexpr,
    OUT_FP8: gl.constexpr,
    SMEM_FP8: gl.constexpr,
    ROW_LAYOUT: gl.constexpr,
):
    # Adjacent value heads reuse the same Q and K tiles.
    pid_bh = gl.program_id(0)
    pid_m = gl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H
    if IS_VARLEN and HAS_NT:  # noqa: SIM102 (a constexpr, then a device load)
        if pid_m >= gl.load(nt_ptr):
            return
    if IS_VARLEN:
        chunk = pid_m
        sequence = gl.load(chunk_indices + pid_m * 2).to(gl.int32)
        pid_m = gl.load(chunk_indices + pid_m * 2 + 1).to(gl.int32)
        bos = gl.load(cu_seqlens + sequence).to(gl.int32)
        eos = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
        T = eos - bos
    else:
        chunk = batch * gl.cdiv(T, BLOCK_M) + pid_m
        bos = batch * T
    start_m = pid_m * BLOCK_M
    start_token = bos.to(gl.int64)

    q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=q + (start_token * Hg + head // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_ROW_K,
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k + (start_token * Hg + head // (H // Hg)) * K,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_ROW_K,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v + (start_token * H + head) * V,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, V),
        layout=SMEM_ROW_V,
    )
    h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h + (chunk.to(gl.int64) * H + head) * K * V,
        shape=(K, V),
        strides=(V, 1),
        block_shape=(K, V),
        layout=SMEM_H,
    )
    if OUT_FP8:
        o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=o_fp8 + (start_token * H + head) * V,
            shape=(T, V),
            strides=(H * V, 1),
            block_shape=(BLOCK_M, V),
            layout=SMEM_FP8,
        )
    else:
        o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=o + (start_token * H + head) * V,
            shape=(T, V),
            strides=(H * V, 1),
            block_shape=(BLOCK_M, V),
            layout=SMEM_ROW_K,
        )
    q_smem = gl.allocate_shared_memory(q.type.element_ty, [BLOCK_M, K], SMEM_ROW_K)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [BLOCK_M, K], SMEM_ROW_K)
    h_smem = gl.allocate_shared_memory(h.type.element_ty, [K, V], SMEM_H)
    v_smem = h_smem.slice(0, BLOCK_M, dim=0)
    a_smem = gl.allocate_shared_memory(v.type.element_ty, [BLOCK_M, BLOCK_M], SMEM_A)
    if OUT_FP8:
        o8_smem = gl.allocate_shared_memory(o_fp8.dtype.element_ty, [BLOCK_M, V], SMEM_FP8)
    gl.amd.gfx1250.tdm.async_load(q_desc, [start_m, 0], q_smem)
    gl.amd.gfx1250.tdm.async_load(h_desc, [0, 0], h_smem)
    gl.amd.gfx1250.tdm.async_load(k_desc, [start_m, 0], k_smem)
    if USE_G:
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=g + start_token * H + head,
            shape=(T, 1),
            strides=(H, 1),
            block_shape=(BLOCK_M, 1),
            layout=SMEM_G,
        )
        g_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, 1], SMEM_G)
        gl.amd.gfx1250.tdm.async_load(g_desc, [start_m, 0], g_smem)
    if GATED_NORM:
        z_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=z + start_token * stride_zm + head * V,
            shape=(T, V),
            strides=(stride_zm, 1),
            block_shape=(BLOCK_M, V),
            layout=SMEM_ROW_K if FAST_NORM else SMEM_ROW_V,
        )
        if not FAST_NORM:
            z_smem = gl.allocate_shared_memory(
                z.type.element_ty, [BLOCK_M, V], SMEM_ROW_V
            )
            gl.amd.gfx1250.tdm.async_load(z_desc, [start_m, 0], z_smem)

    offs_m = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA))
    offs_m_k = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(0, WMMA))
    offs_n = gl.arange(0, V, layout=gl.SliceLayout(0, WMMA))
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    b_o = gl.zeros([BLOCK_M, V], gl.float32, layout=WMMA)
    for offs_k in gl.static_range(0, K, BLOCK_K):
        b_q_h = q_smem.slice(offs_k, BLOCK_K, dim=1).load(OP_A)
        b_h = h_smem.slice(offs_k, BLOCK_K, dim=0).load(OP_B)
        b_o = gl.amd.gfx1250.wmma(b_q_h, b_h, b_o)
    if USE_G:
        b_g = g_smem.reshape([BLOCK_M]).load(gl.SliceLayout(1, WMMA))
        b_o = b_o * gl.exp(b_g)[:, None]
    # Every wave finishes its H read before V reuses that storage.
    gl.barrier()
    gl.amd.gfx1250.tdm.async_load(v_desc, [start_m, 0], v_smem)
    b_q = q_smem.load(OP_A)
    b_a = gl.amd.gfx1250.wmma(
        b_q,
        k_smem.permute([1, 0]).load(OP_B),
        gl.zeros([BLOCK_M, BLOCK_M], gl.float32, layout=WMMA),
    )
    if USE_G:
        b_gc = g_smem.reshape([BLOCK_M]).load(gl.SliceLayout(0, WMMA))
        b_a = b_a * gl.exp(b_g[:, None] - b_gc[None, :])
    num_rows = T - start_m
    mask_a = (
        (offs_m[:, None] >= offs_m_k[None, :])
        & (offs_m[:, None] < num_rows)
        & (offs_m_k[None, :] < num_rows)
    )
    b_a = gl.where(mask_a, b_a, 0.0)
    a_smem.store(b_a.to(a_smem.dtype))
    gl.barrier()
    if GATED_NORM and FAST_NORM:
        # The A barrier releases k's buffer for z while Av computes.
        z_smem = k_smem
        gl.amd.gfx1250.tdm.async_load(z_desc, [start_m, 0], z_smem)
        gl.amd.gfx1250.tdm.async_wait(1)
    else:
        gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    b_pv = gl.amd.gfx1250.wmma(
        a_smem.load(OP_A),
        v_smem.load(OP_B),
        gl.zeros([BLOCK_M, V], gl.float32, layout=WMMA),
    )
    b_o = b_o * scale + b_pv * scale
    if GATED_NORM:
        # Normalize the bf16 output, as in the unfused epilogue.
        x_bf16 = b_o.to(v.dtype.element_ty)
        x = x_bf16.to(gl.float32)
        if FAST_NORM:
            # Keep the complete-row reduction order across the WMMA layout.
            norm_smem = h_smem.slice(BLOCK_M, BLOCK_M, dim=0)
            norm_smem.store(x_bf16)
            gl.barrier()
            x_norm = norm_smem.load(FAST_NORM_LAYOUT).to(gl.float32)
            rstd_norm = gl.rsqrt(gl.sum(x_norm * x_norm, 1) / V + eps)
            if USE_G:
                rstd_smem = g_smem.reshape([BLOCK_M])
                rstd_smem.store(rstd_norm)
                gl.barrier()
                rstd = rstd_smem.load(gl.SliceLayout(1, WMMA))
            else:
                rstd = gl.convert_layout(rstd_norm, gl.SliceLayout(1, WMMA))
            weight = gl.load(norm_weight + offs_n).to(gl.float32)
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()
            b_z = z_smem.load(WMMA).to(gl.float32)
            exp_z = gl.inline_asm_elementwise(
                "v_exp_f32 $0, $1",
                "=v,v",
                [-b_z * 1.4426950408889634],
                dtype=gl.float32,
                is_pure=True,
                pack=1,
            )
            silu = gl.fdiv(b_z, 1.0 + exp_z, approx=True)
        else:
            # Match layer_norm_fwd's sum-of-squares reduction order.
            x_rows = gl.convert_layout(x, NORM_LAYOUT)
            rstd = gl.rsqrt(gl.sum(x_rows * x_rows, 1) / V + eps)
            rstd = gl.convert_layout(rstd, gl.SliceLayout(1, WMMA))
            weight = gl.load(norm_weight + offs_n).to(gl.float32)
            b_z = z_smem.load(WMMA).to(gl.float32)
            silu = b_z * (1.0 / (1.0 + gl.exp(-b_z)))
        b_o = x * rstd[:, None] * weight[None, :] * silu
    if OUT_FP8:
        # mxfp8 per (row, value head): amax of the bf16-rounded row from aiter's kAmaxFloor, capped at 448 * 2^119;
        # e8m0 exponent of ceil_pow2(amax / 448) (RoundUp); values RNE(x * 2^-(e - 127)) to e4m3.
        y = b_o.to(v.dtype.element_ty).to(gl.float32)
        amax = gl.minimum(gl.maximum(gl.max(gl.abs(y), axis=1), 1e-10), 448.0 * 6.646139978924579e35)
        u = (amax * (1.0 / 448.0)).to(gl.uint32, bitcast=True)
        e = ((u >> 23) & 0xFF).to(gl.int32)
        e = e + (((u & 0x7FFFFF) != 0) & (e < 0xFF)).to(gl.int32)
        inv = ((254 - gl.minimum(e, 253)).to(gl.uint32) << 23).to(gl.float32, bitcast=True)
        # The tile goes out as the bf16 one does, through LDS and one TDM store that clips the rows past the
        # sequence end. Its own 9 KiB buffer: a TDM store pads once per row, so Q's bf16 buffer (16 B every 256 B)
        # cannot be reinterpreted for 128-byte fp8 rows.
        o_smem = o8_smem
        o_smem.store((y * inv[:, None]).to(o_fp8.dtype.element_ty))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(o_desc, [start_m, 0], o_smem)
        offs_r = gl.arange(0, BLOCK_M, layout=ROW_LAYOUT)
        e_r = gl.convert_layout(e, ROW_LAYOUT).to(gl.uint8)
        gl.store(o_scale + head * scale_rows + start_token + start_m + offs_r, e_r, mask=offs_r < num_rows)
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        # The A barrier released Q's buffer for the output tile.
        o_smem = q_smem
        o_smem.store(b_o.to(o_smem.dtype))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(o_desc, [start_m, 0], o_smem)
        gl.amd.gfx1250.tdm.async_wait(0)


@functools.cache
def _layouts(K, V, BLOCK_M):
    wmma = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=[[1, 0], [0, 1]],
        instr_shape=[16, 16, 32],
    )
    op_a = gl.DotOperandLayout(operand_index=0, parent=wmma, k_width=8)
    op_b = gl.DotOperandLayout(operand_index=1, parent=wmma, k_width=8)
    smem_row_k = gl.PaddedSharedLayout.with_identity_for([[K, 8]], [BLOCK_M, K], [1, 0])
    smem_row_v = gl.PaddedSharedLayout.with_identity_for(
        [[V, 16]], [BLOCK_M, V], [1, 0]
    )
    smem_h = gl.PaddedSharedLayout.with_identity_for([[V, 16]], [K, V], [1, 0])
    smem_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_M, 8]], [BLOCK_M, BLOCK_M], [1, 0]
    )
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    # Match layer_norm_fwd's layout for bf16 rows.
    norm_layout = gl.BlockedLayout([1, 8], [2, 16], [_NUM_WARPS, 1], [1, 0])
    fast_norm_layout = gl.BlockedLayout([1, 8], [16, 2], [_NUM_WARPS, 1], [0, 1])
    # fp8 output tile ([BLOCK_M, V] e4m3 through Q's buffer) and one-row-per-lane scale vector
    smem_fp8 = gl.PaddedSharedLayout.with_identity_for([[V, 16]], [BLOCK_M, V], [1, 0])
    row_layout = gl.BlockedLayout([1], [32], [_NUM_WARPS], [0])
    return (
        wmma,
        op_a,
        op_b,
        smem_row_k,
        smem_row_v,
        smem_h,
        smem_a,
        smem_g,
        norm_layout,
        fast_norm_layout,
        smem_fp8,
        row_layout,
    )


def chunk_o_gfx1250(
    q,
    k,
    v,
    h,
    g,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    nt_ptr,
    NT,
    T,
    B,
    H,
    Hg,
    K,
    V,
    BLOCK_M,
    gate,
    out_fp8=None,
):
    """Write and return chunked attention output in the contiguous bf16 buffer o.

    Q/K have Hg heads; V/H/O have H heads and K=V=128, BLOCK_M=64.
    gate is None or (z, weight[V], eps); z has contiguous [H, V] rows.
    With a gate, write RMSNorm(bf16(o)) * weight * SiLU(z).

    out_fp8: None (default: the bf16 output above), or (o_fp8, o_scale) from mxfp8_act_quant.alloc_mxfp8(rows,
    H * V) with rows >= the B * T token rows written here (rows = the out_proj GEMM's M; extra rows are padding).
    Then o is not written (may be None); the bf16-rounded result goes to o_fp8 [rows, H * V] fp8 e4m3 and its
    e8m0 scales to o_scale at head * rows + token, and (o_fp8, mxfp8_act_quant.scale_view(o_scale, rows, H * V))
    is returned. Padding rows are left alone: fill them with fill_mxfp8_padding.
    """
    layouts = _layouts(K, V, BLOCK_M)
    if gate is not None:
        z, weight, eps = gate
        z2 = z.reshape(-1, *z.shape[-2:]) if z.dim() > 3 else z
        assert (
            z2.stride(-1) == 1 and z2.stride(-2) == V
        ), "z rows must be [H, V] contiguous"
        stride_zm = z2.stride(0)
    else:
        z = weight = None
        eps = 0.0
        stride_zm = 0
    if out_fp8 is not None:
        o_fp8, o_scale = out_fp8
        rows = o_fp8.shape[0]
        assert gate is not None, "the fp8 output is the gated-norm (out_proj input) epilogue"
        assert o_fp8.dtype == torch.float8_e4m3fn and o_fp8.is_contiguous() and o_fp8.shape[1] == H * V
        assert o_scale.dtype == torch.uint8 and o_scale.numel() == H * rows and rows >= B * T
    else:
        o_fp8 = o_scale = None
        rows = 0
    _chunk_o_gfx1250_kernel[(B * H, NT)](
        q,
        k,
        v,
        h,
        g,
        o,
        z,
        weight,
        o_fp8,
        o_scale,
        rows,
        cu_seqlens,
        chunk_indices,
        nt_ptr,
        scale,
        eps,
        T,
        stride_zm,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BLOCK_M=BLOCK_M,
        BLOCK_K=_BLOCK_K,
        USE_G=g is not None,
        GATED_NORM=gate is not None,
        IS_VARLEN=cu_seqlens is not None,
        HAS_NT=nt_ptr is not None,
        FAST_NORM=envs.ATOM_GDN_CHUNK_O_FAST_NORM,
        WMMA=layouts[0],
        OP_A=layouts[1],
        OP_B=layouts[2],
        SMEM_ROW_K=layouts[3],
        SMEM_ROW_V=layouts[4],
        SMEM_H=layouts[5],
        SMEM_A=layouts[6],
        SMEM_G=layouts[7],
        NORM_LAYOUT=layouts[8],
        FAST_NORM_LAYOUT=layouts[9],
        OUT_FP8=out_fp8 is not None,
        SMEM_FP8=layouts[10],
        ROW_LAYOUT=layouts[11],
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )
    if out_fp8 is not None:
        from atom.model_ops.mxfp8_act_quant import scale_view

        return o_fp8, scale_view(o_scale, rows, H * V)
    return o


def fill_mxfp8_padding(o_fp8, o_scale, num_actual_tokens):
    """The fp8-mode counterpart of `core_attn_out[num_actual_tokens:].zero_()`: rows past num_actual_tokens become
    what the unfused quant makes of zero rows -- fp8 0 and e8m0 byte 85 (aiter's kAmaxFloor 1e-10 -> 2^-42)."""
    rows, cols = o_fp8.shape
    if num_actual_tokens >= rows:
        return
    o_fp8[num_actual_tokens:].view(torch.uint8).zero_()
    o_scale.view(cols // 128, rows)[:, num_actual_tokens:].fill_(85)
