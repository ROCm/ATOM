# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chunked Gated DeltaNet output with a fused gated RMSNorm on gfx1250.

Each program computes one chunk of one value head:
    o = scale * (exp(g) * (q @ h) + tril(exp(g_i - g_j) * (q @ k.T)) @ v)
The gated epilogue computes RMSNorm(bf16(o)) * weight * SiLU(z).
OUT_FP8 writes the bf16-rounded output as fp8 e4m3 with one e8m0 scale per (row, value head) instead, bit-identical
to the bf16 output followed by mxfp8_act_quant.reference.
"""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language import core

from atom.model_ops.mxfp8_act_quant import e8m0_group_scale
from atom.utils import envs

_BLOCK_K = 32
_NUM_WARPS = 4
_WAVES_PER_EU = 4


@core.extern
def _rcp(x, _semantic=None):
    # v_rcp_f32 alone, without the range scaling of the fast division.
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.dtype("fp32"),): ("llvm.amdgcn.rcp.f32", core.dtype("fp32"))},
        is_pure=True,
        _semantic=_semantic,
    )


@core.extern
def _exp2(x, _semantic=None):
    # v_exp_f32 alone, without the denormal range reduction of exp2.
    return core.extern_elementwise(
        "",
        "",
        [x],
        {(core.dtype("fp32"),): ("llvm.amdgcn.exp2.f32", core.dtype("fp32"))},
        is_pure=True,
        _semantic=_semantic,
    )


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
    OUT_FP8: gl.constexpr,
    XCD_PAIRS: gl.constexpr,
):
    WMMA: gl.constexpr = _layout("WMMA", K, V, BLOCK_M)
    OP_A: gl.constexpr = _layout("OP_A", K, V, BLOCK_M)
    OP_B: gl.constexpr = _layout("OP_B", K, V, BLOCK_M)
    SMEM_ROW_K: gl.constexpr = _layout("SMEM_ROW_K", K, V, BLOCK_M)
    SMEM_ROW_V: gl.constexpr = _layout("SMEM_ROW_V", K, V, BLOCK_M)
    SMEM_H: gl.constexpr = _layout("SMEM_H", K, V, BLOCK_M)
    SMEM_A: gl.constexpr = _layout("SMEM_A", K, V, BLOCK_M)
    SMEM_G: gl.constexpr = _layout("SMEM_G", K, V, BLOCK_M)
    NORM_LAYOUT: gl.constexpr = _layout("NORM_LAYOUT", K, V, BLOCK_M)
    FAST_NORM_LAYOUT: gl.constexpr = _layout("FAST_NORM_LAYOUT", K, V, BLOCK_M)
    SMEM_FP8: gl.constexpr = _layout("SMEM_FP8", K, V, BLOCK_M)
    ROW_LAYOUT: gl.constexpr = _layout("ROW_LAYOUT", K, V, BLOCK_M)
    # Adjacent value heads reuse the same Q and K tiles.
    pid_bh = gl.program_id(0)
    pid_m = gl.program_id(1)
    batch = pid_bh // H
    head = pid_bh % H
    if XCD_PAIRS:
        # Workgroups go round-robin over 8 XCDs: the two value heads of a k head go 8 workgroups apart, one XCD.
        wg = pid_m * H + pid_bh
        pair = (wg // 16) * 8 + wg % 8
        pid_m = pair // Hg
        head = (pair % Hg) * 2 + (wg // 8) % 2
    if IS_VARLEN and HAS_NT:  # noqa: SIM102 (a constexpr, then a device load)
        if pid_m >= gl.load(nt_ptr):
            return
    if IS_VARLEN:
        chunk = pid_m
    else:
        chunk = batch * gl.cdiv(T, BLOCK_M) + pid_m
    # H depends only on the chunk: its load runs under the sequence lookups.
    # H is K-major [V, K] per (chunk, head).
    h_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=h + (chunk.to(gl.int64) * H + head) * K * V,
        shape=(V, K),
        strides=(K, 1),
        block_shape=(V, K),
        layout=SMEM_H,
    )
    h_smem = gl.allocate_shared_memory(h.type.element_ty, [V, K], SMEM_H)
    gl.amd.gfx1250.tdm.async_load(h_desc, [0, 0], h_smem)
    if GATED_NORM and FAST_NORM:
        # The norm weight goes to LDS under the tile loads.
        w_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=norm_weight,
            shape=(1, V),
            strides=(V, 1),
            block_shape=(1, V),
            layout=SMEM_G,
        )
        w_smem = gl.allocate_shared_memory(norm_weight.type.element_ty, [1, V], SMEM_G)
        gl.amd.gfx1250.tdm.async_load(w_desc, [0, 0], w_smem)
    if IS_VARLEN:
        sequence = gl.load(chunk_indices + pid_m * 2).to(gl.int32)
        pid_m = gl.load(chunk_indices + pid_m * 2 + 1).to(gl.int32)
        bos = gl.load(cu_seqlens + sequence).to(gl.int32)
        eos = gl.load(cu_seqlens + sequence + 1).to(gl.int32)
        T = eos - bos
    else:
        bos = batch * T
    start_m = pid_m * BLOCK_M
    start_token = bos.to(gl.int64)
    offs_qk = (start_token * Hg + head // (H // Hg)) * K
    offs_vo = (start_token * H + head) * V

    q_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=q + offs_qk,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_ROW_K,
    )
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k + offs_qk,
        shape=(T, K),
        strides=(Hg * K, 1),
        block_shape=(BLOCK_M, K),
        layout=SMEM_ROW_K,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v + offs_vo,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, V),
        layout=SMEM_ROW_V,
    )
    o_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=(o_fp8 if OUT_FP8 else o) + offs_vo,
        shape=(T, V),
        strides=(H * V, 1),
        block_shape=(BLOCK_M, V),
        layout=SMEM_FP8 if OUT_FP8 else SMEM_ROW_K,
    )
    q_smem = gl.allocate_shared_memory(q.type.element_ty, [BLOCK_M, K], SMEM_ROW_K)
    k_smem = gl.allocate_shared_memory(k.type.element_ty, [BLOCK_M, K], SMEM_ROW_K)
    v_smem = h_smem.slice(0, BLOCK_M, dim=0)
    a_smem = gl.allocate_shared_memory(v.type.element_ty, [BLOCK_M, BLOCK_M], SMEM_A)
    gl.amd.gfx1250.tdm.async_load(q_desc, [start_m, 0], q_smem)
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
        b_h = h_smem.slice(offs_k, BLOCK_K, dim=1).permute([1, 0]).load(OP_B)
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
    # Rows past the sequence end are not stored, and columns past it lie above the diagonal of every stored row.
    b_a = gl.where(offs_m[:, None] >= offs_m_k[None, :], b_a, 0.0)
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
            weight = w_smem.reshape([V]).load(gl.SliceLayout(0, WMMA)).to(gl.float32)
            gl.amd.gfx1250.tdm.async_wait(0)
            gl.barrier()
            b_z = z_smem.load(WMMA).to(gl.float32)
            exp_z = _exp2(b_z * -1.4426950408889634)
            silu = b_z * _rcp(1.0 + exp_z)
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
        # One e8m0 quant group per (row, value head) of the bf16-rounded output.
        y = b_o.to(v.dtype.element_ty).to(gl.float32)
        amax = gl.minimum(
            gl.maximum(gl.max(gl.abs(y), axis=1), 1e-10), 448.0 * 6.646139978924579e35
        )
        e, inv = e8m0_group_scale(amax)
        # Convert the scales first: no LDS scratch may reuse the fp8 tile while the TDM store reads it.
        offs_r = gl.arange(0, BLOCK_M, layout=ROW_LAYOUT)
        e_r = gl.convert_layout(e, ROW_LAYOUT).to(gl.uint8)
        # The fp8 tile (Q's tile is padded for bf16 rows) takes the LDS of the retired tiles.
        o8_smem = gl.allocate_shared_memory(
            o_fp8.dtype.element_ty, [BLOCK_M, V], SMEM_FP8
        )
        o8_smem.store((y * inv[:, None]).to(o_fp8.dtype.element_ty))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(o_desc, [start_m, 0], o8_smem)
        # Scales are transposed: the byte of (token, head) is at head * scale_rows + token.
        gl.store(
            o_scale + head * scale_rows + start_token + start_m + offs_r,
            e_r,
            mask=offs_r < T - start_m,
        )
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        # The A barrier released Q's buffer for the output tile.
        o_smem = q_smem
        o_smem.store(b_o.to(o_smem.dtype))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(o_desc, [start_m, 0], o_smem)
        gl.amd.gfx1250.tdm.async_wait(0)


@gluon.constexpr_function
def _layout(name, K, V, BLOCK_M):
    """Layout `name` of the kernel (WMMA, its operands, LDS tiles, norm rows, fp8 output)."""
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
    smem_h = gl.PaddedSharedLayout.with_identity_for([[K, 16]], [V, K], [1, 0])
    smem_a = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_M, 8]], [BLOCK_M, BLOCK_M], [1, 0]
    )
    smem_g = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0])
    # Match layer_norm_fwd's layout for bf16 rows.
    norm_layout = gl.BlockedLayout([1, 8], [2, 16], [_NUM_WARPS, 1], [1, 0])
    fast_norm_layout = gl.BlockedLayout([1, 8], [16, 2], [_NUM_WARPS, 1], [0, 1])
    # fp8 output tile and one scale per lane
    smem_fp8 = gl.PaddedSharedLayout.with_identity_for([[V, 16]], [BLOCK_M, V], [1, 0])
    row_layout = gl.BlockedLayout([1], [32], [_NUM_WARPS], [0])
    return {
        "WMMA": wmma,
        "OP_A": op_a,
        "OP_B": op_b,
        "SMEM_ROW_K": smem_row_k,
        "SMEM_ROW_V": smem_row_v,
        "SMEM_H": smem_h,
        "SMEM_A": smem_a,
        "SMEM_G": smem_g,
        "NORM_LAYOUT": norm_layout,
        "FAST_NORM_LAYOUT": fast_norm_layout,
        "SMEM_FP8": smem_fp8,
        "ROW_LAYOUT": row_layout,
    }[name]


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
    out_fp8,
):
    """Write the chunked attention output into the contiguous bf16 buffer o.

    Q/K have Hg heads; V/H/O have H heads and K=V=128, BLOCK_M=64.
    h holds K-major chunk states: a [.., K, V] view of [.., V, K] storage.
    gate is None or (z, weight[V], eps); z has contiguous [H, V] rows.
    With a gate, write RMSNorm(bf16(o)) * weight * SiLU(z).
    out_fp8 is None or (o_fp8, o_scale) from mxfp8_act_quant.alloc_mxfp8(rows, H * V), rows >= B * T: then o is not
    written; o_fp8 gets the fp8 rows and o_scale the e8m0 byte of (token, head) at head * rows + token. Rows past
    B * T are not written (fill_mxfp8_padding).
    """
    assert h.stride(-2) == 1, "h must be K-major (chunk_h_gfx1250's layout)"
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
        assert gate is not None, "the fp8 output needs the gated norm"
        assert (
            o_fp8.dtype == torch.float8_e4m3fn
            and o_fp8.is_contiguous()
            and o_fp8.shape[1] == H * V
        )
        assert (
            o_scale.dtype == torch.uint8
            and o_scale.numel() == H * rows
            and rows >= B * T
        )
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
        OUT_FP8=out_fp8 is not None,
        XCD_PAIRS=B == 1 and H == 2 * Hg and Hg % 8 == 0,
        num_warps=_NUM_WARPS,
        waves_per_eu=_WAVES_PER_EU,
    )


def fill_mxfp8_padding(o_fp8, o_scale, num_actual_tokens):
    """The fp8-mode counterpart of `core_attn_out[num_actual_tokens:].zero_()`: rows past num_actual_tokens become
    what the unfused quant makes of zero rows -- fp8 0 and e8m0 byte 85 (aiter's kAmaxFloor 1e-10 -> 2^-42).
    """
    rows, cols = o_fp8.shape
    if num_actual_tokens >= rows:
        return
    o_fp8[num_actual_tokens:].view(torch.uint8).zero_()
    o_scale.view(cols // 128, rows)[:, num_actual_tokens:].fill_(85)
