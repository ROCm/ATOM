# SPDX-License-Identifier: MIT
"""Ops of `model_options.qpath_fuse` (`atom.models.deepseek_v41_plugin.qpath_fuse`).

`dual_rmsnorm_mxfp8_kv_window` is AITER's fused q / kv RMSNorm launch with
the KV row's half of `rope_quant_window` appended to each row's program: the
normed KV row, rounded to its dtype exactly where the norm stores it, is
rotated, group-32 MXFP8-quantized and stored into the decode window ring by
the statements `_rope_window_kernel` runs for it. A decode step then has no
KV work left for the RoPE / window launch.

Bit-identity rests on two things, both checked on the compiled kernels
rather than assumed: the norm half is the AITER kernel's statements over the
same tiles and warps, so its reductions associate as they do there; and the
KV half reads the same BF16 values the standalone kernel loads and rotates
them with the rounding of the standalone kernel's compilation, written out
(`fuse_cos`), because that kernel's own expression compiles to one of two
roundings depending on the table dtype and its program layout.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp8_quant_op

from atom.model_ops.blockscale_kernels.quantization import fp8_groups
from atom.model_ops.v4_kernels.pool_index import window_row

_QUANT_BLOCK_SIZE = 32


@triton.jit
def _dual_rmsnorm_mxfp8_kv_window_kernel(
    q_ptr,
    k_ptr,
    gq_ptr,
    gk_ptr,
    yq_ptr,
    sq_ptr,
    yk_ptr,
    M,
    KQ,
    KK,
    stride_qm,
    stride_qn,
    stride_km,
    stride_kn,
    stride_yqm,
    stride_yqn,
    stride_sqm,
    stride_sqn,
    stride_ykm,
    stride_ykn,
    eps_q,
    eps_k,
    cos,
    sin,
    positions,
    pool,
    pool_scale,
    batches,
    cu,
    slots,
    ring_start,
    BLOCK_SIZE_KQ: tl.constexpr,  # power-of-2 covering full KQ
    BLOCK_SIZE_KK: tl.constexpr,  # == KK: the KV row is rotated whole
    QUANT_BLOCK_SIZE: tl.constexpr,  # =32 (MXFP8 group size)
    NUM_PRGMS,  # row-loop stride; runtime, as in the AITER kernel
    ROPE: tl.constexpr,
    RING_SLOTS: tl.constexpr,
    SLOT_ROWS: tl.constexpr,
    RING_STRIDE: tl.constexpr,
    RUN_ROWS: tl.constexpr,
    KV_FUSE_COS: tl.constexpr,  # the KV rotation's form, see below
    AMAX_FLOOR: tl.constexpr = 0.0,  # Q groups' amax floor (none by default)
):
    """One program per row. The first two sections are
    `_fused_dual_rmsnorm_mxfp8_quant_kernel`'s, statement for statement; the
    third is `_rope_window_kernel`'s KV program (`_quantized_row` and the
    planar ring store) on the row the second just stored."""
    row_start = tl.program_id(0)

    q_col_offsets = tl.arange(0, BLOCK_SIZE_KQ)
    q_mask = q_col_offsets < KQ
    k_col_offsets = tl.arange(0, BLOCK_SIZE_KK)
    k_mask = k_col_offsets < KK

    n_q_groups: tl.constexpr = BLOCK_SIZE_KQ // QUANT_BLOCK_SIZE
    D: tl.constexpr = BLOCK_SIZE_KK
    half: tl.constexpr = ROPE // 2

    for row_idx in tl.range(row_start, M, NUM_PRGMS, num_stages=2):
        # ===== Q side: RMSNorm + MXFP8 quant =====
        x_q = tl.load(
            q_ptr + row_idx * stride_qm + q_col_offsets * stride_qn,
            mask=q_mask,
            other=0.0,
        ).to(tl.float32)
        g_q = tl.load(gq_ptr + q_col_offsets, mask=q_mask, other=0.0).to(tl.float32)

        ss_q = tl.sum(x_q * x_q, axis=-1)
        norm_q = tl.math.rsqrt((ss_q / KQ) + eps_q)
        y_q_fp32 = x_q * norm_q * g_q

        y_q_2d = tl.reshape(y_q_fp32, (n_q_groups, QUANT_BLOCK_SIZE))
        scale_q_e8m0, quant_scale_q = _mxfp8_quant_op(
            y_q_2d, QUANT_AXIS=1, AMAX_FLOOR=AMAX_FLOOR
        )

        qx_q_2d = y_q_2d * quant_scale_q
        qx_q = tl.reshape(qx_q_2d, (BLOCK_SIZE_KQ,))
        y_q_fp8 = qx_q.to(yq_ptr.type.element_ty)

        tl.store(
            yq_ptr + row_idx * stride_yqm + q_col_offsets * stride_yqn,
            y_q_fp8,
            mask=q_mask,
        )

        q_group_offsets = tl.arange(0, n_q_groups)
        q_group_mask = q_group_offsets < (KQ // QUANT_BLOCK_SIZE)
        scale_q_flat = tl.reshape(scale_q_e8m0, (n_q_groups,))
        tl.store(
            sq_ptr + row_idx * stride_sqm + q_group_offsets * stride_sqn,
            scale_q_flat,
            mask=q_group_mask,
        )

        # ===== K side: RMSNorm only, bf16 out =====
        x_k = tl.load(
            k_ptr + row_idx * stride_km + k_col_offsets * stride_kn,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        g_k = tl.load(gk_ptr + k_col_offsets, mask=k_mask, other=0.0).to(tl.float32)

        ss_k = tl.sum(x_k * x_k, axis=-1)
        norm_k = tl.math.rsqrt((ss_k / KK) + eps_k)
        y_k_fp32 = x_k * norm_k * g_k
        y_k_out = y_k_fp32.to(yk_ptr.type.element_ty)

        tl.store(
            yk_ptr + row_idx * stride_ykm + k_col_offsets * stride_ykn,
            y_k_out,
            mask=k_mask,
        )

        # ===== KV row: RoPE, MXFP8 quant, window ring =====
        # `normed` is what the standalone kernel loads back: the stored row.
        activation = yk_ptr.type.element_ty
        position = tl.load(positions + row_idx)
        normed = y_k_out.to(tl.float32)

        even, odd = tl.split(tl.reshape(normed, (D // 2, 2)))
        tl.static_assert((D - ROPE) % ROPE == 0)
        pair = tl.arange(0, D // 2)
        rope_pair = pair >= (D - ROPE) // 2
        cosine = tl.where(
            rope_pair,
            tl.load(cos + position * half + pair % half).to(tl.float32),
            1.0,
        )
        sine = tl.where(
            rope_pair,
            tl.load(sin + position * half + pair % half).to(tl.float32),
            0.0,
        )
        # `even * cosine - odd * sine` and `even * sine + odd * cosine`, with
        # the rounding spelled out. The standalone kernel leaves the choice
        # of the rounded and the fused product of each sum to the compiler,
        # which decides it differently per compilation (table dtype, heads a
        # program); here it is the caller's: the form of the compilation this
        # launch stands in for. Explicit fma with an already rounded addend
        # cannot be contracted another way.
        rot_even = tl.fma(cosine, even, -(sine * odd))
        if KV_FUSE_COS:
            rot_odd = tl.fma(cosine, odd, sine * even)
        else:
            rot_odd = tl.fma(sine, even, cosine * odd)
        groups = (
            tl.reshape(tl.join(rot_even, rot_odd), (D // 32, 32))
            .to(activation)
            .to(tl.float32)
        )
        quantized, code, _ = fp8_groups(groups)

        batch = tl.load(batches + row_idx)
        if batch >= 0 and row_idx >= tl.load(cu + batch + 1) - RING_SLOTS:
            row = window_row(
                tl.load(slots + batch).to(tl.int64),
                position,
                ring_start,
                RING_SLOTS,
                SLOT_ROWS,
                RING_STRIDE,
                RUN_ROWS,
            )
            dim = tl.arange(0, D)
            tl.store(
                pool + row * D + dim,
                tl.reshape(quantized, (D,)).to(tl.uint8, bitcast=True),
            )
            tl.store(pool_scale + row * (D // 32) + tl.arange(0, D // 32), code)


def dual_rmsnorm_mxfp8_kv_window(
    q,
    k,
    q_weight,
    k_weight,
    eps_q,
    eps_k,
    amax_floor,
    cos,
    sin,
    positions,
    ring,
    *,
    rope_dim,
    fuse_cos=False,
):
    """`fused_dual_rmsnorm_mxfp8_quant(q, k, ...)` -- returns its `(yq, sq,
    yk)` -- that also runs each normed KV row through the KV half of
    `rope_quant_window(..., ring=ring)`: rotated at `positions`, quantized
    and stored into the decode window.

    `q` `[M, KQ]`, `k` `[M, KK]` (KK a power of two); `ring` is
    `rope_quant_window`'s `(pool, step, window)` with a planar `(data,
    scale)` pool of contiguous uint8 planes. `cos` / `sin`: the tables
    `rope_quant_window` would read, fp32 or bf16 (widened as it widens them).

    `fuse_cos`: how the KV rotation rounds. False: `even' = fma(cos, even,
    -(sin * odd))`, `odd' = fma(sin, even, cos * odd)`; True: the same `even'`
    and `odd' = fma(cos, odd, sin * even)`. Which of the two the standalone
    kernel computes depends on its compilation; `qpath_fuse` knows.
    """
    M, KQ = q.shape
    Mk, KK = k.shape
    (pool, pool_scale), step, window = ring
    if M != Mk or KQ % _QUANT_BLOCK_SIZE or KK & (KK - 1) or KK % _QUANT_BLOCK_SIZE:
        raise ValueError(f"unsupported q / kv shapes {tuple(q.shape)} {tuple(k.shape)}")
    if pool.dtype != torch.uint8 or pool_scale.dtype != torch.uint8:
        raise ValueError("the window ring must be separate uint8 (data, scale) planes")
    yq = torch.empty((M, KQ), dtype=torch.float8_e4m3fn, device=q.device)
    sq = torch.empty((M, KQ // _QUANT_BLOCK_SIZE), dtype=torch.uint8, device=q.device)
    yk = torch.empty((M, KK), dtype=k.dtype, device=k.device)
    _dual_rmsnorm_mxfp8_kv_window_kernel[(M,)](
        q,
        k,
        q_weight,
        k_weight,
        yq,
        sq,
        yk,
        M,
        KQ,
        KK,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        yq.stride(0),
        yq.stride(1),
        sq.stride(0),
        sq.stride(1),
        yk.stride(0),
        yk.stride(1),
        eps_q,
        eps_k,
        cos,
        sin,
        positions,
        pool,
        pool_scale,
        step.batch_ids,
        step.cu_seqlens_q,
        step.slots,
        window.ring_start,
        BLOCK_SIZE_KQ=triton.next_power_of_2(KQ),
        BLOCK_SIZE_KK=KK,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=M,
        ROPE=rope_dim,
        RING_SLOTS=window.ring_slots,
        SLOT_ROWS=window.slot_rows,
        RING_STRIDE=window.ring_stride,
        RUN_ROWS=window.run_rows,
        KV_FUSE_COS=bool(fuse_cos),
        AMAX_FLOOR=amax_floor,
    )
    return yq, sq, yk
