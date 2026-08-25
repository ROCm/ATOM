# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# This file contains code adapted from the flash-linear-attention project
# (fla/ops/attnres/fused.py). The original source code was licensed under the
# MIT license and included the following copyright notice:
# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Fused attention-residual operations for Kimi-K3.

The algorithm is flash-linear-attention's ``fused_attnres``
(``fla/ops/attnres/fused.py``, MIT; read against fla 0.5.2), which is what the
reference KDA model calls -- see ``fla/models/kda/modeling_kda.py:135``.
Attention Residuals: https://arxiv.org/abs/2603.15031

Four deliberate divergences from that reference:

* ``residuals`` is a Sequence of separate ``[..., D]`` tensors there; here it is
  one packed ``[T, B, H]`` block_residual plus ``prefix_sum`` read as the final
  candidate. Not just cheaper -- their pointer-table gather cannot run on ROCm
  at all (details at the ``Adapted from FLA`` note in the kernel body).
* the caller's ``prefix_sum = prefix_sum + ...`` adds are folded into the last
  candidate's on-load (``DO_ADD``/``DO_ADD2``), which fla leaves to the caller.
  That fold is what lets the decoder layers defer their FFN and routed/shared
  expert adds across the layer boundary.
* ``score_weight`` arrives pre-multiplied: fla passes ``query`` and
  ``rms_weight`` separately, while ``AttnRes`` folds their product once at load
  time (see ``AttnRes.process_weights_after_loading``).
* forward only -- ATOM is inference-only, so there is no bwd and no
  ``checkpoint_level`` counterpart.
"""

from __future__ import annotations

import torch

from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _attn_res_fused_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        B,
        Bp,
        H,
        eps,
        out_eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_hs2_t,
        stride_pref_t,
        BL: tl.constexpr,  # candidates per tile
        BD: tl.constexpr,  # next_pow2(H) -- one tile spans all of H
        DO_ADD: tl.constexpr,  # fold prefix += add_hidden on-load
        DO_ADD2: tl.constexpr,  # fold a second addend (shared-expert output)
        WRITE_PREF: tl.constexpr,  # write the (summed) prefix back to pref_ptr
        OUT_NORM: tl.constexpr,  # fold the caller's output rmsnorm into the store
        PEEL: tl.constexpr = False,  # prefill form: peel the prefix candidate out of the loop
        CG: tl.constexpr = False,  # .cg on the block_residual stream (prefill only)
        CG2: tl.constexpr = False,  # .cg on the prefix / addend streams
    ):
        # One program per row t: rmsnorm each of the Bp = B+1 candidates, score =
        # <normed, score_weight>, softmax over Bp, then weighted sum -> y[t].
        # Candidates 0..B-1 are block_residual rows; candidate B is prefix_sum.
        # Read both source tensors directly (no torch.cat materialization).
        #
        # SINGLE PASS. The tile spans all of H and the running output stays in
        # registers, so the softmax runs online (flash-style): each new candidate
        # tile rescales the accumulator by exp(m_prev - m_new) instead of waiting
        # for a completed reduction over the candidate axis. Every candidate is
        # therefore read exactly ONCE.
        #
        # The tiling axis is what makes that work: tiling over CANDIDATES (not
        # over H, as an earlier version did) is what lets the whole output live in
        # registers. Tiling over H forces two passes -- probs aren't known until
        # the H-reduction completes, so the combine has to re-read everything --
        # and forces a third to fold OUT_NORM, since sum_h y_h^2 needs a formed y.
        # Here y is already formed in registers when the loop ends, so OUT_NORM is
        # free, and the token-count gate that the H-tiled version needed (the fold
        # stopped paying once a row's [Bp, H] reload spilled L2) is gone with it.
        #
        # Cost is registers: BD = next_pow2(H) floats of accumulator, 32 KB of VGPR
        # at H=7168, plus the [BL, BD] tile. BL is kept small for that reason.
        #
        # Adapted from FLA's fused_attnres (see the module docstring). Theirs
        # gathers from a tuple of separate residual tensors via a pointer table;
        # ours indexes one contiguous [T, B, H] block_residual, which is both
        # cheaper and necessary here -- the pointer-table form miscompiles on
        # ROCm (TritonAMDGPUCanonicalizePointers rejects arith.select on
        # tensor<Nx!tt.ptr>), so their kernel cannot run on this backend at all.
        #
        # DO_ADD folds the caller's ``prefix_sum = prefix_sum + add_hidden``
        # elementwise add into the last-candidate on-load (saving a separate
        # kernel launch + HBM round-trip); WRITE_PREF then stores that summed
        # prefix so downstream layers reuse it. DO_ADD2 folds a SECOND addend the
        # same way, so an MoE layer can hand over its routed and shared expert
        # outputs unsummed and skip an entire [T, H] elementwise kernel.
        t = tl.program_id(0)
        o_d = tl.arange(0, BD)
        m_d = o_d < H
        sw = tl.load(sw_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

        if PEEL:
            # ---- PREFILL form (T > 512). Same math, same single pass, same
            # byte count -- every candidate is still read exactly once -- but the
            # prefix candidate is PEELED out of the candidate loop instead of
            # being carried in registers across every iteration and selected in
            # with tl.where(is_last, ...).  Two things fall out:
            #   * `ps` (a [BD] fp32 array = BD/threads VGPR/thread) is dead the
            #     moment it seeds the accumulator, so it does not co-exist with
            #     the [BL, BD] tile.  At num_warps=4 that is 32 VGPR/thread back,
            #     which is what takes the prefill body off the 1-wave/SIMD
            #     occupancy cliff (n_regs 258 -> see the report).
            #   * the loop now runs over the B block_residual rows only, so the
            #     tile-raggedness that made odd Bp collapse keys on B, and BL=1
            #     degenerates to a flat 1-D row load with no candidate-axis mask
            #     and no select at all.
            # Prefill is bandwidth-bound (3 GB working set vs 4 MB L2), so the
            # single-pass property is non-negotiable: nothing here re-reads.
            pp = tl.multiple_of(ps_ptr + t * stride_ps_t + o_d, (16,))
            if CG2:
                ps = tl.load(pp, mask=m_d, other=0.0, cache_modifier=".cg").to(tl.float32)
            else:
                ps = tl.load(pp, mask=m_d, other=0.0).to(tl.float32)
            if DO_ADD:
                hp = tl.multiple_of(hs_ptr + t * stride_hs_t + o_d, (16,))
                if CG2:
                    ps += tl.load(hp, mask=m_d, other=0.0, cache_modifier=".cg").to(tl.float32)
                else:
                    ps += tl.load(hp, mask=m_d, other=0.0).to(tl.float32)
            if DO_ADD2:
                h2p = tl.multiple_of(hs2_ptr + t * stride_hs2_t + o_d, (16,))
                if CG2:
                    ps += tl.load(h2p, mask=m_d, other=0.0, cache_modifier=".cg").to(tl.float32)
                else:
                    ps += tl.load(h2p, mask=m_d, other=0.0).to(tl.float32)
            if WRITE_PREF:
                tl.store(
                    tl.multiple_of(pref_ptr + t * stride_pref_t + o_d, (16,)),
                    ps.to(pref_ptr.dtype.element_ty),
                    mask=m_d,
                )

            # seed the online softmax with the prefix candidate, then drop `ps`
            b_m = tl.sum(ps * sw, axis=0) * tl.rsqrt(tl.sum(ps * ps, axis=0) / H + eps)
            b_acc = tl.full([], 1.0, tl.float32)
            b_o = ps

            if BL == 1:
                for i_l in range(B):
                    bp = tl.multiple_of(
                        br_ptr + t * stride_br_t + i_l * stride_br_b + o_d, (16,)
                    )
                    if CG:
                        v = tl.load(bp, mask=m_d, other=0.0, cache_modifier=".cg").to(tl.float32)
                    else:
                        v = tl.load(bp, mask=m_d, other=0.0).to(tl.float32)
                    s = tl.sum(v * sw, axis=0) * tl.rsqrt(tl.sum(v * v, axis=0) / H + eps)
                    b_mp = b_m
                    b_m = tl.maximum(b_m, s)
                    r = tl.exp(b_mp - b_m)
                    p = tl.exp(s - b_m)
                    b_acc = b_acc * r + p
                    b_o = b_o * r + p * v
            else:
                for i_l in range(tl.cdiv(B, BL)):
                    o_l = i_l * BL + tl.arange(0, BL)
                    m_l = o_l < B
                    bp = tl.multiple_of(
                        br_ptr
                        + t * stride_br_t
                        + o_l[:, None] * stride_br_b
                        + o_d[None, :],
                        (1, 16),
                    )
                    msk = m_l[:, None] & m_d[None, :]
                    if CG:
                        v = tl.load(bp, mask=msk, other=0.0, cache_modifier=".cg").to(tl.float32)
                    else:
                        v = tl.load(bp, mask=msk, other=0.0).to(tl.float32)
                    rstd = tl.rsqrt(tl.sum(v * v, axis=1) / H + eps)
                    s = tl.where(m_l, tl.sum(v * sw[None, :], axis=1) * rstd, float("-inf"))
                    b_m, b_mp = tl.maximum(b_m, tl.max(s, axis=0)), b_m
                    r = tl.exp(b_mp - b_m)
                    p = tl.exp(s - b_m)
                    b_acc = b_acc * r + tl.sum(p, axis=0)
                    b_o = b_o * r + tl.sum(p[:, None] * v, axis=0)
        else:
            # prefix (the last candidate) is loaded once and reused across tiles;
            # re-reading it per tile would undo the single-pass property.
            ps = tl.load(ps_ptr + t * stride_ps_t + o_d, mask=m_d, other=0.0).to(tl.float32)
            if DO_ADD:
                ps += tl.load(hs_ptr + t * stride_hs_t + o_d, mask=m_d, other=0.0).to(
                    tl.float32
                )
            if DO_ADD2:
                ps += tl.load(hs2_ptr + t * stride_hs2_t + o_d, mask=m_d, other=0.0).to(
                    tl.float32
                )
            if WRITE_PREF:
                tl.store(
                    pref_ptr + t * stride_pref_t + o_d,
                    ps.to(pref_ptr.dtype.element_ty),
                    mask=m_d,
                )

            b_m = tl.full([], float("-inf"), dtype=tl.float32)  # running max
            b_acc = tl.zeros([], dtype=tl.float32)  # running softmax denominator
            b_o = tl.zeros([BD], dtype=tl.float32)  # running weighted sum

            for i_l in range(tl.cdiv(Bp, BL)):
                o_l = i_l * BL + tl.arange(0, BL)
                m_l = o_l < Bp
                is_last = o_l == B
                v = tl.load(
                    br_ptr + t * stride_br_t + o_l[:, None] * stride_br_b + o_d[None, :],
                    mask=(o_l < B)[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                v = tl.where(is_last[:, None], ps[None, :], v)

                # score_weight = norm_weight * proj_weight, precomputed at load time
                rstd = tl.rsqrt(tl.sum(v * v, axis=1) / H + eps)
                s = tl.where(m_l, tl.sum(v * sw[None, :], axis=1) * rstd, float("-inf"))

                b_m, b_mp = tl.maximum(b_m, tl.max(s, axis=0)), b_m
                r = tl.exp(b_mp - b_m)  # rescale for the new max
                p = tl.exp(s - b_m)
                b_acc = b_acc * r + tl.sum(p, axis=0)
                b_o = b_o * r + tl.sum(p[:, None] * v, axis=0)

        b_o = b_o / b_acc
        if OUT_NORM:
            # Free: b_o is already fully formed in registers.
            rs = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / H + out_eps)
            b_o = b_o * rs * tl.load(ow_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
        tl.store(y_ptr + t * stride_yt + o_d, b_o.to(y_ptr.dtype.element_ty), mask=m_d)

    # ------------------------------------------------------------------ decode
    # DECODE FAST PATH (small T).  The baseline body above walks the candidate
    # axis in cdiv(Bp, BL) online-softmax steps; every step's rescale waits on
    # the previous step's max/sum, so the 2*Bp full-width cross-lane reduction
    # trees sit end-to-end on ONE serial dependency chain.  At T <= 64 the grid
    # (one program per token, 64 CTAs on 256 CUs) cannot hide that chain behind
    # other work, which is why baseline decode time is flat in T and sits ~3x
    # above its own streaming floor.
    #
    # This kernel puts the WHOLE candidate axis in a single [NR, BD] tile, so
    # the two score reductions become exactly two BATCHED trees -- independent
    # of each other, so they pipeline -- and the softmax is a plain NR-wide
    # vector op: no rescale chain, no b_m/b_acc carry, and block_residual is
    # still read exactly once.
    #
    # PIT ("prefix in tile") folds the prefix candidate into a spare tile row
    # when next_pow2(B+1) == next_pow2(B) (e.g. B=1 -> 2 rows, B=15 -> 16 rows),
    # which is strictly cheaper than reducing it separately.  Otherwise the
    # prefix stays in fp32 registers and gets its own two (independent) trees --
    # that keeps its arithmetic exact AND keeps the tile at next_pow2(B) rows
    # rather than next_pow2(B+1), which for B=8 is a 2x tile saving.
    @triton.jit
    def _attn_res_decode_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        B,
        H,
        eps,
        out_eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_hs2_t,
        stride_pref_t,
        NR: tl.constexpr,  # rows of the candidate tile
        BD: tl.constexpr,  # next_pow2(H) -- one tile spans all of H
        PIT: tl.constexpr,  # the prefix occupies tile row B
        GRAM: tl.constexpr,  # NR==2 PIT: get sum(y*y) from the 2x2 Gram matrix
        DO_ADD: tl.constexpr,
        DO_ADD2: tl.constexpr,
        WRITE_PREF: tl.constexpr,
        OUT_NORM: tl.constexpr,
    ):
        t = tl.program_id(0)
        o_d = tl.arange(0, BD)
        m_d = o_d < H
        sw = tl.load(sw_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

        ps = tl.load(ps_ptr + t * stride_ps_t + o_d, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if DO_ADD2:
            ps += tl.load(hs2_ptr + t * stride_hs2_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                pref_ptr + t * stride_pref_t + o_d,
                ps.to(pref_ptr.dtype.element_ty),
                mask=m_d,
            )

        o_l = tl.arange(0, NR)
        v = tl.load(
            br_ptr + t * stride_br_t + o_l[:, None] * stride_br_b + o_d[None, :],
            mask=(o_l < B)[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if PIT:
            v = tl.where((o_l == B)[:, None], ps[None, :], v)
            m_s = o_l <= B
        else:
            m_s = o_l < B

        s1 = tl.sum(v * v, axis=1)
        s2 = tl.sum(v * sw[None, :], axis=1)
        if GRAM:
            # third INDEPENDENT batched tree, issued alongside s1/s2: the only
            # Gram entry the [c; p] tile does not already produce.  It replaces
            # the OUT_NORM tree, which is a DEPENDENT full-width reduction at
            # the very end of the kernel (measured 0.71 us of a 4.34 us kernel).
            cv = tl.sum(tl.where((o_l == 0)[:, None], v, 0.0), axis=0)
            vr = tl.where((o_l == B)[:, None], cv[None, :], ps[None, :])
            s3 = tl.sum(v * vr, axis=1)
        sc = tl.where(m_s, s2 * tl.rsqrt(s1 / H + eps), float("-inf"))

        if PIT:
            b_m = tl.max(sc, axis=0)
            e = tl.exp(sc - b_m)
            zz = tl.sum(e, axis=0)
            b_o = tl.sum(e[:, None] * v, axis=0) / zz
        else:
            q1 = tl.sum(ps * ps, axis=0)
            q2 = tl.sum(ps * sw, axis=0)
            sp = q2 * tl.rsqrt(q1 / H + eps)
            b_m = tl.maximum(tl.max(sc, axis=0), sp)
            e = tl.exp(sc - b_m)
            ep = tl.exp(sp - b_m)
            b_o = (tl.sum(e[:, None] * v, axis=0) + ep * ps) / (tl.sum(e, axis=0) + ep)

        if OUT_NORM:
            if GRAM:
                # y = (e0*c + e1*p)/z  =>  <y,y> = (e0^2<c,c> + 2 e0 e1 <c,p>
                #                                  + e1^2<p,p>) / z^2
                e0 = tl.sum(tl.where(o_l == 0, e, 0.0), axis=0)
                e1 = tl.sum(tl.where(o_l == B, e, 0.0), axis=0)
                g_cc = tl.sum(tl.where(o_l == 0, s1, 0.0), axis=0)
                g_pp = tl.sum(tl.where(o_l == B, s1, 0.0), axis=0)
                g_cp = tl.sum(tl.where(o_l == 0, s3, 0.0), axis=0)
                qn = (e0 * e0 * g_cc + 2.0 * e0 * e1 * g_cp + e1 * e1 * g_pp) / (zz * zz)
            else:
                qn = tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0)
            rs = tl.rsqrt(qn / H + out_eps)
            b_o = b_o * rs * tl.load(ow_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
        tl.store(y_ptr + t * stride_yt + o_d, b_o.to(y_ptr.dtype.element_ty), mask=m_d)

    # -------------------------------------------------- decode, online variant
    # At NR = 16 the single tile above does not fit: [16, 8192] fp32 is 512 KB,
    # i.e. the ENTIRE 4-SIMD register file of one CU, and Triton spills 200+
    # registers whatever num_warps is used.  Holding it in bf16 does not help
    # (the backend widens it anyway) and re-reading block_residual for the mix
    # costs a second full pass over the largest tensor.
    #
    # So: keep ONE [GR, BD] tile live at a time and fold each group into the
    # output flash-style.  block_residual is still read exactly once, the tile
    # is a fraction of the candidate axis, and -- the point -- the dependency
    # chain is NG steps long (2), not the baseline's cdiv(B+1, BL) = 8.
    @triton.jit
    def _attn_res_decode_online_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        B,
        H,
        eps,
        out_eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_hs2_t,
        stride_pref_t,
        BD: tl.constexpr,
        GR: tl.constexpr,  # candidate rows per group
        NG: tl.constexpr,  # groups (GR * NG rows in total)
        PIT: tl.constexpr,  # the prefix occupies tile row B
        PIT0: tl.constexpr,  # the prefix is candidate 0, in a PEELED group 0
        DO_ADD: tl.constexpr,
        DO_ADD2: tl.constexpr,
        WRITE_PREF: tl.constexpr,
        OUT_NORM: tl.constexpr,
    ):
        t = tl.program_id(0)
        o_d = tl.arange(0, BD)
        m_d = o_d < H
        sw = tl.load(sw_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

        ps = tl.load(ps_ptr + t * stride_ps_t + o_d, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if DO_ADD2:
            ps += tl.load(hs2_ptr + t * stride_hs2_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                pref_ptr + t * stride_pref_t + o_d,
                ps.to(pref_ptr.dtype.element_ty),
                mask=m_d,
            )

        o_g = tl.arange(0, GR)
        if PIT0:
            # r2_d0.  Make the prefix CANDIDATE 0 and PEEL its group out of the
            # loop.  Two things fall out at once:
            #   * the two full-width prefix trees (sum(ps*ps), sum(ps*sw)) leave
            #     the serial prologue -- they become two more rows of a tree the
            #     group was going to run anyway, and the flash seed is free
            #     because step 0 needs no rescale;
            #   * `ps` never enters the LOOP's live set, so the [BD] fp32 prefix
            #     stops competing with the [GR, BD] tile and the [BD] output
            #     accumulator for the register file (the same PEEL that took the
            #     prefill kernel from 258 to 152 registers).
            # Trailing-row PIT loses here precisely because it keeps `ps` live
            # to the LAST group; putting the prefix first kills it after one.
            brow = o_g - 1
            v = tl.load(
                br_ptr + t * stride_br_t + brow[:, None] * stride_br_b + o_d[None, :],
                mask=((o_g >= 1) & (o_g <= B))[:, None] & m_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v = tl.where((o_g == 0)[:, None], ps[None, :], v)
            s1 = tl.sum(v * v, axis=1)
            s2 = tl.sum(v * sw[None, :], axis=1)
            sc = tl.where(o_g <= B, s2 * tl.rsqrt(s1 / H + eps), float("-inf"))
            b_m = tl.max(sc, axis=0)
            e = tl.exp(sc - b_m)
            b_acc = tl.sum(e, axis=0)
            b_o = tl.sum(e[:, None] * v, axis=0)
        elif PIT:
            b_m = tl.full([], float("-inf"), dtype=tl.float32)
            b_acc = tl.zeros([], dtype=tl.float32)
            b_o = tl.zeros([BD], dtype=tl.float32)
        else:
            # seed the accumulator with the prefix candidate: exp(sp - sp) = 1
            q1 = tl.sum(ps * ps, axis=0)
            q2 = tl.sum(ps * sw, axis=0)
            b_m = q2 * tl.rsqrt(q1 / H + eps)
            b_acc = tl.full([], 1.0, dtype=tl.float32)
            b_o = ps

        # a dynamic (not static_range) loop on purpose: unrolling lets the
        # backend hoist every group's load, which puts the whole [GR*NG, BD]
        # tile live at once and brings the spills straight back.
        for g in range(1 if PIT0 else 0, NG):
            rows = g * GR + o_g
            if PIT0:
                # candidate k = block_residual row k - 1 (candidate 0 = prefix,
                # already consumed by the peeled group above)
                brow = rows - 1
                v = tl.load(
                    br_ptr
                    + t * stride_br_t
                    + brow[:, None] * stride_br_b
                    + o_d[None, :],
                    mask=(rows <= B)[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                m_s = rows <= B
            else:
                v = tl.load(
                    br_ptr
                    + t * stride_br_t
                    + rows[:, None] * stride_br_b
                    + o_d[None, :],
                    mask=(rows < B)[:, None] & m_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                if PIT:
                    v = tl.where((rows == B)[:, None], ps[None, :], v)
                    m_s = rows <= B
                else:
                    m_s = rows < B
            s1 = tl.sum(v * v, axis=1)
            s2 = tl.sum(v * sw[None, :], axis=1)
            sc = tl.where(m_s, s2 * tl.rsqrt(s1 / H + eps), float("-inf"))
            b_mp = b_m
            b_m = tl.maximum(b_m, tl.max(sc, axis=0))
            r = tl.exp(b_mp - b_m)
            e = tl.exp(sc - b_m)
            b_acc = b_acc * r + tl.sum(e, axis=0)
            b_o = b_o * r + tl.sum(e[:, None] * v, axis=0)
        b_o = b_o / b_acc

        if OUT_NORM:
            rs = tl.rsqrt(tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0) / H + out_eps)
            b_o = b_o * rs * tl.load(ow_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
        tl.store(y_ptr + t * stride_yt + o_d, b_o.to(y_ptr.dtype.element_ty), mask=m_d)


    # ----------------------------------------------- decode, split-H variant
    # GRID FIX.  Every decode kernel above launches exactly T programs, so at
    # T=64 only 64 of the 256 CUs have work and at T=1 exactly one does.  Decode
    # device time is consequently FLAT in T.
    #
    # Here the grid is (NS, T): each program owns ONE H-chunk of length
    # CL = H / NS for ALL Bp candidates of one token, so T * NS CTAs run.  NS is
    # chosen on the host so that T * NS <= 256 == the CU count, which makes every
    # participating CTA RESIDENT and therefore makes a device-scope spin barrier
    # between them safe (no CTA can be waiting on a CTA that has not been
    # scheduled).  That is what keeps this to ONE kernel launch: a separate
    # combine kernel would cost a second HIP-graph node (~1.7 us, 14% of
    # T64_B15), and the two-pass form that re-reads block_residual from HBM was
    # measured strictly slower.
    #
    # Protocol (deterministic, allocation-free, no zero-fill dispatch):
    #   * each split writes its OWN slot in `scr` -- no float atomic_add, so the
    #     combine order is fixed and the result is bit-reproducible run to run;
    #   * arrival is signalled with an int32 `tl.atomic_add(.., 1, release, gpu)`
    #     on a MONOTONIC per-(NS, token) counter that is never reset.  The ticket
    #     `old` returned by the fetch-add gives this call's base for free as
    #     `old - old % NS` (the counter only ever advances by NS per call and NS
    #     is a power of two), so no host-side zeroing kernel is needed;
    #   * the spin re-reads through `tl.atomic_add(.., 0, acquire, gpu)`, which is
    #     a coherent read by construction -- no reliance on a cache modifier
    #     surviving the optimizer.
    #
    # After the barrier every program already holds its own [NR, CL] slice of
    # block_residual IN REGISTERS, so applying the softmax weights costs no
    # second read of the largest tensor.  OUT_NORM needs sum_h y^2 over the full
    # row, so it takes a second (identical) barrier round -- skipped entirely
    # when OUT_NORM is off.
    @triton.jit
    def _attn_res_decode_split_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        scr_ptr,  # fp32 workspace, >= T*NS*2*NCP + T*NS floats
        cnt_ptr,  # int32 monotonic arrival counters
        cnt_off,  # phase-1 counter base (phase 2 lives at cnt_off + CNT_STRIDE)
        B,
        H,
        eps,
        out_eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_hs2_t,
        stride_pref_t,
        NS: tl.constexpr,  # H splits (== CTAs per token)
        CL: tl.constexpr,  # H / NS, exact
        BDc: tl.constexpr,  # next_pow2(CL)
        NR: tl.constexpr,  # candidate tile rows
        NCP: tl.constexpr,  # next_pow2(B+1) -- scratch row pitch
        NSP: tl.constexpr,  # next_pow2(NS)
        NL: tl.constexpr,  # num_warps * 64 -- spin-value broadcast width
        SCR2: tl.constexpr,  # float offset of the phase-2 scratch
        CNT_STRIDE: tl.constexpr,
        PIT: tl.constexpr,
        DO_ADD: tl.constexpr,
        DO_ADD2: tl.constexpr,
        WRITE_PREF: tl.constexpr,
        OUT_NORM: tl.constexpr,
    ):
        c = tl.program_id(0)
        t = tl.program_id(1)

        o_i = tl.arange(0, BDc)
        m_d = o_i < CL
        o_d = c * CL + o_i

        sw = tl.load(sw_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)
        ps = tl.load(ps_ptr + t * stride_ps_t + o_d, mask=m_d, other=0.0).to(tl.float32)
        if DO_ADD:
            ps += tl.load(hs_ptr + t * stride_hs_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if DO_ADD2:
            ps += tl.load(hs2_ptr + t * stride_hs2_t + o_d, mask=m_d, other=0.0).to(
                tl.float32
            )
        if WRITE_PREF:
            tl.store(
                pref_ptr + t * stride_pref_t + o_d,
                ps.to(pref_ptr.dtype.element_ty),
                mask=m_d,
            )

        o_l = tl.arange(0, NR)
        v = tl.load(
            br_ptr + t * stride_br_t + o_l[:, None] * stride_br_b + o_d[None, :],
            mask=(o_l < B)[:, None] & m_d[None, :],
            other=0.0,
        ).to(tl.float32)
        if PIT:
            v = tl.where((o_l == B)[:, None], ps[None, :], v)

        s1 = tl.sum(v * v, axis=1)
        s2 = tl.sum(v * sw[None, :], axis=1)

        pb = scr_ptr + (t * NS + c) * (2 * NCP)
        tl.store(pb + o_l, s1)
        tl.store(pb + NCP + o_l, s2)
        if not PIT:
            tl.store(pb + B, tl.sum(ps * ps, axis=0))
            tl.store(pb + NCP + B, tl.sum(ps * sw, axis=0))

        # ---- barrier 1 -------------------------------------------------
        # tl.debug_barrier() first: the release atomic below is issued by ONE
        # elected lane and its s_waitcnt vmcnt(0) only covers that lane's WAVE,
        # so the other waves' partial stores need the block barrier (which the
        # AMDGPU backend prefixes with a full vmcnt wait) to be complete.
        # Three things about the spin, each of them load-bearing (all measured
        # the hard way -- see the report):
        #  1. `.cv` is the ONLY cache modifier that lowers to
        #     `global_load_dword ... sc0 sc1`, i.e. past both L1 and the
        #     per-XCD L2 (gfx950 has 8 of them).  `volatile=True` is an
        #     NVIDIA-only flag here and is silently dropped.
        #  2. The load must not be hoisted.  A bare `.cv` load of a
        #     loop-invariant address IS hoisted by LICM and the backend then
        #     emits a literal `s_cbranch_vccz .` infinite loop.  Reducing over
        #     a tensor keeps a barrier in the loop body, which blocks the hoist.
        #  3. The exit test must be BLOCK-uniform.  Waves poll independently, so
        #     a per-wave test lets wave 0 leave the loop an iteration before
        #     wave 1 -- after which every later `s_barrier` (Triton puts them in
        #     every cross-warp reduction) is mispaired and the LDS reductions
        #     return garbage or NaN.  `tl.max(...)` over an [NL] tensor forces
        #     the polled value through the LDS reduction, so all waves branch on
        #     the identical value.  A scalar `tl.atomic_*` in the loop is not an
        #     option either: Triton lowers it as elect-one-lane + LDS broadcast
        #     + s_barrier, which hangs outright when nested in a loop.
        ck = cnt_ptr + cnt_off + t
        o_n = tl.arange(0, NL)
        m_n = o_n == 0
        tl.debug_barrier()
        tk = tl.atomic_add(ck, 1, sem="release", scope="gpu")
        tgt = tk - (tk % NS) + NS
        cur = tl.max(tl.load(ck + o_n, mask=m_n, other=0, cache_modifier=".cv"), axis=0)
        while cur < tgt:
            tl.debug_barrier()
            cur = tl.max(
                tl.load(ck + o_n, mask=m_n, other=0, cache_modifier=".cv"), axis=0
            )
        tl.debug_barrier()

        o_s = tl.arange(0, NSP)
        m_ss = o_s < NS
        rb = scr_ptr + (t * NS + o_s) * (2 * NCP)
        # Read-back with `.cv` loads (sc0 sc1, past L1 and the per-XCD L2).
        # The obvious alternative -- tl.atomic_add(.., 0.0), which is also
        # coherent -- is CORRECT but costs 32 us per launch at 256 CTAs
        # (measured: 21-35 us for the read alone versus 4 us for the whole
        # barrier), because each device-scope atomic is ~500 ns and they do not
        # pipeline.  Each split owns its own slot and every program folds the
        # NS slots in the same fixed order, so the combine is bit-deterministic
        # -- no fp32 atomic_add ordering can perturb it run to run.
        R1 = tl.sum(
            tl.load(rb[:, None] + o_l[None, :], mask=m_ss[:, None], other=0.0,
                    cache_modifier=".cv"), axis=0
        )
        R2 = tl.sum(
            tl.load(rb[:, None] + NCP + o_l[None, :], mask=m_ss[:, None], other=0.0,
                    cache_modifier=".cv"),
            axis=0,
        )

        if PIT:
            sc = tl.where(o_l <= B, R2 * tl.rsqrt(R1 / H + eps), float("-inf"))
            b_m = tl.max(sc, axis=0)
            e = tl.exp(sc - b_m)
            b_o = tl.sum(e[:, None] * v, axis=0) / tl.sum(e, axis=0)
        else:
            q1 = tl.sum(tl.load(rb + B, mask=m_ss, other=0.0,
                                cache_modifier=".cv"), axis=0)
            q2 = tl.sum(tl.load(rb + NCP + B, mask=m_ss, other=0.0,
                                cache_modifier=".cv"), axis=0)
            sp = q2 * tl.rsqrt(q1 / H + eps)
            sc = tl.where(o_l < B, R2 * tl.rsqrt(R1 / H + eps), float("-inf"))
            b_m = tl.maximum(tl.max(sc, axis=0), sp)
            e = tl.exp(sc - b_m)
            ep = tl.exp(sp - b_m)
            b_o = (tl.sum(e[:, None] * v, axis=0) + ep * ps) / (tl.sum(e, axis=0) + ep)

        if OUT_NORM:
            # ---- barrier 2 (only OUT_NORM needs the full-row sum of y^2) ----
            s2p = scr_ptr + SCR2
            tl.store(s2p + (t * NS + c), tl.sum(tl.where(m_d, b_o * b_o, 0.0), axis=0))
            ck2 = cnt_ptr + cnt_off + CNT_STRIDE + t
            tl.debug_barrier()
            tk2 = tl.atomic_add(ck2, 1, sem="release", scope="gpu")
            tgt2 = tk2 - (tk2 % NS) + NS
            cur2 = tl.max(
                tl.load(ck2 + o_n, mask=m_n, other=0, cache_modifier=".cv"), axis=0
            )
            while cur2 < tgt2:
                tl.debug_barrier()
                cur2 = tl.max(
                    tl.load(ck2 + o_n, mask=m_n, other=0, cache_modifier=".cv"), axis=0
                )
            tl.debug_barrier()
            tot = tl.sum(tl.load(s2p + t * NS + o_s, mask=m_ss, other=0.0,
                                 cache_modifier=".cv"), axis=0)
            rs = tl.rsqrt(tot / H + out_eps)
            b_o = b_o * rs * tl.load(ow_ptr + o_d, mask=m_d, other=0.0).to(tl.float32)

        tl.store(y_ptr + t * stride_yt + o_d, b_o.to(y_ptr.dtype.element_ty), mask=m_d)


# (num_warps, num_stages, BL) by token count. One program per token, so at small T
# the grid alone cannot fill the GPU and wider warps are what recover occupancy;
# BL stays small throughout because the [BL, BD] tile competes with the [BD]
# accumulator for the register file.
_ATTN_RES_CONFIGS = (
    (8, 8, 2, 2),  # T <= 8
    (64, 8, 2, 2),
    (512, 8, 2, 2),
    (2048, 4, 2, 2),
)
_ATTN_RES_CATCHALL = (4, 2, 2)  # T > largest bucket


def _pick_attn_res_config(tokens: int):
    for max_tokens, nw, ns, bl in _ATTN_RES_CONFIGS:
        if tokens <= max_tokens:
            return nw, ns, bl
    return _ATTN_RES_CATCHALL


# ---------------------------------------------------------------- PREFILL ---
# T > _PREFILL_T is a different machine entirely: the grid alone is 4-64 CTAs per
# CU, so nothing is latency-starved and the ONLY thing that matters is how close
# to the 5.2 TB/s this box actually delivers the kernel streams its 3 GB working
# set.  The knobs that decide that are (BL, num_warps, num_stages, waves_per_eu)
# and they key on B -- with the prefix candidate peeled out (PEEL=True) the
# candidate loop covers exactly the B block_residual rows, so a BL that does not
# divide B leaves the last tile half empty and throughput collapses.  Entries are
# (num_warps, num_stages, BL, waves_per_eu); waves_per_eu=0 means "let the
# compiler choose".
_PREFILL_T = 512
# (num_warps, num_stages, BL, waves_per_eu, PEEL, CG(block_residual), CG2(prefix))
# Measured on gfx950 (graph replay, 4 interleaved passes, best-of): BL=1 + PEEL is
# worth 1.78-1.83x on every case whose Bp made the old [2, 8192] tile ragged and
# 1.05-1.10x on the three that were already at 97-102% of achievable, so there is
# no regression anywhere.  `.cg` on the block_residual stream (the read-once bulk)
# pays; `.cg` on the prefix/addend streams costs 4-10% and is off.  At T=2048 with
# B<=4 block_residual stops being the bulk and `.cg` stops paying there too.
_ATTN_RES_PREFILL_DEFAULT = (4, 2, 1, 0, True, True, False)
_ATTN_RES_PREFILL_CONFIGS = {
    (2048, 4): (4, 2, 1, 0, True, False, False),
    (2048, 1): (4, 2, 1, 0, True, False, False),
}


def _pick_attn_res_prefill(tokens: int, b: int):
    cfg = _ATTN_RES_PREFILL_CONFIGS.get((tokens, b))
    if cfg is None:
        cfg = _ATTN_RES_PREFILL_CONFIGS.get(b)
    if cfg is None:
        cfg = _ATTN_RES_PREFILL_DEFAULT
    return cfg


# ---------------------------------------------------------------- decode path
# Token count at or below which the batched-reduction decode kernels are used.
# Above it the tile-per-token form loses to the streaming prefill kernel.
_DECODE_MAX_T = 512

# next_pow2 tile rows -> (num_warps, num_stages).  A static table, never
# triton.autotune: HIP-graph replay forbids JIT at replay time, so every
# reachable specialization must be compiled on the host warm-up path.
_DECODE_WARPS = {1: 4, 2: 4, 4: 4, 8: 8, 16: 4}
_DECODE_STAGES = {1: 1, 2: 1, 4: 1, 8: 1, 16: 1}
# r1_d2's measured waves_per_eu finding, re-keyed onto the decode tile: the
# "1,1" hint pays where one workgroup owns the CU and the tile is mid-sized.
# 0 == let the compiler choose (the previous behaviour).
_DECODE_WPE = {1: 0, 2: 0, 4: 1, 8: 0, 16: 0}
# candidate rows per online-kernel group.  Measured on gfx950 with the PIT0
# prefix-first form: 4 beats 2 (12.63 us) and 8 (13.67 us) at T64_B15.
_DECODE_GR = 4


_DECODE_PLAN_CACHE: dict[int, tuple] = {}


def _decode_plan(B: int):
    """(tile_rows, prefix_in_tile, groups, num_warps, num_stages, wpe, GR,
    prefix_is_candidate_0, gram) for this B.

    Memoized: this runs on every eager call, and the eager path is already
    launch-bound (~19 us of the ~25 us per call is the raw Triton launch).
    """
    plan = _DECODE_PLAN_CACHE.get(B)
    if plan is None:
        nr = triton.next_power_of_2(B)
        pit = triton.next_power_of_2(B + 1) == nr
        if B == 1:
            # [1, BD] plus two separate prefix trees costs more than a [2, BD] tile
            nr, pit = 2, True
        # measured: folding the prefix into the LAST tile row costs more than
        # it saves in the online kernel (it keeps the fp32 prefix live across
        # the whole group loop); PIT0 below puts it FIRST instead, which wins.
        gr = _DECODE_GR
        pit0 = nr > 8 and nr > B
        if nr <= 8:
            ng = 1
        elif pit0:
            # groups cover the B+1 candidates (prefix first), GR rows each
            ng = (B + 1 + gr - 1) // gr
        else:
            ng = nr // gr
        pit = pit and ng == 1
        plan = (
            nr,
            pit,
            ng,
            _DECODE_WARPS.get(nr, 4),
            _DECODE_STAGES.get(nr, 1),
            _DECODE_WPE.get(nr, 0),
            gr,
            pit0,
            # r2_d2: B==1 only -- derive sum(y*y) from the 2x2 Gram matrix of
            # the [c; p] tile instead of a dependent full-width OUT_NORM tree.
            bool(pit and ng == 1 and nr == 2 and B == 1),
        )
        _DECODE_PLAN_CACHE[B] = plan
    return plan


# ------------------------------------------------------------- split-H decode
# The decode grid is (NS, T) with NS H-splits per token.  The device-scope spin
# barrier inside the kernel is only safe while every participating CTA is
# resident, so NS is capped by T * NS <= _CU_COUNT.  NS is also kept a power of
# two that divides H exactly (H = 7168 = 2^10 * 7, so every power of two up to
# 1024 divides it) -- next_pow2(7168) = 8192 would waste 12.5% of every lane.
_CU_COUNT = 256
_SPLIT_MIN_CL = 224   # do not split H finer than this many elements per CTA
_SPLIT_MAX_NS = 16
_SPLIT_MAX_B = 31     # scratch row pitch is 2*next_pow2(B+1) <= 64 floats
# WHERE THIS PAYS, and why the window is narrow.  An isolated probe of the
# barrier alone (store + release ticket + `.cv` spin + `.cv` read-back, no real
# work) costs 4.0 us per round at 256 CTAs and 1.6 us at 32, i.e. ~3-8 us for
# the two rounds an OUT_NORM decode needs.  The decode kernels it would replace
# run in 4.4-12 us TOTAL, so the barrier eats the entire saving everywhere the
# per-token work is already small.  Measured decode-only sweep (us, candidate
# before -> after): T64_B15 12.11 -> 12.05, T32_B15 11.96 -> 12.04,
# T1_B8 7.67 -> 8.75, T1_B4 5.40 -> 8.21, T1_B1 4.43 -> 7.09 -- all flat or
# worse.  The single case where the token's own work is big enough to amortise
# two barriers is T=1 with a full candidate set: T1_B15 11.78 -> 8.29 (1.42x).
# So the split is gated to exactly that: one token, many candidates.
_SPLIT_MAX_T = 1
_SPLIT_MIN_B = 12
_SPLIT_MAX_CTAS = _CU_COUNT

# scratch: [T*NS, 2*NCP] fp32 partials + [T*NS] fp32 out-norm partials.
# Allocated ONCE per device and cached (never per call -- the eager decode path
# is launch-bound).  Never zero-filled: every slot is written before it is read,
# and the arrival counters are monotonic so they never need resetting either.
_SPLIT_SCR_FLOATS = _CU_COUNT * 64
_SPLIT_SCR2_OFF = _SPLIT_SCR_FLOATS
_SPLIT_CNT_STRIDE = 12 * _CU_COUNT
_SPLIT_SCRATCH: dict = {}


def _split_scratch(device):
    key = (device.type, device.index)
    e = _SPLIT_SCRATCH.get(key)
    if e is None:
        scr = torch.empty(_SPLIT_SCR_FLOATS + _CU_COUNT, dtype=torch.float32,
                          device=device)
        cnt = torch.zeros(2 * _SPLIT_CNT_STRIDE, dtype=torch.int32, device=device)
        e = (scr, cnt)
        _SPLIT_SCRATCH[key] = e
    return e


_SPLIT_WARPS = {128: 8, 256: 8, 512: 8, 1024: 8, 2048: 8, 4096: 8}
_SPLIT_PLAN_CACHE: dict = {}


def _split_plan(T: int, B: int, H: int):
    """(NS, CL, BDc, NR, NCP, NSP, num_warps) or None when splitting is off."""
    key = (T, B, H)
    plan = _SPLIT_PLAN_CACHE.get(key)
    if plan is None:
        ns = 1
        if _SPLIT_MIN_B <= B <= _SPLIT_MAX_B and T <= _SPLIT_MAX_T:
            while (ns * 2 * T <= _SPLIT_MAX_CTAS and ns * 2 <= _SPLIT_MAX_NS
                   and H % (ns * 2) == 0 and H // (ns * 2) >= _SPLIT_MIN_CL):
                ns *= 2
        if ns == 1:
            plan = False
        else:
            cl = H // ns
            bdc = triton.next_power_of_2(cl)
            nr = triton.next_power_of_2(B)
            pit = triton.next_power_of_2(B + 1) == nr
            if B == 1:
                nr, pit = 2, True
            ncp = triton.next_power_of_2(B + 1)
            plan = (ns, cl, bdc, nr, ncp, triton.next_power_of_2(ns), pit,
                    _SPLIT_WARPS.get(bdc, 4))
        _SPLIT_PLAN_CACHE[key] = plan
    return plan


def _attn_res_decode(
    prefix_sum,
    block_residual,
    score_weight,
    eps,
    add_hidden,
    out_norm_weight,
    out_eps,
    add_hidden2,
    T,
    B,
    H,
):
    """One-launch decode path; see ``_attn_res_decode_kernel``."""
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    out_norm = out_norm_weight is not None
    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    ow = out_norm_weight.contiguous() if out_norm else sw
    hs = add_hidden.contiguous() if do_add else ps
    hs2 = add_hidden2.contiguous() if do_add2 else ps
    pref = torch.empty_like(ps) if do_add else ps

    sp = _split_plan(T, B, H)
    if sp:
        s_ns, s_cl, s_bdc, s_nr, s_ncp, s_nsp, s_pit, s_nw = sp
        scr, cnt = _split_scratch(block_residual.device)
        _attn_res_decode_split_kernel[(s_ns, T)](
            br,
            ps,
            sw,
            y,
            hs,
            hs2,
            pref,
            ow,
            scr,
            cnt,
            (s_ns.bit_length() - 1) * _CU_COUNT,
            B,
            H,
            float(eps),
            float(out_eps),
            br.stride(0),
            br.stride(1),
            ps.stride(0),
            y.stride(0),
            hs.stride(0),
            hs2.stride(0),
            pref.stride(0),
            NS=s_ns,
            CL=s_cl,
            BDc=s_bdc,
            NR=s_nr,
            NCP=s_ncp,
            NSP=s_nsp,
            NL=s_nw * 64,
            SCR2=_SPLIT_SCR2_OFF,
            CNT_STRIDE=_SPLIT_CNT_STRIDE,
            PIT=s_pit,
            num_warps=s_nw,
            num_stages=1,
            DO_ADD=do_add,
            DO_ADD2=do_add2,
            WRITE_PREF=do_add,
            OUT_NORM=out_norm,
        )
        return y, (pref if do_add else prefix_sum)

    nr, pit, ng, nw, ns, wpe, gr, pit0, gram = _decode_plan(B)
    _wpe = {"waves_per_eu": wpe} if wpe else {}
    bd = triton.next_power_of_2(H)
    if ng == 1:
        _attn_res_decode_kernel[(T,)](
            br,
            ps,
            sw,
            y,
            hs,
            hs2,
            pref,
            ow,
            B,
            H,
            float(eps),
            float(out_eps),
            br.stride(0),
            br.stride(1),
            ps.stride(0),
            y.stride(0),
            hs.stride(0),
            hs2.stride(0),
            pref.stride(0),
            NR=nr,
            BD=bd,
            PIT=pit,
            GRAM=gram,
            num_stages=ns,
            num_warps=nw,
            DO_ADD=do_add,
            DO_ADD2=do_add2,
            WRITE_PREF=do_add,
            OUT_NORM=out_norm,
            **_wpe,
        )
    else:
        _attn_res_decode_online_kernel[(T,)](
            br,
            ps,
            sw,
            y,
            hs,
            hs2,
            pref,
            ow,
            B,
            H,
            float(eps),
            float(out_eps),
            br.stride(0),
            br.stride(1),
            ps.stride(0),
            y.stride(0),
            hs.stride(0),
            hs2.stride(0),
            pref.stride(0),
            BD=bd,
            GR=gr,
            NG=ng,
            PIT=pit,
            PIT0=pit0,
            num_stages=ns,
            num_warps=nw,
            DO_ADD=do_add,
            DO_ADD2=do_add2,
            WRITE_PREF=do_add,
            OUT_NORM=out_norm,
            **_wpe,
        )
    return y, (pref if do_add else prefix_sum)


def _apply_attn_res_impl(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]
    score_weight: torch.Tensor,  # [H] (norm_weight * proj_weight, precomputed)
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H], folded: prefix += add_hidden
    out_norm_weight: torch.Tensor | None = None,  # [H], folded: y = rmsnorm(y)
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,  # [T, H], folded the same way
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-residual soft-attention mix: rmsnorm each of the B+1 candidates,
    score = <normed, score_weight>, softmax over B+1, weighted sum.

    Candidates are the B rows of ``block_residual`` plus ``prefix_sum``, so
    ``score_weight`` must already fold the rmsnorm gain into the scoring
    projection (see ``_attn_res_score_weight`` on the model side).

    Returns ``(mixed_output, prefix_out)``. When ``add_hidden`` (and optionally
    ``add_hidden2``) is given, the caller's ``prefix_sum = prefix_sum + ...``
    elementwise add is folded into the kernel on-load and ``prefix_out`` is that
    sum; otherwise ``prefix_out`` is ``prefix_sum`` unchanged. Two addends exist
    so an MoE layer can pass its routed and shared expert outputs separately and
    skip the [T, H] elementwise add that would otherwise combine them.

    When ``out_norm_weight`` is given, the caller's rmsnorm OF THE RESULT (every
    apply_attn_res call site in kimi_k3.py feeds one) is folded in too, so the
    returned ``y`` is already normed and scaled.
    """
    T, B, H = block_residual.shape
    if _HAS_TRITON and T <= _DECODE_MAX_T:
        if add_hidden2 is not None and add_hidden is None:
            raise ValueError("add_hidden2 requires add_hidden")
        return _attn_res_decode(
            prefix_sum,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            out_norm_weight,
            out_eps,
            add_hidden2,
            T,
            B,
            H,
        )
    Bp = B + 1
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    if do_add2 and not do_add:
        raise ValueError("add_hidden2 requires add_hidden")
    out_norm = out_norm_weight is not None
    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    ow = out_norm_weight.contiguous() if out_norm else sw
    # hs/hs2/pref pointers are always passed (triton needs a tensor); when not
    # adding they alias ps and are never dereferenced (DO_ADD / DO_ADD2 /
    # WRITE_PREF are False).
    hs = add_hidden.contiguous() if do_add else ps
    hs2 = add_hidden2.contiguous() if do_add2 else ps
    pref = torch.empty_like(ps) if do_add else ps

    if T > _PREFILL_T:
        nw, ns, bl, wpe, peel, cg, cg2 = _pick_attn_res_prefill(T, B)
    else:
        nw, ns, bl = _pick_attn_res_config(T)
        wpe, peel, cg, cg2 = 0, False, False, False
    _attn_res_fused_kernel[(T,)](
        br,
        ps,
        sw,
        y,
        hs,
        hs2,
        pref,
        ow,
        B,
        Bp,
        H,
        float(eps),
        float(out_eps),
        br.stride(0),
        br.stride(1),
        ps.stride(0),
        y.stride(0),
        hs.stride(0),
        hs2.stride(0),
        pref.stride(0),
        BL=bl,
        BD=triton.next_power_of_2(H),
        num_stages=ns,
        num_warps=nw,
        DO_ADD=do_add,
        DO_ADD2=do_add2,
        WRITE_PREF=do_add,
        OUT_NORM=out_norm,
        PEEL=peel,
        CG=cg,
        CG2=cg2,
        **({"waves_per_eu": wpe} if wpe else {}),
    )
    return y, (pref if do_add else prefix_sum)


def _apply_attn_res_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
) -> torch.Tensor:
    mixed_output, _ = _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
    )
    return mixed_output


def _apply_attn_res_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
) -> torch.Tensor:
    return torch.empty_like(prefix_sum)


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res",
    op_func=_apply_attn_res_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_op_fake,
)


def _apply_attn_res_add_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        add_hidden2=add_hidden2,
    )


def _apply_attn_res_add_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(prefix_sum), torch.empty_like(prefix_sum)


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_add",
    op_func=_apply_attn_res_add_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_add_op_fake,
)


@mark_trace
def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor | None = None,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch an opaque custom op whose CUDA implementation selects by concrete T.

    ``out_norm_weight`` folds the caller's rmsnorm of the result into the kernel;
    the returned mixed output is then already normed and scaled by it.
    ``add_hidden2`` folds a second addend into the prefix (see the impl)."""
    if add_hidden is None:
        if add_hidden2 is not None:
            raise ValueError("add_hidden2 requires add_hidden")
        return (
            torch.ops.aiter.kimi_k3_apply_attn_res(
                prefix_sum, block_residual, score_weight, eps, out_norm_weight, out_eps
            ),
            prefix_sum,
        )
    return torch.ops.aiter.kimi_k3_apply_attn_res_add(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight,
        out_eps,
        add_hidden2,
    )
