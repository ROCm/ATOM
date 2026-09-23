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
        ys_ptr,
        hs_ptr,
        hs2_ptr,
        pref_ptr,
        ow_ptr,
        B,
        Bp,
        H,
        eps,
        out_eps,
        fp8_max,
        inv_fp8_max,
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
        QUANT: tl.constexpr,  # fold the consumer's per-token quant into the store
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
        if QUANT:
            # Also free, and for the same reason OUT_NORM is: the row is already
            # in registers, so the per-token amax is a register reduction and the
            # store just narrows. Folding it here is what lets the consuming GEMM
            # skip a standalone quant of this same [T, H] -- the fusion the
            # RMSNorm module does on the paths where it, rather than this kernel,
            # applies out_norm.
            amax = tl.max(tl.where(m_d, tl.abs(b_o), 0.0), axis=0)
            # amax * (1/fp8_max), not amax / fp8_max -- triton's ROCm fdiv is
            # 1 ulp off IEEE and would put this row's scale on a different code
            # than aiter's per-token quant, which the non-fused path uses for the
            # same activation. See the same note in kimi_k3/quant.py.
            scale = amax * inv_fp8_max
            inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
            b_o = tl.minimum(tl.maximum(b_o * inv, -fp8_max), fp8_max)
            tl.store(ys_ptr + t, scale)
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


def _apply_attn_res_impl(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]
    score_weight: torch.Tensor,  # [H] (norm_weight * proj_weight, precomputed)
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H], folded: prefix += add_hidden
    out_norm_weight: torch.Tensor | None = None,  # [H], folded: y = rmsnorm(y)
    out_eps: float = 1e-6,
    add_hidden2: torch.Tensor | None = None,  # [T, H], folded the same way
    quant_dtype: torch.dtype | None = None,  # folded: y = per-token quant(y)
) -> (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
):
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

    ``quant_dtype`` folds the CONSUMING GEMM's per-token activation quant in as
    well, returning ``(y_quantized, y_scale, prefix_out)`` instead of
    ``(y, prefix_out)``. It requires ``out_norm_weight``: quantizing an unnormed
    mix would hand the consumer an activation on the wrong scale entirely.
    """
    T, B, H = block_residual.shape
    Bp = B + 1
    do_add = add_hidden is not None
    do_add2 = add_hidden2 is not None
    if do_add2 and not do_add:
        raise ValueError("add_hidden2 requires add_hidden")
    out_norm = out_norm_weight is not None
    quant = quant_dtype is not None
    if quant and not out_norm:
        raise ValueError("quant_dtype requires out_norm_weight")
    fp8_max = float(torch.finfo(quant_dtype).max) if quant else 1.0
    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty(
        (T, H),
        device=block_residual.device,
        dtype=quant_dtype if quant else prefix_sum.dtype,
    )
    # Always allocated (triton needs a tensor); size 1 and never dereferenced
    # when QUANT is False.
    y_scale = torch.empty(
        (T, 1) if quant else (1,), device=block_residual.device, dtype=torch.float32
    )
    ow = out_norm_weight.contiguous() if out_norm else sw
    # hs/hs2/pref pointers are always passed (triton needs a tensor); when not
    # adding they alias ps and are never dereferenced (DO_ADD / DO_ADD2 /
    # WRITE_PREF are False).
    hs = add_hidden.contiguous() if do_add else ps
    hs2 = add_hidden2.contiguous() if do_add2 else ps
    pref = torch.empty_like(ps) if do_add else ps

    nw, ns, bl = _pick_attn_res_config(T)
    _attn_res_fused_kernel[(T,)](
        br,
        ps,
        sw,
        y,
        y_scale,
        hs,
        hs2,
        pref,
        ow,
        B,
        Bp,
        H,
        float(eps),
        float(out_eps),
        fp8_max,
        1.0 / fp8_max,
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
        QUANT=quant,
    )
    prefix_out = pref if do_add else prefix_sum
    if quant:
        return y, y_scale, prefix_out
    return y, prefix_out


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


# Quantizing variants. Separate ops rather than an optional `quant_dtype` on the
# two above because the return arity differs (the scale), and a schema's return
# type is fixed at registration.


def _apply_attn_res_quant_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    y, y_scale, _ = _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        quant_dtype=quant_dtype,
    )
    return y, y_scale


def _apply_attn_res_quant_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(prefix_sum, dtype=quant_dtype),
        torch.empty(
            (prefix_sum.shape[0], 1), device=prefix_sum.device, dtype=torch.float32
        ),
    )


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_quant",
    op_func=_apply_attn_res_quant_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_quant_op_fake,
)


def _apply_attn_res_add_quant_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
    add_hidden2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _apply_attn_res_impl(
        prefix_sum,
        block_residual,
        score_weight,
        eps,
        add_hidden,
        out_norm_weight=out_norm_weight,
        out_eps=out_eps,
        add_hidden2=add_hidden2,
        quant_dtype=quant_dtype,
    )


def _apply_attn_res_add_quant_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
    out_norm_weight: torch.Tensor,
    out_eps: float,
    quant_dtype: torch.dtype,
    add_hidden2: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(prefix_sum, dtype=quant_dtype),
        torch.empty(
            (prefix_sum.shape[0], 1), device=prefix_sum.device, dtype=torch.float32
        ),
        torch.empty_like(prefix_sum),
    )


direct_register_custom_op(
    op_name="kimi_k3_apply_attn_res_add_quant",
    op_func=_apply_attn_res_add_quant_op,
    mutates_args=[],
    fake_impl=_apply_attn_res_add_quant_op_fake,
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
    quant_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Dispatch an opaque custom op whose CUDA implementation selects by concrete T.

    ``out_norm_weight`` folds the caller's rmsnorm of the result into the kernel;
    the returned mixed output is then already normed and scaled by it.
    ``add_hidden2`` folds a second addend into the prefix (see the impl).

    ``quant_dtype`` additionally folds the consuming GEMM's per-token quant in,
    making the first element of the returned pair a ``(quantized, scale)`` tuple
    rather than a tensor -- the same shape the fused-quant RMSNorm hands its
    consumers, so the call sites that already unpack one need no new branch."""
    if quant_dtype is not None:
        if add_hidden is None:
            if add_hidden2 is not None:
                raise ValueError("add_hidden2 requires add_hidden")
            y, y_scale = torch.ops.aiter.kimi_k3_apply_attn_res_quant(
                prefix_sum,
                block_residual,
                score_weight,
                eps,
                out_norm_weight,
                out_eps,
                quant_dtype,
            )
            return (y, y_scale), prefix_sum
        y, y_scale, prefix_out = torch.ops.aiter.kimi_k3_apply_attn_res_add_quant(
            prefix_sum,
            block_residual,
            score_weight,
            eps,
            add_hidden,
            out_norm_weight,
            out_eps,
            quant_dtype,
            add_hidden2,
        )
        return (y, y_scale), prefix_out
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
