"""Fused Triton kernels for Kimi-K3 small ops (gfx1250 perf).

Drop-in replacements for the elementwise/reduction glue that otherwise runs as
several separate torch ops (each a kernel launch + full HBM round-trip):

- ``situ_and_mul``      : SiTU(gate) * linear(up)  (dense/shared-expert MLP act)
- ``rmsnorm_gated``     : rmsnorm(x) * weight * sigmoid(gate)  (KDA o_norm)
- ``apply_attn_res``    : block-residual soft-attention mix (attn_res boundaries)
- ``dual_rmsnorm``      : q + k RMSNorm in one AITER kernel (MLA a_layernorm)

The activation and gated-rmsnorm kernels remain gated behind ``ATOM_K3_FUSED``
in kimi_k3.py and retain local torch fallbacks. ``apply_attn_res`` is the
production path. ``dual_rmsnorm`` has no environment gate or fallback and always
launches the fused AITER q/k RMSNorm kernel.
"""

from __future__ import annotations

import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.fused_qk_norm_rope_cache_quant import (
    _fused_qk_rmsnorm_kernel as _aiter_fused_qk_rmsnorm_kernel,
)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _situ_and_mul_kernel(
        x_ptr,
        y_ptr,
        M,
        D,
        stride_xm,
        stride_ym,
        beta,
        inv_beta,
        linear_beta,
        inv_linear_beta,
        HAS_LINEAR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = col < D
        g = tl.load(x_ptr + row * stride_xm + col, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(x_ptr + row * stride_xm + D + col, mask=mask, other=0.0).to(
            tl.float32
        )
        # SiTUv2 gate: beta * tanh(gate/beta) * sigmoid(gate); tanh via sigmoid
        # identity (tanh(z) = 2*sigmoid(2z) - 1) for portability across triton.
        out = beta * (2.0 * tl.sigmoid(2.0 * g * inv_beta) - 1.0) * tl.sigmoid(g)
        if HAS_LINEAR:
            u = linear_beta * (2.0 * tl.sigmoid(2.0 * u * inv_linear_beta) - 1.0)
        y = out * u
        tl.store(y_ptr + row * stride_ym + col, y.to(y_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _rmsnorm_gated_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        y_ptr,
        H,
        eps,
        stride_xm,
        stride_ym,
        stride_g_outer,
        stride_g_head,
        HEADS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        # x / y are row-contiguous [M, H]; gate may be strided. Its logical row
        # `row` decomposes into (outer, head) so the token-boundary jump
        # (stride_g_outer) and per-head step (stride_g_head) are read directly,
        # avoiding a contiguous copy of the strided gate slice.
        g_off = (row // HEADS) * stride_g_outer + (row % HEADS) * stride_g_head + cols
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w) * tl.sigmoid(gate)
        tl.store(
            y_ptr + row * stride_ym + cols, y.to(y_ptr.dtype.element_ty), mask=mask
        )


if _HAS_TRITON:

    @triton.jit
    def _attn_res_fused_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        y_ptr,
        hs_ptr,
        pref_ptr,
        B,
        Bp,
        H,
        eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_yt,
        stride_hs_t,
        stride_pref_t,
        BP: tl.constexpr,  # Bp padded to a power of 2 (vectorized candidate axis)
        BLOCK_H: tl.constexpr,
        NS: tl.constexpr,  # num_stages for the H-loop software pipeline
        DO_ADD: tl.constexpr,  # fold prefix += hidden_states on-load
        WRITE_PREF: tl.constexpr,  # write the (summed) prefix back to pref_ptr
    ):
        # One program per row t: rmsnorm each of the Bp = B+1 candidates, score =
        # <normed, score_weight>, softmax over Bp, then weighted sum -> y[t].
        # Candidates 0..B-1 are block_residual rows; candidate B is prefix_sum.
        # Read both source tensors directly (no torch.cat materialization); the
        # Bp axis is vectorized, so scores/probs stay in registers and softmax +
        # weighted-sum never touch HBM.
        #
        # DO_ADD folds the caller's ``prefix_sum = prefix_sum + hidden_states``
        # elementwise add into the last-candidate on-load (saving a separate
        # kernel launch + HBM round-trip); WRITE_PREF then stores that summed
        # prefix once (first pass) so downstream layers reuse it.
        #
        # Two HBM passes over H: the softmax + weighted-sum combine is over the
        # small Bp axis, so probs isn't known until the whole H-reduction is done.
        # The second pass re-reads br/ps -- but for one row that footprint is only
        # ~Bp*H*2B (L2-resident), so the reload is served from cache, not HBM.
        # (Holding the [BP, H] tile in registers to avoid the reload was measured
        # slower: it blows the VGPR file and collapses occupancy.) num_stages
        # pipelines each pass so the next chunk's load overlaps the current
        # reduce -- the only win at small T where occupancy alone can't hide it.
        t = tl.program_id(0)
        b_idx = tl.arange(0, BP)
        b_mask = b_idx < Bp
        is_last = b_idx == B  # prefix_sum candidate
        br_base = t * stride_br_t + b_idx * stride_br_b  # [BP]
        ps_base = t * stride_ps_t

        acc_sq = tl.zeros((BP,), dtype=tl.float32)
        acc_dot = tl.zeros((BP,), dtype=tl.float32)
        for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(
                br_ptr + br_base[:, None] + cols[None, :],
                mask=(b_idx < B)[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(
                tl.float32
            )  # [BLOCK_H]
            if DO_ADD:
                ps += tl.load(
                    hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0
                ).to(tl.float32)
            if WRITE_PREF:
                tl.store(
                    pref_ptr + t * stride_pref_t + cols,
                    ps.to(pref_ptr.dtype.element_ty),
                    mask=h_mask,
                )
            v = tl.where(
                is_last[:, None], ps[None, :], br
            )  # [BP, BLOCK_H], ps broadcast in-reg
            # score_weight = norm_weight * proj_weight, precomputed at load time
            sw = tl.load(sw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
            acc_sq += tl.sum(v * v, axis=1)  # [BP]
            acc_dot += tl.sum(v * sw[None, :], axis=1)  # [BP]

        rstd = 1.0 / tl.sqrt(acc_sq / H + eps)
        scores = tl.where(b_mask, rstd * acc_dot, float("-inf"))
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=0)  # [BP], softmax over Bp

        for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(
                br_ptr + br_base[:, None] + cols[None, :],
                mask=(b_idx < B)[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
            if DO_ADD:
                ps += tl.load(
                    hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0
                ).to(tl.float32)
            v = tl.where(is_last[:, None], ps[None, :], br)
            out = tl.sum(probs[:, None] * v, axis=0)  # [BLOCK_H]
            tl.store(
                y_ptr + t * stride_yt + cols,
                out.to(y_ptr.dtype.element_ty),
                mask=h_mask,
            )

    @triton.jit
    def _attn_res_reduce_kernel(
        br_ptr,
        ps_ptr,
        sw_ptr,
        psq_ptr,
        pdot_ptr,
        hs_ptr,
        pref_ptr,
        B,
        H,
        S,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_o_t,
        stride_o_s,
        stride_hs_t,
        stride_pref_t,
        BP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NS: tl.constexpr,
        DO_ADD: tl.constexpr,
        WRITE_PREF: tl.constexpr,
    ):
        # Split-H stage 1, grid=(T, S): S workgroups cooperate on one row t, each
        # owning a block-cyclic subset of the H-chunks. They emit PARTIAL sums
        # psq/pdot[t, s] -- no softmax here, the reduction axis (H) is orthogonal
        # to the softmax axis (Bp), so we stop strictly before the softmax fence.
        # This multiplies the grid by S to fill the GPU at small T (where the
        # grid=(T,) kernel launches too few workgroups to reach full occupancy).
        #
        # DO_ADD folds prefix += hidden_states on-load; each (t, s) owns a disjoint
        # block-cyclic slice of H, so WRITE_PREF here stores that slice of the
        # summed prefix with no overlap -- together the S programs cover all of H.
        t = tl.program_id(0)
        s = tl.program_id(1)
        b_idx = tl.arange(0, BP)
        is_last = b_idx == B
        br_base = t * stride_br_t + b_idx * stride_br_b
        ps_base = t * stride_ps_t
        acc_sq = tl.zeros((BP,), dtype=tl.float32)
        acc_dot = tl.zeros((BP,), dtype=tl.float32)
        for h0 in tl.range(s * BLOCK_H, H, S * BLOCK_H, num_stages=NS):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(
                br_ptr + br_base[:, None] + cols[None, :],
                mask=(b_idx < B)[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
            if DO_ADD:
                ps += tl.load(
                    hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0
                ).to(tl.float32)
            if WRITE_PREF:
                tl.store(
                    pref_ptr + t * stride_pref_t + cols,
                    ps.to(pref_ptr.dtype.element_ty),
                    mask=h_mask,
                )
            v = tl.where(is_last[:, None], ps[None, :], br)
            sw = tl.load(sw_ptr + cols, mask=h_mask, other=0.0).to(tl.float32)
            acc_sq += tl.sum(v * v, axis=1)
            acc_dot += tl.sum(v * sw[None, :], axis=1)
        o = t * stride_o_t + s * stride_o_s + b_idx
        tl.store(psq_ptr + o, acc_sq)
        tl.store(pdot_ptr + o, acc_dot)

    @triton.jit
    def _attn_res_combine_kernel(
        br_ptr,
        ps_ptr,
        psq_ptr,
        pdot_ptr,
        y_ptr,
        hs_ptr,
        B,
        Bp,
        H,
        S,
        eps,
        stride_br_t,
        stride_br_b,
        stride_ps_t,
        stride_i_t,
        stride_i_s,
        stride_yt,
        stride_hs_t,
        BP: tl.constexpr,
        BLOCK_H: tl.constexpr,
        NS: tl.constexpr,
        DO_ADD: tl.constexpr,
    ):
        # Split-H stage 2, grid=(T,): sum the S partials back to the full
        # H-reduction (associative, so exact), then the identical softmax +
        # weighted-sum tail as the single-kernel path.
        t = tl.program_id(0)
        b_idx = tl.arange(0, BP)
        b_mask = b_idx < Bp
        is_last = b_idx == B
        acc_sq = tl.zeros((BP,), dtype=tl.float32)
        acc_dot = tl.zeros((BP,), dtype=tl.float32)
        for s in range(S):
            o = t * stride_i_t + s * stride_i_s + b_idx
            acc_sq += tl.load(psq_ptr + o)
            acc_dot += tl.load(pdot_ptr + o)
        rstd = 1.0 / tl.sqrt(acc_sq / H + eps)
        scores = tl.where(b_mask, rstd * acc_dot, float("-inf"))
        scores = scores - tl.max(scores, axis=0)
        probs = tl.exp(scores)
        probs = probs / tl.sum(probs, axis=0)
        br_base = t * stride_br_t + b_idx * stride_br_b
        ps_base = t * stride_ps_t
        for h0 in tl.range(0, H, BLOCK_H, num_stages=NS):
            cols = h0 + tl.arange(0, BLOCK_H)
            h_mask = cols < H
            br = tl.load(
                br_ptr + br_base[:, None] + cols[None, :],
                mask=(b_idx < B)[:, None] & h_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            ps = tl.load(ps_ptr + ps_base + cols, mask=h_mask, other=0.0).to(tl.float32)
            if DO_ADD:
                ps += tl.load(
                    hs_ptr + t * stride_hs_t + cols, mask=h_mask, other=0.0
                ).to(tl.float32)
            v = tl.where(is_last[:, None], ps[None, :], br)
            out = tl.sum(probs[:, None] * v, axis=0)
            tl.store(
                y_ptr + t * stride_yt + cols,
                out.to(y_ptr.dtype.element_ty),
                mask=h_mask,
            )


# Per-token-count tuning for apply_attn_res at H=7168 (gfx1250). Each bucket maps
# an upper-bound token count to (split, S, num_stages, num_warps):
#   split=True  -> two-kernel split-H (S workgroups per row); wins at small T by
#                  filling the GPU that grid=(T,) leaves idle (~1.3-1.55x).
#   split=False -> single grid=(T,) two-pass kernel with num_stages pipelining;
#                  wins once T alone saturates the machine (the split-H partial
#                  round-trip then becomes pure overhead).
# Dispatch rounds T UP to the smallest bucket >= T (ceil-to-bucket), matching how
# CUDAGraph captures a handful of fixed batch sizes; T above the largest bucket
# falls through to the catch-all two-pass path.
_ATTN_RES_CONFIGS = (
    # (max_tokens, split, S, num_stages, num_warps)
    (8, True, 7, 1, 2),
    (16, True, 7, 1, 4),
    (32, True, 7, 1, 4),
    (64, True, 7, 1, 4),
    (128, True, 6, 1, 4),
    (256, False, 1, 2, 4),
)
_ATTN_RES_CATCHALL = (False, 1, 2, 4)  # T > largest bucket
_ATTN_RES_BLOCK_H = 1024


def _pick_attn_res_config(tokens: int):
    for max_tokens, split, s, ns, nw in _ATTN_RES_CONFIGS:
        if tokens <= max_tokens:
            return split, s, ns, nw
    return _ATTN_RES_CATCHALL


def _apply_attn_res_impl(
    prefix_sum: torch.Tensor,  # [T, H]
    block_residual: torch.Tensor,  # [T, B, H]
    score_weight: torch.Tensor,  # [H] (norm_weight * proj_weight, precomputed)
    eps: float,
    add_hidden: torch.Tensor | None = None,  # [T, H], folded: prefix += add_hidden
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-residual soft-attention mix: rmsnorm each of the B+1 candidates,
    score = <normed, score_weight>, softmax over B+1, weighted sum.

    ``score_weight`` folds ``norm_weight * proj_weight`` once at load time (both
    are static loaded weights), so the kernel loads a single vector instead of
    two and drops the per-forward multiply.

    When ``add_hidden`` is given, the caller's ``prefix_sum = prefix_sum +
    hidden_states`` elementwise add is folded into the kernel (added on-load to
    the prefix candidate, one HBM read + one launch saved) and the summed prefix
    is written back. Returns ``(y, prefix_out)`` where ``prefix_out`` is the
    summed prefix (or the unchanged ``prefix_sum`` when ``add_hidden`` is None),
    so downstream layers reuse it.

    Dispatches by token count (see ``_ATTN_RES_CONFIGS``): split-H at small T to
    fill the GPU, the single-pass pipelined kernel once T saturates it."""
    T, B, H = block_residual.shape
    Bp = B + 1
    do_add = add_hidden is not None
    br = block_residual.contiguous()
    ps = prefix_sum.contiguous()
    sw = score_weight.contiguous()
    y = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    # hs/pref pointers are always passed (triton needs a tensor); when not adding
    # they alias ps and are never dereferenced (DO_ADD / WRITE_PREF are False).
    if do_add:
        hs = add_hidden.contiguous()
        pref = torch.empty((T, H), device=block_residual.device, dtype=prefix_sum.dtype)
    else:
        hs = ps
        pref = ps
    BP = triton.next_power_of_2(Bp)
    BLOCK_H = _ATTN_RES_BLOCK_H
    nchunk = triton.cdiv(H, BLOCK_H)

    split, s, ns, nw = _pick_attn_res_config(T)
    # S can't exceed the chunk count (a workgroup with no chunk to own is wasted).
    s = min(s, nchunk)
    if split and s > 1:
        psq = torch.empty((T, s, BP), device=br.device, dtype=torch.float32)
        pdot = torch.empty((T, s, BP), device=br.device, dtype=torch.float32)
        _attn_res_reduce_kernel[(T, s)](
            br,
            ps,
            sw,
            psq,
            pdot,
            hs,
            pref,
            B,
            H,
            s,
            br.stride(0),
            br.stride(1),
            ps.stride(0),
            psq.stride(0),
            psq.stride(1),
            hs.stride(0),
            pref.stride(0),
            BP=BP,
            BLOCK_H=BLOCK_H,
            NS=ns,
            num_warps=nw,
            DO_ADD=do_add,
            WRITE_PREF=do_add,
        )
        _attn_res_combine_kernel[(T,)](
            br,
            ps,
            psq,
            pdot,
            y,
            hs,
            B,
            Bp,
            H,
            s,
            eps,
            br.stride(0),
            br.stride(1),
            ps.stride(0),
            psq.stride(0),
            psq.stride(1),
            y.stride(0),
            hs.stride(0),
            BP=BP,
            BLOCK_H=BLOCK_H,
            NS=ns,
            num_warps=nw,
            DO_ADD=do_add,
        )
        return y, (pref if do_add else prefix_sum)

    _attn_res_fused_kernel[(T,)](
        br,
        ps,
        sw,
        y,
        hs,
        pref,
        B,
        Bp,
        H,
        float(eps),
        br.stride(0),
        br.stride(1),
        ps.stride(0),
        y.stride(0),
        hs.stride(0),
        pref.stride(0),
        BP=BP,
        BLOCK_H=BLOCK_H,
        NS=ns,
        num_warps=nw,
        DO_ADD=do_add,
        WRITE_PREF=do_add,
    )
    return y, (pref if do_add else prefix_sum)


@torch.library.custom_op("atom::apply_attn_res", mutates_args=(), device_types="cuda")
def _apply_attn_res_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mixed_output, _ = _apply_attn_res_impl(
        prefix_sum, block_residual, score_weight, eps
    )
    return mixed_output


@_apply_attn_res_op.register_fake
def _apply_attn_res_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.empty_like(prefix_sum)


@torch.library.custom_op(
    "atom::apply_attn_res_add", mutates_args=(), device_types="cuda"
)
def _apply_attn_res_add_op(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _apply_attn_res_impl(
        prefix_sum, block_residual, score_weight, eps, add_hidden
    )


@_apply_attn_res_add_op.register_fake
def _apply_attn_res_add_op_fake(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(prefix_sum), torch.empty_like(prefix_sum)


def apply_attn_res(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float,
    add_hidden: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch an opaque custom op whose CUDA implementation selects by concrete T."""
    if add_hidden is None:
        return (
            _apply_attn_res_op(prefix_sum, block_residual, score_weight, eps),
            prefix_sum,
        )
    return _apply_attn_res_add_op(
        prefix_sum, block_residual, score_weight, eps, add_hidden
    )


def situ_and_mul(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    """SiTUv2 gated activation over the last dim (x[..., :D] gate, x[..., D:] up)."""
    *lead, two_d = x.shape
    assert two_d % 2 == 0
    d = two_d // 2
    x2 = x.reshape(-1, two_d)
    m = x2.shape[0]
    y = torch.empty((m, d), dtype=x.dtype, device=x.device)
    if not _HAS_TRITON or m == 0:
        return _situ_and_mul_torch(x, beta, linear_beta)
    BLOCK = 1024
    grid = (m, triton.cdiv(d, BLOCK))
    has_linear = linear_beta is not None
    _situ_and_mul_kernel[grid](
        x2,
        y,
        m,
        d,
        x2.stride(0),
        y.stride(0),
        float(beta),
        1.0 / float(beta),
        float(linear_beta) if has_linear else 0.0,
        (1.0 / float(linear_beta)) if has_linear else 0.0,
        HAS_LINEAR=has_linear,
        BLOCK=BLOCK,
    )
    return y.reshape(*lead, d)


def rmsnorm_gated(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    """rmsnorm(x) over last dim * weight * sigmoid(gate).

    ``gate`` may be strided (e.g. a column slice of a fused GEMM output): the
    kernel reads it via (outer, head) strides so no contiguous copy is needed.
    ``x`` is normed row-wise and is made contiguous (cheap; the caller's ``out``
    already is). Supports a 2D ``[M, H]`` or 3D ``[outer, heads, H]`` gate.
    """
    h = x.shape[-1]
    x2 = x.reshape(-1, h)
    m = x2.shape[0]
    if not _HAS_TRITON or m == 0 or h > 8192:
        return _rmsnorm_gated_torch(x, weight, gate, eps)
    if gate.ndim == 3:
        heads = gate.shape[1]
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        # 2D: one logical head per row; the head term drops out (row % 1 == 0).
        heads = 1
        stride_g_outer, stride_g_head = gate.stride(0), 0
    x2 = x2.contiguous()
    y = torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(h)
    _rmsnorm_gated_kernel[(m,)](
        x2,
        weight,
        gate,
        y,
        h,
        float(eps),
        x2.stride(0),
        y.stride(0),
        stride_g_outer,
        stride_g_head,
        HEADS=heads,
        BLOCK=BLOCK,
    )
    return y.reshape_as(x)


# --------------------------------------------------------------------------- #
# torch references (also the fallback when triton is unavailable)
# --------------------------------------------------------------------------- #
def _situ_and_mul_torch(
    x: torch.Tensor, beta: float, linear_beta: float | None
) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate_f = gate.float()
    up_f = up.float()
    out = beta * torch.tanh(gate_f / beta) * torch.sigmoid(gate_f)
    if linear_beta is not None:
        up_f = linear_beta * torch.tanh(up_f / linear_beta)
    return (out * up_f).to(x.dtype)


def _rmsnorm_gated_torch(
    x: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor, eps: float
) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float()
    var = x_f.pow(2).mean(dim=-1, keepdim=True)
    xn = x_f * torch.rsqrt(var + eps)
    return (xn.to(dtype) * weight.to(dtype)) * torch.sigmoid(gate)


# --------------------------------------------------------------------------- #
# KDA initial-state gather (masked)                                            #
# --------------------------------------------------------------------------- #
if _HAS_TRITON:

    @triton.jit
    def _gather_kda_state_kernel(
        src_ptr,  # ssm_state viewed as [num_lines, S]
        idx_ptr,  # state_indices [num_seqs]
        mask_ptr,  # has_initial_state [num_seqs] int8 (unused when HAS_MASK=False)
        dst_ptr,  # initial viewed as [num_seqs, S]
        S,
        stride_src,
        stride_dst,
        HAS_MASK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        seq = tl.program_id(0)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        col_mask = offs < S
        if HAS_MASK:
            keep = tl.load(mask_ptr + seq) != 0
            load_mask = col_mask & keep
        else:
            load_mask = col_mask
        # int64 offsets: line * S can exceed int32 for large state caches.
        line = tl.load(idx_ptr + seq).to(tl.int64)
        vals = tl.load(src_ptr + line * stride_src + offs, mask=load_mask, other=0.0)
        tl.store(dst_ptr + seq.to(tl.int64) * stride_dst + offs, vals, mask=col_mask)


def gather_kda_initial_state(
    ssm_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``ssm_state[state_indices]`` into a packed ``[num_seqs, ...]``
    initial-state tensor, zeroing sequences whose ``has_initial_state`` is
    False -- in a single kernel.

    Fuses the ``ssm_state[idx].contiguous()`` gather with the
    ``initial[~has_initial_state] = 0`` masking so fresh sequences are written
    as zeros in the same pass instead of a gather followed by a separate
    zero-write pass. Falls back to the torch path when triton is unavailable or
    there are no sequences.
    """
    num_seqs = int(state_indices.shape[0])
    if not _HAS_TRITON or num_seqs == 0:
        initial = ssm_state[state_indices].contiguous()
        if has_initial_state is not None:
            initial[~has_initial_state] = 0
        return initial
    tail = ssm_state.shape[1:]
    src = ssm_state.reshape(ssm_state.shape[0], -1)
    S = src.shape[1]
    initial = torch.empty(
        (num_seqs, *tail), dtype=ssm_state.dtype, device=ssm_state.device
    )
    dst = initial.reshape(num_seqs, -1)
    has_mask = has_initial_state is not None
    mask_arg = has_initial_state.to(torch.int8) if has_mask else src
    BLOCK = 1024
    grid = (num_seqs, triton.cdiv(S, BLOCK))
    _gather_kda_state_kernel[grid](
        src,
        state_indices,
        mask_arg,
        dst,
        S,
        src.stride(0),
        dst.stride(0),
        HAS_MASK=has_mask,
        BLOCK=BLOCK,
    )
    return initial


def _dual_rmsnorm_fake(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return q.new_empty(q.shape), k.new_empty(k.shape)


@torch_compile_guard(gen_fake=_dual_rmsnorm_fake, mutates_args=[])
def dual_rmsnorm(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    q_eps: float,
    k: torch.Tensor,
    k_weight: torch.Tensor,
    k_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # The AITER kernel honors row strides; K3 split views have inner stride 1.
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    _aiter_fused_qk_rmsnorm_kernel(
        q,
        q_weight,
        q_eps,
        k,
        k_weight,
        k_eps,
        q_out,
        k_out,
    )
    return q_out, k_out
