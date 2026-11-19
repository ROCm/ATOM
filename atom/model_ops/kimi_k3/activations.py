# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused activation and gated RMSNorm operations for Kimi-K3."""

from __future__ import annotations

import functools

import torch
from aiter import QuantType, dtypes, get_hip_quant
from aiter.jit.utils.torch_guard import torch_compile_guard

from atom.utils.decorators import mark_trace

try:
    import triton
    import triton.language as tl

    from atom.model_ops.triton_e8m0_quant import fp8_e8m0_group_scale

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

_RMSNORM_MULTIROW_MIN_ROWS = 43_008
_RMSNORM_MULTIROW_BLOCK_M = 8
_RMSNORM_MULTIROW_LONG_ROWS = 900_000
_RMSNORM_MULTIROW_LONG_BLOCK_M = 32


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
        s_ptr=None,
        stride_s_m=0,
        stride_s_g=0,
        E8M0_SCALE: tl.constexpr = False,
        FP8_MAX: tl.constexpr = 0.0,
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
        if E8M0_SCALE:
            # quantize the bf16 the unfused path stores; groups past D are skipped
            y = y.to(x_ptr.dtype.element_ty).to(tl.float32)
            y = y.reshape(BLOCK // 128, 128)
            code, multiplier = fp8_e8m0_group_scale(tl.max(tl.abs(y), axis=1), FP8_MAX)
            y = y * multiplier[:, None]
            y = tl.clamp(y, -FP8_MAX, FP8_MAX, propagate_nan=tl.PropagateNan.ALL)
            y = y.reshape(BLOCK)
            groups = tl.program_id(1) * (BLOCK // 128) + tl.arange(0, BLOCK // 128)
            tl.store(
                s_ptr + row * stride_s_m + groups * stride_s_g,
                code,
                mask=groups < D // 128,
            )
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
        row_i64 = row.to(tl.int64)
        cols = tl.arange(0, BLOCK)
        mask = cols < H
        # x / y are row-contiguous [M, H]; gate may be strided. Its logical row
        # `row` decomposes into (outer, head) so the token-boundary jump
        # (stride_g_outer) and per-head step (stride_g_head) are read directly,
        # avoiding a contiguous copy of the strided gate slice.
        outer_i64 = (row // HEADS).to(tl.int64)
        head_i64 = (row % HEADS).to(tl.int64)
        g_off = outer_i64 * stride_g_outer + head_i64 * stride_g_head + cols
        x = tl.load(x_ptr + row_i64 * stride_xm + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        var = tl.sum(x * x, axis=0) / H
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd * w) * tl.sigmoid(gate)
        tl.store(
            y_ptr + row_i64 * stride_ym + cols,
            y.to(y_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_gated_multirow_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        y_ptr,
        M,
        H,
        eps,
        stride_xm,
        stride_ym,
        stride_g_outer,
        stride_g_head,
        HEADS: tl.constexpr,
        BLOCK: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = tl.arange(0, BLOCK)
        row_mask = rows < M
        mask = row_mask[:, None] & (cols[None, :] < H)
        # A masked load/store may still form its pointer on ROCm. Clamp padded
        # rows before any address arithmetic; `mask` remains based on the real
        # row ids, so those lanes still perform no memory access.
        safe_rows = tl.where(row_mask, rows, 0)
        safe_rows_i64 = safe_rows.to(tl.int64)
        x_off = safe_rows_i64[:, None] * stride_xm + cols[None, :]
        outer_i64 = (safe_rows // HEADS).to(tl.int64)
        head_i64 = (safe_rows % HEADS).to(tl.int64)
        g_off = (
            outer_i64[:, None] * stride_g_outer
            + head_i64[:, None] * stride_g_head
            + cols[None, :]
        )
        x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / H
        rstd = tl.rsqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        y = (x * rstd[:, None] * w[None, :]) * tl.sigmoid(gate)
        tl.store(
            y_ptr + safe_rows_i64[:, None] * stride_ym + cols[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=mask,
        )

    @triton.jit
    def _rmsnorm_gated_fp8_per_token_kernel(
        x_ptr,
        w_ptr,
        g_ptr,
        y_ptr,
        s_ptr,
        H,
        eps,
        fp8_max,
        stride_xm,
        stride_xh,
        stride_g_outer,
        stride_g_head,
        stride_ym,
        HEADS: tl.constexpr,
        HEADS_POW2: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        tok = tl.program_id(0)
        tok_i64 = tok.to(tl.int64)
        head_ids = tl.arange(0, HEADS_POW2)
        cols = tl.arange(0, BLOCK)
        mask = (head_ids[:, None] < HEADS) & (cols[None, :] < H)  # [HEADS_POW2, BLOCK]
        # Padding heads (head_ids >= HEADS) are masked out on every load/store,
        # but their raw offset (head_ids * stride) can still address past the end
        # of the buffer -- forming an out-of-bounds pointer is UB on ROCm/triton
        # and faults when the allocation abuts an unmapped page. Clamp the head
        # index used for addressing to a valid row; the mask (other=0.0) still
        # discards the value, so numerics are unchanged.
        h_safe = tl.where(head_ids < HEADS, head_ids, 0)
        h_safe_i64 = h_safe.to(tl.int64)
        x_off = tok_i64 * stride_xm + h_safe_i64[:, None] * stride_xh + cols[None, :]
        x = tl.load(x_ptr + x_off, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(x * x, axis=1) / H  # [HEADS]
        rstd = 1.0 / tl.sqrt(var + eps)  # [HEADS]
        w = tl.load(w_ptr + cols, mask=cols < H, other=0.0).to(tl.float32)  # [BLOCK]
        g_off = (
            tok_i64 * stride_g_outer
            + h_safe_i64[:, None] * stride_g_head
            + cols[None, :]
        )
        gate = tl.load(g_ptr + g_off, mask=mask, other=0.0).to(tl.float32)
        normed = (x * rstd[:, None] * w[None, :]) * tl.sigmoid(gate)  # [HEADS, BLOCK]
        amax = tl.max(tl.abs(normed))  # scalar per token
        scale = amax / fp8_max
        inv = tl.where(scale > 0.0, 1.0 / scale, 0.0)
        q = normed * inv
        q = tl.minimum(tl.maximum(q, -fp8_max), fp8_max)
        y_off = tok_i64 * stride_ym + h_safe_i64[:, None] * H + cols[None, :]
        tl.store(y_ptr + y_off, q.to(y_ptr.dtype.element_ty), mask=mask)
        tl.store(s_ptr + tok, scale)


@mark_trace
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
    _launch_situ_and_mul(x2, y, beta, linear_beta)
    return y.reshape(*lead, d)


def _launch_situ_and_mul(x2, y, beta, linear_beta, **e8m0_args) -> None:
    """_situ_and_mul_kernel over [m, 2*d] ``x2`` into [m, d] ``y``."""
    m, d = y.shape
    BLOCK = 1024
    has_linear = linear_beta is not None
    _situ_and_mul_kernel[(m, triton.cdiv(d, BLOCK))](
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
        **e8m0_args,
    )


def _situ_and_mul_quant_fake(
    x: torch.Tensor, beta: float, linear_beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    *lead, two_d = x.shape
    m = 1
    for s in lead:
        m *= s
    d = two_d // 2
    out = torch.empty((m, d), dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    return out, scale


# mutates_args=[] -- out/scale are allocated inside, so the op is functional and
# torch.compile treats it as opaque via the fake above (mirrors
# atom.model_ops.activation.mxfp4_act_mul_quant_fuse).
@torch_compile_guard(gen_fake=_situ_and_mul_quant_fake, mutates_args=[])
def situ_and_mul_quant(
    x: torch.Tensor, beta: float, linear_beta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """SiTUv2 gated activation fused with per-token FP8 quant.

    ``x`` is ``[m, 2*d]`` bf16 (``x[..., :d]`` gate, ``x[..., d:]`` up). Returns
    ``(fp8 [m, d], scale [m, 1] float32)`` ready for the consuming a8w8 per-token
    GEMM's ``x_scale=`` path (dense-MLP / shared-expert down_proj under ptpc_fp8).
    The aiter kernel folds SiTUv2 and the amax/quant into one pass, so the bf16
    activation never round-trips through HBM. ``linear_beta`` must be set: the
    kernel always applies the linear-beta tanh to the up half.
    """
    from aiter.ops.activation import situv2_and_mul_quant as _aiter_situ_quant

    assert linear_beta is not None, "situ_and_mul_quant requires linear_beta"
    two_d = x.shape[-1]
    d = two_d // 2
    x2 = x.reshape(-1, two_d)
    m = x2.shape[0]
    out = torch.empty((m, d), dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m > 0:
        _aiter_situ_quant(
            out, x2.contiguous(), scale, d, float(beta), float(linear_beta)
        )
    return out, scale


def _situ_and_mul_fp8_e8m0_quant_fake(
    x: torch.Tensor, beta: float, linear_beta: float | None, transpose_scale: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    m, d = x.numel() // x.shape[-1], x.shape[-1] // 2
    out = torch.empty((m, d), dtype=dtypes.fp8, device=x.device)
    scale = torch.empty((m, d // 128), dtype=dtypes.fp8_e8m0, device=x.device)
    return out, scale


# Functional, like per_1x128_e8m0_quant: Inductor cannot functionalize an E8M0
# mutation.
@torch_compile_guard(gen_fake=_situ_and_mul_fp8_e8m0_quant_fake, mutates_args=[])
def situ_and_mul_fp8_e8m0_quant(
    x: torch.Tensor, beta: float, linear_beta: float | None, transpose_scale: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """``per_1x128_e8m0_quant(situ_and_mul(x, ...))`` in one kernel; d % 128 == 0."""
    x2 = x.reshape(-1, x.shape[-1])
    m, d = x2.shape[0], x2.shape[1] // 2
    assert d % 128 == 0, f"per-1x128 groups need d % 128 == 0, got d={d}"
    out = torch.empty((m, d), dtype=dtypes.fp8, device=x.device)
    # uint8, viewed as E8M0 below; transpose_scale: column-major, as aiter's
    scale = torch.empty(
        (d // 128, m) if transpose_scale else (m, d // 128),
        dtype=torch.uint8,
        device=x.device,
    )
    if m:
        _launch_situ_and_mul(
            x2,
            out,
            beta,
            linear_beta,
            s_ptr=scale,
            stride_s_m=1 if transpose_scale else d // 128,
            stride_s_g=m if transpose_scale else 1,
            E8M0_SCALE=True,
            FP8_MAX=float(torch.finfo(dtypes.fp8).max),
        )
    return out, scale.view(m, d // 128).view(dtypes.fp8_e8m0)


def situ_and_mul_maybe_quant(
    x: torch.Tensor,
    beta: float,
    linear_beta: float | None,
    quant: bool = False,
    e8m0_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SiTUv2 gated activation, optionally fused with down_proj's FP8 quant.

    Unifies the quant and non-quant paths behind one call site so the caller
    always does ``down_proj(x, x_scale=scale)``:

    - ``quant=False``: return ``(bf16 [..., d], None)``. The ``None`` scale makes
      ``down_proj(x, x_scale=None)`` fall back to its own standalone activation
      quant.
    - ``quant=True``: fuse SiTUv2 + per-token FP8 quant into one aiter kernel and
      return ``(fp8 [m, d], scale [m, 1])`` for down_proj's ``x_scale=`` path
      (dense-MLP / shared-expert under ptpc_fp8).
    - ``e8m0_scale``: per-1x128 FP8 with E8M0 scales instead, from ATOM's kernel
      (aiter's ``situv2_and_mul_quant`` is per-token only).
    """
    if quant and e8m0_scale:
        from atom.utils import envs

        return situ_and_mul_fp8_e8m0_quant(
            x, beta, linear_beta, envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        )
    if quant:
        return situ_and_mul_quant(x, beta, linear_beta)
    return situ_and_mul(x, beta, linear_beta), None


@functools.cache
def _onorm_gluon_module():
    """The gfx1250 Gluon o_norm + E8M0 quant kernel module, or None elsewhere."""
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        if get_arch() != "gfx1250":
            return None
        from atom.model_ops.kimi_k3 import rmsnorm_gated_gluon
    except ImportError:
        return None
    return rmsnorm_gated_gluon


@mark_trace
def rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    quant_type: QuantType | None = None,
    quant_dtype: torch.dtype | None = None,
    is_prefill: bool = False,
    e8m0_scale: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """rmsnorm(x) over last dim * weight * sigmoid(gate).

    When ``(quant_type, quant_dtype)`` is the per-token FP8 scheme, the normed
    output is also quantized and the function returns
    ``(fp8 [t, heads*H], scale [t, 1])`` ready for the consuming GEMM's
    ``x_scale=`` path. With no quant (``None``/``QuantType.No``) it returns the
    bf16 tensor shaped like ``x``. Per-token FP8 is the only fused scheme
    besides ``e8m0_scale`` (per-1x128 FP8, one group per 128-wide head:
    ``(fp8 [t, heads*H], E8M0 scale [t, heads])``, bit-identical to
    ``per_1x128_e8m0_quant`` of the bf16 output); any other scheme asserts.

    ``gate`` may be strided (e.g. a column slice of a fused GEMM output): the
    kernel reads it via (outer, head) strides so no contiguous copy is needed.
    ``x`` is normed row-wise and is made contiguous (cheap; the caller's ``out``
    already is). Supports a 2D ``[M, H]`` or 3D ``[outer, heads, H]`` gate.
    """
    if quant_type == QuantType.per_Token and quant_dtype == dtypes.fp8:
        return _rmsnorm_gated_per_token_quant(x, weight, gate, eps, quant_dtype)
    if quant_type == QuantType.per_1x128 and quant_dtype == dtypes.fp8 and e8m0_scale:
        assert x.ndim == 3 and gate.ndim == 3 and x.shape[-1] == 128, (
            x.shape,
            gate.shape,
        )
        from atom.utils import envs

        col_major = envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE
        gluon = None if is_prefill else _onorm_gluon_module()
        if gluon is not None:
            return gluon.rmsnorm_gated_e8m0(
                x, weight, gate, eps, quant_dtype, col_major
            )
        # Prefill (the multirow kernel) and other archs quantize separately.
        from atom.model_ops.linear import per_1x128_e8m0_quant

        y = rmsnorm_gated(x, weight, gate, eps, is_prefill=is_prefill)
        return per_1x128_e8m0_quant(y.view(x.shape[0], -1), quant_dtype, col_major)
    # Only the no-quant (bf16) path remains. Any other requested scheme is
    # unsupported here -- fail loud rather than silently feed bf16 activations to
    # a GEMM that expects quantized input.
    assert quant_type in (None, QuantType.No), (
        "rmsnorm_gated only fuses per-token FP8 quant; got "
        f"quant_type={quant_type}, quant_dtype={quant_dtype}"
    )
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
    # BF16 prefill keeps the same per-head reduction layout at every batch
    # size. Switching between the one-row and multirow layouts changes FP32
    # variance reduction and sqrt rounding; a BF16 midpoint can then change
    # downstream expert selection after a prefix is resumed in a smaller batch.
    # Reuse the existing eight-row kernel, including for short prefills.
    # Other dtypes retain the measured flattened-row (tokens * heads) cutoffs.
    stable_bf16 = x.dtype == torch.bfloat16
    if is_prefill and h == 128 and (stable_bf16 or m >= _RMSNORM_MULTIROW_MIN_ROWS):
        is_long = not stable_bf16 and m >= _RMSNORM_MULTIROW_LONG_ROWS
        block_m = (
            _RMSNORM_MULTIROW_LONG_BLOCK_M if is_long else _RMSNORM_MULTIROW_BLOCK_M
        )
        _rmsnorm_gated_multirow_kernel[(triton.cdiv(m, block_m),)](
            x2,
            weight,
            gate,
            y,
            m,
            h,
            float(eps),
            x2.stride(0),
            y.stride(0),
            stride_g_outer,
            stride_g_head,
            HEADS=heads,
            BLOCK=BLOCK,
            BLOCK_M=block_m,
            num_warps=2 if is_long else 1,
            waves_per_eu=4,
        )
        return y.reshape_as(x)
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


def _rmsnorm_gated_per_token_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor,
    eps: float,
    quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head sigmoid-gated RMSNorm fused to per-token quant.

    Same math as ``rmsnorm_gated`` (rmsnorm(x) over the last dim * weight *
    sigmoid(gate)), but instead of a bf16 output it emits a ``quant_dtype`` tensor
    plus one per-token scale (amax / dtype_max) over the flattened
    ``heads * head_dim`` row, ready for o_proj's per-token a8w8 GEMM. ``gate`` may
    be strided (a column slice of the fused in_proj output).

    Returns ``(out [t, heads*head_dim], scale [t, 1] float32)``.
    """
    assert x.ndim == 3, f"expected [t, heads, head_dim], got {tuple(x.shape)}"
    t, heads, H = x.shape
    fp8_max = float(torch.finfo(quant_dtype).max)
    if not _HAS_TRITON or t == 0 or H > 8192:
        normed = _rmsnorm_gated_torch(x, weight, gate, eps).reshape(t, heads * H)
        return get_hip_quant(QuantType.per_Token)(normed, quant_dtype=quant_dtype)
    x = x.contiguous()
    out = torch.empty((t, heads * H), dtype=quant_dtype, device=x.device)
    scale = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    if gate.ndim == 3:
        stride_g_outer, stride_g_head = gate.stride(0), gate.stride(1)
    else:
        # 2D [t, heads*H]: one logical head per row; head term drops out.
        stride_g_outer, stride_g_head = gate.stride(0), 0
    BLOCK = triton.next_power_of_2(H)
    _rmsnorm_gated_fp8_per_token_kernel[(t,)](
        x,
        weight,
        gate,
        out,
        scale,
        H,
        float(eps),
        fp8_max,
        x.stride(0),
        x.stride(1),
        stride_g_outer,
        stride_g_head,
        out.stride(0),
        HEADS=heads,
        HEADS_POW2=triton.next_power_of_2(heads),
        BLOCK=BLOCK,
    )
    return out, scale


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
