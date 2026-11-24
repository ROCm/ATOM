# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""RMSNorm with Triton auto-fallback and fused quantization for all 6 scaling modes.

All backends are AITER implementations — no torch.nn.functional fallbacks.

Supported quantization modes (fused into norm forward where possible):
    - ``delayed``    — fused per-tensor static quant (Triton)
    - ``dynamic``    — fused per-token dynamic quant (Triton)
    - ``blockwise``  — fused per-block group quant (Triton)
    - ``per_token``  — fused per-token dynamic quant (Triton)
    - ``mxfp8``      — norm then standalone MXFP8 conversion (unfused)
    - ``none``       — no quantization

Backward is always unquantized and handled by autograd.

Provides:
    - :func:`rmsnorm` — functional API  (autograd-aware, no quant fusion)
    - :func:`rmsnorm_with_quant` — functional API with fused quantization
    - :class:`LumenRMSNorm` — ``nn.Module`` API
"""

import logging
import os

import torch
from torch import nn

from atomrl.lumen.core.grad_quant import quantize_grad_tensor
from atomrl.lumen.kernels.rmsnorm import (
    NARROW_RMSNORM_MAX_N,
    narrow_rmsnorm_backward,
    narrow_rmsnorm_forward,
)
from atomrl.lumen.ops.dispatch import Backend, build_fallback_chain, try_backends


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape to 2D, only calling .contiguous() when necessary."""
    x_2d = x.reshape(-1, x.shape[-1])
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports — each backend guarded by availability
# ---------------------------------------------------------------------------


def _get_triton_rms_norm():
    from aiter.ops.triton.normalization.rmsnorm import rms_norm

    return rms_norm


def _get_triton_rmsnorm2d_fwd_with_dynamicquant():
    from aiter.ops.triton.normalization.rmsnorm import rmsnorm2d_fwd_with_dynamicquant

    return rmsnorm2d_fwd_with_dynamicquant


def _get_fused_rms_fp8_per_tensor_static_quant():
    from aiter.ops.triton.quant.fused_fp8_quant import (
        fused_rms_fp8_per_tensor_static_quant,
    )

    return fused_rms_fp8_per_tensor_static_quant


def _get_fused_rms_fp8_group_quant():
    from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant

    return fused_rms_fp8_group_quant


def _get_triton_per_tensor_quant():
    from aiter.ops.quant import per_tensor_quant_triton

    return per_tensor_quant_triton


def _get_triton_per_token_quant():
    from aiter.ops.quant import pertoken_quant

    return pertoken_quant


def _get_triton_fused_add_rmsnorm_pad():
    from aiter.ops.triton.normalization.fused_add_rmsnorm_pad import (
        fused_add_rmsnorm_pad,
    )

    return fused_add_rmsnorm_pad


# ---------------------------------------------------------------------------
# Unquantized RMSNorm with fallback (Triton via AITER)
# ---------------------------------------------------------------------------


def _rmsnorm_triton(x_2d, weight, eps):
    fn = _get_triton_rms_norm()
    return fn(x_2d, weight, eps)


_num_sms_cached = None


def _num_sms(device) -> int:
    global _num_sms_cached
    if _num_sms_cached is None:
        _num_sms_cached = torch.cuda.get_device_properties(device).multi_processor_count
    return _num_sms_cached


def _rows_are_short(x_2d: torch.Tensor, weight: torch.Tensor) -> bool:
    """True when the row is too short to fill a wavefront.

    AITER specialises its forward for this case but not its backward, so per-head
    norms (``N = head_dim``) spend most of the backward with idle lanes.
    """
    return (
        x_2d.shape[-1] <= NARROW_RMSNORM_MAX_N
        and x_2d.is_cuda
        and weight.dim() == 1
        and weight.is_contiguous()
    )


class _NarrowRMSNorm(torch.autograd.Function):
    """RMSNorm over short rows, using Lumen's row-tiling kernels."""

    @staticmethod
    def forward(ctx, x_2d, weight, eps):
        y, rsigma = narrow_rmsnorm_forward(x_2d, weight, eps)
        ctx.save_for_backward(x_2d, weight, rsigma)
        return y

    @staticmethod
    def backward(ctx, dy):
        x_2d, weight, rsigma = ctx.saved_tensors
        if not dy.is_contiguous():
            dy = dy.contiguous()
        dx, dw = narrow_rmsnorm_backward(
            dy, x_2d, weight, rsigma, _num_sms(x_2d.device)
        )
        return dx, dw, None


def _build_rmsnorm_chain():
    return build_fallback_chain({Backend.TRITON: _rmsnorm_triton})


_rmsnorm_chain = None


def _get_rmsnorm_chain():
    global _rmsnorm_chain
    if _rmsnorm_chain is None:
        _rmsnorm_chain = _build_rmsnorm_chain()
    return _rmsnorm_chain


# ---------------------------------------------------------------------------
# Grad-quantized wrapper (autograd Function)
# ---------------------------------------------------------------------------


class _RMSNormGradQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, grad_quant_type):
        ctx.grad_quant_type = grad_quant_type
        ctx.eps = eps
        ctx.save_for_backward(x, weight)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        y = try_backends(_get_rmsnorm_chain(), x_2d, weight, eps, op_name="rmsnorm")
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        gqt = ctx.grad_quant_type

        with torch.enable_grad():
            x_detached = x.detach().requires_grad_(True)
            w_detached = weight.detach().requires_grad_(True)
            orig_shape = x_detached.shape
            x_2d = x_detached.reshape(-1, x_detached.shape[-1])
            y = _rmsnorm_triton(x_2d, w_detached, ctx.eps)
            y = y.reshape(orig_shape)
            torch.autograd.backward(y, grad_output)

        dx = quantize_grad_tensor(x_detached.grad, gqt)
        dw = quantize_grad_tensor(w_detached.grad, gqt)
        return dx, dw, None, None


# ---------------------------------------------------------------------------
# Public functional API — unquantized
# ---------------------------------------------------------------------------


_USE_APEX_RMSNORM = os.environ.get("LUMEN_USE_APEX_RMSNORM", "0") == "1"


def _rmsnorm_apex(x_2d, weight, eps):
    """Apex fused RMSNorm — matches AMD MLPerf TE reference."""
    from apex.normalization.fused_layer_norm import fused_rms_norm_affine

    return fused_rms_norm_affine(x_2d, weight, weight.shape, eps, False)


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    grad_quant_type: str | None = None,
) -> torch.Tensor:
    """Apply RMSNorm with automatic backend fallback (Triton via AITER).

    When any input requires grad, the Triton backend is preferred because
    its ``_RMSNorm`` autograd.Function provides forward+backward support.

    Set ``LUMEN_USE_APEX_RMSNORM=1`` to use apex's ``fused_rms_norm_affine``
    instead of AITER, matching the AMD MLPerf reference (TE + apex).

    Args:
        x: Input tensor ``(*, hidden_size)``.
        weight: Learnable scale ``(hidden_size,)``.
        eps: Epsilon for numerical stability.
        grad_quant_type: Gradient quantization format.

    Returns:
        Normalised tensor with same shape as *x*.
    """
    if grad_quant_type is not None:
        return _RMSNormGradQuant.apply(x, weight, eps, grad_quant_type)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    needs_grad = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)

    if _USE_APEX_RMSNORM:
        y = _rmsnorm_apex(x_2d, weight, eps)
    elif needs_grad and _rows_are_short(x_2d, weight):
        y = _NarrowRMSNorm.apply(_to_2d(x), weight, eps)
    elif needs_grad:
        y = _rmsnorm_triton(x_2d, weight, eps)
    else:
        y = try_backends(_get_rmsnorm_chain(), x_2d, weight, eps, op_name="rmsnorm")
    return y.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Fused residual-add + RMSNorm  (Triton → unfused fallback)
# ---------------------------------------------------------------------------


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual add + RMSNorm: ``residual = x + residual; y = RMSNorm(residual)``.

    Dispatches to AITER Triton ``fused_add_rmsnorm_pad`` (allocates output)
    when the probe succeeds, then falls back to unfused ``add`` + :func:`rmsnorm`.

    Args:
        x: Input tensor ``(*, hidden_size)``.
        residual: Residual tensor (same shape as *x*).
        weight: RMSNorm scale ``(hidden_size,)``.
        eps: Epsilon for numerical stability.

    Returns:
        ``(normed, residual_out)`` — the normalised output and the
        updated residual stream.
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)
    res_2d = _to_2d(residual)
    _catchable = (RuntimeError, NotImplementedError, TypeError, ValueError)

    try:
        fn = _get_triton_fused_add_rmsnorm_pad()
        normed, res_out = fn(x_2d, weight, eps, res=res_2d)
        return normed.reshape(orig_shape), res_out.reshape(orig_shape)
    except _catchable as e:
        logger.warning("fused_add_rmsnorm: Triton pad failed (%s), falling back", e)

    res_out = x + residual
    normed = rmsnorm(res_out, weight, eps)
    return normed, res_out


# ---------------------------------------------------------------------------
# Fused RMSNorm + Quantization for each scaling mode
# ---------------------------------------------------------------------------


def rmsnorm_delayed_per_tensor(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + delayed per-tensor FP8 quant (fused Triton kernel).

    Uses ``fused_rms_fp8_per_tensor_static_quant`` which applies a
    pre-computed scale (from amax history).

    Returns:
        ``(x_fp8, scale)`` — quantized output and the scale used.
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)

    fn = _get_fused_rms_fp8_per_tensor_static_quant()
    out_fp8, _out_bf16, _, _ = fn(
        x_2d,
        weight,
        eps,
        scale,
        dtype_quant=fp8_dtype,
    )
    return out_fp8.reshape(orig_shape), scale


def rmsnorm_add_delayed_per_tensor(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm + delayed per-tensor FP8 quant.

    ``fused_rms_fp8_per_tensor_static_quant`` supports an optional
    ``res1`` parameter: ``residual_out = x + residual`` is computed
    inside the kernel, then RMSNorm + FP8 quant is applied to the result.

    Falls back to ``fused_add_rmsnorm`` + separate quant if the fused
    kernel is unavailable.

    Returns:
        ``(x_fp8, scale, residual_out)``
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)
    res_2d = _to_2d(residual)

    try:
        fn = _get_fused_rms_fp8_per_tensor_static_quant()
        out_fp8, _, _, res_out = fn(
            x_2d,
            weight,
            eps,
            scale,
            dtype_quant=fp8_dtype,
            res1=res_2d,
        )
        return out_fp8.reshape(orig_shape), scale, res_out.reshape(orig_shape)
    except (RuntimeError, NotImplementedError, TypeError, ValueError) as e:
        logger.warning("rmsnorm_add_delayed: fused kernel failed (%s), falling back", e)

    normed, res_out = fused_add_rmsnorm(
        x.reshape(orig_shape),
        residual.reshape(orig_shape),
        weight,
        eps,
    )
    normed_2d = _to_2d(normed)
    fn = _get_triton_per_tensor_quant()
    out_fp8, _ = fn(normed_2d, scale=scale, quant_dtype=fp8_dtype)
    return out_fp8.reshape(orig_shape), scale, res_out


def rmsnorm_current_per_tensor(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + current per-tensor FP8 quant.

    Uses per-token dynamic quant kernel, then derives per-tensor scale
    from the per-token scales (takes max).

    Returns:
        ``(x_fp8, tensor_scale)``
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)
    M, _N = x_2d.shape

    out_fp8 = torch.empty_like(x_2d, dtype=fp8_dtype)
    yscale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    fused = False
    try:
        fn = _get_triton_rmsnorm2d_fwd_with_dynamicquant()
        fn(out_fp8, x_2d, yscale, weight, eps)
        fused = True
    except (RuntimeError, NotImplementedError):
        pass

    if fused:
        tensor_scale = yscale.max()
        return out_fp8.reshape(orig_shape), tensor_scale.reshape(1)

    normed = rmsnorm(x, weight, eps)
    normed_2d = _to_2d(normed)
    fn = _get_triton_per_tensor_quant()
    out_fp8, scale = fn(normed_2d, quant_dtype=fp8_dtype)
    return out_fp8.reshape(orig_shape), scale


def rmsnorm_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + per-token (per-row) FP8 dynamic quant (fused Triton).

    Returns:
        ``(x_fp8, yscale)`` where yscale is ``[M, 1]``.
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)
    M, _N = x_2d.shape

    out_fp8 = torch.empty_like(x_2d, dtype=fp8_dtype)
    yscale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    def _tri(out, inp, ys, w, e):
        fn = _get_triton_rmsnorm2d_fwd_with_dynamicquant()
        fn(out, inp, ys, w, e)
        return out, ys

    result_fp8, result_scale = try_backends(
        [(Backend.TRITON, lambda: _tri(out_fp8, x_2d, yscale, weight, eps))],
        op_name="rmsnorm_per_token",
    )
    return result_fp8.reshape(orig_shape), result_scale


def rmsnorm_blockwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + blockwise FP8 group quant (fused Triton kernel).

    Returns:
        ``(x_fp8, block_scales)``
    """
    orig_shape = x.shape
    x_2d = _to_2d(x)

    try:
        fn = _get_fused_rms_fp8_group_quant()
        (out_fp8, out_scales), _, _, _ = fn(
            x_2d,
            weight,
            eps,
            group_size=block_size,
            dtype_quant=fp8_dtype,
        )
        return out_fp8.reshape(orig_shape), out_scales
    except (RuntimeError, NotImplementedError):
        pass

    normed = rmsnorm(x, weight, eps)
    from atomrl.lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    normed_2d = _to_2d(normed)
    out_fp8, out_scales = quant_fp8_blockwise_impl(
        normed_2d, dtype=fp8_dtype, axis=1, block_size=block_size
    )
    return out_fp8.reshape(orig_shape), out_scales


def rmsnorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    block_size: int = 32,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RMSNorm + MXFP8 conversion (unfused: AITER norm then AITER MXFP8 quant).

    Returns:
        ``(x_mxfp8, scales)``
    """
    normed = rmsnorm(x, weight, eps)
    from atomrl.lumen.ops.quantize.ops import convert_to_mxfp8

    return convert_to_mxfp8(
        normed,
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=fp8_dtype,
    )


# ---------------------------------------------------------------------------
# Unified dispatch: rmsnorm_with_quant
# ---------------------------------------------------------------------------


def rmsnorm_with_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scaling_type: str = "none",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale: torch.Tensor | None = None,
    block_size: int = 128,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unified RMSNorm + quantization dispatch.

    Args:
        scaling_type: One of ``"delayed"``, ``"dynamic"``, ``"per_token"``,
            ``"blockwise"``, ``"blockwise2d"``, ``"mxfp8"``, ``"none"``.
        scale: Pre-computed scale (required for ``"delayed"``).
        block_size: Block size for ``"blockwise"``, ``"blockwise2d"``, or ``"mxfp8"``.

    Returns:
        For ``"none"``: just the normalized tensor.
        For all others: ``(quantized_tensor, scale_or_scales)``.
    """
    if scaling_type == "none":
        return rmsnorm(x, weight, eps)
    elif scaling_type == "delayed":
        assert scale is not None, "delayed scaling requires a pre-computed scale"
        return rmsnorm_delayed_per_tensor(x, weight, eps, scale, fp8_dtype)
    elif scaling_type == "dynamic":
        return rmsnorm_current_per_tensor(x, weight, eps, fp8_dtype)
    elif scaling_type == "per_token":
        return rmsnorm_per_token(x, weight, eps, fp8_dtype)
    elif scaling_type in ("blockwise", "blockwise2d"):
        return rmsnorm_blockwise(x, weight, eps, block_size, fp8_dtype)
    elif scaling_type == "mxfp8":
        mxfp8_block = 32
        return rmsnorm_mxfp8(x, weight, eps, mxfp8_block, fp8_dtype)
    else:
        raise ValueError(f"Unknown scaling_type={scaling_type!r}")


# ---------------------------------------------------------------------------
# nn.Module API
# ---------------------------------------------------------------------------


class LumenRMSNorm(nn.Module):
    """RMSNorm backed by AITER Triton kernels.

    RMSNorm implementation backed by AITER kernels.

    Args:
        hidden_size: Last dimension of the input.
        eps: Epsilon for numerical stability.
        grad_quant_type: Gradient quantization format.
        sequence_parallel: Whether the input sequence is split across the tensor
            parallel group. Each rank then holds a partial weight gradient, which
            Megatron only all-reduces for parameters tagged with this flag.

    Example::

        norm = LumenRMSNorm(4096)
        y = norm(x)  # x: (batch, seq, 4096)
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        grad_quant_type: str | None = None,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.grad_quant_type = grad_quant_type
        self.weight.sequence_parallel = sequence_parallel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight.to(x.dtype), self.eps, self.grad_quant_type)

    def extra_repr(self) -> str:
        gq = f", grad_quant={self.grad_quant_type}" if self.grad_quant_type else ""
        return f"{self.weight.shape[0]}, eps={self.eps}{gq}"


# ---------------------------------------------------------------------------
# Diffusers class-level route
# ---------------------------------------------------------------------------

_diffusers_compiled = None
_diffusers_original_forward = None
_diffusers_stats = {"fused": 0, "fallback": 0}


def _diffusers_rmsnorm_weighted(hidden_states, weight, eps):
    """Diffusers weighted RMSNorm expression with FP32 intermediates."""
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return hidden_states.to(weight.dtype) * weight


def fused_diffusers_rmsnorm(hidden_states, weight, eps):
    """Compile the narrow Diffusers RMSNorm expression on first use."""
    global _diffusers_compiled
    if _diffusers_compiled is None:
        _diffusers_compiled = torch.compile(
            _diffusers_rmsnorm_weighted,
            dynamic=False,
        )
    return _diffusers_compiled(hidden_states, weight, eps)


def patch_diffusers_rmsnorm(min_seq: int = 1024) -> bool:
    """Patch weighted Diffusers RMSNorm forwards without replacing modules."""
    global _diffusers_original_forward
    if _diffusers_original_forward is not None:
        return True
    try:
        from diffusers.models.normalization import RMSNorm
    except (ImportError, OSError) as error:
        logger.info(
            "Diffusers RMSNorm unavailable (%s); not patching",
            type(error).__name__,
        )
        return False
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()

    _diffusers_original_forward = RMSNorm.forward
    original = _diffusers_original_forward
    threshold = int(min_seq)

    def forward(self, hidden_states):
        weight = getattr(self, "weight", None)
        if (
            weight is not None
            and getattr(self, "bias", None) is None
            and hidden_states.dim() == 4
            and hidden_states.shape[1] >= threshold
            and not isinstance(weight, DTensor)
            and not isinstance(hidden_states, DTensor)
        ):
            _diffusers_stats["fused"] += 1
            return fused_diffusers_rmsnorm(
                hidden_states,
                weight,
                self.eps,
            )
        _diffusers_stats["fallback"] += 1
        return original(self, hidden_states)

    RMSNorm.forward = forward
    logger.info(
        "Diffusers RMSNorm forward fused for 4-D sequences >= %d",
        threshold,
    )
    return True


def unpatch_diffusers_rmsnorm() -> None:
    """Restore Diffusers RMSNorm's original class method."""
    global _diffusers_original_forward
    if _diffusers_original_forward is None:
        return
    from diffusers.models.normalization import RMSNorm

    RMSNorm.forward = _diffusers_original_forward
    _diffusers_original_forward = None


def diffusers_rmsnorm_stats() -> dict[str, int]:
    """Return fused and fallback call counts for the Diffusers route."""
    return dict(_diffusers_stats)
