# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""LayerNorm with Triton auto-fallback and fused quantization for all 6 scaling modes.

All backends are AITER implementations — no torch.nn.functional fallbacks.

Supported quantization modes (fused into norm forward where possible):
    - ``delayed``    — norm then per-tensor static quant (unfused)
    - ``dynamic``    — fused per-token dynamic quant (Triton)
    - ``blockwise``  — norm then blockwise quant (unfused)
    - ``per_token``  — fused per-token dynamic quant (Triton)
    - ``mxfp8``      — norm then standalone MXFP8 conversion (unfused)
    - ``none``       — no quantization

Backward is always unquantized and handled by autograd.

Provides:
    - :func:`layernorm` — functional API  (autograd-aware, no quant fusion)
    - :func:`layernorm_with_quant` — functional API with fused quantization
    - :class:`LumenLayerNorm` — ``nn.Module`` API
"""

import logging

import torch
from torch import nn

from atomrl.lumen.core.grad_quant import quantize_grad_tensor
from atomrl.lumen.ops.dispatch import Backend, build_fallback_chain, try_backends

logger = logging.getLogger(__name__)

_zero_bias_cache: dict = {}


def _get_zero_bias(weight: torch.Tensor) -> torch.Tensor:
    """Return a cached all-zeros tensor matching *weight* (shape, dtype, device).

    Avoids allocating ``torch.zeros_like(weight)`` on every forward call
    when ``bias is None`` (~480 calls/step).
    """
    key = (weight.shape, weight.dtype, weight.device)
    buf = _zero_bias_cache.get(key)
    if buf is None:
        buf = torch.zeros_like(weight)
        _zero_bias_cache[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


def _get_triton_layer_norm():
    from aiter.ops.triton.normalization.norm import layer_norm

    return layer_norm


def _get_triton_layernorm2d_fwd_with_dynamicquant():
    from aiter.ops.triton.normalization.norm import layernorm2d_fwd_with_dynamicquant

    return layernorm2d_fwd_with_dynamicquant


def _get_triton_per_tensor_quant():
    from aiter.ops.quant import per_tensor_quant_triton

    return per_tensor_quant_triton


def _get_triton_per_token_quant():
    from aiter.ops.quant import pertoken_quant

    return pertoken_quant


# ---------------------------------------------------------------------------
# Unquantized LayerNorm with fallback (Triton via AITER)
# ---------------------------------------------------------------------------


def _layernorm_triton(x_2d, weight, bias, eps):
    fn = _get_triton_layer_norm()
    return fn(x_2d, weight, bias, eps)


_layernorm_chain = None


def _get_layernorm_chain():
    global _layernorm_chain
    if _layernorm_chain is None:
        candidates = {Backend.TRITON: _layernorm_triton}
        _layernorm_chain = build_fallback_chain(candidates)
    return _layernorm_chain


# ---------------------------------------------------------------------------
# Grad-quantized wrapper (autograd Function)
# ---------------------------------------------------------------------------


class _LayerNormGradQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps, grad_quant_type):
        ctx.grad_quant_type = grad_quant_type
        ctx.eps = eps
        ctx.save_for_backward(x, weight, bias)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        y = try_backends(
            _get_layernorm_chain(), x_2d, weight, bias, eps, op_name="layernorm"
        )
        return y.reshape(orig_shape)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        gqt = ctx.grad_quant_type

        with torch.enable_grad():
            x_d = x.detach().requires_grad_(True)
            w_d = weight.detach().requires_grad_(True)
            b_d = bias.detach().requires_grad_(True) if bias is not None else None
            orig_shape = x_d.shape
            x_2d = x_d.reshape(-1, x_d.shape[-1])
            y = _layernorm_triton(x_2d, w_d, b_d, ctx.eps)
            y = y.reshape(orig_shape)
            torch.autograd.backward(y, grad_output)

        dx = quantize_grad_tensor(x_d.grad, gqt)
        dw = quantize_grad_tensor(w_d.grad, gqt)
        db = (
            quantize_grad_tensor(b_d.grad, gqt)
            if b_d is not None and b_d.grad is not None
            else None
        )
        return dx, dw, db, None, None


# ---------------------------------------------------------------------------
# Public functional API — unquantized
# ---------------------------------------------------------------------------


def layernorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
    grad_quant_type: str | None = None,
) -> torch.Tensor:
    """Apply LayerNorm with automatic backend fallback (Triton via AITER).

    When any input requires grad, the Triton backend is preferred because
    its ``_LayerNorm`` autograd.Function provides forward+backward support.

    Args:
        x: Input tensor ``(*, hidden_size)``.
        weight: Learnable scale ``(hidden_size,)``.
        bias: Learnable bias ``(hidden_size,)`` or ``None``.
        eps: Epsilon for numerical stability.
        grad_quant_type: Gradient quantization format.

    Returns:
        Normalised tensor with same shape as *x*.
    """
    if bias is None:
        bias = _get_zero_bias(weight)

    if grad_quant_type is not None:
        return _LayerNormGradQuant.apply(x, weight, bias, eps, grad_quant_type)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])

    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad
        or weight.requires_grad
        or (bias is not None and bias.requires_grad)
    )
    if needs_grad:
        y = _layernorm_triton(x_2d, weight, bias, eps)
    else:
        y = try_backends(
            _get_layernorm_chain(), x_2d, weight, bias, eps, op_name="layernorm"
        )
    return y.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Fused LayerNorm + Quantization for each scaling mode
# ---------------------------------------------------------------------------


def layernorm_delayed_per_tensor(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm + delayed per-tensor FP8 quant (unfused: norm then quant).

    AITER has no fused LayerNorm + static per-tensor quant kernel,
    so we run AITER norm then apply AITER per-tensor quant.

    Returns:
        ``(x_fp8, scale)``
    """
    normed = layernorm(x, weight, bias, eps)
    normed_2d = normed.reshape(-1, normed.shape[-1]).contiguous()
    fn = _get_triton_per_tensor_quant()
    out_fp8, _ = fn(normed_2d, scale=scale, quant_dtype=fp8_dtype)
    return out_fp8.reshape(normed.shape), scale


def layernorm_current_per_tensor(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm + current per-tensor FP8 quant.

    Uses per-token dynamic quant kernel and derives per-tensor scale
    from max of per-token scales.

    Returns:
        ``(x_fp8, tensor_scale)``
    """
    if bias is None:
        bias = _get_zero_bias(weight)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    M, _N = x_2d.shape

    out_fp8 = torch.empty_like(x_2d, dtype=fp8_dtype)
    yscale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    fused = False
    try:
        fn = _get_triton_layernorm2d_fwd_with_dynamicquant()
        fn(out_fp8, x_2d, yscale, weight, bias, eps)
        fused = True
    except (RuntimeError, NotImplementedError):
        pass

    if fused:
        tensor_scale = yscale.max()
        return out_fp8.reshape(orig_shape), tensor_scale.reshape(1)

    # Unfused: AITER norm then AITER per-tensor quant
    normed = layernorm(x, weight, bias, eps)
    normed_2d = normed.reshape(-1, normed.shape[-1]).contiguous()
    fn = _get_triton_per_tensor_quant()
    out_fp8, scale = fn(normed_2d, quant_dtype=fp8_dtype)
    return out_fp8.reshape(orig_shape), scale


def layernorm_per_token(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm + per-token (per-row) FP8 dynamic quant (fused Triton).

    Returns:
        ``(x_fp8, yscale)`` where yscale is ``[M, 1]``.
    """
    if bias is None:
        bias = _get_zero_bias(weight)

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    M, _N = x_2d.shape

    out_fp8 = torch.empty_like(x_2d, dtype=fp8_dtype)
    yscale = torch.empty(M, 1, dtype=torch.float32, device=x.device)

    def _tri(out, inp, ys, w, b, e):
        fn = _get_triton_layernorm2d_fwd_with_dynamicquant()
        fn(out, inp, ys, w, b, e)
        return out, ys

    result_fp8, result_scale = try_backends(
        [(Backend.TRITON, lambda: _tri(out_fp8, x_2d, yscale, weight, bias, eps))],
        op_name="layernorm_per_token",
    )
    return result_fp8.reshape(orig_shape), result_scale


def layernorm_blockwise(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    block_size: int = 128,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm + blockwise FP8 quant (unfused: AITER norm then blockwise quant).

    Returns:
        ``(x_fp8, block_scales)``
    """
    normed = layernorm(x, weight, bias, eps)
    from atomrl.lumen.ops.quantize.ops import quant_fp8_blockwise_impl

    normed_2d = normed.reshape(-1, normed.shape[-1]).contiguous()
    out_fp8, out_scales = quant_fp8_blockwise_impl(
        normed_2d, dtype=fp8_dtype, axis=1, block_size=block_size
    )
    return out_fp8.reshape(normed.shape), out_scales


def layernorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    block_size: int = 32,
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LayerNorm + MXFP8 conversion (unfused: AITER norm then AITER MXFP8 quant).

    Returns:
        ``(x_mxfp8, scales)``
    """
    normed = layernorm(x, weight, bias, eps)
    from atomrl.lumen.ops.quantize.ops import convert_to_mxfp8

    return convert_to_mxfp8(
        normed,
        block_size=block_size,
        axis=-1,
        float8_dtype_pt=fp8_dtype,
    )


# ---------------------------------------------------------------------------
# Unified dispatch: layernorm_with_quant
# ---------------------------------------------------------------------------


def layernorm_with_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    eps: float,
    scaling_type: str = "none",
    fp8_dtype: torch.dtype = torch.float8_e4m3fn,
    scale: torch.Tensor | None = None,
    block_size: int = 128,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Unified LayerNorm + quantization dispatch.

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
        return layernorm(x, weight, bias, eps)
    elif scaling_type == "delayed":
        assert scale is not None, "delayed scaling requires a pre-computed scale"
        return layernorm_delayed_per_tensor(x, weight, bias, eps, scale, fp8_dtype)
    elif scaling_type == "dynamic":
        return layernorm_current_per_tensor(x, weight, bias, eps, fp8_dtype)
    elif scaling_type == "per_token":
        return layernorm_per_token(x, weight, bias, eps, fp8_dtype)
    elif scaling_type in ("blockwise", "blockwise2d"):
        return layernorm_blockwise(x, weight, bias, eps, block_size, fp8_dtype)
    elif scaling_type == "mxfp8":
        mxfp8_block = 32
        return layernorm_mxfp8(x, weight, bias, eps, mxfp8_block, fp8_dtype)
    else:
        raise ValueError(f"Unknown scaling_type={scaling_type!r}")


# ---------------------------------------------------------------------------
# nn.Module API
# ---------------------------------------------------------------------------


class LumenLayerNorm(nn.Module):
    """LayerNorm backed by AITER Triton kernels.

    LayerNorm implementation backed by AITER kernels.

    Args:
        hidden_size: Last dimension of the input.
        eps: Epsilon for numerical stability.
        elementwise_affine: Whether to learn scale/bias.
        grad_quant_type: Gradient quantization format.
        sequence_parallel: Whether the input sequence is split across the tensor
            parallel group. Each rank then holds a partial weight gradient, which
            Megatron only all-reduces for parameters tagged with this flag.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        grad_quant_type: str | None = None,
        sequence_parallel: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.grad_quant_type = grad_quant_type
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.weight.sequence_parallel = sequence_parallel
            self.bias.sequence_parallel = sequence_parallel
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = (
            self.weight
            if self.weight is not None
            else torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
        )
        w = w.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return layernorm(x, w, b, self.eps, self.grad_quant_type)

    def extra_repr(self) -> str:
        sz = self.weight.shape[0] if self.weight is not None else "?"
        gq = f", grad_quant={self.grad_quant_type}" if self.grad_quant_type else ""
        return f"{sz}, eps={self.eps}{gq}"
