# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused MLP operations with AITER backend dispatch.

Provides fused gated and ungated MLP forward operations that dispatch
to AITER Triton kernels when available, with decomposed AITER GEMM fallback.
"""

import logging

import torch
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_gated import (
    ff_a16w16_fused_gated,
)
from aiter.ops.triton.gemm.feed_forward.ff_a16w16_fused_ungated import (
    ff_a16w16_fused_ungated,
)
from torch.autograd.function import once_differentiable
from torch.nn.functional import gelu, linear, relu, silu

from atomrl.lumen.ops.quantize.linear import gemm_bf16, quantize_input

logger = logging.getLogger(__name__)


_ACTIVATION_TO_AITER = {
    "swiglu": "silu",
    "geglu": "gelu",
    "reglu": "relu",
    "gelu": "gelu",
    "relu": "relu",
    "silu": "silu",
}

_ACTIVATION_FNS = {
    "swiglu": silu,
    "geglu": gelu,
    "reglu": relu,
    "gelu": gelu,
    "relu": relu,
    "silu": silu,
}


def _needs_autograd(*tensors: torch.Tensor | None) -> bool:
    return torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    )


def _aiter_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    name = _ACTIVATION_TO_AITER[activation]
    if name == "silu":
        return silu(x)
    if name == "gelu":
        return gelu(x)
    if name == "relu":
        return relu(x)
    raise ValueError(f"Unsupported AITER activation: {activation}")


def _activation_with_grad(
    value: torch.Tensor, activation: str
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        tracked = value.detach().requires_grad_(True)
        activated = _aiter_activation(tracked, activation)
        derivative = torch.autograd.grad(activated.sum(), tracked)[0]
    return activated.detach(), derivative


def _gemm(
    x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """N-D input GEMM via AITER: reshapes to 2D, dispatches gemm_bf16, reshapes back."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    out = gemm_bf16(x_2d, w, bias)
    return out.reshape(orig_shape[:-1] + (out.shape[-1],))


class _AITERFusedGatedMLP(torch.autograd.Function):
    """AITER fused forward with explicit AITER GEMM backward."""

    @staticmethod
    def forward(ctx, x, w_up, w_gate, w_down, activation):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        w_combined = torch.cat([w_gate, w_up], dim=0)
        out = ff_a16w16_fused_gated(
            x_2d,
            w_combined,
            w_down.T.contiguous(),
            dtype=x.dtype,
            activation=_ACTIVATION_TO_AITER[activation],
        )
        ctx.save_for_backward(x_2d, w_up, w_gate, w_down)
        ctx.activation = activation
        ctx.x_shape = orig_shape
        return out.reshape(orig_shape[:-1] + (out.shape[-1],))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, w_up, w_gate, w_down = ctx.saved_tensors
        grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        gate = gemm_bf16(x, w_gate)
        up = gemm_bf16(x, w_up)
        activated, activation_grad = _activation_with_grad(gate, ctx.activation)
        hidden = activated * up

        grad_hidden = gemm_bf16(grad, w_down.T.contiguous())
        grad_gate = grad_hidden * up * activation_grad
        grad_up = grad_hidden * activated

        grad_x = gemm_bf16(grad_gate, w_gate.T.contiguous()) + gemm_bf16(
            grad_up, w_up.T.contiguous()
        )
        grad_w_gate = gemm_bf16(grad_gate.T.contiguous(), x.T.contiguous())
        grad_w_up = gemm_bf16(grad_up.T.contiguous(), x.T.contiguous())
        grad_w_down = gemm_bf16(grad.T.contiguous(), hidden.T.contiguous())
        return (
            grad_x.reshape(ctx.x_shape),
            grad_w_up,
            grad_w_gate,
            grad_w_down,
            None,
        )


class _AITERFusedMLP(torch.autograd.Function):
    """AITER fused ungated forward with explicit AITER GEMM backward."""

    @staticmethod
    def forward(ctx, x, w_up, w_down, activation):
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        out = ff_a16w16_fused_ungated(
            x_2d,
            w_up,
            w_down.T.contiguous(),
            dtype=x.dtype,
            activation=_ACTIVATION_TO_AITER[activation],
        )
        ctx.save_for_backward(x_2d, w_up, w_down)
        ctx.activation = activation
        ctx.x_shape = orig_shape
        return out.reshape(orig_shape[:-1] + (out.shape[-1],))

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        x, w_up, w_down = ctx.saved_tensors
        grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        pre_activation = gemm_bf16(x, w_up)
        activated, activation_grad = _activation_with_grad(
            pre_activation, ctx.activation
        )
        grad_hidden = gemm_bf16(grad, w_down.T.contiguous())
        grad_pre_activation = grad_hidden * activation_grad

        grad_x = gemm_bf16(grad_pre_activation, w_up.T.contiguous())
        grad_w_up = gemm_bf16(grad_pre_activation.T.contiguous(), x.T.contiguous())
        grad_w_down = gemm_bf16(grad.T.contiguous(), activated.T.contiguous())
        return grad_x.reshape(ctx.x_shape), grad_w_up, grad_w_down, None


def fused_gated_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "swiglu",
    bias_up: torch.Tensor | None = None,
    bias_gate: torch.Tensor | None = None,
    bias_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused gated MLP: down(gate_act(gate(x)) * up(x)).

    Bias-free execution uses AITER ``ff_a16w16_fused_gated`` with an explicit
    AITER GEMM backward. Biased training uses differentiable PyTorch linear
    operations; biased inference uses decomposed AITER BF16 GEMMs.
    """
    act_fn = _ACTIVATION_FNS[activation]
    if x.device.type != "cuda":
        gate_out = linear(x, w_gate, bias_gate)
        up_out = linear(x, w_up, bias_up)
        return linear(act_fn(gate_out) * up_out, w_down, bias_down)

    use_bias = bias_up is not None or bias_gate is not None or bias_down is not None
    if use_bias and _needs_autograd(
        x, w_up, w_gate, w_down, bias_up, bias_gate, bias_down
    ):
        gate_out = linear(x, w_gate, bias_gate)
        up_out = linear(x, w_up, bias_up)
        return linear(act_fn(gate_out) * up_out, w_down, bias_down)

    if not use_bias:
        return _AITERFusedGatedMLP.apply(x, w_up, w_gate, w_down, activation)

    gate_out = _gemm(x, w_gate, bias_gate)
    up_out = _gemm(x, w_up, bias_up)
    hidden = act_fn(gate_out) * up_out
    return _gemm(hidden, w_down, bias_down)


def fused_mlp(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "gelu",
    bias_up: torch.Tensor | None = None,
    bias_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused ungated MLP: down(act(up(x))).

    Bias-free execution uses AITER ``ff_a16w16_fused_ungated`` with an explicit
    AITER GEMM backward. Biased training uses differentiable PyTorch linear
    operations; biased inference uses decomposed AITER BF16 GEMMs.
    """
    act_fn = _ACTIVATION_FNS[activation]
    if x.device.type != "cuda":
        return linear(act_fn(linear(x, w_up, bias_up)), w_down, bias_down)

    use_bias = bias_up is not None or bias_down is not None
    if use_bias and _needs_autograd(x, w_up, w_down, bias_up, bias_down):
        return linear(act_fn(linear(x, w_up, bias_up)), w_down, bias_down)

    if not use_bias:
        return _AITERFusedMLP.apply(x, w_up, w_down, activation)

    hidden = act_fn(_gemm(x, w_up, bias_up))
    return _gemm(hidden, w_down, bias_down)


# ---------------------------------------------------------------------------
# FP8 activation storage for reduced memory in backward
# ---------------------------------------------------------------------------


def _quant_activation(t: torch.Tensor, fp8_dtype: torch.dtype):
    """Quantize an activation tensor to FP8 via AITER dynamic per-tensor quant.

    Returns ``(fp8_uint8_view, scale)`` where scale follows AITER convention
    (``dequant = fp8.to(float32) * scale``).
    """
    t_2d = t.reshape(-1, t.shape[-1]).contiguous()
    desc = quantize_input(t_2d, "dynamic", fp8_dtype)
    return desc.data.view(torch.uint8), desc.scale


def _dequant_activation(
    t_u8: torch.Tensor,
    scale: torch.Tensor,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype,
):
    """Dequantize a uint8-stored FP8 activation back to working precision."""
    return (t_u8.view(fp8_dtype).to(torch.float32) * scale).to(out_dtype)


class _FusedGatedMLPFP8Store(torch.autograd.Function):
    """Fused gated MLP with FP8 activation storage for backward."""

    @staticmethod
    def forward(
        ctx,
        x,
        w_gate,
        w_up,
        w_down,
        activation,
        bias_gate,
        bias_up,
        bias_down,
        fp8_dtype,
    ):
        act_fn = _ACTIVATION_FNS.get(activation, silu)
        gate_out = _gemm(x, w_gate, bias_gate)
        up_out = _gemm(x, w_up, bias_up)
        hidden = act_fn(gate_out) * up_out
        output = _gemm(hidden, w_down, bias_down)

        x_u8, scale_x = _quant_activation(x, fp8_dtype)
        hidden_u8, scale_h = _quant_activation(hidden, fp8_dtype)

        ctx.save_for_backward(
            x_u8,
            hidden_u8,
            w_gate,
            w_up,
            w_down,
            scale_x,
            scale_h,
        )
        ctx.activation = activation
        ctx.has_bias = (
            bias_gate is not None,
            bias_up is not None,
            bias_down is not None,
        )
        ctx.fp8_dtype = fp8_dtype
        ctx.x_shape = x.shape
        ctx.bias_gate = bias_gate
        ctx.bias_up = bias_up

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_u8, hidden_u8, w_gate, w_up, w_down, scale_x, scale_h = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        dtype = grad_output.dtype

        x_2d = _dequant_activation(x_u8, scale_x, fp8_dtype, dtype)
        hidden_2d = _dequant_activation(hidden_u8, scale_h, fp8_dtype, dtype)

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        # dgrad through down projection
        grad_hidden_2d = gemm_bf16(grad_2d, w_down.t().contiguous())
        # wgrad for down projection
        grad_w_down = gemm_bf16(grad_2d.t().contiguous(), hidden_2d.t().contiguous())
        grad_bias_down = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias[2]
            else None
        )

        # Recompute gate/up from dequantized x (include bias for correct activation grad)
        act_fn = _ACTIVATION_FNS.get(ctx.activation, silu)
        gate_out_2d = gemm_bf16(x_2d, w_gate, ctx.bias_gate)
        up_out_2d = gemm_bf16(x_2d, w_up, ctx.bias_up)

        with torch.enable_grad():
            gate_out_g = gate_out_2d.detach().requires_grad_(True)
            act_val = act_fn(gate_out_g)
            act_grad = torch.autograd.grad(
                act_val.sum(), gate_out_g, retain_graph=False
            )[0]

        grad_gate_2d = grad_hidden_2d * up_out_2d * act_grad
        grad_up_2d = grad_hidden_2d * act_fn(gate_out_2d)

        # dgrad through gate+up projections
        grad_x_2d = gemm_bf16(grad_gate_2d, w_gate.t().contiguous()) + gemm_bf16(
            grad_up_2d, w_up.t().contiguous()
        )

        # wgrad for gate and up projections
        grad_w_gate = gemm_bf16(grad_gate_2d.t().contiguous(), x_2d.t().contiguous())
        grad_w_up = gemm_bf16(grad_up_2d.t().contiguous(), x_2d.t().contiguous())

        grad_bias_gate = grad_gate_2d.sum(dim=0) if ctx.has_bias[0] else None
        grad_bias_up = grad_up_2d.sum(dim=0) if ctx.has_bias[1] else None

        grad_x = grad_x_2d.reshape(ctx.x_shape)

        return (
            grad_x,
            grad_w_gate,
            grad_w_up,
            grad_w_down,
            None,
            grad_bias_gate,
            grad_bias_up,
            grad_bias_down,
            None,
        )


def fused_gated_mlp_fp8_store(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_gate: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "swiglu",
    bias_up: torch.Tensor | None = None,
    bias_gate: torch.Tensor | None = None,
    bias_down: torch.Tensor | None = None,
    fp8_dtype=torch.float8_e4m3fn,
) -> torch.Tensor:
    """Fused gated MLP with FP8 activation storage."""
    return _FusedGatedMLPFP8Store.apply(
        x, w_gate, w_up, w_down, activation, bias_gate, bias_up, bias_down, fp8_dtype
    )


class _FusedMLPFP8Store(torch.autograd.Function):
    """Fused ungated MLP with FP8 activation storage for backward."""

    @staticmethod
    def forward(ctx, x, w_up, w_down, activation, bias_up, bias_down, fp8_dtype):
        act_fn = _ACTIVATION_FNS.get(activation, gelu)
        hidden = act_fn(_gemm(x, w_up, bias_up))
        output = _gemm(hidden, w_down, bias_down)

        x_u8, scale_x = _quant_activation(x, fp8_dtype)
        hidden_u8, scale_h = _quant_activation(hidden, fp8_dtype)

        ctx.save_for_backward(
            x_u8,
            hidden_u8,
            w_up,
            w_down,
            scale_x,
            scale_h,
        )
        ctx.activation = activation
        ctx.has_bias = (bias_up is not None, bias_down is not None)
        ctx.fp8_dtype = fp8_dtype
        ctx.x_shape = x.shape
        ctx.bias_up = bias_up

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x_u8, hidden_u8, w_up, w_down, scale_x, scale_h = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype
        dtype = grad_output.dtype

        x_2d = _dequant_activation(x_u8, scale_x, fp8_dtype, dtype)
        hidden_2d = _dequant_activation(hidden_u8, scale_h, fp8_dtype, dtype)

        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        # dgrad through down projection
        grad_hidden_2d = gemm_bf16(grad_2d, w_down.t().contiguous())
        # wgrad for down projection
        grad_w_down = gemm_bf16(grad_2d.t().contiguous(), hidden_2d.t().contiguous())
        grad_bias_down = (
            grad_output.sum(dim=tuple(range(grad_output.dim() - 1)))
            if ctx.has_bias[1]
            else None
        )

        # Recompute pre-activation from dequantized x (include bias for correct activation grad)
        act_fn = _ACTIVATION_FNS.get(ctx.activation, gelu)
        up_out_2d = gemm_bf16(x_2d, w_up, ctx.bias_up)

        with torch.enable_grad():
            up_out_g = up_out_2d.detach().requires_grad_(True)
            act_val = act_fn(up_out_g)
            act_grad = torch.autograd.grad(act_val.sum(), up_out_g, retain_graph=False)[
                0
            ]

        grad_pre_act_2d = grad_hidden_2d * act_grad

        # dgrad through up projection
        grad_x_2d = gemm_bf16(grad_pre_act_2d, w_up.t().contiguous())
        # wgrad for up projection
        grad_w_up = gemm_bf16(grad_pre_act_2d.t().contiguous(), x_2d.t().contiguous())
        grad_bias_up = grad_pre_act_2d.sum(dim=0) if ctx.has_bias[0] else None

        grad_x = grad_x_2d.reshape(ctx.x_shape)

        return (
            grad_x,
            grad_w_up,
            grad_w_down,
            None,
            grad_bias_up,
            grad_bias_down,
            None,
        )


def fused_mlp_fp8_store(
    x: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    activation: str = "gelu",
    bias_up: torch.Tensor | None = None,
    bias_down: torch.Tensor | None = None,
    fp8_dtype=torch.float8_e4m3fn,
) -> torch.Tensor:
    """Fused ungated MLP with FP8 activation storage."""
    return _FusedMLPFP8Store.apply(
        x, w_up, w_down, activation, bias_up, bias_down, fp8_dtype
    )
