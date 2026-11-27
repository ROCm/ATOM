# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Fused LayerNorm/RMSNorm + column-parallel linear.

Applies normalization, optional FP8 quantization, then a column-parallel
GEMM — all in one module with a single set of TP communication calls.

The norm runs on the *local* (possibly sequence-parallel) input; the
all-gather (if sequence_parallel) happens *after* the norm but *before*
the GEMM.
"""

import logging
import os
import warnings
from collections.abc import Callable

import torch
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    set_tensor_model_parallel_attributes,
)
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
)
from megatron.core.tensor_parallel.utils import divide
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from torch import nn
from torch.autograd.function import once_differentiable
from torch.nn.parameter import Parameter

from atomrl.lumen.modules.parallel_linear import (
    _DeferredWgrad,
    _do_gemm,
    _get_tp_group,
    _pg_rank,
    _pg_size,
    _use_sdma_from_args,
)

_logger = logging.getLogger(__name__)


def _to_2d(x: torch.Tensor) -> torch.Tensor:
    """Reshape to 2D, only calling .contiguous() when necessary."""
    x_2d = x.reshape(-1, x.shape[-1])
    return x_2d if x_2d.is_contiguous() else x_2d.contiguous()


_FUSED_NORM_QUANT = os.environ.get("LUMEN_FUSED_NORM_QUANT", "0") == "1"
_FUSED_NORM_QUANT_V2 = os.environ.get("LUMEN_FUSED_NORM_QUANT_V2", "1") == "1"
_SAVE_RSIGMA = os.environ.get("LUMEN_SAVE_RSIGMA", "1") == "1"  # Opt A


# Thread-local storage for passing residual from TransformerLayer into
# LumenLayerNormLinear without changing Megatron's MLP interface.
import threading

_tls = threading.local()


def _set_pending_residual(residual: torch.Tensor) -> None:
    """Store a residual tensor for the next LumenLayerNormLinear.forward() call."""
    _tls.pending_residual = residual


def _pop_pending_residual():
    """Pop the pending residual (returns None if not set)."""
    res = getattr(_tls, "pending_residual", None)
    _tls.pending_residual = None
    return res


def _pop_residual_out():
    """Pop the fused residual_out computed by the fused norm+quant path."""
    res = getattr(_tls, "residual_out", None)
    _tls.residual_out = None
    return res


def _set_residual_out(residual_out: torch.Tensor) -> None:
    _tls.residual_out = residual_out


def _pop_cached_ln_out():
    """Pop the cached normalized output from the last LumenLayerNormLinear forward.

    Used by the LoRA adapter patch to avoid recomputing RMSNorm for lora_a.
    """
    res = getattr(_tls, "cached_ln_out", None)
    _tls.cached_ln_out = None
    return res


def _set_cached_ln_out(ln_out: torch.Tensor) -> None:
    _tls.cached_ln_out = ln_out


def _aiter_fused_rms_quant(
    x_2d, weight, eps, scale, fp8_dtype, *, res1=None, save_rsigma=False
):
    """Call AITER's fused RMS quant using the pinned public call contract."""
    from aiter.ops.triton.quant.fused_fp8_quant import (
        fused_rms_fp8_per_tensor_static_quant,
    )

    kwargs = {
        "dtype_quant": fp8_dtype,
        "output_unquantized_inp1": True,
    }
    if res1 is not None:
        kwargs["res1"] = res1
    out_fp8, out_bf16, _, res_out = fused_rms_fp8_per_tensor_static_quant(
        x_2d, weight, eps, scale, **kwargs
    )
    norm_input = res_out if res_out is not None else x_2d
    rsigma = (
        torch.rsqrt(norm_input.float().square().mean(dim=-1) + eps)
        if save_rsigma
        else None
    )
    amax = out_bf16.float().abs().amax().reshape(1)
    return out_fp8, out_bf16, res_out, rsigma, amax


class _FusedResidualRMSNormFP8Quant(torch.autograd.Function):
    """Fused residual-add + RMSNorm + FP8 quant via a single Triton kernel.

    Computes ``residual_out = x + residual``, then ``norm_out = RMSNorm(residual_out)``,
    then ``fp8_out = quant(norm_out, scale)`` in ONE kernel launch.

    Returns ``(norm_out_bf16, fp8_out, scale, residual_out)`` where
    ``norm_out_bf16`` participates in autograd. Backward uses rsigma
    saved from forward to avoid recomputing RMSNorm statistics.
    """

    @staticmethod
    def forward(ctx, x, residual, weight, eps, scale, fp8_dtype):
        x_2d = _to_2d(x)
        res_2d = _to_2d(residual)
        out_fp8, out_bf16, res_out, rsigma, amax = _aiter_fused_rms_quant(
            x_2d,
            weight,
            eps,
            scale,
            fp8_dtype,
            res1=res_2d,
            save_rsigma=_SAVE_RSIGMA,
        )
        if _SAVE_RSIGMA:
            ctx.save_for_backward(res_out, weight, rsigma)
        else:
            ctx.save_for_backward(res_out, weight)
        ctx.eps = eps
        ctx.x_shape = x.shape

        return out_bf16.reshape(x.shape), out_fp8, scale, res_out.reshape(x.shape), amax

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_bf16, _grad_fp8, _grad_scale, grad_res_out, _grad_amax):
        from aiter.ops.triton.normalization.rmsnorm import (
            _rmsnorm_backward,
        )

        if _SAVE_RSIGMA:
            res_out, weight, rsigma = ctx.saved_tensors
        else:
            res_out, weight = ctx.saved_tensors
            from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_forward

            _, rsigma = _rmsnorm_forward(
                res_out.reshape(-1, res_out.shape[-1]), weight, ctx.eps
            )

        x_2d = res_out.reshape(-1, res_out.shape[-1])
        grad_2d = grad_bf16.reshape(-1, grad_bf16.shape[-1])

        dx, dw = _rmsnorm_backward(grad_2d, x_2d, weight, rsigma)
        dx = dx.reshape(ctx.x_shape)
        if grad_res_out is not None:
            dx = dx + grad_res_out
        return dx, dx, dw, None, None, None


class _FusedRMSNormFP8QuantV2(torch.autograd.Function):
    """Fused RMSNorm + FP8 quant + amax via a single Triton kernel.

    Reads input once, produces BF16 norm output (for backward) and FP8
    quantized output + amax (for scaling manager).  Eliminates:
      - separate RMSNorm kernel
      - separate cast_amax_fp8 / quant+amax kernel
      - separate amax(abs(x)) kernel

    Backward uses rsigma saved from forward to skip recompute.
    """

    @staticmethod
    def forward(ctx, x, weight, eps, scale, fp8_dtype):
        from atomrl.lumen.ops.quantize.cast_transpose import rmsnorm_quant_amax_fp8

        x_2d = _to_2d(x)
        if _SAVE_RSIGMA:
            out_fp8, out_bf16, amax, rsigma = rmsnorm_quant_amax_fp8(
                x_2d,
                weight,
                eps,
                scale,
                fp8_dtype,
                output_bf16=True,
                output_rsigma=True,
            )
            ctx.save_for_backward(x, weight, rsigma)
        else:
            out_fp8, out_bf16, amax = rmsnorm_quant_amax_fp8(
                x_2d,
                weight,
                eps,
                scale,
                fp8_dtype,
                output_bf16=True,
                output_rsigma=False,
            )
            ctx.save_for_backward(x, weight)
        ctx.eps = eps

        return out_bf16.reshape(x.shape), out_fp8, scale, amax

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_bf16, _grad_fp8, _grad_scale, _grad_amax):
        if _SAVE_RSIGMA:
            x, weight, rsigma = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_forward

            _, rsigma = _rmsnorm_forward(x.reshape(-1, x.shape[-1]), weight, ctx.eps)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_2d = grad_bf16.reshape(-1, grad_bf16.shape[-1])

        from aiter.ops.triton.normalization.rmsnorm import (
            _rmsnorm_backward,
        )

        dx, dw = _rmsnorm_backward(grad_2d, x_2d, weight, rsigma)
        return dx.reshape(x.shape), dw, None, None, None


class _FusedRMSNormFP8Quant(torch.autograd.Function):
    """Fused RMSNorm + per-tensor FP8 quant with correct autograd (AITER path).

    Forward calls the fused Triton kernel which produces both the BF16
    norm output (for autograd) and the FP8 quantized output.
    Returns ``(ln_out_bf16, out_fp8, scale)`` — only the first output
    gets gradients; the FP8 tensor and scale are constants w.r.t.
    autograd.

    Backward uses rsigma saved from forward to skip recompute.
    """

    @staticmethod
    def forward(ctx, x, weight, eps, scale, fp8_dtype):
        x_2d = _to_2d(x)
        out_fp8, out_bf16, _, rsigma, amax = _aiter_fused_rms_quant(
            x_2d,
            weight,
            eps,
            scale,
            fp8_dtype,
            save_rsigma=_SAVE_RSIGMA,
        )
        if _SAVE_RSIGMA:
            ctx.save_for_backward(x, weight, rsigma)
        else:
            ctx.save_for_backward(x, weight)
        ctx.eps = eps

        return out_bf16.reshape(x.shape), out_fp8, scale, amax

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_bf16, _grad_fp8, _grad_scale, _grad_amax):
        if _SAVE_RSIGMA:
            x, weight, rsigma = ctx.saved_tensors
        else:
            x, weight = ctx.saved_tensors
            from aiter.ops.triton.normalization.rmsnorm import _rmsnorm_forward

            _, rsigma = _rmsnorm_forward(x.reshape(-1, x.shape[-1]), weight, ctx.eps)

        x_2d = x.reshape(-1, x.shape[-1])
        grad_2d = grad_bf16.reshape(-1, grad_bf16.shape[-1])

        from aiter.ops.triton.normalization.rmsnorm import (
            _rmsnorm_backward,
        )

        dx, dw = _rmsnorm_backward(grad_2d, x_2d, weight, rsigma)
        return dx.reshape(x.shape), dw, None, None, None


__all__ = ["LumenLayerNormLinear"]


class LumenLayerNormLinear(nn.Module):
    """Fused Norm + ColumnParallelLinear using Lumen GEMM.

    Norm type is selected via ``config.normalization`` (``"LayerNorm"`` or
    ``"RMSNorm"``).
    """

    _lora_tp_mode = "column"

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config,
        init_method: Callable,
        gather_output: bool = False,
        bias: bool = True,
        skip_bias_add: bool = False,
        is_expert: bool = False,
        skip_weight_param_allocation: bool = False,
        tp_comm_buffer_name: str | None = None,
        tp_group: torch.distributed.ProcessGroup | None = None,
    ):
        super().__init__()

        if gather_output:
            raise ValueError("LumenLayerNormLinear does not support gather_output=True")
        if is_expert:
            raise ValueError("LumenLayerNormLinear does not support MoE experts")
        if skip_weight_param_allocation:
            raise ValueError(
                "LumenLayerNormLinear does not support skip_weight_param_allocation"
            )

        self.config = config
        self.input_size = input_size
        self.output_size = output_size
        self.skip_bias_add = skip_bias_add

        self.tp_group = _get_tp_group(tp_group, is_expert=False)
        self.tp_size = _pg_size(self.tp_group)
        tp_rank = _pg_rank(self.tp_group)
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.sequence_parallel = getattr(config, "sequence_parallel", False)
        if self.sequence_parallel and self.tp_size <= 1:
            warnings.warn("sequence_parallel disabled: tp_size <= 1")
            self.sequence_parallel = False

        self.use_sdma = _use_sdma_from_args()
        self._sdma_comm = None

        # Norm type
        normalization = getattr(config, "normalization", "LayerNorm")
        self.use_rmsnorm = normalization == "RMSNorm"
        eps = getattr(config, "layernorm_epsilon", 1e-5)
        zero_centered = getattr(config, "layernorm_zero_centered_gamma", False)

        # Norm parameters
        self.layer_norm_weight = Parameter(
            torch.ones(
                input_size,
                dtype=config.params_dtype,
                device=torch.cuda.current_device(),
            )
        )
        if self.use_rmsnorm:
            self.register_parameter("layer_norm_bias", None)
        else:
            self.layer_norm_bias = Parameter(
                torch.zeros(
                    input_size,
                    dtype=config.params_dtype,
                    device=torch.cuda.current_device(),
                )
            )
        self.ln_weight = self.layer_norm_weight
        self.ln_bias = self.layer_norm_bias
        self.ln_eps = eps
        self.zero_centered_gamma = zero_centered

        # FP8 config
        self.scaling_type = "none"
        self.scaling_manager = None
        from atomrl.lumen.quantize.config import _get_float8_e4m3

        self.fp8_dtype = _get_float8_e4m3()
        self.block_size = 128
        self.gradient_accumulation_fusion = False
        self.delay_wgrad = False
        self.fp8_activation_store = False
        self._deferred_wgrad = _DeferredWgrad()

        # Linear weight [output_size_per_partition, input_size]
        if getattr(config, "use_cpu_initialization", False):
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    input_size,
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                from megatron.core.tensor_parallel.layers import condition_init_method

                _initialize_affine_weight_cpu(
                    self.weight,
                    output_size,
                    input_size,
                    self.output_size_per_partition,
                    0,
                    condition_init_method(config, init_method),
                    stride=1,
                    return_master_weight=False,
                    rank=tp_rank,
                    world_size=self.tp_size,
                    skip_set_tensor_parallel_attributes=True,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if getattr(config, "perform_initialization", True):
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=1,
                )
        self.weight.allreduce = True

        if bias:
            if getattr(config, "use_cpu_initialization", False):
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition, dtype=config.params_dtype
                    )
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=torch.cuda.current_device(),
                        dtype=config.params_dtype,
                    )
                )
            set_tensor_model_parallel_attributes(self.bias, True, 0, 1)
            if getattr(config, "perform_initialization", True):
                with torch.no_grad():
                    self.bias.zero_()
            self.bias.allreduce = True
        else:
            self.register_parameter("bias", None)

        self._register_load_state_dict_pre_hook(
            lambda state_dict, prefix, *args, **kwargs: state_dict.setdefault(
                f"{prefix}_extra_state"
            )
        )

    def _norm(self, x):
        """Apply normalization."""
        w = self.layer_norm_weight
        if self.zero_centered_gamma:
            w = w + 1.0
        if self.use_rmsnorm:
            from atomrl.lumen.ops.normalization.rmsnorm import rmsnorm

            return rmsnorm(x, w, self.ln_eps)
        else:
            from atomrl.lumen.ops.normalization.layernorm import layernorm

            return layernorm(x, w, self.layer_norm_bias, self.ln_eps)

    def _get_sdma_comm(self):
        if self._sdma_comm is None:
            from atomrl.lumen.modules.sdma_comm import SdmaTpComm

            self._sdma_comm = SdmaTpComm.get(self.tp_group)
        return self._sdma_comm

    def _try_fused_norm_quant(self, x: torch.Tensor):
        """Attempt fused RMSNorm + FP8 quantization.

        Returns ``(ln_out_bf16, fp8_data, scale)`` on success, or
        ``None`` if the fused path is not applicable.

        ``ln_out_bf16`` participates in autograd (gradients flow back
        to ``x`` and ``ln_weight``).  ``fp8_data`` and ``scale`` are
        passed downstream as ``pre_quantized_input`` to skip the
        standalone FP8 quant kernel.

        Two implementations are available:

        **V2** (``LUMEN_FUSED_NORM_QUANT_V2=1``, default): A custom
        Triton kernel that fuses RMSNorm + FP8 cast + amax into a
        single launch (one read, two writes: FP8 + BF16).  The amax
        is computed on the *normalized* output inside the kernel,
        eliminating the separate ``update_amax`` call.

        **V1** (fallback): AITER ``fused_rms_fp8_per_tensor_static_quant``
        which also produces FP8 + BF16 but doesn't compute amax —
        requires a separate ``update_amax`` call.
        """
        if not (
            _FUSED_NORM_QUANT
            and self.use_rmsnorm
            and self.scaling_type == "delayed"
            and self.scaling_manager is not None
            and not self.sequence_parallel
        ):
            return None

        w = self.layer_norm_weight
        if self.zero_centered_gamma:
            w = w + 1.0

        x_2d = _to_2d(x)
        scale, precomputed_amax = self.scaling_manager.get_scale(
            "activation",
            x_2d,
            return_amax=True,
        )
        if scale is None:
            return None

        if _FUSED_NORM_QUANT_V2:
            try:
                ln_out_bf16, out_fp8, out_scale, amax = _FusedRMSNormFP8QuantV2.apply(
                    x,
                    w,
                    self.ln_eps,
                    scale,
                    self.fp8_dtype,
                )
                self.scaling_manager.update_amax_value("activation", amax)
                return (ln_out_bf16, out_fp8, out_scale)
            except RuntimeError as e:
                _logger.debug("Fused norm+quant v2 failed: %s, trying v1", e)

        try:
            ln_out_bf16, out_fp8, out_scale, kernel_amax = _FusedRMSNormFP8Quant.apply(
                x,
                w,
                self.ln_eps,
                scale,
                self.fp8_dtype,
            )

            if precomputed_amax is not None:
                self.scaling_manager.update_amax_value("activation", precomputed_amax)
            else:
                self.scaling_manager.update_amax_value("activation", kernel_amax)

            return (ln_out_bf16, out_fp8, out_scale)
        except RuntimeError as e:
            _logger.debug("Fused norm+quant v1 failed: %s, falling back", e)
            return None

    def _try_fused_norm_quant_with_residual(self, x, residual):
        """Fused residual_add + RMSNorm + FP8 quant in a single AITER kernel.

        Uses ``_FusedResidualRMSNormFP8Quant`` autograd Function which calls
        ``fused_rms_fp8_per_tensor_static_quant(res1=residual,
        output_unquantized_inp1=True)`` computing in ONE kernel:
          1. ``residual_out = x + residual``
          2. ``norm_out = RMSNorm(residual_out)``
          3. ``fp8_out = quant(norm_out, scale)``

        Returns ``(ln_out_bf16, fp8_data, scale, residual_out)`` on success,
        or ``None`` if the fused path is not applicable.
        """
        if not (
            _FUSED_NORM_QUANT
            and self.use_rmsnorm
            and self.scaling_type == "delayed"
            and self.scaling_manager is not None
            and not self.sequence_parallel
        ):
            return None

        w = self.layer_norm_weight
        if self.zero_centered_gamma:
            w = w + 1.0

        x_2d = _to_2d(x)
        scale, precomputed_amax = self.scaling_manager.get_scale(
            "activation",
            x_2d,
            return_amax=True,
        )
        if scale is None:
            return None

        try:
            ln_out_bf16, out_fp8, out_scale, residual_out, kernel_amax = (
                _FusedResidualRMSNormFP8Quant.apply(
                    x,
                    residual,
                    w,
                    self.ln_eps,
                    scale,
                    self.fp8_dtype,
                )
            )
            if precomputed_amax is not None:
                self.scaling_manager.update_amax_value("activation", precomputed_amax)
            else:
                self.scaling_manager.update_amax_value("activation", kernel_amax)

            return (ln_out_bf16, out_fp8, out_scale, residual_out)
        except RuntimeError as e:
            _logger.debug("Fused residual+norm+quant failed: %s, falling back", e)
            return None

    def forward(self, x: torch.Tensor):
        """Forward: Norm → (all-gather if SP) → GEMM.

        When ``LUMEN_FUSED_NORM_QUANT=1`` and this layer uses RMSNorm with
        FP8, the norm and activation quantization are fused into a single
        AITER kernel call, eliminating one full read+write of the hidden
        state from HBM.

        Returns:
            (output, bias) where bias is ``None`` unless ``skip_bias_add``.
        """
        pre_quantized_input = None

        pending_residual = _pop_pending_residual()

        if pending_residual is not None:
            # Try fused add+norm+quant (single kernel: eliminates aten::add)
            fused_res_result = self._try_fused_norm_quant_with_residual(
                x, pending_residual
            )
            if fused_res_result is not None:
                ln_out, out_fp8, out_scale, residual_out = fused_res_result
                pre_quantized_input = (out_fp8, out_scale)
                _set_residual_out(residual_out)
            else:
                # Fallback: standalone add + norm
                x = x + pending_residual
                fused_result = self._try_fused_norm_quant(x)
                if fused_result is not None:
                    ln_out, out_fp8, out_scale = fused_result
                    pre_quantized_input = (out_fp8, out_scale)
                else:
                    ln_out = self._norm(x)
                _set_residual_out(x)
        else:
            fused_result = self._try_fused_norm_quant(x)
            if fused_result is not None:
                ln_out, out_fp8, out_scale = fused_result
                pre_quantized_input = (out_fp8, out_scale)
            else:
                ln_out = self._norm(x)

        if self.sequence_parallel:
            if self.use_sdma and self.tp_size > 1:
                from atomrl.lumen.modules.sdma_comm import (
                    sdma_gather_from_sequence_parallel_region,
                )

                ln_out = sdma_gather_from_sequence_parallel_region(
                    ln_out,
                    self._get_sdma_comm(),
                )
            else:
                ln_out = gather_from_sequence_parallel_region(
                    ln_out,
                    tensor_parallel_output_grad=True,
                    group=self.tp_group,
                )

        _set_cached_ln_out(ln_out)

        gemm_bias = self.bias if not self.skip_bias_add else None
        output = _do_gemm(
            ln_out,
            self.weight,
            gemm_bias,
            self.scaling_manager,
            self.scaling_type,
            self.fp8_dtype,
            self.block_size,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            delay_wgrad=self.delay_wgrad,
            deferred_wgrad=self._deferred_wgrad if self.delay_wgrad else None,
            pre_quantized_input=pre_quantized_input,
        )

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    def enable_fp8(
        self,
        scaling_manager=None,
        scaling_type="dynamic",
        fp8_dtype=None,
        block_size=None,
    ):
        from atomrl.lumen.quantize import QuantConfig, ScalingManager

        self.scaling_type = scaling_type
        self.scaling_manager = scaling_manager or ScalingManager(QuantConfig())
        if fp8_dtype is not None:
            self.fp8_dtype = fp8_dtype
        if block_size is not None:
            self.block_size = block_size

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        state_dict = self.state_dict(prefix="", keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            state_dict,
            prefix,
            {"weight": 0, "bias": 0},
            sharded_offsets,
        )

    def set_extra_state(self, state):
        pass

    def get_extra_state(self):
        return None

    def execute_deferred_wgrad(self):
        """Execute any deferred weight gradient computation."""
        self._deferred_wgrad.execute()

    def backward_dw(self):
        """Megatron-compatible API: execute deferred weight gradient."""
        self._deferred_wgrad.execute()

    def __repr__(self):
        norm = "RMSNorm" if self.use_rmsnorm else "LayerNorm"
        return (
            f"{type(self).__name__}({norm}, in={self.input_size}, out={self.output_size}, "
            f"bias={self.bias is not None}, TP={self.tp_size}, fp8={self.scaling_type})"
        )
