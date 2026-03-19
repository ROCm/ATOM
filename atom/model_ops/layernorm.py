# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple, Optional
import torch
from torch import Tensor
from torch.overrides import (
    has_torch_function_unary,
    handle_torch_function,
)
from atom.config import QuantizationConfig, LayerQuantConfig
from atom.utils.decorators import mark_trace
from torch import nn
from aiter import (
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
    layernorm2d_fwd,
    layernorm2d_fwd_with_add,
)
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
from aiter.dist.parallel_state import get_tensor_model_parallel_world_size
from aiter.ops.triton.fused_add_rmsnorm_pad import fused_add_rmsnorm_pad
from aiter.jit.utils.torch_guard import torch_compile_guard

from aiter import (
    QuantType,
)
from atom.model_ops.utils import MXFP4_QUANT_BLOCK_SIZE

try:
    from aiter import dtypes as _aiter_dtypes
except ImportError:
    _aiter_dtypes = None


def silu(input: Tensor, inplace: bool = False) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.

    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}

    .. note::
        See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_
        where the SiLU (Sigmoid Linear Unit) was originally coined, and see
        `Sigmoid-Weighted Linear Units for Neural Network Function Approximation
        in Reinforcement Learning <https://arxiv.org/abs/1702.03118>`_ and `Swish:
        a Self-Gated Activation Function <https://arxiv.org/abs/1710.05941v1>`_
        where the SiLU was experimented with later.

    See :class:`~torch.nn.SiLU` for more details.
    """
    if has_torch_function_unary(input):
        return handle_torch_function(silu, (input,), input, inplace=inplace)
    if inplace:
        return torch._C._nn.silu_(input)
    return torch._C._nn.silu(input)


@torch_compile_guard()
def rmsnorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return rmsnorm2d_fwd(x, weight, eps).view(ori_shape)


@torch_compile_guard()
def rmsnorm2d_fwd_with_add_(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, eps: float, dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(out, x, residual, residual_out, weight, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


def fused_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    return out


@torch_compile_guard(gen_fake=fused_rmsnorm_pad_fake_tensors)
def fused_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    x_pad_to_multiple: int = 0,
) -> torch.Tensor:
    return fused_add_rmsnorm_pad(x, weight, epsilon, None, x_pad_to_multiple)


def fused_add_rmsnorm_pad_fake_tensors(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    M, N = x.shape
    N_out = (N + x_pad_to_multiple - 1) // x_pad_to_multiple * x_pad_to_multiple
    out = torch.empty((M, N_out), dtype=x.dtype, device=x.device)
    res_out = torch.empty((M, N), dtype=res.dtype, device=res.device)
    return out, res_out


@torch_compile_guard(gen_fake=fused_add_rmsnorm_pad_fake_tensors)
def fused_add_rmsnorm_pad_(
    x: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    res: torch.Tensor,
    x_pad_to_multiple: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return fused_add_rmsnorm_pad(x, weight, epsilon, res, x_pad_to_multiple)


def mxfp4_rms_quant_fuse_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shuffle: bool = False,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    M, N = x.shape
    out = torch.empty((M, N // 2), dtype=torch.float4_e2m1fn_x2, device=x.device)
    MXFP4_QUANT_BLOCK_SIZE = 32
    SCALE_N_valid = (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    use_scale_shuffle_padding = shuffle
    if use_scale_shuffle_padding:
        SCALE_M = ((M + 255) // 256) * 256
        SCALE_N = ((SCALE_N_valid + 7) // 8) * 8
    else:
        SCALE_M = M
        SCALE_N = SCALE_N_valid
    scale = torch.empty(
        (SCALE_M, SCALE_N),
        dtype=torch.float8_e8m0fnu,
        device=x.device,
    )
    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty_like(res1)
    return (out, scale, out_res1)


# It's important to use mutates_args=[] to avoid functionized_v2 op generation
@torch_compile_guard(gen_fake=mxfp4_rms_quant_fuse_fake, mutates_args=[])
def mxfp4_rms_quant_fuse(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    shuffle: bool = False,
    res1: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_mxfp4_quant import (
        fused_rms_mxfp4_quant,
    )

    (x_quant, x_scale), _, _, residual_out = fused_rms_mxfp4_quant(
        x, weight, eps, shuffle=shuffle, res1=res1
    )

    return x_quant, x_scale, residual_out


# ---------------------------------------------------------------------------
# Group-quant fused RMSNorm kernels (moved from deepseek_v2.py)
# ---------------------------------------------------------------------------


def _fuse_rmsnorm_fp4_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n1 = x1.shape
    n2 = x2.shape[1] if x2 is not None else 0

    out1_quantized = torch.empty((m, n1 // 2), dtype=torch.uint8, device=x1.device)
    scale_n_valid = (n1 + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
    scale_m = ((m + 255) // 256) * 256
    scale_n = ((scale_n_valid + 7) // 8) * 8
    out1_bs = torch.empty((scale_m, scale_n), dtype=torch.uint8, device=x1.device)

    out2 = None
    if x2 is not None:
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)
    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)
    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def _fused_rms_fp8_group_quant_fake(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: "torch.dtype | None" = None,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if dtype_quant is None and _aiter_dtypes is not None:
        dtype_quant = _aiter_dtypes.fp8
    m, n1 = x1.shape
    out1_quantized = torch.empty((m, n1), dtype=dtype_quant, device=x1.device)
    out1_bs = torch.empty(
        (m, (n1 + group_size - 1) // group_size), dtype=torch.float32, device=x1.device
    )
    if transpose_scale:
        out1_bs = out1_bs.transpose(0, 1).contiguous().view(*out1_bs.shape)
    out1_unquantized = None
    if output_unquantized_inp1:
        out1_unquantized = torch.empty_like(x1)
    out2 = None
    if x2 is not None:
        _, n2 = x2.shape
        out2 = torch.empty((m, n2), dtype=x1.dtype, device=x1.device)
    out_res1 = None
    if res1 is not None:
        out_res1 = torch.empty((m, n1), dtype=x1.dtype, device=x1.device)
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fuse_rmsnorm_fp4_quant_fake)
def _fuse_rmsnorm_fp4_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = True,
    output_unquantized_inp1: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_mxfp4_quant import fused_rms_mxfp4_quant

    m = x1.shape[0]
    shuffle_bool = shuffle and (m >= MXFP4_QUANT_BLOCK_SIZE)

    (out1_quantized, out1_bs), _out1_unquantized, out2, out_res1 = (
        fused_rms_mxfp4_quant(
            x1=x1,
            x1_weight=x1_weight,
            x1_epsilon=x1_epsilon,
            x2=x2,
            x2_weight=x2_weight,
            x2_epsilon=0.0 if x2_epsilon is None else x2_epsilon,
            res1=res1,
            shuffle=shuffle_bool,
            scale_shuffle_padding=scale_shuffle_padding,
            output_unquantized_inp1=output_unquantized_inp1,
        )
    )
    out1_unquantized = None
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


@torch_compile_guard(gen_fake=_fused_rms_fp8_group_quant_fake)
def _fused_rms_fp8_group_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: "torch.dtype | None" = None,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_group_quant

    if dtype_quant is None and _aiter_dtypes is not None:
        dtype_quant = _aiter_dtypes.fp8

    (out1_quantized, out1_bs), out1_unquantized, out2, out_res1 = (
        fused_rms_fp8_group_quant(
            x1,
            x1_weight,
            x1_epsilon,
            x2,
            x2_weight,
            x2_epsilon,
            group_size,
            dtype_quant,
            res1,
            output_unquantized_inp1,
            transpose_scale,
        )
    )
    return out1_quantized, out1_bs, out1_unquantized, out2, out_res1


def fuse_rmsnorm_group_quant(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: Optional[torch.Tensor] = None,
    x2_weight: Optional[torch.Tensor] = None,
    x2_epsilon: Optional[float] = None,
    res1: Optional[torch.Tensor] = None,
    dtype_quant: "torch.dtype | None" = None,
    shuffle: bool = True,
    scale_shuffle_padding: bool = False,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = False,
):
    """Dispatch fused RMSNorm + group quantization to the correct kernel.

    Supports FP8 per-group and MXFP4 quantization. Optionally normalizes a
    second tensor (x2) and fuses residual-add (res1) in the same kernel call.

    Returns: (out1_quantized, out1_scale), out1_unquantized, out2_normed, residual_out
    """
    if _aiter_dtypes is None:
        raise RuntimeError("aiter.dtypes not available")

    if dtype_quant is None:
        dtype_quant = _aiter_dtypes.fp8

    if dtype_quant == _aiter_dtypes.fp4x2:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fuse_rmsnorm_fp4_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                shuffle,
                scale_shuffle_padding,
                output_unquantized_inp1,
            )
        )
    elif dtype_quant == _aiter_dtypes.fp8:
        out1_quantized, out1_bs, out1_unquantized, out2, out_res1 = (
            _fused_rms_fp8_group_quant(
                x1,
                x1_weight,
                x1_epsilon,
                x2,
                x2_weight,
                x2_epsilon,
                res1,
                dtype_quant,
                group_size,
                output_unquantized_inp1,
                transpose_scale,
            )
        )
    else:
        raise ValueError(
            f"No fused rmsnorm quant kernel available for quant dtype: {dtype_quant}."
        )
    return (out1_quantized, out1_bs), out1_unquantized, out2, out_res1


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        x_pad_to_multiple: int = 0,
        fused_allreduce: bool = False,
        fused_quant: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        transpose_scale: bool = False,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.x_pad_to_multiple = x_pad_to_multiple
        self.fused_allreduce = fused_allreduce
        self.use_fused_quant = fused_quant
        self.tp_size = get_tensor_model_parallel_world_size()
        self.transpose_scale = transpose_scale
        self.shuffle = shuffle

        layer_quant_config = (
            LayerQuantConfig()
            if quant_config is None
            else quant_config.global_quant_config
        )
        quant_type = layer_quant_config["quant_type"]
        params_dtype = layer_quant_config["quant_dtype"]
        self.quant_type = quant_type
        self.params_dtype = params_dtype

        # Determine the fused quant path based on quant_config:
        # - "group": FP8 per-group (per_1x128/per_Token) or MXFP4 group (fp4x2)
        # - "per_tensor": FP8 per-tensor static (requires x_scale at forward time)
        # - "simple_mxfp4": MXFP4 simple (per_1x32 without group quant)
        # - None: no quantization fusion
        self._quant_path = None
        if fused_quant and _aiter_dtypes is not None:
            if params_dtype == _aiter_dtypes.fp8 and quant_type in (
                QuantType.per_1x128,
                QuantType.per_Token,
            ):
                self._quant_path = "group"
            elif (
                params_dtype == _aiter_dtypes.fp8 and quant_type == QuantType.per_Tensor
            ):
                self._quant_path = "per_tensor"
            elif params_dtype == _aiter_dtypes.fp4x2:
                self._quant_path = "group"
            elif quant_type == QuantType.per_1x32:
                self._quant_path = "simple_mxfp4"

    @mark_trace(prefix="rmsnorm", torch_compile=True)
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.x_pad_to_multiple > 0:
            assert (
                not self.fused_allreduce
            ), "fused_allreduce_rmsnorm is not supported with rms_norm padding!"
            if residual is None:
                x = fused_rmsnorm_pad_(x, self.weight, self.eps, self.x_pad_to_multiple)
                return x
            else:
                x, residual = fused_add_rmsnorm_pad_(
                    x, self.weight, self.eps, residual, self.x_pad_to_multiple
                )
                return x, residual
        if self.fused_allreduce and self.tp_size > 1:
            assert (
                residual is not None
            ), "fused_allreduce_rmsnorm requires residual input!"
            x, residual = tensor_model_parallel_fused_allreduce_rmsnorm(
                x,
                residual,
                self.weight,
                self.eps,
            )
            return x, residual

        # --- Fused quant paths (dispatched by _quant_path) ---
        if self._quant_path == "group":
            # FP8 per-group or MXFP4 group quantization
            (x_quant, x_scale), _, _, res_out = fuse_rmsnorm_group_quant(
                x,
                self.weight,
                self.eps,
                res1=residual,
                dtype_quant=self.params_dtype,
                shuffle=self.shuffle,
                scale_shuffle_padding=self.shuffle,
                group_size=128,
                transpose_scale=self.transpose_scale,
            )
            if residual is None:
                return (x_quant, x_scale)
            else:
                return (x_quant, x_scale), res_out

        if self._quant_path == "per_tensor" and x_scale is not None:
            from aiter.ops.triton.fused_fp8_quant import (
                fused_rms_fp8_per_tensor_static_quant,
            )
            import aiter as rocm_aiter

            rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8
            if residual is None:
                x, _, _, _ = fused_rms_fp8_per_tensor_static_quant(
                    x,
                    self.weight,
                    self.eps,
                    x_scale,
                    None,
                    None,
                    self.eps,
                    dtype_quant=rocm_aiter_fp8_dtype,
                    res1=None,
                )
                return (x, x_scale)
            else:
                x, _, _, residual = fused_rms_fp8_per_tensor_static_quant(
                    x,
                    self.weight,
                    self.eps,
                    x_scale,
                    None,
                    None,
                    self.eps,
                    dtype_quant=rocm_aiter_fp8_dtype,
                    res1=residual,
                )
                return (x, x_scale), residual

        if self._quant_path == "simple_mxfp4":
            if residual is None:
                x, x_scale, _ = mxfp4_rms_quant_fuse(
                    x, self.weight, self.eps, shuffle=self.shuffle
                )
                return x, x_scale
            else:
                x, x_scale, residual = mxfp4_rms_quant_fuse(
                    x, self.weight, self.eps, shuffle=self.shuffle, res1=residual
                )
                return (x, x_scale), residual

        # --- Plain RMSNorm (no fusion) ---
        if residual is None:
            x = rmsnorm2d_fwd_(x, self.weight, self.eps, self.dim)
            return x
        else:
            x, residual = rmsnorm2d_fwd_with_add_(
                x, self.weight, residual, self.eps, self.dim
            )
            return x, residual


class DualRMSNorm(nn.Module):
    """Fused dual RMSNorm + quantization for two inputs.

    Uses a single AITER kernel call to normalize both tensors and quantize
    the first, reducing kernel launch overhead vs two separate RMSNorm calls.
    In MLA attention, normalizes q_c via q_a_layernorm and kv_c via
    kv_a_layernorm in one fused call.

    Does NOT own weight parameters — references existing RMSNorm modules'
    weights so that checkpoint loading works correctly.
    """

    def __init__(
        self,
        norm1: RMSNorm,
        norm2: RMSNorm,
        quant_config: Optional[QuantizationConfig] = None,
        transpose_scale: bool = False,
        shuffle: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm1
        self.norm2 = norm2
        self.transpose_scale = transpose_scale
        self.shuffle = shuffle

        layer_quant_config = (
            LayerQuantConfig()
            if quant_config is None
            else quant_config.global_quant_config
        )
        self.params_dtype = layer_quant_config["quant_dtype"]

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> tuple:
        """Normalize x1 and x2, quantize x1.

        Returns: (x1_quant, x1_scale), x2_normed
        """
        (x1_quant, x1_scale), _, x2_normed, _ = fuse_rmsnorm_group_quant(
            x1,
            self.norm1.weight,
            self.norm1.eps,
            x2=x2,
            x2_weight=self.norm2.weight,
            x2_epsilon=self.norm2.eps,
            dtype_quant=self.params_dtype,
            shuffle=self.shuffle,
            group_size=128,
            transpose_scale=self.transpose_scale,
        )
        return (x1_quant, x1_scale), x2_normed


class RMSNormGated(nn.Module):
    """RMS Normalization with optional gating.

    This is a native PyTorch implementation that supports:
    - Standard RMS normalization
    - Group RMS normalization
    - Optional gating with SiLU activation
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        group_size: int | None = None,
        norm_before_gate: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Initialize RMSNormGated.

        Args:
            hidden_size: Size of the hidden dimension
            eps: Epsilon for numerical stability
            group_size: If not None, do GroupNorm with each group
                        having group_size elements.
                        group_size=None is equivalent to group_size=hidden_size
                        (i.e. there's only 1 group).
            norm_before_gate: If True and z is provided: out = norm(x) * silu(z)
                              If False and z is provided: out = norm(x * silu(z))
            dtype: Data type for parameters
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward_native(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Native PyTorch implementation of RMS normalization with gating.

        Args:
            x: Input tensor
            z: Optional gating tensor

        Returns:
            Normalized (and optionally gated) tensor

        If z is not None:
            - norm_before_gate=True: out = norm(x) * silu(z)
            - norm_before_gate=False: out = norm(x * silu(z))
        """
        # Apply gating before normalization if needed
        if z is not None and not self.norm_before_gate:
            x = x * silu(z)

        # RMS Normalization
        if self.group_size is None:
            # Standard RMS norm across the last dimension
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps)
            out = x_normed * self.weight
        else:
            # Group RMS norm
            from einops import rearrange

            x_group = rearrange(x, "... (g d) -> ... g d", d=self.group_size)
            variance = x_group.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x_group * torch.rsqrt(variance + self.eps)
            out = rearrange(x_normed, "... g d -> ... (g d)") * self.weight

        # Apply gating after normalization if needed
        if z is not None and self.norm_before_gate:
            out = out * silu(z)

        return out

    def forward_cuda(
        self, x: torch.Tensor, z: torch.Tensor | None = None
    ) -> torch.Tensor:

        if torch.compiler.is_compiling():
            return self.forward_native(x, z)
        return self.forward_native(x, z)

        # from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

        # return rmsnorm_fn(
        #     x,
        #     self.weight,
        #     self.bias,
        #     z=z,
        #     eps=self.eps,
        #     group_size=self.group_size,
        #     norm_before_gate=self.norm_before_gate,
        # )

    def forward(self, x: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:

        return self.forward_cuda(x, z)


class GemmaRMSNorm(nn.Module):
    """RMS normalization for Gemma.

    Two differences from the above RMSNorm:
        1. x * (1 + w) instead of x * w.
        2. (x * w).to(orig_dtype) instead of x.to(orig_dtype) * w.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    @staticmethod
    def forward_static(
        weight: torch.Tensor,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        orig_dtype = x.dtype
        if residual is not None:
            x = (
                x.float() + residual.float()
                if orig_dtype == torch.float16
                else x + residual
            )
            residual = x

        x = x.float()
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + variance_epsilon)
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        x = x * (1.0 + weight.float())
        x = x.to(orig_dtype)
        return x if residual is None else (x, residual)

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        return self.forward_static(self.weight.data, self.variance_epsilon, x, residual)

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if torch.compiler.is_compiling():
            return self.forward_native(x, residual)

        if not getattr(self, "_is_compiled", False):
            self.forward_static = torch.compile(self.forward_static)  # type: ignore
            self._is_compiled = True
        return self.forward_native(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.forward_cuda(x, residual)


@torch_compile_guard()
def layernorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return layernorm2d_fwd(x, weight, bias, eps).view(ori_shape)


@torch_compile_guard()
def layernorm2d_fwd_with_add_(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    layernorm2d_fwd_with_add(out, x, residual, residual_out, weight, bias, eps)
    return out.view(ori_shape), residual_out.view(ori_shape)


class LayerNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return layernorm2d_fwd_(x, self.weight, self.bias, self.eps, self.dim)
        else:
            return layernorm2d_fwd_with_add_(
                x, self.weight, residual, self.bias, self.eps, self.dim
            )
