# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""nn.Module wrappers for quantized linear — mirrors ``attention.py``'s
:class:`LumenAttention`.

Usage::

    from atomrl.lumen.modules import LumenLinear

    linear = LumenLinear(
        in_features=4096,
        out_features=4096,
        backend_type="aiter",        # or "triton"
    )
    output = linear(x)               # FP8 quant/dequant handled internally
"""

import torch
from torch import nn

from atomrl.lumen.ops.quantize import quantized_linear
from atomrl.lumen.quantize import QuantConfig, ScalingManager
from atomrl.lumen.quantize.config import _get_float8_e4m3

__all__ = ["LumenLinear"]


class LumenLinear(nn.Module):
    """``nn.Linear`` with FP8 quantized forward + backward.

    Supports AITER (hipBLASLt) and Triton (blockwise) backends, mirroring
    :class:`~atomrl.lumen.modules.attention.LumenAttention`.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If ``True``, adds a learnable bias. Default: ``True``.
        backend_type: ``"aiter"`` or ``"triton"``.
        fp8_dtype: Target FP8 dtype. ``None`` auto-detects based on GPU
            architecture.
        block_size: Block size for blockwise quantization (triton backend).
        config: Optional :class:`~atomrl.lumen.quantize.QuantConfig`.
                If not provided, a default config is created.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        backend_type: str = "aiter",
        fp8_dtype: torch.dtype | None = None,
        block_size: int = 128,
        config: QuantConfig | None = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.backend_type = backend_type
        self.fp8_dtype = fp8_dtype if fp8_dtype is not None else _get_float8_e4m3()
        self.block_size = block_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.scaling_manager = ScalingManager(config or QuantConfig())

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pqw = None
        w = self.weight
        if hasattr(w, "_fp8_desc") and w.dtype in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
        ):
            _sc = w._fp8_desc.scale
            # blockwise(2d) stores a 2D block scale used directly; per-tensor
            # uses the reciprocal (dequant factor).
            pqw = (w._fp8_desc.data, _sc if _sc.numel() > 1 else 1.0 / _sc)
        return quantized_linear(
            input,
            self.weight,
            self.bias,
            scaling_manager=self.scaling_manager,
            backend=self.backend_type,
            fp8_dtype=self.fp8_dtype,
            block_size=self.block_size,
            pre_quantized_weight=pqw,
            activation_tensor_id=getattr(self, "_activation_tensor_id", None),
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self.backend_type}, "
            f"fp8_dtype={self.fp8_dtype}, block_size={self.block_size}"
        )
