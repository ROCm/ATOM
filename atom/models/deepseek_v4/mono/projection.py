# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Read-only schema for the native BF16 indexer head-weight projection."""

import torch
from aiter import QuantType


class MonoIndexerProjection(torch.nn.Module):
    """Share the loaded weight and the native GEMM without mutating either input."""

    def __init__(self, native):
        super().__init__()
        if (
            native.quant_type != QuantType.No
            or native.bias is not None
            or native.weight.dtype != torch.bfloat16
        ):
            raise ValueError("V4 indexer projection must be unquantized bias-free BF16")
        self.native = native

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.native, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Native Linear's existing out= path declares only the destination
        # mutable. Its allocating GEMM conservatively marks weights mutable,
        # which invalidates Mono's binding during TR_HYBRID warmup.
        output = x.new_empty(
            (*x.shape[:-1], self.weight.shape[0]), dtype=torch.bfloat16
        )
        return self.native(x, out=output)
