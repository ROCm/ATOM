# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The format a rank's routed rows travel in.

Routing runs on the rank that owns the rows, before any of them move -- an
expert-parallel dispatch, a DP gather, a PCP gather -- so the rows that move
need only be in the format the experts take. Where the experts take MXFP8,
the rows travel quantized with their e8m0 scales (half the bytes of bf16) and
fused_moe takes them as is.
"""

from dataclasses import dataclass
from typing import Any

import torch
from aiter import QuantType, dtypes, get_hip_quant


@dataclass(frozen=True)
class RoutedRowsWire:
    dtype: torch.dtype  # the rows as they travel
    quant_type: Any | None  # aiter QuantType for the quantizer; None: as is
    scale_dim: int
    scale_dtype: torch.dtype

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`x` as this wire sends it, with its scales (None when unquantized)."""
        if self.quant_type is None:
            return x, None
        if x.shape[0] == 0:
            return (
                torch.empty(x.shape, dtype=self.dtype, device=x.device),
                torch.empty(
                    (0, self.scale_dim), dtype=self.scale_dtype, device=x.device
                ),
            )
        # per_1x32 to fp8 defaults to fp32 scales; MXFP8 is the e8m0 ones
        return get_hip_quant(self.quant_type)(
            x, quant_dtype=self.dtype, scale_type=self.scale_dtype
        )


def pack_routed_rows(
    wire: RoutedRowsWire | None,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    """The rows a gather moves, in the format the experts take: the hidden
    states in `wire` (unless `x_scale` says they are already), their top-k
    weights and ids, then the wire's scales."""
    if x_scale is None and wire is not None:
        hidden_states, x_scale = wire.quantize(hidden_states)
    rows = (hidden_states, topk_weights, topk_ids)
    return rows if x_scale is None else (*rows, x_scale)


_CARRIERS = {4: torch.float32, 2: torch.bfloat16, 1: torch.bfloat16}


def gather_rows(gather, rows):
    """`gather(list of tensors) -> list of tensors` over `rows`, each carried
    as the float dtype of its byte width (int32 ids as fp32; fp8 / e8m0 byte
    pairs as bf16, so a one-byte row's width must be even): a gather moves
    bytes, but PyNCCL and the custom all-gather take float dtypes only."""
    gathered = gather([t.view(_CARRIERS[t.element_size()]) for t in rows])
    return tuple(g.view(t.dtype) for g, t in zip(gathered, rows))


def resolve_routed_rows_wire(
    in_dtype: torch.dtype,
    hidden_dim: int,
    *,
    fp4: bool = False,
    mxfp8_activations: bool = False,
) -> RoutedRowsWire:
    """The wire for rows of `hidden_dim` in `in_dtype`. Call once per layer.

    `fp4`: MXFP4 (`ATOM_MORI_FP4_DISPATCH`). `mxfp8_activations`: the experts
    take MXFP8 activations, so the rows travel in it.
    """
    if fp4:
        # fp4 blockwise is one scale per 32 elements, not per 128 as fp8 uses,
        # and get_hip_quant(per_1x32) emits e8m0 scales (1 byte), not fp32.
        return RoutedRowsWire(
            dtype=dtypes.fp4x2,
            quant_type=QuantType.per_1x32,
            scale_dim=hidden_dim // 32,
            scale_dtype=torch.float8_e8m0fnu,
        )
    if mxfp8_activations:
        return RoutedRowsWire(
            dtype=dtypes.fp8,
            quant_type=QuantType.per_1x32,
            scale_dim=hidden_dim // 32,
            scale_dtype=torch.float8_e8m0fnu,
        )
    return RoutedRowsWire(
        dtype=in_dtype, quant_type=None, scale_dim=0, scale_dtype=torch.float32
    )
