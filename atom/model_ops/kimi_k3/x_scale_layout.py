# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-T layout of the E8M0 per-1x128 scale a Kimi-K3 norm's fused quant (and
the attn-res kernel folding that norm) hands its GEMMs.

The scale is column-major by default. aiter's gfx1250 mxfp8_128 GEMM also reads
it row-major, which lets the producer write contiguous per-token bytes, but that
is not slower for the GEMM only with some kernel configs, and the config depends
on M. So the layout is picked per M, for a producer and every GEMM reading its
scale at once.
"""

import weakref

from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_blockscale_bpreshuffle_prefers_row_major_x_scale as _prefers_row_major,
)
from torch import nn

from atom.model_ops.layernorm import RMSNorm
from atom.model_ops.linear import LinearBase, register_x_scale_row_major_rule


def share_x_scale_layout(norm: RMSNorm, consumers: list[nn.Module]) -> None:
    """Hand ``norm``'s scale to ``consumers`` row-major at M rows exactly when
    every one of their GEMMs reads that layout at least as fast; ``consumers``
    must be every GEMM the scale reaches in forward. A consumer that is not a
    ``LinearBase`` keeps the scale column-major.

    The norm and the GEMMs share one ``linear.x_scale_row_major`` key, so they
    agree per M. The rule is first asked at the first forward, when loading has
    fixed the quant schemes, so it follows online quantisation."""
    if not consumers or not all(isinstance(c, LinearBase) for c in consumers):
        return
    refs = [weakref.ref(m) for m in (norm, *consumers)]

    def row_major(m: int) -> bool:
        producer, *gemms = (r() for r in refs)
        return producer.emits_col_major_e8m0_scale and all(
            c.takes_row_major_x_scale
            and _prefers_row_major(m, c.weight.shape[0], c.weight.shape[1])
            for c in gemms
        )

    key = register_x_scale_row_major_rule(row_major)
    norm.x_scale_row_major_key = key
    for c in consumers:
        c.x_scale_row_major_key = key
