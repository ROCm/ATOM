# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Prepare/Finalize over aiter's FlyDSL intranode dispatch/combine.

Selected with ``--all2all-backend flydsl``. The FlyDSL op returns what MoRI's
IntraNode op returns (compact receive buffer of ``total_recv`` rows, scales,
weights, global top-k ids, ``total_recv`` as a 1-element tensor) and combine
takes the experts' output in the same receive order, so this reuses
:class:`MoriPrepareAndFinalize` through a call-surface shim. The kernels
choose their launch geometry by token count (aiter's tuned
``dispatch_combine_intranode`` table), so the MoRI geometry policy is unused.
mori is not imported.
"""

from typing import Any

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.mori_prepare_finalize import MoriPrepareAndFinalize
from atom.model_ops.fused_moe.routed_rows import RoutedRowsWire


class _FlyDSLOpShim:
    """MoRI ``EpDispatchCombineOp`` call surface over a FlyDSL op."""

    def __init__(self, op: Any):
        self._op = op

    def dispatch(self, x, weights, scales, indices, block_num=-1, warp_per_block=-1):
        if weights.dtype != torch.float32:
            weights = weights.float()
        return self._op.dispatch(x, weights, scales, indices)

    def combine(self, x, weights, indices, block_num=-1, warp_per_block=-1):
        return self._op.combine(x, weights, indices)


class FlyDSLPrepareAndFinalize(MoriPrepareAndFinalize):
    def __init__(
        self,
        flydsl_ops: list[Any],
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_format: RoutedRowsWire,
        quant_dtype: torch.dtype = None,
    ):
        if not flydsl_ops:
            raise ValueError("flydsl_ops must contain at least one handle")
        mk.FusedMoEPrepareAndFinalize.__init__(self)
        self._mori_ops = tuple(_FlyDSLOpShim(op) for op in flydsl_ops)
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_format = dispatch_format
        self.quant_dtype = quant_dtype
        self._use_split_send_recv = False
        self._use_handle_launch = False

    def _get_dispatch_config(self, num_tokens: int | None = None) -> tuple[int, int]:
        return -1, -1
