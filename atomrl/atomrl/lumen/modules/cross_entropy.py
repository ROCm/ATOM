# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Lumen parallel cross-entropy loss wrapper.

When ``--use-sdma`` is enabled the TP all-gather inside the cross-entropy
forward is routed through mori SDMA.
"""

import torch

from atomrl.lumen.modules.parallel_linear import _use_sdma_from_args
from atomrl.lumen.ops.cross_entropy import parallel_cross_entropy as _parallel_ce

__all__ = ["lumen_parallel_cross_entropy"]


def _ce_chunk_rows_from_args() -> int:
    """Return --lumen-ce-chunk-rows (0 = disabled) without hard-importing Megatron."""
    try:
        from megatron.training import get_args

        return getattr(get_args(), "lumen_ce_chunk_rows", 0)
    except (AssertionError, ImportError, RuntimeError):
        return 0


def lumen_parallel_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    tp_group: torch.distributed.ProcessGroup,
    is_cg_capturable: bool = False,
) -> torch.Tensor:
    """Vocab-parallel cross-entropy using Lumen Triton kernels.

    Args:
        logits: ``[B, SQ, V // tp_size]``
        labels: ``[B, SQ]`` with values in ``[0, V)``
        tp_group: tensor-parallel process group.
        is_cg_capturable: if ``True``, skip a ``torch.equal`` check
            that would break CUDA-graph capture.

    Returns:
        Per-token loss ``[B, SQ]``.
    """
    return _parallel_ce(
        logits,
        labels,
        0.0,  # label_smoothing
        False,  # reduce_loss
        tp_group,
        -100,  # ignore_idx
        is_cg_capturable,
        _use_sdma_from_args(),
        _ce_chunk_rows_from_args(),
    )
