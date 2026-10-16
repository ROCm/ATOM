# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""BLASST block-skip threshold resolution.

BLASST (Block-skipping via Softmax Thresholding) lets a flash-attention kernel
drop a K/V tile whose contribution to the softmax is negligible: if the tile's
per-row max score sits more than ``log(threshold)`` below the running max, the
tile's V load and P@V matmul are skipped.

No torch or aiter imports here, so the threshold logic stays testable on CPU.
"""

import math

from atom.utils import envs

__all__ = ["blasst_threshold", "resolve_threshold"]


def blasst_threshold(alpha: float, beta: float, sparsity: float, seqlen: int) -> float:
    """Calibrated threshold for a target skip fraction at a given length.

        threshold = alpha * exp(beta * sparsity) / seqlen

    ``alpha``/``beta`` are calibration inputs; nothing in ATOM produces them.
    They are per-model -- see docs/environment_variables.md.

    ``sparsity`` is a target, not a guarantee: a tile elides only when every row
    in it agrees to skip, so achieved elision runs below the target.

    Args:
        alpha: Fitted scale coefficient. Must be > 0.
        beta: Fitted exponent coefficient.
        sparsity: Target fraction of K/V tiles to skip, in [0, 1).
        seqlen: Sequence length the threshold applies to. Must be > 0.

    Returns:
        The threshold, always > 0.

    Raises:
        ValueError: If any argument is outside the domain above. A bad value
            means a bad config, so failing beats silently degrading accuracy.
    """
    if alpha <= 0.0:
        raise ValueError(f"BLASST alpha must be > 0, got {alpha}")
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"BLASST sparsity must be in [0, 1), got {sparsity}")
    if seqlen <= 0:
        raise ValueError(f"BLASST seqlen must be > 0, got {seqlen}")
    return alpha * math.exp(beta * sparsity) / seqlen


def resolve_threshold(seqlen: int) -> float:
    """Resolve the block-skip threshold for this forward pass from env config.

    Precedence: an explicit ``ATOM_BLASST_THRESHOLD`` wins; otherwise a
    calibrated ALPHA/BETA/SPARSITY fit evaluated at ``seqlen``; otherwise off.

    Returns:
        The threshold, or 0.0 to run dense.
    """
    fixed = envs.ATOM_BLASST_THRESHOLD
    if fixed > 0.0:
        return fixed

    alpha = envs.ATOM_BLASST_ALPHA
    sparsity = envs.ATOM_BLASST_SPARSITY
    if alpha > 0.0 and sparsity > 0.0:
        return blasst_threshold(alpha, envs.ATOM_BLASST_BETA, sparsity, seqlen)

    return 0.0
