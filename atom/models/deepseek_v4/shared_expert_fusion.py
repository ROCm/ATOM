# SPDX-License-Identifier: MIT
"""Shared-expert fused activation: scale layout and the width gate.

Kept beside ``Expert`` (V4 and V4.1) and out of ``deepseek_v4/__init__.py``
so a CPU test can load this file without importing the model.
"""

import logging
from functools import cache

from atom.utils import envs

logger = logging.getLogger(__name__)

# A native MX w2 (V4.1, 32x32 weights) reads a row-major 1x32 E8M0 x_scale.
# A blockscale w2 reads an FP32 1x128 one, transposed.
_FUSED_SCALE_NATIVE_MX = {"quant_block_size": 32, "scale_dtype_fmt": "ue8m0"}
_FUSED_SCALE_BLOCKSCALE = {"transpose_scale": True}


def fused_scale_kwargs(native_a8_group_rows: int | None) -> dict:
    """Scale kwargs for ``fused_clamp_act_mul`` given ``w2``'s activation layout."""
    if native_a8_group_rows is not None:
        return _FUSED_SCALE_NATIVE_MX
    return _FUSED_SCALE_BLOCKSCALE


@cache
def _log_unfused_width(intermediate_per_rank: int) -> None:
    """Log a rejected blockscale width once. Every layer on a rank shares it."""
    logger.info(
        "ATOM_V4_USE_TRITON_FUSION is set, but this rank's shared-expert "
        "intermediate is %d (a blockscale w2 needs fused_clamp_act_mul to see "
        "a multiple of 128). Using the unfused path.",
        intermediate_per_rank,
    )


def shared_expert_fused_act_enabled(
    intermediate_per_rank: int, native_a8_group_rows: int | None
) -> bool:
    """Whether this rank's shared expert takes ``fused_clamp_act_mul``.

    A native MX ``w2`` (V4.1-Flash, 32x32 weights) is always a whole number of
    32-column quant groups, so the fused path always applies: ``LinearBase``
    rejects an ``input_size`` that is not a multiple of 32
    (``atom/model_ops/linear.py``), and ``input_size`` is already the per-rank
    intermediate. A blockscale ``w2`` instead needs a multiple of 128; with the
    flag on and a width that is not a whole number of 128-groups, log once per
    width and keep the unfused path.
    """
    if not envs.ATOM_V4_USE_TRITON_FUSION:
        return False
    if native_a8_group_rows is not None:
        return True
    if intermediate_per_rank >= 128 and intermediate_per_rank % 128 == 0:
        return True
    _log_unfused_width(intermediate_per_rank)
    return False
