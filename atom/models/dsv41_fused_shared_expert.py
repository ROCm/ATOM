# SPDX-License-Identifier: MIT
"""Shared-expert fused activation: scale layout and the width gate.

Kept out of ``deepseek_v4/__init__.py`` so a CPU test can import it without
loading the model.
"""

import logging
from typing import Any

from atom.utils import envs

logger = logging.getLogger(__name__)

# A native MX w2 (V4.1, 32x32 weights) reads a row-major 1x32 E8M0 x_scale.
# A blockscale w2 reads an FP32 1x128 one, transposed.
_FUSED_SCALE_NATIVE_MX = {"quant_block_size": 32, "scale_dtype_fmt": "ue8m0"}
_FUSED_SCALE_BLOCKSCALE = {"transpose_scale": True}
_fused_width_fallback_logged = False


def fused_scale_kwargs(native_a8_group_rows: Any | None) -> dict:
    """Scale kwargs for ``fused_clamp_act_mul`` given ``w2``'s activation layout."""
    if native_a8_group_rows is not None:
        return _FUSED_SCALE_NATIVE_MX
    return _FUSED_SCALE_BLOCKSCALE


def shared_expert_fused_act_enabled(
    intermediate_per_rank: int, native_mx: bool = False
) -> bool:
    """Whether this rank's shared expert takes ``fused_clamp_act_mul``.

    The kernel quantizes whole scale groups of the half width: 32 columns for a
    native MX ``w2`` (1x32 E8M0 activations), 128 for a blockscale one. With the
    flag on and a width that is not a whole number of groups, log once and keep
    the unfused path. V4.1-Flash's shared expert (2304) is native MX, so every TP
    up to 8 qualifies (576 at TP4, 288 at TP8).
    """
    global _fused_width_fallback_logged
    if not envs.ATOM_V4_USE_TRITON_FUSION:
        return False
    group = 32 if native_mx else 128
    if intermediate_per_rank >= group and intermediate_per_rank % group == 0:
        return True
    if not _fused_width_fallback_logged:
        _fused_width_fallback_logged = True
        logger.info(
            "ATOM_V4_USE_TRITON_FUSION is set, but this rank's shared-expert "
            "intermediate is %d (fused_clamp_act_mul needs a multiple of %d, "
            "at least %d). Using the unfused path.",
            intermediate_per_rank,
            group,
            group,
        )
    return False
