# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""IMPORT-phase patch: Lumen fused MoE router ops in Megatron moe_utils."""

from __future__ import annotations

import logging

from atomrl.lumen.patches.registry import PatchPhase, register_patch

logger = logging.getLogger(__name__)


def install_moe_fused_router() -> None:
    """Monkey-patch Megatron-Core ``moe_utils`` with Lumen fused router ops."""
    try:
        from megatron.core.transformer.moe import moe_utils
    except ImportError:
        logger.debug("Megatron-Core moe_utils not found, skipping MoE router patch")
        return

    if getattr(moe_utils, "_lumen_fused_router_patched", False):
        return

    try:
        from atomrl.lumen.ops.moe.fused_router import (
            fused_compute_score_for_moe_aux_loss,
            fused_moe_aux_loss,
            fused_topk_with_score_function,
        )
    except ImportError as exc:
        # IMPORT-phase patches are applied unconditionally, so a dense run --
        # which never touches a router -- would otherwise die here on its way to
        # GPTModel. Megatron keeps its own router; only MoE runs lose the AITER
        # kernels, and they are no worse off than without this patch.
        #
        # Only this module's own absence is expected, and it arrives as a plain
        # ImportError rather than ModuleNotFoundError: the module is present and
        # the three names are not. Anything attributable to another module -- a
        # broken AITER, say -- has to propagate, because swallowing it would turn
        # a bad environment into a silent slow path for every MoE run.
        if exc.name is not None and not exc.name.startswith("atomrl.lumen.ops.moe"):
            raise
        logger.warning(
            "Lumen fused router ops unavailable (%s), skipping MoE router patch",
            exc,
        )
        # Mark it handled so a later apply_patches pass does not warn again.
        moe_utils._lumen_fused_router_patched = True
        return

    moe_utils.fused_topk_with_score_function = fused_topk_with_score_function
    moe_utils.fused_compute_score_for_moe_aux_loss = (
        fused_compute_score_for_moe_aux_loss
    )
    moe_utils.fused_moe_aux_loss = fused_moe_aux_loss

    try:
        import megatron.core.extensions.transformer_engine as te_ext

        te_ext.fused_topk_with_score_function = fused_topk_with_score_function
        te_ext.fused_compute_score_for_moe_aux_loss = (
            fused_compute_score_for_moe_aux_loss
        )
        te_ext.fused_moe_aux_loss = fused_moe_aux_loss
    except ImportError:
        pass

    moe_utils._lumen_fused_router_patched = True
    logger.info("Patched Megatron-Core moe_utils with Lumen fused router ops")


register_patch(
    "moe_fused_router",
    PatchPhase.IMPORT,
    description="Lumen fused MoE router top-k and aux-loss ops in moe_utils",
    tags=frozenset({"core", "moe", "megatron"}),
)(install_moe_fused_router)
