# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Backward-compatible re-export of Megatron IMPORT patches.

Implementations live in :mod:`atomrl.lumen.patches.runtime.megatron_import`.
Import this module or ``atomrl.lumen.patches.runtime`` to register IMPORT patches.
"""

import atomrl.lumen.patches.runtime
import atomrl.lumen.patches.runtime.megatron_import  # noqa: F401 — Megatron IMPORT registrations
from atomrl.lumen.patches.runtime.megatron_import import (
    _env_flag,
    _LumenSplitAlongDim,
    _post_eval_rewarm,
    install_all,
    install_cross_entropy,
    install_eval_recompute,
    install_fused_layer_norm,
    install_fused_residual_norm,
    install_fused_rope,
    install_fused_swiglu_triton,
    install_language_module_checkpoint_guard,
    install_mlp_fp8_store,
    install_mlp_recompute,
    install_mmap_checkpoint,
    install_optimizer_patches,
    install_post_eval_cache_clear,
    install_requires_grad_fix,
    install_split_along_dim,
    install_swiglu_fp8,
    remap_lora_state_dict,
)

__all__ = [
    "_LumenSplitAlongDim",
    "_env_flag",
    "_post_eval_rewarm",
    "install_all",
    "install_cross_entropy",
    "install_eval_recompute",
    "install_fused_layer_norm",
    "install_fused_residual_norm",
    "install_fused_rope",
    "install_fused_swiglu_triton",
    "install_language_module_checkpoint_guard",
    "install_mlp_fp8_store",
    "install_mlp_recompute",
    "install_mmap_checkpoint",
    "install_optimizer_patches",
    "install_post_eval_cache_clear",
    "install_requires_grad_fix",
    "install_split_along_dim",
    "install_swiglu_fp8",
    "remap_lora_state_dict",
]
