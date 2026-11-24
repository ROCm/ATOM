# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Unified patch registry for Lumen.

Patches are registered with :func:`register_patch` and applied via
:func:`apply_patches` according to their :class:`PatchPhase`.
"""

from atomrl.lumen.patches.builders import (
    apply_args_patches,
    apply_config_build,
    apply_model_build,
)
from atomrl.lumen.patches.cli import apply_megatron_source_patches, print_patch_report
from atomrl.lumen.patches.registry import (
    PatchRegistry,
    apply_patches,
    list_patches,
    register_patch,
)
from atomrl.lumen.patches.training import apply_training_patches
from atomrl.lumen.patches.types import PatchPhase, PatchResult, PatchSpec

__all__ = [
    "PatchPhase",
    "PatchRegistry",
    "PatchResult",
    "PatchSpec",
    "apply_args_patches",
    "apply_config_build",
    "apply_megatron_source_patches",
    "apply_model_build",
    "apply_patches",
    "apply_training_patches",
    "list_patches",
    "print_patch_report",
    "register_patch",
]
