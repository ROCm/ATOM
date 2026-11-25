# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Runtime (import-time) patch registrations."""

from atomrl.lumen.patches.runtime import (
    moe_fused_router as moe_fused_router,
)

__all__ = ["moe_fused_router"]


def apply_import_patches(*, default_only: bool = True, dry_run: bool = False):
    """Apply registered IMPORT-phase Megatron monkey-patches."""
    import atomrl.lumen.patches.runtime.megatron_import  # noqa: F401
    from atomrl.lumen.patches import PatchPhase, apply_patches

    return apply_patches(
        PatchPhase.IMPORT,
        default_only=default_only,
        dry_run=dry_run,
    )
