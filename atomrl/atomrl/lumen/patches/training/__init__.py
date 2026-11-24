# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""TRAINING-phase patch helpers."""

from __future__ import annotations

from typing import Any

from atomrl.lumen.patches.registry import PatchPhase, apply_patches
from atomrl.lumen.patches.types import PatchResult

_training_loaded = False


def _ensure_training() -> None:
    global _training_loaded
    from atomrl.lumen.patches.registry import PatchRegistry

    if _training_loaded and not PatchRegistry.all():
        import importlib

        import atomrl.lumen.patches.training.megatron_hooks as _hooks

        importlib.reload(_hooks)
        return

    if _training_loaded:
        return

    import atomrl.lumen.patches.training.megatron_hooks  # noqa: F401

    _training_loaded = True


def apply_training_patches(
    *,
    tags: set[str] | None = None,
    names: set[str] | None = None,
    **kwargs: Any,
) -> dict[str, PatchResult]:
    """Apply TRAINING-phase patches (Megatron setup/eval hooks)."""
    _ensure_training()
    return apply_patches(
        PatchPhase.TRAINING,
        tags=tags,
        names=names,
        default_only=False,
        **kwargs,
    )
