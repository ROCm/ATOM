# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""CONFIG_BUILD / ARGS / MODEL_BUILD patch helpers."""

from __future__ import annotations

from typing import Any

from atomrl.lumen.patches.registry import PatchPhase, apply_patches
from atomrl.lumen.patches.types import PatchResult

_builders_loaded = False
_model_builders_loaded = False


def _ensure_builders() -> None:
    import importlib

    global _builders_loaded
    from atomrl.lumen.patches.registry import PatchRegistry

    # Re-register when the registry has been cleared (tests only).
    if _builders_loaded and not PatchRegistry.all():
        import atomrl.lumen.patches.builders.dsv4 as _m_dsv4
        import atomrl.lumen.patches.builders.llama as _m_llama
        import atomrl.lumen.patches.builders.megatron_args as _m_megatron_args

        importlib.reload(_m_megatron_args)
        importlib.reload(_m_dsv4)
        importlib.reload(_m_llama)
        _reload_model_builders()
        return

    if _builders_loaded:
        return

    from atomrl.lumen.patches.builders import dsv4 as _dsv4  # noqa: F401
    from atomrl.lumen.patches.builders import llama as _llama  # noqa: F401
    from atomrl.lumen.patches.builders import (
        megatron_args as _megatron_args,  # noqa: F401
    )

    _builders_loaded = True


def _reload_model_builders() -> None:
    import importlib

    global _model_builders_loaded
    import atomrl.lumen.patches.builders.megatron_model as _megatron_model

    importlib.reload(_megatron_model)
    _model_builders_loaded = True


def _ensure_model_builders() -> None:
    global _model_builders_loaded

    _ensure_builders()
    if _model_builders_loaded:
        return

    import atomrl.lumen.patches.builders.megatron_model as _megatron_model  # noqa: F401

    _model_builders_loaded = True


def apply_config_build(
    config: Any,
    args: Any,
    *,
    tags: set[str] | None = None,
) -> dict[str, PatchResult]:
    """Apply CONFIG_BUILD patches to a Megatron ``TransformerConfig``."""
    _ensure_builders()
    return apply_patches(
        PatchPhase.CONFIG_BUILD,
        config=config,
        args=args,
        tags=tags,
    )


def apply_args_patches(
    parser: Any,
    *,
    tags: set[str] | None = None,
    names: set[str] | None = None,
    **kwargs: Any,
) -> dict[str, PatchResult]:
    """Register CLI argument groups via ARGS-phase patches."""
    _ensure_builders()
    return apply_patches(
        PatchPhase.ARGS,
        parser=parser,
        tags=tags,
        names=names,
        default_only=False,
        **kwargs,
    )


def apply_model_build(
    *,
    config: Any,
    args: Any,
    tags: set[str] | None = None,
    names: set[str] | None = None,
    **kwargs: Any,
) -> dict[str, PatchResult]:
    """Apply MODEL_BUILD side-effect patches before/after module construction."""
    _ensure_model_builders()
    return apply_patches(
        PatchPhase.MODEL_BUILD,
        config=config,
        args=args,
        tags=tags,
        names=names,
        default_only=False,
        **kwargs,
    )
