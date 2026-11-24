# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Shared types for the Lumen patch registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class PatchPhase(Enum):
    """When a patch is applied in the training stack."""

    SOURCE = auto()  # Megatron source on disk (container bootstrap)
    IMPORT = auto()  # import-time monkey patch
    ARGS = auto()  # CLI arg registration
    CONFIG_BUILD = auto()  # after core_transformer_config_from_args
    MODEL_BUILD = auto()  # model provider / builder
    TRAINING = auto()  # training loop hooks


@dataclass(frozen=True)
class PatchSpec:
    """Declarative metadata for a registered patch."""

    name: str
    phase: PatchPhase
    fn: Callable[..., Any]
    description: str = ""
    enabled: Callable[[], bool] = field(default=lambda: True)
    depends_on: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    config_fields: tuple[str, ...] = ()
    default: bool = True


@dataclass
class PatchResult:
    """Outcome of applying a single patch."""

    name: str
    applied: bool = False
    skipped: bool = False
    reason: str = ""
    return_value: Any = None
