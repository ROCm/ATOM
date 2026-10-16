# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 model and optional execution backends.

The production model remains available through the historical package exports,
for example ``from atom.models.deepseek_v4 import DeepseekV4ForCausalLM``.
Those exports are loaded on first access so importing the independent
``deepseek_v4.mono`` admission and lifecycle modules does not require AITER or
FlyDSL during CPU-only test collection.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType

_MODEL_MODULE = f"{__name__}.model"


def _model() -> ModuleType:
    return importlib.import_module(_MODEL_MODULE)


class _DeepseekV4Package(ModuleType):
    """Lazy compatibility facade for the former single-file model module."""

    def __getattr__(self, name: str):
        model = _model()
        if name == "model":
            return model
        try:
            return getattr(model, name)
        except AttributeError as error:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from error

    def __setattr__(self, name: str, value) -> None:
        model = sys.modules.get(_MODEL_MODULE)
        if model is not None and hasattr(model, name):
            setattr(model, name, value)
            return
        super().__setattr__(name, value)

    def __dir__(self):
        model = sys.modules.get(_MODEL_MODULE)
        if model is None:
            return super().__dir__()
        return sorted(set(super().__dir__()) | set(dir(model)))


sys.modules[__name__].__class__ = _DeepseekV4Package
