# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next (``qwen4_exp``): the original implementation (``model``).
Every name this module used to hold still resolves here, so
``from atom.models.qwen4_exp import Qwen4ExpForConditionalGeneration`` is
unchanged.

Code that *patches* module globals (the SGLang plugin swaps
``install_stacked_expert_loaders``) must patch ``atom.models.qwen4_exp.model``:
the classes resolve their globals there, and an assignment on this package
would bind a name nothing reads.

Resolution is deferred to first access, so importing a GPU-free submodule does
not drag in the model and AITER.
"""

import importlib


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(importlib.import_module(f"{__name__}.model"), name)


def __dir__():
    return sorted(dir(importlib.import_module(f"{__name__}.model")))
