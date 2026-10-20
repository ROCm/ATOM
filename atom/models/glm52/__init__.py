# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM-5.2. The model itself is ``deepseek_v2.GlmMoeDsaForCausalLM`` (GLM-5.2
shares DeepSeek-V2's decoder); this package holds what is GLM-5.2's alone.

Resolution is deferred to first access. ``deepseek_v2`` imports AITER at module
load, so binding the class here eagerly would mean that importing any GPU-free
submodule -- ``mono.config``, ``mono.dispatch``, ``mono.kernels.layout`` --
drags in AITER and fails on a CPU runner, taking ten unit tests with it.
"""

import importlib

__all__ = ["GlmMoeDsaForCausalLM"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    return getattr(importlib.import_module("atom.models.deepseek_v2"), name)


def __dir__():
    return sorted([*globals(), *__all__])
