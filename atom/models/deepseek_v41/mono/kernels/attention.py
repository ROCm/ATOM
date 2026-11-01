# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compatibility import for the gfx950 kernel implementation."""
import importlib
import sys

sys.modules[__name__] = importlib.import_module("atom.models.deepseek_v41.mono.gfx950.kernels.attention")
