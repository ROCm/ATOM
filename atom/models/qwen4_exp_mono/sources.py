# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sources carried in every Qwen FlyDSL MonoKernel cache key."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("mono", "models/qwen4_exp_mono")
