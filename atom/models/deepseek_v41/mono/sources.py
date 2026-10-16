# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono kernels' sources, in every build's JIT cache key
(``atom.mono.plan.build_key``): this package, the shared mono framework, and the
decode router whose device helpers the MoE kernel calls."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest(
    "mono", "models/deepseek_v41/mono", "model_ops/deepseek_v41/router.py"
)
