# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step tail's kernel sources' digest, part of every build's JIT key
(``atom.mono.plan.build_key``): this package and the device helpers its
kernels are written with."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest(
    "models/kimi_k3/step_tail/kernels.py",
    "models/kimi_k3/step_tail/layout.py",
    "models/deepseek_v41/mono/gfx1250/kernels/device.py",
)

# the draft launch's: the head's files and its own
MARKOV_SOURCES = source_digest(
    "models/kimi_k3/step_tail/kernels.py",
    "models/kimi_k3/step_tail/layout.py",
    "models/kimi_k3/step_tail/markov_kernels.py",
    "models/kimi_k3/step_tail/markov_layout.py",
    "models/deepseek_v41/mono/gfx1250/kernels/device.py",
)
