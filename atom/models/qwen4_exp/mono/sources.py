# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono kernels' sources, in every build's JIT cache key
(``atom.mono.plan.build_key``): the ``kernels`` package, which imports nothing
else of ATOM's. FlyDSL's own key misses an edit to a constant or helper in an
imported module, and the old binary then runs."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("models/qwen4_exp/mono/kernels")
