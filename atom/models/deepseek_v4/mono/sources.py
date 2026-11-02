# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device build dependencies, including plans outside FlyDSL's source walk.

Serving-only edits do not rebuild kernels. Device helpers, arithmetic profiles,
formats and task/layout plans do, even when reached inside a traced branch.
"""

from atom.mono.plan.build_key import source_digest

SOURCE_PATHS = (
    "mono/device",
    "models/deepseek_v4/mono/config.py",
    "models/deepseek_v4/mono/kernels",
    "models/deepseek_v4/mono/sources.py",
)
SOURCES = source_digest(*SOURCE_PATHS)
