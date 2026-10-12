# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The K3 mono kernels' sources, in every build's JIT cache key
(``atom.mono.plan.build_key``): this package and the shared mono framework.

Narrow on purpose. The port this grew from digested ``source_digest(".")``,
which resolves to the whole ``atom`` package: every edit anywhere in ATOM
invalidated every K3 kernel binary, and the rebuild storm hid which edit
actually changed a kernel.
"""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("mono", "models/kimi_k3/mono")
