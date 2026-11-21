# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The step-gap kernels' sources' digest, part of every build's JIT key: this
package, and the mono kernels and device helpers its kernels are written with
(their own digest)."""

from atom.models.kimi_k3.mono.sources import SOURCES as MONO_SOURCES
from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("models/kimi_k3/step_gap") + MONO_SOURCES
