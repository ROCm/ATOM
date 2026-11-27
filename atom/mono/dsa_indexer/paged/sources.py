# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Model-independent paged indexer dependencies in the JIT cache key."""

from atom.mono.plan.build_key import source_digest

SOURCES = source_digest("mono")
