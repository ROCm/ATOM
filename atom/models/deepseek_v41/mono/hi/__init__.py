# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The high-interactivity (hi) monokernel runtime of DeepSeek-V4.1-Flash on gfx950 (MI355X) at TP4, as the V4.1 mono
decode's hi hook serves it (``ATOM_DSV41_MONO_HI``: ``atom/models/deepseek_v41/mono/hi_runner.py``).

The layer recipe and schedules (``v41``), the gfx950 blocks and device code (``blocks.gfx950``,
``device.gfx950``), the planner (``plan``), the emitter and compiler driver (``emit``) and the runtime (``runtime``:
kernels, peer memory, dual-copy and uncached scratch). Imports within the package are relative; it imports nothing
from ``atom``.
"""
