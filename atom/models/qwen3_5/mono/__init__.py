# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.5-MoE mono decode: each decoder layer one persistent kernel launch (``kernels``), over
ATOM's own weights and state (``runner``), routed by ``dispatch``."""
