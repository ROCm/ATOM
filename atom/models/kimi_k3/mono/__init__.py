# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3's fused per-layer decode: the KDA layers' two persistent kernels.

Nothing is imported here. ``config``, ``sources`` and ``contract`` must stay
importable without a GPU (the contract check builds compile-only), so the model,
AITER and the runner are reached only through their own modules.
"""
