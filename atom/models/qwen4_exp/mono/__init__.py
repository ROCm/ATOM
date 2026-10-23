# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next decode monokernel: one FlyDSL launch per decoder layer.

``kernels/`` holds the device code; the host side binds ATOM's loaded
weights and caches.
"""
