# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's V4.1 mono decode on gfx950 (``k1k2``): every decoder layer of a decode step of up to 48 rows as two fused
kernels, K1 and K2 (``kernels``), the DSpark draft's block backbone likewise (``draft_runner``); the wrappers that route
those steps (``dispatch``) and the deployments it serves (``config``). Selected by ``mono.selection``.
"""
