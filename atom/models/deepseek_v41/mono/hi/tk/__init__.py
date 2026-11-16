# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's own host for the TileRT-derived DSpark decode step at TP8 on gfx950: the buffers (``alloc``), the code
objects and their launch (``codeobj``), one rank's step, its kernargs and graphs (``host``), and the CPU checks
(``check``). The step's launches are ``plan.decode_launches``'; the parameters ``weights.params``'.
"""
