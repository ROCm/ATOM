# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4.1 mono decode on gfx950 (MI355X): ATOM's per-layer K1 / K2 kernels (``k1k2``), the TileRT-structured
decode step (``tk``) and its PD decode server (``pd``); the high-interactivity monokernels stay in ``mono/hi``. Which
backend a deployment takes is ``mono.selection``'s; nothing here is imported unless it is chosen.
"""
