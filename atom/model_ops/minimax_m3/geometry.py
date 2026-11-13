# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's sparse block size, in a module that imports nothing: the startup
checks of `Config` (which must not import aiter) and the page layouts read it
beside the kernels that own it."""

# One sparse block == one KV page == one MXFP4 page of tokens.
SPARSE_BLOCK_SIZE = 128
