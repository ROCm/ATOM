# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3's sparse block and ASM page sizes, in a module that imports
nothing: the startup checks of `Config` (which must not import aiter), the page
layouts and the draft read window read them beside the kernels that own them."""

# One sparse block == one KV page == one MXFP4 page of tokens.
SPARSE_BLOCK_SIZE = 128
# The ASM paged-attention kernel's physical page (the page-16 SHUFFLE layout).
ASM_PAGE_SIZE = 16
