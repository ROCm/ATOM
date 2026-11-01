# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Complete K1/K2 builders for the measured target and draft widths."""

TARGET_WIDTHS = (6, 12, 18, 24, 30, 36, 42, 48)
SQ_WORDS = 48 * 160 * 2


def builders(rows, layers):
    if layers == 40 and rows in TARGET_WIDTHS:
        if rows == 48:
            from .kernels.gfx1250_dspark_ct2_index_down import build_dspark_down
            from .kernels.gfx1250_dspark_wide_accum_k1 import build_dspark_k1
        else:
            from .kernels.gfx1250_dspark_dynamic_index_down import build_dspark_down
            from .kernels.gfx1250_dspark_dynamic_k1 import build_dspark_k1
    else:
        from .kernels.gfx1250_dspark_k1 import build_dspark_k1

        if layers == 3 and rows == 40:
            from .kernels.gfx1250_dspark_ct2_draft_index_down import build_dspark_down
        else:
            from .kernels.gfx1250_dspark_index_down import build_dspark_down
    return build_dspark_k1, build_dspark_down
