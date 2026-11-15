# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Index cache layout names, shared without importing any GPU kernels."""

PRESHUFFLE = "preshuffle"
# gfx1250: FP4 rows in natural order, scored by the OPUS kernels.
NATURAL = "natural"
# gfx950 OPUS: MFMA-permuted scales, distinct from FlyDSL's preshuffle.
MFMA = "mfma"
