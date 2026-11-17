# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono layer kernels (K1, K2a, K2b) and the stages they are built from (the per-rank widths: ``mono.kernels.dims``).

Each stage reproduces the rounding of the original kernel it replaces (the
reduction trees, the contractions, the quantization rules), as recorded in
each module; only the split of a long reduction may differ.
"""
