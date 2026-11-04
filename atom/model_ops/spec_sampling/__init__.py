# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lossless speculative sampling: the sampled DSpark draft and the
probability-ratio verify (`ATOM_SPEC_LOSSLESS_SAMPLING`).

The modules here share private kernels, constants and layout rules (the
`SCRATCH_ALIGN` boundary, the Philox key words, the -1 row-base sentinel);
callers import what they use from the modules themselves. Nothing on the
default path depends on this package: the helpers it shares with the
exact-match verify and the greedy draft live outside it (`verify_kernels`,
`tile_argmax`).
"""
