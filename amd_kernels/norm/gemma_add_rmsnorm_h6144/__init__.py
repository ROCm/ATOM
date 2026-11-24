# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Residual add + Gemma RMSNorm at hidden 6144 on gfx950, FlyDSL: ``add_rmsnorm`` (one GPU; bf16 or MXFP8 out, the
split-K GEMM's slab reduce optionally in front) and ``allreduce_add_rmsnorm`` (the TP4 all-reduce in front, eager and
graphs). The torch contract is ``reference.py``."""
