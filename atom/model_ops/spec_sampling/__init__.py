# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lossless speculative sampling: the sampled DSpark draft and the
probability-ratio verify (`ATOM_SPEC_LOSSLESS_SAMPLING`).

The modules here share private kernels, constants and layout rules (the
`SCRATCH_ALIGN` boundary, the Philox key words, the -1 row-base sentinel);
callers import what they use from the modules themselves. It depends on
the default path, not the reverse: the helpers it shares with the exact-match
verify and the greedy draft live outside it (`verify_kernels`, `tile_argmax`,
and the Markov ops' shared stage 1, `dspark_markov_stage1`), and no default
kernel calls into it. `rejection_sampler` and the DSpark
proposer import `ratio_verify` / `dspark_lossless` at module scope only to
dispatch to them when the flag arms them.
"""
