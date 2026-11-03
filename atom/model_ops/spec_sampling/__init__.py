# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lossless speculative sampling: the sampled DSpark draft and the
probability-ratio verify (`ATOM_SPEC_LOSSLESS_SAMPLING`).

The modules here share private kernels, constants and layout rules (the
`SCRATCH_ALIGN` boundary, the Philox key words, the -1 row-base sentinel), so
only what the rejection sampler and the drafter call is exported below; the
rest is this package's own.

The exports resolve lazily, so a module that needs no Triton (the scratch
allocator, the store) imports without it, and the sampled draft op, which
builds on the greedy op in `dspark_markov_argmax` that itself imports
`softmax_tiles` from here, is imported from its own module.
"""

import importlib

_EXPORTS = {
    "RatioVerifyInputs": "ratio_verify",
    "_ratio_sample": "ratio_verify",
    "ratio_verify_scratch": "ratio_verify",
    "warmup_ratio_verify": "ratio_verify",
    "SamplingScratch": "sampling_scratch",
    "Q_WIDTH": "sparse_q_store",
    "SparseQStore": "sparse_q_store",
    "_draft_token": "verify_kernels",
    "_finish_request": "verify_kernels",
}

__all__ = [
    "Q_WIDTH",
    "RatioVerifyInputs",
    "SamplingScratch",
    "SparseQStore",
    "_draft_token",
    "_finish_request",
    "_ratio_sample",
    "ratio_verify_scratch",
    "warmup_ratio_verify",
]


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
