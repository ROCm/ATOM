# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Diffusion (video/audio generation) subsystem for ATOM.

A sibling of ``atom.model_engine``, not an extension: no KV cache, no
prefill/decode split, no token-level batching. One request is one minutes-long
job of N denoise steps over several heterogeneous components, parallelised by
splitting a *single* request's sequence across every GPU (Ulysses).
"""

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.request import DiffusionJob, JobStatus

__all__ = [
    "DiffusionConfig",
    "DiffusionJob",
    "JobStatus",
]
