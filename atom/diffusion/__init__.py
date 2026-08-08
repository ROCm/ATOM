# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Diffusion (video/audio generation) subsystem for ATOM.

A sibling of ``atom.model_engine``, not an extension of it: no KV cache, no
prefill/decode split, no token-level batching. One request is one job of N
denoise steps over several heterogeneous components, taking minutes, and
parallelised by splitting a *single* request's sequence across every GPU
(Ulysses) rather than sharding a replicated batch.

Reused from the LLM side: ``atom.model_ops``, ``atom.model_loader``,
``atom.utils.distributed``, and the ZMQ multi-process pattern.
"""

from atom.diffusion.config import ComponentConfig, DiffusionConfig
from atom.diffusion.request import DiffusionJob, JobStatus

__all__ = [
    "ComponentConfig",
    "DiffusionConfig",
    "DiffusionJob",
    "JobStatus",
]
