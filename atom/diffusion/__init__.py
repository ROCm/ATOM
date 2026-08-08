# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Diffusion (video/image generation) subsystem for ATOM.

ATOM's LLM engine assumes an autoregressive execution model: a KV cache, a
prefill/decode split, and token-level continuous batching. Diffusion inference
shares none of that. A request is one job that runs a fixed number of denoise
steps over several heterogeneous components (text encoder, DiT, VAEs), takes
minutes rather than milliseconds, and is parallelised by splitting a *single*
request's sequence across every GPU (Ulysses) rather than by sharding a
replicated batch.

So this is a sibling of ``atom.model_engine``, not an extension of it. What it
does reuse from the LLM side: ``atom.model_ops`` layers, ``atom.model_loader``,
``atom.utils.distributed``, and the ZMQ multi-process pattern.

Entry points:
    ``DiffusionConfig``  -- :mod:`atom.diffusion.config`
    ``DiffusionJob``     -- :mod:`atom.diffusion.request`
    ``PipelineStage``    -- :mod:`atom.diffusion.stages.base`
    ``ComposedPipeline`` -- :mod:`atom.diffusion.pipelines.base`
    ``UlyssesGroup``     -- :mod:`atom.diffusion.distributed.ulysses`
    ``JobScheduler``     -- :mod:`atom.diffusion.engine.job_scheduler`
    ``PipelineRunner``   -- :mod:`atom.diffusion.engine.pipeline_runner`
"""

from atom.diffusion.config import ComponentConfig, DiffusionConfig
from atom.diffusion.request import DiffusionJob, JobStatus

__all__ = [
    "ComponentConfig",
    "DiffusionConfig",
    "DiffusionJob",
    "JobStatus",
]
