# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Configuration for the ATOM diffusion subsystem."""

from dataclasses import dataclass
from enum import Enum, auto


class PerformanceMode(Enum):
    """Component placement policy.

    Only ``SPEED`` (everything resident) is implemented: 192 GB per GPU against
    a ~85 GB peak means offload and FSDP buy nothing, and omitting them is what
    keeps this subsystem small.
    """

    SPEED = auto()


@dataclass
class DiffusionConfig:
    """Top-level configuration for a diffusion pipeline replica.

    One replica owns ``num_gpus`` devices and serves one model variant. Variants
    that are separate checkpoint partitions (e.g. MiniMax-H3 ``fl2va`` vs
    ``ref2va``) are separate replicas, not two branches of one load.
    """

    model_path: str
    pipeline_class: str
    """Dotted path to the ``ComposedPipeline`` subclass to instantiate."""
    model_variant: str | None = None
    num_gpus: int = 1
    ulysses_degree: int = 1

    num_inference_steps: int = 50
    performance_mode: PerformanceMode = PerformanceMode.SPEED

    max_queued_jobs: int = 32
    """Admission cap. Beyond this the scheduler rejects rather than queues, so
    callers get backpressure instead of an unbounded wait on a minutes-long
    job."""
    max_concurrent_jobs: int = 1
    """In-flight generations per replica. The resident DiT plus activations
    dominate the GPU, so anything above 1 mostly trades latency for nothing."""

    warmup: bool = True
    """Run one throwaway denoise step at load, before reporting ready.

    The first DiT forward on a fresh process is ~11 s of aiter kernel JIT,
    allocator growth and GEMM selection against 563 ms for every later step.
    Free during a multi-minute load; 11 s of latency inside a request."""

    seed: int | None = None
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")
        if self.ulysses_degree < 1:
            raise ValueError(f"ulysses_degree must be >= 1, got {self.ulysses_degree}")
        # Ulysses splits one request's sequence across the group; the group
        # must tile the device set exactly.
        if self.ulysses_degree != self.num_gpus:
            raise ValueError(
                f"ulysses_degree must equal num_gpus: "
                f"{self.ulysses_degree} != {self.num_gpus}"
            )

        if self.max_concurrent_jobs < 1:
            raise ValueError(
                f"max_concurrent_jobs must be >= 1, got {self.max_concurrent_jobs}"
            )
        if self.max_queued_jobs < self.max_concurrent_jobs:
            raise ValueError(
                f"max_queued_jobs ({self.max_queued_jobs}) must be >= "
                f"max_concurrent_jobs ({self.max_concurrent_jobs})"
            )
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {self.num_inference_steps}"
            )
