# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Configuration for the ATOM diffusion subsystem."""

from dataclasses import dataclass, field
from enum import Enum, auto


class PerformanceMode(Enum):
    """Component placement policy.

    Only ``SPEED`` (everything resident on device) is implemented. MI300-class
    parts have 192 GB per GPU against a ~85 GB peak for MiniMax-H3 at
    Ulysses-8, so offload and FSDP buy nothing here and are deliberately out of
    scope -- that omission is what keeps this subsystem small.
    """

    SPEED = auto()


@dataclass
class ComponentConfig:
    """One loadable model component of a diffusion pipeline.

    A pipeline is several independent networks, not one model, so each carries
    its own checkpoint subfolder and dtype. ``params_dtype`` is a string
    (``"bfloat16"``, ``"float32"``) so configs stay picklable across the ZMQ
    boundary without importing torch.
    """

    name: str
    """Registry key the pipeline uses to look this component up."""
    class_path: str
    """Dotted path to the ``nn.Module`` subclass implementing the component."""
    subfolder: str = ""
    """Checkpoint subfolder relative to the model root ("" = root)."""
    params_dtype: str = "bfloat16"
    """Parameter dtype name, resolved against torch at load time."""
    tp_sharded: bool = False
    """Whether this component's linears shard across the tensor-parallel group.

    The DiT is Ulysses-parallel with ``tp_size == 1``, so its linears are
    replicated and this stays False; a wide text encoder may set it True.
    """

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ComponentConfig.name must be non-empty")
        if not self.class_path or "." not in self.class_path:
            raise ValueError(
                f"ComponentConfig.class_path must be a dotted path, "
                f"got {self.class_path!r}"
            )


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
    components: list[ComponentConfig] = field(default_factory=list)

    model_variant: str | None = None
    num_gpus: int = 1
    ulysses_degree: int = 1
    tp_size: int = 1

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
    """Run one throwaway denoise step at load, before the replica reports ready.

    The first DiT forward on a fresh process costs far more than the rest: on
    gfx950 at Ulysses-8, step 1 is 11.6 s against 563 ms for every later step,
    which is aiter kernel JIT, allocator growth and GEMM selection, not model
    work. Paying it during a multi-minute load is free; paying it inside the
    first generation is 11 s of that request's latency."""

    seed: int | None = None
    output_dir: str = "outputs"

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {self.num_gpus}")
        if self.ulysses_degree < 1:
            raise ValueError(f"ulysses_degree must be >= 1, got {self.ulysses_degree}")
        if self.tp_size < 1:
            raise ValueError(f"tp_size must be >= 1, got {self.tp_size}")

        # Ulysses splits one request's sequence across the group and trades it
        # for heads inside attention; TP shards the same tokens across a
        # different axis. The two groups must tile the device set exactly.
        if self.ulysses_degree * self.tp_size != self.num_gpus:
            raise ValueError(
                f"ulysses_degree * tp_size must equal num_gpus: "
                f"{self.ulysses_degree} * {self.tp_size} != {self.num_gpus}"
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

        names = [c.name for c in self.components]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ValueError(f"duplicate component names: {duplicates}")

    def component(self, name: str) -> ComponentConfig:
        """Look up a component config by registry key."""
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(
            f"no component named {name!r}; have {sorted(c.name for c in self.components)}"
        )
