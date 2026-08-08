# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Composed pipeline: an ordered list of stages plus a component registry."""

import logging
import time
from abc import ABC, abstractmethod

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.distributed.ulysses import UlyssesGroup
from atom.diffusion.stages.base import (
    DiffusionBatch,
    PipelineStage,
    StageParallelism,
)

logger = logging.getLogger(__name__)


class ComposedPipeline(ABC):
    """Base class for diffusion pipelines.

    Subclasses declare their stage order and required components; this class
    owns execution, parallelism dispatch and per-stage timing.
    """

    pipeline_name: str = "ComposedPipeline"

    #: Component registry keys this pipeline cannot run without.
    required_components: tuple[str, ...] = ()

    def __init__(
        self,
        config: DiffusionConfig,
        ulysses: UlyssesGroup | None = None,
        *,
        model_root: str = "",
    ) -> None:
        self.config = config
        self.ulysses = ulysses or UlyssesGroup()
        # Every diffusion pipeline loads from a checkpoint root, and the worker
        # constructs whatever class the config names -- so this belongs here
        # rather than as a kwarg only one subclass happens to accept.
        self.model_root = model_root or config.model_path
        self.components: dict[str, object] = {}
        self.stages: list[PipelineStage] = list(self.build_stages())
        if not self.stages:
            raise ValueError(f"{self.pipeline_name} declared no stages")
        self.last_stage_times: dict[str, float] = {}

    @abstractmethod
    def build_stages(self) -> list[PipelineStage]:
        """Return the pipeline's stages, in execution order."""

    # ------------------------------------------------------------------
    # components
    # ------------------------------------------------------------------

    host_staged_components: tuple[str, ...] = ()
    """Components deliberately kept in host memory and moved in per use.

    Resident placement is the default and the right one for anything on the
    denoise critical path. This is the exception list for components that run
    once per request and are large enough that co-residency does not fit.
    """

    def register_component(self, name: str, module: object) -> None:
        self.components[name] = module

    def component(self, name: str) -> object:
        if name not in self.components:
            raise KeyError(
                f"component {name!r} not registered; have {sorted(self.components)}"
            )
        return self.components[name]

    def load_components(self) -> None:
        """Build every component from the checkpoint and register it.

        Left to subclasses on purpose. A diffusion pipeline is several
        independent networks with different loaders -- H3's DiT needs a
        grouped-QKV reorder, its VAEs load through the checkpoint's own
        ``auto_map`` classes, and its text encoder is a truncated Qwen3-VL --
        so a generic "instantiate class_path and load a state dict" would fit
        none of them.
        """
        raise NotImplementedError(
            f"{self.pipeline_name} does not implement load_components(); "
            "register components explicitly or implement the hook"
        )

    def verify_components(self) -> None:
        missing = [c for c in self.required_components if c not in self.components]
        if missing:
            raise RuntimeError(
                f"{self.pipeline_name} is missing required components: {missing}"
            )

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------

    def _should_run(self, stage: PipelineStage) -> bool:
        if stage.parallelism is StageParallelism.REPLICATED:
            return True
        return self.ulysses.is_main

    def forward(self, batch: DiffusionBatch) -> DiffusionBatch:
        """Run every stage in order and return the final batch."""
        self.verify_components()
        self.last_stage_times = {}

        for stage in self.stages:
            t0 = time.perf_counter()

            if self._should_run(stage):
                batch = stage(batch, self.config)
            elif stage.parallelism is StageParallelism.MAIN_RANK_BROADCAST:
                # Non-main ranks still need the declared outputs, so they wait
                # here rather than skipping ahead with a hollow batch.
                pass

            if stage.parallelism is StageParallelism.MAIN_RANK_BROADCAST:
                payload = (
                    {k: batch.tensors.get(k) for k in stage.produces}
                    if self.ulysses.is_main
                    else None
                )
                payload = self.ulysses.broadcast_object(payload)
                if not self.ulysses.is_main and payload:
                    batch.tensors.update(payload)
                stage.verify_outputs(batch)

            self.last_stage_times[stage.name] = time.perf_counter() - t0

        return batch

    def stage_timing_report(self) -> str:
        """Human-readable per-stage timing from the last ``forward``."""
        if not self.last_stage_times:
            return "(no stages run yet)"
        total = sum(self.last_stage_times.values())
        lines = [f"{self.pipeline_name} total {total:.3f}s"]
        for name, secs in sorted(self.last_stage_times.items(), key=lambda kv: -kv[1]):
            share = (secs / total * 100) if total else 0.0
            lines.append(f"  {name:<40s} {secs:8.3f}s {share:5.1f}%")
        return "\n".join(lines)
