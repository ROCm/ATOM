# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Pipeline and stage abstractions for diffusion.

A pipeline is an ordered list of stages, each transforming a shared
:class:`DiffusionBatch` in place. The declaration with no LLM analogue is
*where* a stage runs: replicated on every rank (the denoise loop, collectively
parallel), rank 0 then broadcast (text encoding -- a 66 GB encoder on 8 ranks is
waste), or rank 0 alone (writing the file).

Stages declare ``requires``/``produces`` and are checked on both sides, so an
ordering mistake fails at the boundary rather than as a ``None`` several stages
later.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.ulysses import UlyssesGroup

if TYPE_CHECKING:  # pragma: no cover - typing only
    from atom.diffusion.request import DiffusionJob


class StageParallelism(Enum):
    """How a stage is distributed across the replica's ranks."""

    REPLICATED = auto()
    """Runs on every rank. Use for collectively-parallel work (denoise)."""

    MAIN_RANK_ONLY = auto()
    """Runs on rank 0 only; other ranks skip. Output is not shared, so only
    valid for terminal side effects such as writing a file."""

    MAIN_RANK_BROADCAST = auto()
    """Runs on rank 0, result broadcast to the rest. Use when the work is
    inherently serial but downstream stages need the output everywhere."""


@dataclass
class DiffusionBatch:
    """Mutable state threaded through a pipeline's stages.

    Deliberately a bag rather than a typed struct: stages are model-specific
    and the set of intermediates differs per pipeline. Keys are namespaced by
    convention (``"latents"``, ``"prompt_embeds"``, ``"packed_seq_params"``).
    """

    job: "DiffusionJob"
    tensors: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    is_warmup: bool = False

    def get(self, key: str, default: Any = None) -> Any:
        return self.tensors.get(key, default)

    def require(self, key: str) -> Any:
        """Fetch a required intermediate, failing loudly if a stage is missing.

        Stage ordering bugs otherwise surface as an opaque ``None`` several
        stages later.
        """
        if key not in self.tensors:
            raise KeyError(
                f"{key!r} not in batch; produced-so-far: " f"{sorted(self.tensors)}"
            )
        return self.tensors[key]

    def set(self, key: str, value: Any) -> None:
        self.tensors[key] = value


class PipelineStage(ABC):
    """Base class for a diffusion pipeline stage."""

    parallelism: StageParallelism = StageParallelism.REPLICATED

    #: Keys this stage expects in the batch before it runs.
    requires: tuple[str, ...] = ()
    #: Keys this stage adds to the batch.
    produces: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        return type(self).__name__

    def verify_inputs(self, batch: DiffusionBatch) -> None:
        """Check declared preconditions. Called by the pipeline before forward."""
        missing = [k for k in self.requires if k not in batch.tensors]
        if missing:
            raise KeyError(
                f"{self.name} requires {missing} which no earlier stage produced; "
                f"batch has {sorted(batch.tensors)}"
            )

    def verify_outputs(self, batch: DiffusionBatch) -> None:
        """Check the stage produced what it declared."""
        missing = [k for k in self.produces if k not in batch.tensors]
        if missing:
            raise KeyError(f"{self.name} declared but did not produce {missing}")

    @abstractmethod
    def forward(
        self, batch: DiffusionBatch, config: "DiffusionConfig"
    ) -> DiffusionBatch:
        """Transform the batch. Implementations mutate and return it."""

    def __call__(
        self, batch: DiffusionBatch, config: "DiffusionConfig"
    ) -> DiffusionBatch:
        self.verify_inputs(batch)
        out = self.forward(batch, config)
        if out is None:
            raise TypeError(f"{self.name}.forward returned None; must return the batch")
        self.verify_outputs(out)
        return out


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

        Left to subclasses: a diffusion pipeline is several networks with different
        loaders, so a generic "instantiate class_path and load a state dict" fits
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
