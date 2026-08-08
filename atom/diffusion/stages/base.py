# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Stage abstraction for composed diffusion pipelines.

A diffusion pipeline is an ordered list of stages -- validate, encode text,
prepare latents, prepare timesteps, denoise, decode, present -- each of which
transforms a shared :class:`DiffusionBatch` in place and returns it.

The one thing a stage must declare that has no LLM analogue is *where* it runs.
Some stages are replicated on every rank (the denoise loop, which is
collectively parallel), some run only on rank 0 and broadcast (text encoding,
where replicating a 66 GB encoder's work on 8 ranks is pure waste), and some
run only on rank 0 with no broadcast at all (writing the output file).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from atom.diffusion.config import DiffusionConfig
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
