# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Per-request state for diffusion jobs.

``DiffusionJob`` plays the role ``atom.model_engine.sequence.Sequence`` plays on
the LLM side, but the shape of the state is different: there are no tokens, no
KV blocks and no incremental output. A job is a fixed amount of work whose only
observable progress is "step k of n", and whose result is a file.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import count
from typing import Any


class JobStatus(Enum):
    QUEUED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    # Client went away. The job keeps its slot until the current stage returns
    # -- a denoise step cannot be interrupted mid-kernel -- then the scheduler
    # drops it without producing output.
    ABORTED = auto()

    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.ABORTED)


@dataclass
class DiffusionJob:
    """One generation request.

    Timings are recorded as monotonic seconds by the scheduler and runner; they
    are left as ``None`` rather than 0.0 so "never happened" is distinguishable
    from "happened at t=0".
    """

    counter = count()

    prompt: str = ""
    task: str = ""
    """Pipeline-specific task discriminator (e.g. MiniMax-H3 t2va/fl2va/ref2va).

    The offline path must carry this explicitly: it is the field whose absence
    makes sglang's ``generate`` CLI unable to drive H3 at all.
    """
    conditions: list[dict[str, Any]] = field(default_factory=list)
    target: dict[str, Any] = field(default_factory=dict)
    num_inference_steps: int = 50
    seed: int | None = None

    job_id: str = ""
    status: JobStatus = JobStatus.QUEUED
    error: str | None = None

    arrive_time: float | None = None
    start_time: float | None = None
    finish_time: float | None = None

    current_step: int = 0
    total_steps: int = 0

    output_path: str | None = None
    peak_memory_mb: float | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            self.job_id = f"job-{next(DiffusionJob.counter)}"
        if self.num_inference_steps < 1:
            raise ValueError(
                f"num_inference_steps must be >= 1, got {self.num_inference_steps}"
            )
        self.total_steps = self.total_steps or self.num_inference_steps

    @property
    def is_finished(self) -> bool:
        return self.status.is_terminal()

    @property
    def progress(self) -> float:
        """Fraction of denoise steps done, in [0, 1]."""
        if self.total_steps <= 0:
            return 0.0
        return min(self.current_step / self.total_steps, 1.0)

    @property
    def elapsed(self) -> float | None:
        """Wall seconds from start to finish (or None if not yet started)."""
        if self.start_time is None:
            return None
        end = self.finish_time
        return None if end is None else end - self.start_time

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.error = error

    def to_dict(self) -> dict[str, Any]:
        """Serialisable view for the HTTP layer."""
        return {
            "id": self.job_id,
            "status": self.status.name.lower(),
            "progress": self.progress,
            "task": self.task,
            "error": self.error,
            "output_path": self.output_path,
            "inference_time_s": self.elapsed,
            "peak_memory_mb": self.peak_memory_mb,
        }
