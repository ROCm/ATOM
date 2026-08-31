# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Wire protocol between the diffusion API process and its GPU workers.

One job in, one result out, minutes apart -- so pickle over ZMQ costs nothing.

Two properties the engine depends on: every rank receives every request
(Ulysses is collective, so a job reaching 3 of 4 ranks hangs rather than
running slower), and only rank 0 replies (the others would be duplicates).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from atom.diffusion.request import DiffusionJob, JobStatus


class RequestType(str, Enum):
    """API process -> worker."""

    ADD = "add"
    ABORT = "abort"
    SHUTDOWN = "shutdown"


class OutputType(str, Enum):
    """Worker -> API process."""

    READY = "ready"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    DEAD = "dead"


@dataclass
class EngineRequest:
    """One instruction, broadcast to every rank of a replica."""

    type: RequestType
    job: DiffusionJob | None = None
    job_id: str = ""

    def __post_init__(self) -> None:
        if self.type is RequestType.ADD and self.job is None:
            raise ValueError("ADD requires a job")
        if self.type is RequestType.ABORT and not self.job_id:
            raise ValueError("ABORT requires a job_id")
        if self.job is not None and not self.job_id:
            self.job_id = self.job.job_id


@dataclass
class EngineOutput:
    """One report from a worker.

    ``rank`` is carried on every message so a failure can be attributed: an
    error raised on rank 3 during an all-to-all is a very different bug from
    the same error on rank 0.
    """

    type: OutputType
    rank: int = 0
    job_id: str = ""
    status: JobStatus | None = None
    current_step: int = 0
    total_steps: int = 0
    output_path: str | None = None
    error: str | None = None
    elapsed_s: float | None = None
    peak_memory_mb: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_job(
        cls, job: DiffusionJob, output_type: OutputType, *, rank: int = 0, **kwargs: Any
    ) -> "EngineOutput":
        return cls(
            type=output_type,
            rank=rank,
            job_id=job.job_id,
            status=job.status,
            current_step=job.current_step,
            total_steps=job.total_steps,
            output_path=job.output_path,
            error=job.error,
            elapsed_s=job.elapsed,
            peak_memory_mb=job.peak_memory_mb,
            **kwargs,
        )
