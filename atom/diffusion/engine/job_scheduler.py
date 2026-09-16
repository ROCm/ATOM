# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""FIFO admission and queueing for diffusion jobs.

Much simpler than the LLM scheduler on purpose: a job occupies the whole
replica for minutes, so there is nothing to pack and no KV to juggle. The only
decisions are whether to admit a job and which one runs next.
"""

import logging
import time
from collections import deque

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.request import DiffusionJob, JobStatus

logger = logging.getLogger(__name__)


class AdmissionError(RuntimeError):
    """Raised when a job cannot be queued (full, or wrong model variant)."""


class JobScheduler:
    """FIFO queue with a hard admission cap and a variant gate."""

    def __init__(self, config: DiffusionConfig) -> None:
        self.config = config
        self.waiting: deque[DiffusionJob] = deque()
        self.running: list[DiffusionJob] = []
        self.finished: dict[str, DiffusionJob] = {}

    # ------------------------------------------------------------------
    # admission
    # ------------------------------------------------------------------

    def add_job(self, job: DiffusionJob) -> DiffusionJob:
        """Queue a job, or raise ``AdmissionError``.

        Rejecting is deliberate: with minutes-long jobs an unbounded queue turns
        into an unbounded wait that the caller cannot see, so backpressure is
        more useful than acceptance.
        """
        depth = len(self.waiting) + len(self.running)
        if depth >= self.config.max_queued_jobs:
            raise AdmissionError(
                f"queue full ({depth}/{self.config.max_queued_jobs}); " f"retry later"
            )

        job.arrive_time = time.monotonic()
        job.status = JobStatus.QUEUED
        self.waiting.append(job)
        logger.info(
            "queued %s (task=%s, steps=%d); depth now %d",
            job.job_id,
            job.task or "-",
            job.num_inference_steps,
            depth + 1,
        )
        return job

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------

    @property
    def has_capacity(self) -> bool:
        return len(self.running) < self.config.max_concurrent_jobs

    def schedule(self) -> DiffusionJob | None:
        """Pop the next runnable job, or None.

        Aborted jobs are dropped here rather than run: unlike the LLM path there
        is no KV state to reclaim, so they need no cleanup pass.
        """
        while self.waiting:
            if not self.has_capacity:
                return None
            job = self.waiting.popleft()
            # Any terminal state, not just ABORTED: a rank dying FAILs queued
            # jobs too, and dispatching one would flip it back to RUNNING.
            if job.is_finished:
                self.finished[job.job_id] = job
                logger.info(
                    "dropping %s %s before start",
                    job.status.name.lower(),
                    job.job_id,
                )
                continue
            job.status = JobStatus.RUNNING
            job.start_time = time.monotonic()
            self.running.append(job)
            return job
        return None

    def complete(
        self,
        job: DiffusionJob,
        *,
        output_path: str | None = None,
        error: str | None = None,
    ) -> None:
        """Retire a running job as completed or failed."""
        if job in self.running:
            self.running.remove(job)
        job.finish_time = time.monotonic()
        if error is not None:
            job.mark_failed(error)
            logger.error("job %s failed: %s", job.job_id, error)
        elif job.status is not JobStatus.ABORTED:
            job.status = JobStatus.COMPLETED
            job.output_path = output_path
            job.current_step = job.total_steps
            logger.info("job %s completed in %.1fs", job.job_id, job.elapsed or 0.0)
        self.finished[job.job_id] = job

    def abort(self, job_id: str) -> bool:
        """Mark a job aborted. Returns False if it is unknown or already done.

        A running job keeps its slot until the current stage returns -- a
        denoise step cannot be interrupted mid-kernel.
        """
        for job in list(self.waiting) + self.running:
            if job.job_id == job_id:
                job.status = JobStatus.ABORTED
                return True
        return False

    def get(self, job_id: str) -> DiffusionJob | None:
        if job_id in self.finished:
            return self.finished[job_id]
        for job in list(self.waiting) + self.running:
            if job.job_id == job_id:
                return job
        return None

    def stats(self) -> dict[str, int]:
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "finished": len(self.finished),
        }
